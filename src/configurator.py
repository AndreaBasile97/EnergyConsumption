import polars as pl
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from rpy2 import robjects as r
import rpy2.robjects.numpy2ri
from datetime import datetime, timedelta
from src.Utils import min_max_scale_polarsdf
from sklearn.metrics import r2_score
import tensorflow as tf
import pandas as pd
import pickle
import torch
from src.LSTM import *
from collections import defaultdict
import networkx as nx
from tensorflow import keras

rpy2.robjects.numpy2ri.activate()

class Configurator:
    def __init__(self, configuration,windows_size,n_targets, target, key, histFeatures, dateCol, spatial=None, temporal=None):
        self.configuration = configuration
        self.windows_size = int(windows_size)
        self.n_targets = int(n_targets)
        self.target = target
        self.key = key
        self.histFeatures = histFeatures
        self.dateCol = dateCol
        self.spatial_method = spatial
        self.temporal_method = temporal

    def spatial(self, df, type):
        print('Applying PCNM')
        if type == "PCNM":
            r.r.source('src/pcnm.R')
            cabina = df.select(["lat", "lon"]).unique()
            matrix = haversine_distances(cabina)
            threshold = np.median(matrix)
            # 0.150 metri threshold
            # res = r.r['calculate_pcnm'](matrix, (0.15*1000)/6371000)
            res = r.r['calculate_pcnm'](matrix, threshold)

            # extraction of 15 eigenvectors from PCNM
            eigenvectors = pl.DataFrame(res[0], schema=['pcnm_' + str(elem) for elem in range(res[0].shape[1])])[:, :15]
            # add eigenvectors as features
            df = df.join(pl.concat([cabina, eigenvectors], how="horizontal"), on=['lat', 'lon'])

        return df


    def add_features(self, df, key, window_size, features, type='lag'):
        """
        funzione che mi permette di aggiungere, come features descrittive del dataset, i valori di "features" dei "window_size" timestep precedenti al target
        key: rappresenta la feature sulla quale devo raggruppare
        window_size: rappresenta il numero di timestep precedenti che devo considerare come feature
        features: rappresenta le features di cui vogliamo prendere i valori precedenti
        Type = "lag"  mi permette di prendere la misurazione precedente invece Type = "lead" mi permette di prendere la misurazione successiva
        """
        df = df.sort([key, self.dateCol])

        if type == 'lag':
            for i in np.arange(1, window_size + 1):
                for feature in features:
                    df = df.with_columns(pl.col(feature).shift(i).over(key).alias(feature + '_hist_' + str(i)))
        if type == 'lead':
            for i in np.arange(1, window_size):
                for feature in features:
                    df = df.with_columns(pl.col(feature).shift(-i).over(key).alias(feature + '_' + str(i + 1)))
        return df


    def transform(self, df_):
        print(df_.dtypes)

         # Convert 'city' column to category and then to ordinal numbers
        df = df_.with_columns([pl.col('city').cast(pl.Categorical).cast(pl.UInt32)])

        df = df.with_columns([pl.col(["lat", "lon"]).map(np.radians)])
        # In this scenario we consider the hour and minutes as an important feature
        df = df.with_columns([pl.col('date').dt.hour().alias('hour')])
        df = df.with_columns([pl.col('date').dt.minute().alias('minutes')])
        # Since we got only 2019, this feature is useless
        df = df.drop('year')
        df.to_pandas().to_csv('energy_consumption_transformed.csv', index=False)
        #Nel caso di granularità quart'oraria con 16 valori di target
        # filtro ogni 4 ore per ottenere il setting considerato
        # df = df.filter(pl.col(self.dateCol).dt.hour().is_in(list(np.arange(0, 23, 4))) & (pl.col(self.dateCol).dt.minute() == 0)).drop_nulls()

        if 'MULTI-STEP' in self.configuration:
            df = self.add_features(df, self.key, self.windows_size, self.histFeatures, "lag")
            df = self.add_features(df, self.key, self.n_targets, [self.target], "lead").drop_nulls()

        if self.spatial_method in ["PCNM", "LISA"]:
            df = self.spatial(df, self.spatial_method)
            self.configuration = self.configuration + '_' + self.spatial_method

        print("configuration:", self.configuration)
        
        return df, self.configuration

        
    def kfold_prediction(self, df, method, num_months_per_fold, week=False):
        if 'MULTI-STEP' in self.configuration and week:
            preds = self.MT_learning_prediction_week(df.sort(self.dateCol), method)
            return preds
        elif 'MULTI-STEP' in self.configuration:
            preds = self.MT_learning_prediction_cv(df.sort(self.dateCol), method, num_months_per_fold)
            return preds         


    def MT_learning_prediction_week(self, df, method):
        prediction_results = []
        trained_models = {}

        min_date = df[self.dateCol].min()
        max_date = df[self.dateCol].max()

        train_end_date = min_date + timedelta(days=30)
        test_start_date = train_end_date + timedelta(hours=28, minutes=15)
        test_end_date = test_start_date + timedelta(days=7)

        while test_end_date <= max_date:
            train = df.filter((pl.col(self.dateCol) >= min_date) & (pl.col(self.dateCol) < train_end_date))
            test = df.filter((pl.col(self.dateCol) >= test_start_date) & (pl.col(self.dateCol) < test_end_date))

            print(f'Training date range: {min_date} - {train_end_date}')
            print(f'Testing date range: {test_start_date} - {test_end_date}')
            
            col_target = [col for col in df.columns if self.target in col and "hist" not in col]
            
            try:
                X_train, y_train = train.select(pl.col(list(set(train.columns).difference(col_target)))).drop([self.key, self.dateCol]),train.select(pl.col(list(col_target)))
                X_test, y_test = test.select(pl.col(list(set(test.columns).difference(col_target)))).drop([self.key, self.dateCol]),test.select(pl.col(list(col_target)))

                method.fit(X_train, y_train)
                y_pred = method.predict(X_test)

                # Convert datetime to string format for filename
                filename = f"{test_start_date.strftime('%Y_%m_%d_%H_%M_%S')}_to_{test_end_date.strftime('%Y_%m_%d_%H_%M_%S')}"

                prediction_results.append((y_test, pl.DataFrame(y_pred), test[self.dateCol], test[self.key], filename))
                
                trained_models[train_end_date] = method  # save the trained model for this fold
            except Exception as e:
                print(f"An error occurred: {e}")
                pass

            # Update the start and end dates for the next loop iteration
            min_date = min_date + timedelta(days=7)
            train_end_date = min_date + timedelta(days=30)
            test_start_date = train_end_date + timedelta(hours=28, minutes=15)
            test_end_date = test_start_date + timedelta(days=7)
            
            # Save the trained model
        with open(method + '.pkl', 'wb') as f:
            pickle.dump(method, f)

        return prediction_results, trained_models
    

    
    # Prediction using regressors
    def MT_learning_prediction_cv(self, df, method, num_months_per_fold = 3):
        print('monthly predictions')
        prediction_results = []
        trained_models = {}

        months = np.unique(df['month'].to_numpy())
        assert num_months_per_fold > 1 and num_months_per_fold <= len(months), f'The number of months {num_months_per_fold} is not valid since it must be at least 2 and max {len(months)}'

        # Creating sliding windows
        month_windows = [months[i:i + num_months_per_fold] for i in range(len(months) - num_months_per_fold + 1)]

        print(month_windows)

        col_target = [col for col in df.columns if self.target in col and "hist" not in col]
        for i, window in enumerate(month_windows):

            train = df.filter((pl.col('month') != window[2]) & (pl.col('month').is_in(window)))
            # Save the last date of the train set
            train_last_date = train[self.dateCol].max()

            # Change: Filter the data that comes before 20:15:00 of the latest date.
            train = train.filter((pl.col(self.dateCol) < train_last_date.replace(hour=20, minute=15, second=0)))
            print(f'first date of traning set: {train[self.dateCol].min()}')
            print(f'last date of traning set: {train[self.dateCol].max()}')

            X_train, y_train = train.select(pl.col(list(set(train.columns).difference(col_target)))).drop([self.key, self.dateCol]),train.select(pl.col(list(col_target)))

            # The test set must start from the next month
            test = df.filter((pl.col('month') == window[2]))

            # Print the first and last date of the test set
            test_first_date = test[self.dateCol].min()
            test_last_date = test[self.dateCol].max()
            print(f'first date of test set: {test_first_date}')
            print(f'last date of test set: {test_last_date}')

            try:
                X_test, y_test = test.select(pl.col(list(set(test.columns).difference(col_target)))).drop([self.key, self.dateCol]),test.select(pl.col(list(col_target)))
                method.fit(X_train, y_train)
                y_pred = method.predict(X_test)
                prediction_results.append((y_test, pl.DataFrame(y_pred), test[self.dateCol], test[self.key], window))
                trained_models[i] = method  # save the trained model for this fold
                #with open(f'model_{method}.pkl', 'wb') as f:  # Save model to disk
                #    pickle.dump(method, f)
                model_name = type(method).__name__
                pickle.dump(method, open(f"{model_name}_regressor.pkl", "wb"))

            except Exception as e:
                print(e)
                pass
        
        return prediction_results, trained_models


    def MT_LSTM_prediction_cv(self, df, num_months_per_fold = 3):

        print('LSTM starting learning...')

        # create a pandas DataFrame to save the scores
        scores_df = pd.DataFrame(columns=['RMSE', 'RSE', 'R2'])

        # Create and compile the model
        model = create_model()
        model = compile_model(model)

        prediction_results = []

        months = np.unique(df['month'].to_numpy())
        assert num_months_per_fold > 1 and num_months_per_fold <= len(months), f'The number of months {num_months_per_fold} is not valid since it must be at least 2 and max {len(months)}'

        # Creating sliding windows
        month_windows = [months[i:i + num_months_per_fold] for i in range(len(months) - num_months_per_fold + 1)]

        print(month_windows)

        col_target = [col for col in df.columns if self.target in col and "hist" not in col]
        for window in month_windows:

            train = df.filter((pl.col('month') != window[2]) & (pl.col('month').is_in(window)))
            # Save the last date of the train set
            train_last_date = train[self.dateCol].max()

            # Change: Filter the data that comes before 20:15:00 of the latest date.
            # train = train.filter((pl.col(self.dateCol) < train_last_date.replace(hour=20, minute=15, second=0)))
            print(f'last date of traning set: {train[self.dateCol].max()}')

            X_train, y_train = train.select(pl.col(list(set(train.columns).difference(col_target)))).drop([self.key, self.dateCol]),train.select(pl.col(list(col_target)))

            X_train_array = min_max_scale_polarsdf(X_train)
            X_train_reshaped = X_train_array.reshape((X_train_array.shape[0], 1, X_train_array.shape[1])) # shape should be (143500, 1, 204)


            try:
                # The test set must start from the next month AND 28h later the last date of the training set to avoid overlapping
                # test = df.filter((pl.col('month') == window[2]) & (pl.col('date') > train_last_date_shifted))
                test = df.filter((pl.col('month') == window[2]))
                print(f'first date of test set: {test[self.dateCol].min()}')
                X_test, y_test = test.select(pl.col(list(set(test.columns).difference(col_target)))).drop([self.key, self.dateCol]),test.select(pl.col(list(col_target)))

                X_test_array = min_max_scale_polarsdf(X_test)
                X_test_reshaped = X_test_array.reshape((X_test_array.shape[0], 1, X_test_array.shape[1]))            

                y_pred = train_and_predict(model, X_train_reshaped, y_train, X_test_reshaped)

                prediction_results.append(y_pred)

                # Compute metrics on the test set
                rmse, rse, r2 = compute_metrics(y_test, y_pred)

                print(rmse, rse, r2)

                # Save metrics to scores_df
                new_row = pd.DataFrame({'RMSE': [rmse],
                                        'RSE': [rse],
                                        'R2': [r2]})
                scores_df = pd.concat([scores_df, new_row], ignore_index=True)
            except Exception as e:
                print(e)
                break

            print('prediction concluded')

        # Save scores_df and y_pred to CSV
        scores_df.to_csv('scores.csv', index=False)
        pd.DataFrame(y_pred).to_csv('y_pred.csv', index=False)

        return prediction_results