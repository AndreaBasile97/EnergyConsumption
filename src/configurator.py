import polars as pl
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from rpy2 import robjects as r
import rpy2.robjects.numpy2ri
from datetime import datetime, timedelta
from src.Utils import min_max_scale_polarsdf
from sklearn.metrics import r2_score
import tensorflow as tf
from src.lstm import create_model, compile_model, compute_metrics, train_and_predict
import pandas as pd

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
        if type == "LISA":
            lat_long = df[[self.key, "lat", "lon"]].unique().drop(self.key)

            matrix = haversine_distances(lat_long)
            # conversione in km
            matrix = matrix * 6371000 / 1000
            threshold = np.mean(matrix)

            col_target = [col for col in df.columns if self.target in col and "hist" not in col]
            # seleziono le colonne su cui calcolare LISA
            col = [c for c in df.columns if c not in col_target and self.target in c]
            df = self.LISA(df, float(threshold), col, matrix)

        elif type == "PCNM":
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

    def Local_Moran(self, transformed_matrix, normalization, row):
        neig = transformed_matrix.filter(pl.col("index") == row[1])[:, transformed_matrix.columns != row[1]]
        norm = normalization.filter((pl.col(self.dateCol) == row[2]) & (pl.col(self.key) != row[1]))
        res = sum((np.array(neig) * np.array(norm["z"])).ravel()) * row[0]
        return res



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
        df = df.with_columns([pl.col('date').dt.hour().alias('minutes')])
        # Since we got only 2019, this feature is useless
        df = df.drop('year')

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

    # def prediction(self, df, start_pred_date, end_pred_date, method):
    #     if 'MULTI-STEP' in self.configuration:
    #         orig, pred, dates, key_val = self.MT_learning_prediction(df.sort(self.dateCol), start_pred_date, end_pred_date, method)
    #         return orig, pred, dates, key_val
        
    def kfold_prediction(self, df, method, num_months_per_fold):
        if 'MULTI-STEP' in self.configuration:
            preds = self.MT_learning_prediction_cv(df.sort(self.dateCol), method, num_months_per_fold)
            return preds

    # def kfold_prediction(self, df):
    #     if 'MULTI-STEP' in self.configuration:
    #         # preds = self.MT_learning_prediction_cv(df.sort(self.dateCol), method, num_months_per_fold)
    #         preds = self.MT_LSTM_prediction_cv(df.sort(self.dateCol))
    #         return preds

    # def MT_learning_prediction(self, df, start_pred_date, end_pred_date, method):
    #     # - selection of target columns
    #     # - train_test split based on the prediction date
    #     # - prediction
    #     col_target = [col for col in df.columns if self.target in col and "hist" not in col]

    #     train = df.filter(pl.col(self.dateCol) < start_pred_date)
    #     X_train, y_train = train.select(pl.col(list(set(train.columns).difference(col_target)))).drop([self.key, self.dateCol]), \
    #                        train.select(pl.col(list(col_target)))

    #     test = df.filter((pl.col(self.dateCol) >= start_pred_date) & (pl.col(self.dateCol) <= end_pred_date)) if end_pred_date \
    #         else df.filter(pl.col(self.dateCol) >= start_pred_date)

    #     X_test, y_test = test.select(pl.col(list(set(test.columns).difference(col_target)))).drop([self.key, self.dateCol]), \
    #                        test.select(pl.col(list(col_target)))
        
    #     X_train.write_csv("ExampleDataframeTrain.csv")
    #     X_test.write_csv("ExampleDataframeTest.csv")

    #     method.fit(X_train, y_train)
    #     y_pred = method.predict(X_test)

    #     return y_test, pl.DataFrame(y_pred), test[self.dateCol], test[self.key]


    # Prediction using regressors
    def MT_learning_prediction_cv(self, df, method, num_months_per_fold = 3):
        
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
            print(f'last date of traning set: {train_last_date}')
            X_train, y_train = train.select(pl.col(list(set(train.columns).difference(col_target)))).drop([self.key, self.dateCol]),train.select(pl.col(list(col_target)))
            # Shift train_last_date by 28 hours and filter test
            train_last_date_shifted = train_last_date + timedelta(hours=28, minutes=15)

            # The test set must start from the next month AND 28h later the last date of the training set to avoid overlapping
            test = df.filter((pl.col('month') == window[2]) & (pl.col('date') > train_last_date_shifted))
            print(f'first date of test set: {test[self.dateCol].min()}')
            try:
                X_test, y_test = test.select(pl.col(list(set(test.columns).difference(col_target)))).drop([self.key, self.dateCol]),test.select(pl.col(list(col_target)))
                method.fit(X_train, y_train)
                y_pred = method.predict(X_test)
                prediction_results.append((y_test, pl.DataFrame(y_pred), test[self.dateCol], test[self.key], window))
                trained_models[i] = method  # save the trained model for this fold
            except:
                pass
        
        return prediction_results, trained_models
    

    # def MT_LSTM_prediction_cv(self, df, num_months_per_fold = 3):
        
    #     print('LSTM starting learning...')

    #     # create a pandas DataFrame to save the scores
    #     scores_df = pd.DataFrame(columns=['RMSE', 'RSE', 'R2'])

    #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    #     # define the model
    #     model = tf.keras.models.Sequential([
    #         tf.keras.layers.LSTM(50, activation='relu', input_shape=(1, 204), return_sequences=True),
    #         tf.keras.layers.LSTM(80, activation='relu', return_sequences=True),
    #         tf.keras.layers.LSTM(80, activation='relu', return_sequences=True),
    #         tf.keras.layers.LSTM(50, activation='relu'),
    #         tf.keras.layers.Dense(16)
    #     ])

        
    #     def RSE(y_true, y_pred):
    #         true_mean = tf.reduce_mean(y_true)
    #         squared_error_num = tf.reduce_sum(tf.square(y_true - y_pred))
    #         squared_error_den = tf.reduce_sum(tf.square(y_true - true_mean))
    #         rse_loss = squared_error_num / squared_error_den
    #         return rse_loss
        
    #     def r2(y_true, y_pred):
    #         SS_res =  tf.reduce_sum(tf.square(y_true - y_pred)) 
    #         SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) 
    #         return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))

    #     # compile the model
    #     model.compile(optimizer=optimizer, loss='mse', metrics=[RSE, r2, tf.keras.metrics.RootMeanSquaredError()])


    #     prediction_results = []

    #     months = np.unique(df['month'].to_numpy())
    #     assert num_months_per_fold > 1 and num_months_per_fold <= len(months), f'The number of months {num_months_per_fold} is not valid since it must be at least 2 and max {len(months)}'

    #     # Creating sliding windows
    #     month_windows = [months[i:i + num_months_per_fold] for i in range(len(months) - num_months_per_fold + 1)]

    #     print(month_windows)

    #     col_target = [col for col in df.columns if self.target in col and "hist" not in col]
    #     for window in month_windows:

    #         train = df.filter((pl.col('month') != window[2]) & (pl.col('month').is_in(window)))
    #         # Save the last date of the train set
    #         train_last_date = train[self.dateCol].max()
    #         print(f'last date of traning set: {train_last_date}')
    #         X_train, y_train = train.select(pl.col(list(set(train.columns).difference(col_target)))).drop([self.key, self.dateCol]),train.select(pl.col(list(col_target)))
            
    #         X_train_array = min_max_scale_polarsdf(X_train)
    #         X_train_reshaped = X_train_array.reshape((X_train_array.shape[0], 1, X_train_array.shape[1])) # shape should be (143500, 1, 204)

    #         # Shift train_last_date by 28 hours and filter test
    #         train_last_date_shifted = train_last_date + timedelta(hours=28, minutes=15)

    #         # The test set must start from the next month AND 28h later the last date of the training set to avoid overlapping
    #         test = df.filter((pl.col('month') == window[2]) & (pl.col('date') > train_last_date_shifted))
    #         print(f'first date of test set: {test[self.dateCol].min()}')
    #         X_test, y_test = test.select(pl.col(list(set(test.columns).difference(col_target)))).drop([self.key, self.dateCol]),test.select(pl.col(list(col_target)))


    #         X_test_array = min_max_scale_polarsdf(X_test)
    #         X_test_reshaped = X_test_array.reshape((X_test_array.shape[0], 1, X_test_array.shape[1]))            
            
    #         model.fit(X_train_reshaped, y_train.to_numpy(), epochs=100, batch_size=32, verbose=2)
    #         y_pred = model.predict(X_test_reshaped)

    #         # Compute metrics on the test set
    #         rmse = tf.keras.metrics.RootMeanSquaredError()
    #         rmse.update_state(y_test.to_numpy(), y_pred)
    #         rse = RSE(y_test.to_numpy(), y_pred)
    #         r2 = r2(y_test.to_numpy(), y_pred)

    #         # Save metrics to scores_df
    #         scores_df = scores_df.append({'RMSE': rmse.result().numpy(),
    #                                     'RSE': rse.numpy(),
    #                                     'R2': r2.numpy()}, ignore_index=True)

    #         print('prediction concluded')

    #     # Save scores_df and y_pred to CSV
    #     scores_df.to_csv('scores.csv', index=False)
    #     pd.DataFrame(y_pred).to_csv('y_pred.csv', index=False)

    #     return prediction_results

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
            print(f'last date of traning set: {train_last_date}')
            X_train, y_train = train.select(pl.col(list(set(train.columns).difference(col_target)))).drop([self.key, self.dateCol]),train.select(pl.col(list(col_target)))
            
            X_train_array = min_max_scale_polarsdf(X_train)
            X_train_reshaped = X_train_array.reshape((X_train_array.shape[0], 1, X_train_array.shape[1])) # shape should be (143500, 1, 204)

            # Shift train_last_date by 28 hours and filter test
            # train_last_date_shifted = train_last_date + timedelta(hours=28, minutes=15)

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
            except:
                pass
            
            print('prediction concluded')

        # Save scores_df and y_pred to CSV
        scores_df.to_csv('scores.csv', index=False)
        pd.DataFrame(y_pred).to_csv('y_pred.csv', index=False)

        return prediction_results