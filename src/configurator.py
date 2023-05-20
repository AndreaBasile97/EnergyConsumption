import polars as pl
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from rpy2 import robjects as r
import rpy2.robjects.numpy2ri
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

        if ("SS-DT" in self.configuration) or ("SS-DTP" in self.configuration):
           df = self.add_features(df, self.key, self.windows_size, self.histFeatures, type='lag').drop_nulls()

        elif 'MULTI-STEP' in self.configuration:
            df = self.add_features(df, self.key, self.windows_size, self.histFeatures, "lag")
            df = self.add_features(df, self.key, self.n_targets, [self.target], "lead").drop_nulls()

        if self.spatial_method in ["PCNM", "LISA"]:
            df = self.spatial(df, self.spatial_method)
            self.configuration = self.configuration + '_' + self.spatial_method

        print("configuration:", self.configuration)

        return df, self.configuration

    def prediction(self, df, start_pred_date, end_pred_date, method):
        if 'MULTI-STEP' in self.configuration:
            orig, pred, dates, key_val = self.MT_learning_prediction(df.sort(self.dateCol), start_pred_date, end_pred_date, method)
            return orig, pred, dates, key_val
        
    def kfold_prediction(self, df, method, num_months_per_fold):
        if 'MULTI-STEP' in self.configuration:
            preds = self.MT_learning_prediction_cv(df.sort(self.dateCol), method, num_months_per_fold)
            return preds

    def MT_learning_prediction(self, df, start_pred_date, end_pred_date, method):
        # - selection of target columns
        # - train_test split based on the prediction date
        # - prediction
        col_target = [col for col in df.columns if self.target in col and "hist" not in col]

        train = df.filter(pl.col(self.dateCol) < start_pred_date)
        X_train, y_train = train.select(pl.col(list(set(train.columns).difference(col_target)))).drop([self.key, self.dateCol]), \
                           train.select(pl.col(list(col_target)))

        test = df.filter((pl.col(self.dateCol) >= start_pred_date) & (pl.col(self.dateCol) <= end_pred_date)) if end_pred_date \
            else df.filter(pl.col(self.dateCol) >= start_pred_date)

        X_test, y_test = test.select(pl.col(list(set(test.columns).difference(col_target)))).drop([self.key, self.dateCol]), \
                           test.select(pl.col(list(col_target)))
        
        X_train.write_csv("ExampleDataframeTrain.csv")
        X_test.write_csv("ExampleDataframeTest.csv")

        method.fit(X_train, y_train)
        y_pred = method.predict(X_test)

        return y_test, pl.DataFrame(y_pred), test[self.dateCol], test[self.key]



    def MT_learning_prediction_cv(self, df, method, num_months_per_fold = 3):
        
        prediction_results = []

        months = np.unique(df['month'].to_numpy())
        assert num_months_per_fold > 1 and num_months_per_fold <= len(months), f'The number of months {num_months_per_fold} is not valid since it must be at least 2 and max {len(months)}'

        # Creating sliding windows
        month_windows = [months[i:i + num_months_per_fold] for i in range(len(months) - num_months_per_fold + 1)]

        print(month_windows)

        col_target = [col for col in df.columns if self.target in col and "hist" not in col]
        for window in month_windows:
            train = df.filter((pl.col('month') != window[2]) & (pl.col('month').is_in(window)))
            X_train, y_train = train.select(pl.col(list(set(train.columns).difference(col_target)))).drop([self.key, self.dateCol]),train.select(pl.col(list(col_target)))
            
            test = df.filter(pl.col('month') == window[2])
            X_test, y_test = test.select(pl.col(list(set(test.columns).difference(col_target)))).drop([self.key, self.dateCol]),test.select(pl.col(list(col_target)))

            method.fit(X_train, y_train)
            y_pred = method.predict(X_test)
            prediction_results.append((y_test, pl.DataFrame(y_pred), test[self.dateCol], test[self.key], window))

        X_train.write_csv("ExampleDataframeTrain.csv")
        X_test.write_csv("ExampleDataframeTest.csv")


        return prediction_results

