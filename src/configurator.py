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

    def LISA(self, df_mod, threshold, features, distance_matrix):
        key_values = df_mod[self.key].unique().cast(pl.Utf8).to_list()
        matrix_km = pl.DataFrame(distance_matrix, schema=key_values)

        matrix_neighborood = matrix_km.select([pl.all().apply(lambda x: 1 if x < threshold else 0)])
        n_pa = np.array(matrix_neighborood.apply(lambda x: (1 / sum(x)))).flatten()
        transformed_matrix = pl.from_numpy(np.multiply(matrix_neighborood, n_pa), schema=matrix_neighborood.columns)
        transformed_matrix = transformed_matrix.with_columns(pl.Series(name="index", values=key_values))

        for feature in features:
            print("Compute LISA for", feature)
            normalization = pl.DataFrame()
            for c in key_values:
                des = df_mod.filter(pl.col(self.key) == int(c)).describe().filter(
                    pl.col("describe").is_in(["mean", "std"]))[:, feature]
                df_cliente = df_mod.filter(pl.col(self.key) == int(c))

                if des[1].item() != 0:
                    normalization = pl.concat(
                        [normalization, pl.DataFrame([((df_cliente[feature] - des[0].item()) / des[1].item()),
                                                      pl.Series(self.key, np.full(len(df_cliente), c)),
                                                      df_cliente[self.dateCol]])])
                else:
                    normalization = pl.concat([normalization, pl.DataFrame([df_cliente[feature],
                                                                            pl.Series(self.key,
                                                                                      np.full(len(df_cliente), c)),
                                                                            df_cliente[self.dateCol]])])

            normalization.columns = ["z", self.key, self.dateCol]
            res = []
            for row in normalization.rows():
                local_moran = self.Local_Moran(transformed_matrix, normalization, row)
                res.append(local_moran)
            df_mod = df_mod.with_columns(pl.Series(name="lisa_" + feature, values=res))
        return df_mod

    def add_features(self, df, key, window_size, features, type='lag'):
        """
        funzione che mi permette di aggiungere come features descrittive del dataset i valori di "features" dei "window_size" timestep precedenti al target
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

        df = df_.with_columns([pl.col(["lat", "lon"]).map(np.radians)])

        if ("SS-DT" in self.configuration) or ("SS-DTP" in self.configuration):
           df = self.add_features(df, self.key, self.windows_size, self.histFeatures, type='lag').drop_nulls()

        elif 'MULTI-STEP' in self.configuration:
            df = self.add_features(df, self.key, self.windows_size, self.histFeatures, "lag")
            df = self.add_features(df, self.key, self.n_targets, [self.target], "lead")

            #nel caso di granularità mensile
            #df = df.filter(pl.col(self.dateCol).dt.month() == 1).drop_nulls()
            df = df.filter(pl.col(self.dateCol).dt.hour().is_in(list(np.arange(0, 23, 4))) & (
                        pl.col(self.dateCol).dt.minute() == 0)).drop_nulls()

            """
            #Nel caso di granularità quart'oraria con 16 valori di target
            # filtro ogni 4 ore per ottenere il setting considerato
            df = df.filter(pl.col(self.dateCol).dt.hour().is_in(list(np.arange(0, 23, 4))) & (pl.col(self.dateCol).dt.minute() == 0)).drop_nulls()
            """

        if self.spatial_method in ["PCNM", "LISA"]:
            df = self.spatial(df, self.spatial_method)
            self.configuration = self.configuration + '_' + self.spatial_method

        print("configuration:", self.configuration)

        return df, self.configuration

    def prediction(self, df, start_pred_date, end_pred_date, method):
        if ("SS-DT" in self.configuration) or ("SS-DTP" in self.configuration):
            df_mod = df.clone()
            orig, pred, dates = self.self_learning_prediction(df_mod.sort(self.dateCol), start_pred_date, end_pred_date, method)
            return orig, pred, dates, None

        elif 'MULTI-STEP' in self.configuration:
            orig, pred, dates, key_val = self.MT_learning_prediction(df.sort(self.dateCol), start_pred_date, end_pred_date, method)
            return orig, pred, dates, key_val

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

        method.fit(X_train, y_train)
        y_pred = method.predict(X_test)

        return y_test, pl.DataFrame(y_pred), test[self.dateCol], test[self.key]


    def self_learning_prediction(self, df, start_pred_date, end_pred_date, method):

        # save original dataset to compare the predictions
        orig = df.filter((pl.col(self.dateCol) >= start_pred_date) & (pl.col(self.dateCol) <= end_pred_date)) if end_pred_date \
            else df.filter(pl.col(self.dateCol) >= start_pred_date)

        dates = orig.select(pl.col(self.dateCol).sort()).unique()
        df = df.sort(self.dateCol)

        # make prediction for each target time-step and update dataframe
        for date in dates.rows():
            train = df.filter(pl.col(self.dateCol) < date)

            X_train, y_train = train.drop([self.key, self.dateCol, self.target]), train.select(pl.col(self.target))

            test = df.filter((pl.col(self.dateCol) == date))

            X_test, y_test = test.drop([self.key, self.dateCol, self.target]), test.select(pl.col(self.target))

            """if type in ["cyclical_month", "both_temporal"]:
                X_train, X_test = cyclical_month(X_train, X_test)"""

            method.fit(X_train, y_train)
            y_pred = method.predict(X_test)

            # update dataframe with the prediction of the single value
            idx = df.select(pl.arg_where(pl.col(self.dateCol) == date)).to_series()
            df_mod = df.with_columns(df[self.target].set_at_idx(idx, pl.Series(y_pred.flatten())))

            dates_to_update = [d for d in dates.rows() if d > date]

            # update also the historical values
            for j, d in enumerate(dates_to_update[:self.windows_size]):
                idx = df.select(pl.arg_where(pl.col(self.dateCol) == d)).to_series()
                df = df_mod.with_columns(df[self.target +"_hist_"+ str(j + 1)].set_at_idx(idx, pl.Series(y_pred.flatten())))

        # save the new dataframe with predictions
        pred = df.filter((pl.col(self.dateCol) >= start_pred_date) & (pl.col(self.dateCol) <= end_pred_date)) if end_pred_date \
            else df.filter(pl.col(self.dateCol) >= start_pred_date)

        return orig, pred, dates



"""
def normal_year(df):
    col_year = [c for c in df.columns if "year" in c]
    normalizer = MinMaxScaler()
    for col in col_year:
        df.loc[:, col] = normalizer.fit_transform(df[col].values.reshape(-1, 1))
    return df


def calculate_rad(actual, target):
    circonferenza = 2 * np.pi
    r1 = 360 * actual / 12
    r2 = 360 * target / 12
    ris = abs(r1 - r2)
    if ris > 180:
        ris = 360 - ris
    return circonferenza - ((circonferenza / 12) * (ris / 30))


def cyclical_month(train, test):
    col_month = [c for c in train.columns if "month" in c]
    for col in col_month:
        train.loc[:, col] = train.loc[:, col].apply(lambda x: calculate_rad(x, test.month.values[0]))
        test.loc[:, col] = test.loc[:, col].apply(lambda x: calculate_rad(x, test.month.values[0]))

    return train, test
"""

