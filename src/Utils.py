import polars as pl
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Model parameters")
    parser.add_argument("--pathCSV", help="csv path", type=str, required=True)
    parser.add_argument("--config", help="configuration", type=str, default="MULTI-STEP")
    parser.add_argument("--target", help="target name", type=str, required=True)
    parser.add_argument("--histFeatures", help="historical features to add", type=str, required=True)
    parser.add_argument("--key", help="unique id name", type=str, required=True)
    parser.add_argument("--dateCol", help="timestamp column name", type=str, default="date")
    parser.add_argument("--windowSize", help="length window size", type=int, default=12)
    parser.add_argument("--numTargets", help="number of targets time-step", type=int, default=12)
    parser.add_argument("--spatial", help="spatial method", type=str, default=None)
    parser.add_argument("--temporal", help="temporal method", type=str, default=None)


    args = parser.parse_args()
    return args.pathCSV, args.config, args.target, args.histFeatures, args.key, args.dateCol,\
           args.windowSize, args.numTargets, args.spatial, args.temporal


def model_evaluation(orig, pred, dates, configuration, model, start_pred_date,end_pred_date, key_val, n_targets, target, key):
    if not os.path.exists("output/predictions/"):
        os.makedirs("output/predictions/")

    score = pl.DataFrame()
    if ("SS-DT" in configuration) or ("SS-DTP" in configuration):

        score = score.with_columns(pl.Series(name="rse",
                values=[1 - r2_score(orig.filter(pl.col("date") == date).select(pl.col(target)),
                                     pred.filter(pl.col("date") == date).select(pl.col(target))) for date in dates.rows()]))
        score = score.with_columns(pl.Series(name="R2",
                                             values=[r2_score(orig.filter(pl.col("date") == date).select(pl.col(target)),
                                                 pred.filter(pl.col("date") == date).select(pl.col(target))) for
                                                     date in dates.rows()]))

        orig[:, (key, "date", target)].join(pred[:, (key, "date", target)], on=[key, 'date'])\
            .write_csv("output/predictions/" + configuration + "_" + type(model).__name__ + ".csv")
        print(score)

    elif 'MULTI-STEP' in configuration:
        score = score.with_columns(pl.Series(name="rmse", values=[mean_squared_error(orig[:, int(i)], pred[:, int(i)], squared=False) for i in np.arange(n_targets)]))
        score = score.with_columns(pl.Series(name="rse",
                                             values=[1 - r2_score(orig[:, int(i)], pred[:, int(i)]) for i in np.arange(n_targets)]))
        score = score.with_columns(pl.Series(name="r2",
                                             values=[r2_score(orig[:, int(i)], pred[:, int(i)]) for i in np.arange(n_targets)]))

        # write on file: key variable, real and predicted values
        real_pred = np.concatenate((list(map(lambda el: [el], np.array(key_val))), orig.to_numpy(), pred.to_numpy()),
                                   axis=1)

        np.savetxt("output/predictions/"+ configuration + "_" + type(model).__name__ + ".csv",
                  real_pred, fmt='%s', delimiter=',')

    media = score.mean().select(pl.all().apply(lambda x: np.round(x,3)))
    media = media.with_columns(start_date=start_pred_date, end_date=end_pred_date)
    media = media.with_columns(pl.lit(configuration).alias('conf'), pl.lit(str(model)).alias('method'))

    print(media)

    file = Path("output/" + type(model).__name__ + "_" + configuration.split("_")[0] + ".csv")
    with open(file, mode="ab") as f:
        media.write_csv(f, has_header=False)

    return score


"""
def create_csv(path, df, year):
    train = df[df.data.dt.year < year].dropna().drop(['data'], axis=1)
    test = df[df.data.dt.year == year].dropna().drop(['data'], axis=1)

    cols = train.columns
    col = [i for i in cols if "year" in i or "month" in i]
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)

    train.to_csv(path + '/train_' + str(year) + '.csv', index=False)
    test.to_csv(path + '/test_' + str(year) + '.csv', index=False)

    print("train.csv / test.csv created for year", year)"""
