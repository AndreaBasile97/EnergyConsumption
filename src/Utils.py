import polars as pl
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import argparse
import os
import calendar

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
    parser.add_argument("--spatial", help="spatial method", type=str, default="PCNM")
    parser.add_argument("--temporal", help="temporal method", type=str, default=None)


    args = parser.parse_args()
    return args.pathCSV, args.config, args.target, args.histFeatures, args.key, args.dateCol,\
           args.windowSize, args.numTargets, args.spatial, args.temporal


def model_evaluation(orig, pred, dates, configuration, model, start_pred_date,end_pred_date, key_val, n_targets, target, key):
    if not os.path.exists("output/predictions/"):
        os.makedirs("output/predictions/")

    score = pl.DataFrame()

    if 'MULTI-STEP' in configuration:
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


def model_evaluation_cv(results, configuration, model, n_targets, target, key):
    if not os.path.exists(f"output/predictions_{len(results)}fold/"):
        os.makedirs(f"output/predictions_{len(results)}fold/")

    scores = []
    windows = []
    for result in results:
        orig, pred, dates, key_val, window = result

        windows.append(window)
        score = pl.DataFrame()

        if 'MULTI-STEP' in configuration:
            score = score.with_columns(pl.Series(name="rmse", values=[mean_squared_error(orig[:, int(i)], pred[:, int(i)], squared=False) for i in np.arange(n_targets)]))
            score = score.with_columns(pl.Series(name="rse",values=[1 - r2_score(orig[:, int(i)], pred[:, int(i)]) for i in np.arange(n_targets)]))
            score = score.with_columns(pl.Series(name="r2",values=[r2_score(orig[:, int(i)], pred[:, int(i)]) for i in np.arange(n_targets)]))
            # write on file: key variable, real and predicted values
            real_pred = np.concatenate((list(map(lambda el: [el], np.array(key_val))), orig.to_numpy(), pred.to_numpy()),axis=1)

            np.savetxt(f"output/predictions_{len(results)}fold/"+ configuration + "_" + type(model).__name__ + "kfold.csv",
                      real_pred, fmt='%s', delimiter=',')
        scores.append(score)

        # Saving the scores for a single fold
        media = score.mean().select(pl.all().apply(lambda x: np.round(x,3)))
        media = media.with_columns(window=pl.lit(str(window)))
        media = media.with_columns(pl.lit(configuration).alias('conf'), pl.lit(str(model)).alias('method'))

        file = Path("output/" + type(model).__name__ + "_" + configuration.split("_")[0] + "_" + str(window) + ".csv")
        with open(file, mode="ab") as f:
            media.write_csv(f, has_header=False)


    # average scores across all results of the cross_fold
    avg_score = pl.concat(scores).mean()

    avg_score = avg_score.select(pl.all().apply(lambda x: np.round(x,3)))
    avg_score = avg_score.with_columns(window=pl.lit(str(windows)))
    avg_score = avg_score.with_columns(pl.lit(configuration).alias('conf'), pl.lit(str(model)).alias('method'))

    print(avg_score)

    file = Path("output/" + type(model).__name__ + "_" + configuration.split("_")[0] + "_" + "k_fold_mean" + ".csv")
    with open(file, mode="ab") as f:
        avg_score.write_csv(f, has_header=False)

    return avg_score
