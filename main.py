import polars as pl
import pyarrow
import sys
from datetime import datetime
from src.configurator import Configurator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.Utils import get_args,model_evaluation, model_evaluation_cv
import numpy as np
#python main.py --pathCSV energy.csv --config MULTI-STEP --target energy_consumption --histFeatures "energy_consumption" --key customer --dateCol date --windowSize 12 --numTargets 12 --spatial None --temporal None
#python main.py --pathCSV "energy_consumption.csv" --config MULTI-STEP --target grid --histFeatures "grid,solar" --key house --dateCol date --windowSize 16 --numTargets 16
# Per SS-DTP in histFeatures avremo "energy_consumption,year,month"

if __name__ == '__main__':

    # get args from the prompt
    pathCSV, config, target, histFeatures, key, dateCol, windowSize, numTargets, spatial, temporal = get_args()
    histFeatures = [str(item) for item in histFeatures.split(',')]
    print(pathCSV, config, target, histFeatures, key, dateCol, windowSize, numTargets)

    # select the methods
    methods = [RandomForestRegressor(n_jobs=-1, random_state=1)]
    # methods = [RandomForestRegressor(n_jobs=-1, random_state=1),
    #            KNeighborsRegressor(n_neighbors=3, n_jobs=-1),
    #            LinearRegression(n_jobs=-1)]

    # set the configurator
    conf = Configurator(configuration=config, windows_size=windowSize, n_targets=numTargets,target=target,key=key,
                        dateCol=dateCol, histFeatures=histFeatures, spatial=spatial, temporal=temporal)

    dataset = pl.read_csv(pathCSV, try_parse_dates=True, use_pyarrow=True)

    # add temporal features (year and month) as numerical 
    dataset = dataset.with_columns(pl.col("date").dt.year().alias("year"))
    dataset = dataset.with_columns(pl.col("date").dt.month().alias("month"))

    # 1) Transformation of the dataset
    df, config = conf.transform(dataset)


    for method in methods:
        #2) predictions
        preds = conf.kfold_prediction(df, method, num_months_per_fold=3)
        # 3) evaluatation
        score = model_evaluation_cv(preds, config, method, numTargets, target, key)
