import polars as pl
import pyarrow
import sys
from datetime import datetime
from src.configurator import Configurator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


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
    #XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05)
    methods = [
                MLPRegressor(hidden_layer_sizes=(128, 128, 128), activation='relu', solver='adam', 
                                alpha=0.0001, batch_size='auto', learning_rate='constant', 
                                learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True, 
                                random_state=None, tol=0.0001, verbose=False, warm_start=False, 
                                momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
                                validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                                n_iter_no_change=10),
               RandomForestRegressor(n_jobs=-1, random_state=1),
               KNeighborsRegressor(n_neighbors=3, n_jobs=-1),
               LinearRegression(n_jobs=-1)
            ]


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
        preds, trained_models = conf.kfold_prediction(df, method, num_months_per_fold=3, week=False)
        # 3) evaluatation
        score = model_evaluation_cv(preds, config, method, numTargets, target, key)
        #score = model_evaluation_cv(preds, config, 'LSTM', numTargets, target, key)


    # 
    # preds = conf.kfold_prediction(df, num_months_per_fold=3)
    # 3) evaluatation
    # score = model_evaluation_cv(preds, config, 'LSTM', numTargets, target, key)