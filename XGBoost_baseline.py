import pandas as pd
import numpy as np

import xgboost as xgb

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm as tq
import os

from commons.load_data import load_train_test_dataframe
from commons.utils import process_missing_values, SMAPE
from xgboost_codes.processing import process

from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils.plotting import plot_series
from xgboost import XGBRegressor

CUR_PATH = 'data'
TRAIN_DF_PATH = 'train.csv'
TEST_DF_PATH = 'test.csv'


if __name__=='__main__':
    train_df, test_df = load_train_test_dataframe(CUR_PATH,TRAIN_DF_PATH,TEST_DF_PATH)

    train_df = process_missing_values(train_df, method='ffill')
    test_df = process_missing_values(test_df, method='ffill')

    #한국어 변수명들 영어로 바꿔줌
    train_df.columns = ['num_date','build_idx','yyyymmdd hh','temp','rain_amount','wind','humid','일조','일사','target']
    test_df.columns = ['num_date','build_idx','yyyymmdd hh','temp','rain_amount','wind','humid']
    
    train_df = process(train_df, name = 'yyyymmdd hh', format ='%Y%m%d %H')
    test_df = process(test_df, name = 'yyyymmdd hh' , format = '%Y%m%d %H')
    
    train_df.drop(['num_date','yyyymmdd hh','일조','일사','hour'], axis = 1, inplace = True)
    test_df.drop(['num_date','yyyymmdd hh','hour'], axis = 1, inplace = True)
    
    
    """
    예시로 1개의 building에 대해 학습/예측 수행
    """
    y = train_df.loc[train_df.build_idx == 1, 'target']
    x = train_df.loc[train_df.build_idx == 1].drop(columns='target').iloc[:,1:]

    y_train, y_valid, x_train, x_valid = temporal_train_test_split(y = y, X = x, test_size = 168) # 24시간*7일 = 168

    print('train data shape\nx:{}, y:{}'.format(x_train.shape, y_train.shape))
    
    
    print("training... w/o validation data...")
    xgb_reg = XGBRegressor(n_estimators = 10000, seed=0, early_stopping_rounds=300)
    xgb_reg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)],
        verbose=False)
    print("training end!")
    
    
    print("predicting...")
    pred = xgb_reg.predict(x_valid)
    pred = pd.Series(pred)
    pred.index = np.arange(y_valid.index[0], y_valid.index[-1]+1)
    plot_series(y_train, y_valid, pd.Series(pred), markers=[',' , ',', ','])
    plt.savefig('xgboost_codes/XGBoost_prediction-about-validation.png')  # 그래프를 이미지 파일로 저장
    plt.close()
    print('SMAPE in validation: {}'.format(SMAPE(y_valid, pred)))
    
    
    print("inference...")
    x_test = test_df.loc[test_df.build_idx == 1, ].iloc[:,1:]
    x_test = x_test[x_train.columns]
    pred = xgb_reg.predict(x_test)
    
    plt.figure(figsize=(20,6))
    plt.plot(np.arange(2040) , train_df.loc[train_df.build_idx == 1, 'target'])
    plt.plot(np.arange(2040, 2040+168 ) , pred)
    plt.savefig('xgboost_codes/XGBoost_prediction-about-test.png')  # 그래프를 이미지 파일로 저장
    plt.close()
    print("inference done!")
    