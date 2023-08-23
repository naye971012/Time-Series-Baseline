import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import tqdm as tq
import os

from commons.load_data import load_train_test_dataframe
from commons.utils import process_missing_values, SMAPE
from xgboost_codes.processing import process


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