import pandas as pd
import numpy as np

from commons.load_data import load_train_test_dataframe
from commons.utils import process_missing_values, SMAPE
from TimesNet_codes.processing import process

from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils.plotting import plot_series

from TimesNet_codes.timesnet import Model

CUR_PATH = 'data'
TRAIN_DF_PATH = 'train.csv'
TEST_DF_PATH = 'test.csv'

class Config:
    def __init__(self):
        self.task_name = "short_term_forecast"
        self.features = "M"
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 128
        self.e_layers = 2
        self.d_layers = 1
        self.factor = 3
        self.enc_in = 11 #21
        self.dec_in = 11 #21
        self.c_out = 11 #21
        self.d_model = 16 #32
        self.d_ff = 16 #32
        self.top_k = 5
        self.num_kernels = 6
        self.des = "Exp"
        self.itr = 1
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.1

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
    
    config = Config()
    model = Model(config)
    print(model)  
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델의 총 파라미터 개수: {total_params}")  

    