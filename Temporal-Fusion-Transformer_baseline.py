import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import tqdm as tq
import os

from commons.load_data import load_train_test_dataframe
from commons.utils import process_missing_values, SMAPE
from TFT_codes.processing import process
from TFT_codes.tft import *

from lightning.pytorch.loggers import TensorBoardLogger

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
    
    
    
    PARAMS = {
            'LOGDIR' : "TFT_codes/model_logs",   # pytorch_forecasting requirs logger
            'CKPTROOT' : 'TFT_codes/model_ckpts',
            'BATCH_SIZE' : 12,
            'MAX_ENCODER_LENGTH' : 256,
            'epoch' : 8,
            'gradient_clip_val': 0.9,
            'hidden_size': 30, 
            'dropout': 0.1,
            'hidden_continuous_size': 15,
            'attention_head_size': 4, 
            'learning_rate': 0.001,
    }
    
    tr_ds , va_ds = get_data(validate=True , train_df=train_df , PARAMS=PARAMS)
    va_loader = va_ds.to_dataloader(batch_size=5, train=False)    
    
    fit(seed=0, tr_ds = tr_ds, va_loader = va_loader, cur_loss=MAPE() , PARAMS=PARAMS)

    model_loc = PARAMS['LOGDIR'] + "/lightning_logs" 
            
    file_list = os.listdir(model_loc) #version_0 ~ version_n 이 몇개 있는지 확인하고
    file_list = sorted(file_list, key=lambda x: int(x.split("_")[1]))[-1] #가장 최근 version 접근
            
    model_path = os.path.join(model_loc, file_list) #가장 최근의 version
    model_path = model_path + "/checkpoints" #checkpoints 접근
    model_path = os.path.join(model_path,os.listdir(model_path)[0]) #그 안에서 첫 번째 모델을 가져옴
            
    validation_pred = validation(model_path,train_df,test_df,PARAMS)