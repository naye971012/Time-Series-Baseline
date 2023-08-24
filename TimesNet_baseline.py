import pandas as pd
import numpy as np
import torch

from commons.load_data import load_train_test_dataframe
from commons.utils import process_missing_values, SMAPE

from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils.plotting import plot_series

from TimesNet_codes.model.timesnet import Model
from TimesNet_codes.dataset_dataloader import CustomDataset
from TimesNet_codes.processing import process
from TimesNet_codes.train_valid_eval import train

from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

CUR_PATH = 'data'
TRAIN_DF_PATH = 'train.csv'
TEST_DF_PATH = 'test.csv'

class Config:
    def __init__(self):
        self.task_name = "short_term_forecast"
        self.features = "M"
        self.seq_len = 48 #
        self.label_len = 48 #
        self.pred_len = 128 #
        self.e_layers = 2
        self.d_layers = 1
        self.factor = 3
        self.enc_in = 11 #
        self.dec_in = 11 #
        self.c_out = 11 #
        self.d_model = 16 
        self.d_ff = 16 
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
    
    
    ###################################################################################
    config = Config()
    model = Model(config)
    print(model)  
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n모델의 총 파라미터 개수: {total_params}")  

    ############################# model testing ########################################
    x = torch.randn((1,config.seq_len,config.enc_in))
    y = torch.randn((1,config.seq_len-config.label_len+config.pred_len,config.dec_in))
    print(f"model input shape : {x.size()}")
    print(f"model label shape : {y.size()}")
    
    #모델이 학습할 때 정답을 보고 학습하면 안되니까 정답부분 마스킹
    dec_inp = torch.zeros_like(y[:, -config.pred_len:, :]).float()
    dec_inp = torch.cat([y[:, :config.label_len, :], dec_inp], dim=1).float()
    
    output = model( x , None, dec_inp , None )
    assert output.size()==y.size()
    print(f"model output shape : {output.size()}")
    ############################# model testing ########################################
    
    print()
    print(f"model variables: {train_df.columns.values}")
    print(f"model target index : {list(train_df.columns).index('target')}")
    
    
    ################ 예시로 build_idx=1 에 대한 예측 수행 ################################
    train_df_1 = train_df[train_df['build_idx']==1].drop(columns=['build_idx'])
    valid_df_1 = train_df_1[- (config.seq_len + config.pred_len ): ]
    train_df_1 = train_df_1[ : - (config.seq_len + config.pred_len )]
    
    test_df_1 = test_df[test_df['build_idx']==1].drop(columns=['build_idx'])
    
    
    ########### scaling #################
    #scaler = MinMaxScaler()
    #train_df_1 = scaler.fit_transform(train_df_1)
    #valid_df_1 = scaler.transform(valid_df_1)
    ########### scaling #################
    
    
    train_dataset = CustomDataset(train_df_1, (config.seq_len, config.label_len, config.pred_len))
    valid_dataset = CustomDataset(valid_df_1, (config.seq_len, config.label_len, config.pred_len))
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, drop_last=True, shuffle=True)
    valid_datalodaer = DataLoader(valid_dataset, batch_size=1, drop_last=False, shuffle=False)
    
    print("\n")
    print(f"train dataset length : {len(train_dataset)}")
    print(f"valid dataset length : {len(valid_dataset)}")
    ####################################################################################
    
    train(model, train_dataloader, valid_datalodaer)