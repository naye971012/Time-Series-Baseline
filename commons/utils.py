import pandas as pd
import numpy as np

def process_missing_values(dataframe,method='ffill'):
    """
    dataframe 에 대한 결측치 처리 수행 함수

    Args:
        dataframe (pd.DataFrame): datafrmae
        method (str, optional): 결측치 처리 방법. Defaults to 'ffill'.

    Returns:
        dataframe
    """
    dataframe = dataframe.fillna(method=method)
    dataframe = dataframe.fillna(0)
    
    return dataframe

def SMAPE(true, pred):
    return np.mean((np.abs(true-pred))/(np.abs(true) + np.abs(pred))) * 200