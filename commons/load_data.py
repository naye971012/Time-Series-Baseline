import pandas as pd
import numpy as np
import os

def load_train_test_dataframe(base_path , train_df_path, test_df_path ):
    """
    단순하게 경로에서 csv 파일 읽어오는 함수

    Args:
        base_path (_type_): _description_
        train_df_path (_type_): _description_
        test_df_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    train_df_loc = os.path.join(base_path,train_df_path)
    test_df_loc = os.path.join(base_path,test_df_path)

    train_df = pd.read_csv(train_df_loc)
    test_df = pd.read_csv(test_df_loc)
    
    return train_df, test_df