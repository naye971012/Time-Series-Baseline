import numpy as np
import pandas as pd

def process(df , name='yyyyymmdd hh', format ='%Y%m%d %H', add_feature=True):
  """
    tabular 형식의 데이터를 시계열 특성에 맞게 변환 및 기타 feature 추가
    Args:
        df (_type_): _description_
        name (str, optional): _description_. Defaults to 'yyyyymmdd hh'.
        format (str, optional): _description_. Defaults to '%Y%m%d %H'.
        add_feature (bool, optional): _description_. Defaults to True.
  """
  df['order'] = range(1, len(df) + 1)
  
  date = pd.to_datetime(df[name], format=format)
  df['hour'] = date.dt.hour
  df['day'] = date.dt.weekday
  df['month'] = date.dt.month
  df['week'] = date.dt.isocalendar().week.astype(int)
  df['holiday'] = df.apply(lambda x : 0 if x['day']<5 else 1, axis = 1) #주말


  ## https://dacon.io/competitions/official/235680/codeshare/2366?page=1&dtype=recent 시간 푸리에 변환
  df['sin_time'] = np.sin(2*np.pi*df.hour/24)
  df['cos_time'] = np.cos(2*np.pi*df.hour/24)
  
  return df