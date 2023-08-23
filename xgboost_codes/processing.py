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
  
  date = pd.to_datetime(df[name], format=format)
  df['hour'] = date.dt.hour
  df['day'] = date.dt.weekday
  df['month'] = date.dt.month
  df['week'] = date.dt.isocalendar().week.astype(int)
  df['holiday'] = df.apply(lambda x : 0 if x['day']<5 else 1, axis = 1) #주말


  ## https://dacon.io/competitions/official/235680/codeshare/2366?page=1&dtype=recent 시간 푸리에 변환
  df['sin_time'] = np.sin(2*np.pi*df.hour/24)
  df['cos_time'] = np.cos(2*np.pi*df.hour/24)


  if add_feature:
    ## https://dacon.io/competitions/official/235736/codeshare/2743?page=1&dtype=recent 불쾌지수
    df['THI'] = 9/5*df['temp'] - 0.55*(1-df['humid']/100)*(9/5*df['humid']-26)+32


    cdhs = np.array([])
    for num in range(1,101,1):
        temp = df[df['build_idx'] == num]
        cdh = CDH(temp['temp'].values)
        cdhs = np.concatenate([cdhs, cdh])
    df['CDH'] = cdhs

  return df

def CDH(xs): # cooling degree hour
      ys = []
      for i in range(len(xs)):
          if i < 11:
              ys.append(np.sum(xs[:(i+1)]-26))
          else:
              ys.append(np.sum(xs[(i-11):(i+1)]-26))
      return np.array(ys)