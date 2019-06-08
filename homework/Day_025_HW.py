#%% [markdown]
# # 範例 : 計程車費率預測
# https://www.kaggle.com/c/new-york-city-taxi-fare-prediction
#%% [markdown]
# # [作業目標]
# - 試著模仿範例寫法, 使用程車費率預測競賽練習時間欄位處理
#%% [markdown]
# # [作業重點]
# - 新增星期幾(day of week)與第幾周(week of year)這兩項特徵, 觀察有什麼影響 (In[4], Out[4], In[5], Out[5])
# - 新增加上年週期與周周期特徵 , 觀察有什麼影響 (In[8], Out[8], In[9], Out[9]) 

#%%
# 做完特徵工程前的所有準備
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

data_path = 'data/'
df = pd.read_csv(data_path + 'taxi_data1.csv')

train_Y = df['fare_amount']
df = df.drop(['fare_amount'] , axis=1)
df.head()


#%%
# 時間特徵分解方式:使用datetime
df['pickup_datetime'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S UTC'))
df['pickup_year'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y')).astype('int64')
df['pickup_month'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%m')).astype('int64')
df['pickup_day'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%d')).astype('int64')
df['pickup_hour'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%H')).astype('int64')
df['pickup_minute'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%M')).astype('int64')
df['pickup_second'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%S')).astype('int64')
df.head()


#%%
# 將結果使用線性迴歸 / 梯度提升樹分別看結果
df_temp = df.drop(['pickup_datetime'] , axis=1)
scaler = MinMaxScaler()
train_X = scaler.fit_transform(df_temp)
Linear = LinearRegression()
print(f'Linear Reg Score : {cross_val_score(Linear, train_X, train_Y, cv=5).mean()}')
GDBT = GradientBoostingRegressor()
print(f'Gradient Boosting Reg Score : {cross_val_score(GDBT, train_X, train_Y, cv=5).mean()}')

#%% [markdown]
# # 作業1
# * 對照範例，試著加入星期幾 (day of week) 與第幾周 (week of year) 這兩項特徵，  
# 看看結果會比原本只有時間特徵分解的結果更好或更差?

#%%
# 加入星期幾與第幾周兩個特徵
#https://docs.python.org/2/library/datetime.html
df['pickup_day_of_week'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%w')).astype('int64')
df['pickup_week_of_year'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%U')).astype('int64')
df['pickup_day_of_year'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%j')).astype('int64')
df.head()


#%%
# 將結果使用線性迴歸 / 梯度提升樹分別看結果
df_temp = df.drop(['pickup_datetime'] , axis=1)
train_X = scaler.fit_transform(df_temp)
print(f'Linear Reg Score : {cross_val_score(Linear, train_X, train_Y, cv=5).mean()}')
print(f'Gradient Boosting Reg Score : {cross_val_score(GDBT, train_X, train_Y, cv=5).mean()}')


#%%
# 加上"日週期"特徵 (參考講義"週期循環特徵")
import math
df['day_cycle'] = df['pickup_hour']/12 + df['pickup_minute']/720 + df['pickup_second']/43200
df['day_cycle'] = df['day_cycle'].map(lambda x:math.sin(x*math.pi))
df.head()


#%%
# 將結果使用線性迴歸 / 梯度提升樹分別看結果
df_temp = df.drop(['pickup_datetime'] , axis=1)
train_X = scaler.fit_transform(df_temp)
print(f'Linear Reg Score : {cross_val_score(Linear, train_X, train_Y, cv=5).mean()}')
print(f'Gradient Boosting Reg Score : {cross_val_score(GDBT, train_X, train_Y, cv=5).mean()}')

#%% [markdown]
# # 作業2
# * 對照範例的日週期效果，試著參考投影片完成年週期與周週期的特徵 (也可以用你自己想到的方式)，  
# 看看結果會比範例中的結果更好或更差?

#%%
# 加上"年週期"與"周週期"特徵
#年週期: cos((⽉/6 + 日/180 )π)
df['year_cycle'] = df['pickup_month']/6 + df['pickup_day_of_year']/180
df['year_cycle'] = df['year_cycle'].map(lambda x:math.cos(x*math.pi))

#週週期: sin((星期幾/3.5 + 小時/84 )π)
df['week_cycle'] = (df['pickup_day_of_week']+1)/3.5 + ((df['pickup_day_of_week']+1)*24+df['pickup_hour'])/84
df['week_cycle'] = df['week_cycle'].map(lambda x:math.sin(x*math.pi))
df.head()


#%%
# 將結果使用線性迴歸 / 梯度提升樹分別看結果
df_temp = df.drop(['pickup_datetime'] , axis=1)
train_X = scaler.fit_transform(df_temp)
print(f'Linear Reg Score : {cross_val_score(Linear, train_X, train_Y, cv=5).mean()}')
print(f'Gradient Boosting Reg Score : {cross_val_score(GDBT, train_X, train_Y, cv=5).mean()}')


#%%



