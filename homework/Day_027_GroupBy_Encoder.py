#%% [markdown]
# # 範例 : (Kaggle)房價預測
#%% [markdown]
# # [教學目標]
# - 以下用房價預測資料, 觀察群聚編碼的效果
#%% [markdown]
# # [範例重點]
# - 了解群聚編碼的寫作方式(In[3], Out[3])
# - 觀察群聚編碼, 搭配線性迴歸以及隨機森林分別有什麼影響 (In[6]~In[9], Out[6]~Out[9])

#%%
# 請先確認您的 sklearn 版本是否相同, 如果版本不是 0.21.1 版, 有可能跑出與本範例不同的結果
import sklearn
sklearn.__version__


#%%
# 做完特徵工程前的所有準備
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

data_path = 'data/'
df = pd.read_csv(data_path + 'house_train.csv.gz')

train_Y = np.log1p(df['SalePrice'])
df = df.drop(['Id', 'SalePrice'] , axis=1)
df.head()


#%%
# 生活總面積(GrLivArea)對販售條件(SaleCondition)做群聚編碼
# 寫法類似均值編碼, 只是對另一個特徵, 而非目標值
df['SaleCondition'] = df['SaleCondition'].fillna('None')
mean_df = df.groupby(['SaleCondition'])['GrLivArea'].mean().reset_index()
mode_df = df.groupby(['SaleCondition'])['GrLivArea'].apply(lambda x: x.mode()[0]).reset_index()
median_df = df.groupby(['SaleCondition'])['GrLivArea'].median().reset_index()
max_df = df.groupby(['SaleCondition'])['GrLivArea'].max().reset_index()
temp = pd.merge(mean_df, mode_df, how='left', on=['SaleCondition'])
temp = pd.merge(temp, median_df, how='left', on=['SaleCondition'])
temp = pd.merge(temp, max_df, how='left', on=['SaleCondition'])
temp.columns = ['SaleCondition', 'Area_Sale_Mean', 'Area_Sale_Mode', 'Area_Sale_Median', 'Area_Sale_Max']
temp


#%%
df = pd.merge(df, temp, how='left', on=['SaleCondition'])
df = df.drop(['SaleCondition'] , axis=1)
df.head()


#%%
#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
df = df.fillna(-1)
MMEncoder = MinMaxScaler()
df.head()


#%%
# 沒有這四個新特徵的 dataframe 稱為 df_minus
df_minus = df.drop(['Area_Sale_Mean', 'Area_Sale_Mode', 'Area_Sale_Median', 'Area_Sale_Max'] , axis=1)

# 原始特徵 + 線性迴歸
train_X = MMEncoder.fit_transform(df_minus)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()


#%%
# 新特徵 + 線性迴歸 : 有些微改善
train_X = MMEncoder.fit_transform(df)
cross_val_score(estimator, train_X, train_Y, cv=5).mean()


#%%
# 原始特徵 + 隨機森林
train_X = MMEncoder.fit_transform(df_minus)
estimator = RandomForestRegressor()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()


#%%
# 新特徵 + 隨機森林
train_X = MMEncoder.fit_transform(df)
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

#%% [markdown]
# # 作業1
# * 試著使用鐵達尼號的例子，創立兩種以上的群聚編碼特徵( mean、median、mode、max、min、count 均可 )
# 
# # 作業2
# * 將上述的新特徵，合併原有的欄位做生存率預估，結果是否有改善?

#%%



