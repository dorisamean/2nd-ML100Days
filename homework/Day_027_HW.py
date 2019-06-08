#%% [markdown]
# # 作業 : (Kaggle)鐵達尼生存預測
#%% [markdown]
# # [作業目標]
# - 試著模仿範例寫法, 在鐵達尼生存預測中, 觀察計數編碼與特徵雜湊的效果
#%% [markdown]
# # [作業重點]
# - 仿造範例, 完成自己挑選特徵的群聚編碼 (In[2], Out[2])
# - 觀察群聚編碼, 搭配邏輯斯回歸, 看看有什麼影響 (In[5], Out[5], In[6], Out[6]) 
#%% [markdown]
# # 作業1
# * 試著使用鐵達尼號的例子，創立兩種以上的群聚編碼特徵( mean、median、mode、max、min、count 均可 )

#%%
# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

data_path = 'data/'
df = pd.read_csv(data_path + 'titanic_train.csv')

train_Y = df['Survived']
df = df.drop(['PassengerId', 'Survived'] , axis=1)
df.head()


#%%
# 取一個類別型欄位, 與一個數值型欄位, 做群聚編碼
"""
Your Code Here
"""


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

#%% [markdown]
# # 作業2
# * 將上述的新特徵，合併原有的欄位做生存率預估，結果是否有改善?

#%%
# 原始特徵 + 邏輯斯迴歸
"""
Your Code Here
"""


#%%
# 新特徵 + 邏輯斯迴歸
"""
Your Code Here
"""


#%%



