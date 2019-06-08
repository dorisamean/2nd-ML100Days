#%% [markdown]
# # 作業 : (Kaggle)鐵達尼生存預測  (Day_024_HW)
# https://www.kaggle.com/c/titanic
#%% [markdown]
# # [作業目標]
# - 試著模仿範例寫法, 在鐵達尼生存預測中, 觀察計數編碼與特徵雜湊的效果
#%% [markdown]
# # [作業重點]
# - 仿造範例, 完成計數編碼以及搭配邏輯斯迴歸的預測 (In[4], Out[4], In[5], Out[5]) 
# - 仿造範例, 完成雜湊編碼, 以及計數編碼+雜湊編碼 搭配邏輯斯迴歸的預測 (In[6], Out[6], In[7], Out[7]) 
# - 試著回答上述執行結果的觀察
#%% [markdown]
# # 作業1
# * 參考範例，將鐵達尼的艙位代碼( 'Cabin' )欄位使用特徵雜湊 / 標籤編碼 / 目標均值編碼三種轉換後，  
# 與其他類別型欄位一起預估生存機率

#%%
# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy, time
import warnings
warnings.filterwarnings('igonre')

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# check scikit-learn version
import sklearn
print('sklearn: %s' % sklearn.__version__)


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

data_path = 'data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()


#%%
#只取類別值 (object) 型欄位, 存於 object_features 中
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'object':
        object_features.append(feature)
print(f'{len(object_features)} Object Features : {object_features}\n')

# 只留類別型欄位
df = df[object_features]
df = df.fillna('None')
train_num = train_Y.shape[0]
df.head()

#%% [markdown]
# # 作業2
# * 承上題，三者比較效果何者最好?

#%%
# 對照組 : 標籤編碼 + 邏輯斯迴歸
df_temp = pd.DataFrame()
for c in df.columns:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
df_temp.head()


#%%
# 加上 'Cabin' 欄位的計數編碼
# 第一行 : df.groupby(['Cabin']) 會輸出 df 以 'Cabin' 群聚後的結果, 
# 但因為群聚一類只會有一個值, 因此必須要定義運算
# 例如 df.groupby(['Cabin']).size(), 但欄位名稱會變成 size, 
# 要取別名就需要用語法 df.groupby(['Cabin']).agg({'Cabin':'size'})
# 這樣出來的計數欄位名稱會叫做 'Cabin_Count', 
# 因為這樣群聚起來的 'Cabin' 是 index, 所以需要 reset_index() 轉成一欄
# 因此第一行的欄位, 在第三行按照 'Cabin_Count' 排序後, 最後的 DataFrame 輸出如 Out[5]

count_df = df.groupby(['Cabin'])['Name'].agg({'Cabin_Count':'size'}).reset_index()

# 但是上面資料表結果只是 'Cabin' 名稱對應的次數, 
# 要做計數編碼還需要第二行 : 將上表結果與原表格 merge, 合併於 'Cabin' 欄位
# 使用 how='left' 是完全保留原資料表的所有 index 與順序
df = pd.merge(df, count_df, on=['Cabin'], how='left')
count_df.sort_values(by=['Cabin_Count'], ascending=False).head(10)

# 印出來看看, 加了計數編碼的資料表 df 有何不同
df.head()



#%%
# 'Cabin'計數編碼 + 邏輯斯迴歸
df_temp = pd.DataFrame()
for c in object_features:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
df_temp['Cabin_Count'] = df['Cabin_Count']
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
df_temp.head()


#%%
# 'Cabin'特徵雜湊 + 邏輯斯迴歸
df_temp = pd.DataFrame()
for c in object_features:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
# 這邊的雜湊編碼, 是直接將 'Cabin' 的名稱放入雜湊函數的輸出數值, 為了要確定是緊密(dense)特徵, 因此除以10後看餘數
# 這邊的 10 是隨機選擇, 不一定要用 10, 同學可以自由選擇購小的數字試看看. 基本上效果都不會太好
df_temp['Cabin_Hash'] = df['Cabin'].map(lambda x:hash(x) % 10)
train_X = df_temp[:train_num]
estimator = LogisticRegression(solver='liblinear')
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
df_temp.head()



#%%
# 'Cabin'計數編碼 + 'Cabin'特徵雜湊 + 邏輯斯迴歸
df_temp = pd.DataFrame()
for c in object_features:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
df_temp['Cabin_Hash'] = df['Cabin'].map(lambda x:hash(x) % 10)
df_temp['Cabin_Count'] = df['Cabin_Count']
train_X = df_temp[:train_num]
estimator = LogisticRegression(solver='liblinear')
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
df_temp.head()

#%%



