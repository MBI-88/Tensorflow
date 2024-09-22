#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np
import scipy.stats as st
from sklearn.metrics import mean_squared_error
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(42)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data_prueba=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/california_housing_test.csv',index_col=False)
data_train=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/california_housing_train.csv',index_col=False)
data_train.shape


# In[3]:


data_train.isna().sum()


# In[4]:


data_train[data_train.duplicated()].shape


# In[5]:


# Outliers
def Outliers(data):
    for col in data:
        if data_train[col].dtype !=np.object:
            outliers=len(data_train[np.abs(st.zscore(data_train[col]))>3])
            print('{} {} {}'.format(data_train[col].name,outliers,data_train[col].dtype))
Outliers(data_train)


# In[6]:


data_train.boxplot(column='total_rooms');


# In[7]:


data_train.plot.scatter(x='median_income',y='housing_median_age');


# In[8]:


data_train.plot.scatter(x='median_income',y='median_house_value');


# In[9]:


data_train.plot.scatter(x='housing_median_age',y='median_house_value');


# In[10]:


data_train.plot.scatter(x='total_rooms',y='median_house_value');


# In[11]:


data_train.plot.scatter(x='total_rooms',y='population');


# In[12]:


data_train.plot.scatter(x='population',y='median_house_value');


# In[13]:


data_train.plot.scatter(x='population',y='housing_median_age');


# In[14]:


data_train.plot.scatter(x='latitude',y='longitude');


# In[15]:


data_train.plot.scatter(x='latitude',y='total_rooms');


# In[16]:


data_train.plot.scatter(x='latitude',y='population');


# In[7]:


def moda_X_columns(data):
    for col in data.columns:
        print('Columns: {}  Mode: {}  Total: {}'.format(col,st.mode(data_train[col])[0],st.mode(data_train[col])[1]))

moda_X_columns(data_train)


# In[18]:


data_train.groupby('housing_median_age')['population'].mean().plot();


# In[19]:


data_train['housing_median_age'].plot.kde();


# In[20]:


data_train['total_rooms'].plot.kde();


# In[8]:


data_train.corr()


# In[9]:


data_train=data_train.reindex(np.random.permutation(data_prueba.index))
data_train.head()


# In[10]:


data_train.describe()


# In[11]:


data_copy=data_train.copy()
data_copy['rooms_x_persons']=(data_train['total_rooms']/data_train['population'])
data_copy['housing_x_persons']=(data_train['housing_median_age']/data_train['population'])
data_copy['median_house_value']/=1000.0


# In[12]:


data_copy.describe()


# In[13]:


data_copy.corr()


# In[14]:


data_copy.info()


# In[15]:


from sklearn.model_selection import train_test_split
label='median_house_value'
features=data_copy.drop(label,axis=1).columns
X_train,y_train,X_val,y_val=train_test_split(data_copy[features],data_copy[label],train_size=0.80)


# In[16]:


lista_columnas=[]

for variable in X_train:
    lista_columnas.append(tf.feature_column.numeric_column(variable,dtype=tf.float32))


# In[20]:


def metrica_train(y,pred):
    train_metric=np.sqrt(mean_squared_error(y,pred))
    return train_metric

def input_fn(variable,objetivo,back_size=50,shuffle=True,num_epoch=150):
    def entrada():
        ds=tf.compat.v2.data.Dataset.from_tensor_slices((dict(variable),objetivo))
        if shuffle:
            ds=ds.shuffle(10000)
        ds=ds.batch(back_size).repeat(num_epoch)
        return ds
    return entrada

set_entrenamiento=input_fn(X_train,X_val)
set_validacion=input_fn(y_train,y_val,shuffle=False,num_epoch=1)


# In[18]:


regresor=tf.estimator.LinearRegressor(feature_columns=lista_columnas,optimizer=tf.keras.optimizers.Ftrl(learning_rate=0.5,l2_regularization_strength=0.004))
regresor.train(set_entrenamiento)
result=regresor.evaluate(set_validacion)
print(result)


# In[22]:


predicciones=list(regresor.predict(set_validacion))
predicciones=pd.Series([pred['predictions'][0] for pred in predicciones])

predicciones.plot(kind='hist', bins=20, title='predicted probabilities');
metrica_train(y_val,predicciones)


# In[23]:


def get_quantiles(feature_columns,num_buckes):
    bondaries=np.arange(1.0,num_buckes)/num_buckes
    quantile=feature_columns.quantile(bondaries)
    return [quantile[q] for q in quantile.keys()]

housing=tf.feature_column.numeric_column('housing_median_age')
bucketized_housing=tf.feature_column.bucketized_column(housing,boundaries=get_quantiles(X_train['housing_median_age'],10))

icome=tf.feature_column.numeric_column('median_income')
bucketized_income=tf.feature_column.bucketized_column(icome,boundaries=get_quantiles(X_train['median_income'],10))

housing_persons=tf.feature_column.numeric_column('housing_x_persons')
bucketized_housing_persons=tf.feature_column.bucketized_column(housing_persons,boundaries=get_quantiles(X_train['housing_x_persons'],10))

rooms_x_persons=tf.feature_column.numeric_column('rooms_x_persons')
bucketized_rooms_x_persons=tf.feature_column.bucketized_column(rooms_x_persons,boundaries=get_quantiles(X_train['rooms_x_persons'],10))


# In[24]:


genr_1=tf.feature_column.crossed_column([bucketized_rooms_x_persons,bucketized_housing_persons],hash_bucket_size=1000)
columns_derivada_1=[genr_1]
regresor_1=tf.estimator.LinearRegressor(feature_columns=lista_columnas+columns_derivada_1,optimizer=tf.keras.optimizers.Ftrl(learning_rate=0.5,l2_regularization_strength=0.004))
regresor_1.train(set_entrenamiento)
result=regresor_1.evaluate(set_validacion)
print(result)


# In[25]:


predicciones=list(regresor_1.predict(set_validacion))
predicciones=pd.Series([pred['predictions'][0] for pred in predicciones])

predicciones.plot(kind='hist', bins=20, title='predicted probabilities');
metrica_train(y_val,predicciones)


# In[26]:


genr_2=tf.feature_column.crossed_column([bucketized_income,bucketized_housing_persons],hash_bucket_size=1000)
columns_derivada_2=[genr_2]
regresor_2=tf.estimator.LinearRegressor(feature_columns=lista_columnas+columns_derivada_2,optimizer=tf.keras.optimizers.Ftrl(learning_rate=0.5,l2_regularization_strength=0.004))
regresor_2.train(set_entrenamiento)
result=regresor_2.evaluate(set_validacion)
print(result)


# In[27]:


predicciones=list(regresor_2.predict(set_validacion))
predicciones=pd.Series([pred['predictions'][0] for pred in predicciones])

predicciones.plot(kind='hist', bins=20, title='predicted probabilities');
metrica_train(y_val,predicciones)


# In[28]:


columns_derivada_3=[genr_1,genr_2]
regresor_3=tf.estimator.LinearRegressor(feature_columns=lista_columnas+columns_derivada_3,optimizer=tf.keras.optimizers.Ftrl(learning_rate=0.5,l2_regularization_strength=0.002))
regresor_3.train(set_entrenamiento)
result=regresor_3.evaluate(set_validacion)
print(result)


# In[29]:


predicciones=list(regresor_3.predict(set_validacion))
predicciones=pd.Series([pred['predictions'][0] for pred in predicciones])

predicciones.plot(kind='hist', bins=20, title='predicted probabilities');
metrica_train(y_val,predicciones)


# In[30]:


data_prueba_copy=data_prueba.copy()
data_prueba_copy['rooms_x_persons']=(data_prueba['total_rooms']/data_prueba['population'])
data_prueba_copy['housing_x_persons']=(data_prueba['housing_median_age']/data_prueba['population'])
data_prueba_copy['median_house_value']/=1000.0
data_prueba_copy.describe()


# In[31]:


data_prueba_copy.isna().sum()


# In[32]:


y_train_prueba=data_prueba_copy.drop(label,axis=1)
y_test=data_prueba_copy[label]
set_prueba=input_fn(y_train_prueba,y_test,shuffle=False,num_epoch=1)


# In[33]:


result=regresor_3.evaluate(set_prueba)
print(result)


# In[34]:


predicciones=list(regresor_3.predict(set_prueba))
predicciones=pd.Series([pred['predictions'][0] for pred in predicciones])

predicciones.plot(kind='hist', bins=20, title='predicted probabilities');
metrica_train(y_test,predicciones)


# In[35]:


y_test[:10],predicciones[:10]


# In[36]:


result=regresor_2.evaluate(set_prueba)
print(result)


# In[37]:


predicciones=list(regresor_2.predict(set_prueba))
predicciones=pd.Series([pred['predictions'][0] for pred in predicciones])

predicciones.plot(kind='hist', bins=20, title='predicted probabilities');
metrica_train(y_test,predicciones)


# In[38]:


result=regresor_1.evaluate(set_prueba)
print(result)


# In[39]:


predicciones=list(regresor_1.predict(set_prueba))
predicciones=pd.Series([pred['predictions'][0] for pred in predicciones])

predicciones.plot(kind='hist', bins=20, title='predicted probabilities');
metrica_train(y_test,predicciones)


# In[40]:


result=regresor.evaluate(set_prueba)
print(result)


# In[41]:


predicciones=list(regresor.predict(set_prueba))
predicciones=pd.Series([pred['predictions'][0] for pred in predicciones])

predicciones.plot(kind='hist', bins=20, title='predicted probabilities');
metrica_train(y_test,predicciones)


# ## Nota el Regresor 2 funciona mejor

# In[ ]:




