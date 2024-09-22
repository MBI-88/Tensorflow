#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf 
import matplotlib.pyplot as plt 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(100)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


data_prueba=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/california_housing_test.csv',index_col=False)
data_train=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/california_housing_train.csv',index_col=False)
data_train.shape,data_prueba.shape


# In[15]:


data_prueba.describe()


# In[16]:


data_train=data_train.reindex(np.random.permutation(data_train.index))
data_copy_train=data_train.copy()
data_copy_train['rooms_x_persons']=(data_train['total_rooms']/data_train['population'])
data_copy_train['median_house_value']/=1000.0

data_prueba=data_prueba.reindex(np.random.permutation(data_prueba.index))
data_copy_test=data_prueba.copy()
data_copy_test['rooms_x_persons']=(data_prueba['total_rooms']/data_prueba['population'])
data_copy_test['median_house_value']/=1000.0
standar=StandardScaler()


# In[17]:


label='median_house_value'
X_test=data_copy_test.drop(label,axis=1)
Columns=X_test.columns
y_test=data_copy_test[label]
X_train,y_train,X_eval,y_eval=train_test_split(data_copy_train.drop(label,axis=1),data_copy_train[label],train_size=0.80)
X_train=standar.fit_transform(X_train)
X_train=pd.DataFrame(X_train,columns=Columns)
X_test=standar.fit_transform(X_test)
X_test=pd.DataFrame(X_test,columns=Columns)
y_train=standar.fit_transform(y_train)
y_train=pd.DataFrame(y_train,columns=Columns)
X_train.shape,y_train.shape,X_eval.shape,y_eval.shape


# In[18]:


def metrica(y,pred):
    train_metric=np.sqrt(mean_squared_error(y,pred))
    return train_metric

lista_columnas=[]
for variable in Columns:
    lista_columnas.append(tf.feature_column.numeric_column(variable,dtype=tf.float32))

def input_fn(X,y,batch_size=10,shuffle=True,num_epo=None):
    ds=tf.data.Dataset.from_tensor_slices((dict(X),y))
    if shuffle:
        ds=ds.shuffle(buffer_size=len(y)).batch(batch_size=batch_size).repeat(num_epo)
    else:
        ds=ds.batch(batch_size=batch_size).repeat(num_epo)
    return ds


# In[27]:


# Crecion del DNNRegressor
num_epochs=100
batch_size=50
step_x_epo=int(np.ceil(len(y_train)/batch_size))
nn=[10,10]

modelo_DNN=tf.estimator.DNNRegressor(hidden_units=nn,feature_columns=lista_columnas,optimizer=tf.keras.optimizers.SGD(learning_rate=0.05,momentum=0.05))


# In[28]:


modelo_DNN.train(input_fn=lambda: input_fn(X_train,X_eval,batch_size=batch_size,num_epo=None),steps=num_epochs*step_x_epo)
resultado=modelo_DNN.evaluate(input_fn=lambda:input_fn(y_train,y_eval,shuffle=False,num_epo=1,batch_size=batch_size))
print(resultado)


# In[29]:


predicciones=list(modelo_DNN.predict(input_fn=lambda:input_fn(y_train,y_eval,shuffle=False,num_epo=1,batch_size=batch_size)))
predicciones=pd.Series([item['predictions'][0] for item in predicciones])
predicciones.plot(kind='hist',bins=20,title='Predicciones');
metrica(y_eval,predicciones)


# In[22]:


predicciones=list(modelo_DNN.predict(lambda: input_fn(X_test,y_test,shuffle=False,num_epo=1,batch_size=batch_size)))
predicciones=pd.Series([item['predictions'][0] for item in predicciones])
predicciones.plot(kind='hist',bins=20,title='Predicciones');
metrica(y_test,predicciones)


# In[ ]:




