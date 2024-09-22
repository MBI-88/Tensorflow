#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import tensorflow as tf 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[2]:


data=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/mnist_train_small.csv',sep=',',index_col=False)
data_test=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/mnist_test.csv',sep=',',index_col=False)
data.shape,data_test.shape


# In[3]:


label='6'
X_test=data_test.drop(label,axis=1)
y_test=data_test[label]
X_train,y_train,X_eval,y_eval=train_test_split(data.drop(label,axis=1),data[label],train_size=0.80)
X_eval.shape,y_eval.shape


# In[4]:


X_train=np.array(X_train)
X_eval=np.array(X_eval)
fig,ax=plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True)
ax=ax.flatten()
for i in range(10):
    img=X_train[X_eval==i][0].reshape(28,28)
    ax[i].imshow(img,cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


# In[5]:


label='6'
X_test=data_test.drop(label,axis=1)
y_test=data_test[label]
X_train,y_train,X_eval,y_eval=train_test_split(data.drop(label,axis=1),data[label],train_size=0.80)
X_train=((X_train/255.)-0.5)*2
y_train=((y_train/255.)-0.5)*2
X_test=((X_test/255.)-0.5)*2


# In[6]:


Columns=X_train.columns
lista_columnas=[]
for variable in Columns:
    lista_columnas.append(tf.feature_column.numeric_column(variable,dtype=tf.float32))

def input_fn(X,y,batch_size=10,shuffle=True,num_epo=None):
    ds=tf.data.Dataset.from_tensor_slices((dict(X),y))
    if shuffle:
        ds=ds.shuffle(buffer_size=10000).batch(batch_size=batch_size).repeat(num_epo)
    else:
        ds=ds.batch(batch_size=batch_size).repeat(num_epo)
    return ds
set_train=lambda: input_fn(X_train,X_eval,batch_size=35,num_epo=None)
set_eval=lambda: input_fn(y_train,y_eval,shuffle=False,num_epo=1)
set_test=lambda: input_fn(X_test,y_test,shuffle=False,num_epo=1)


# In[7]:


regresor_lgt=tf.estimator.LinearClassifier(feature_columns=lista_columnas,n_classes=10,optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.03))
regresor_lgt.train(set_train,max_steps=1000)
resultado=regresor_lgt.evaluate(set_eval)
print(resultado)                           # Nota seguir mejorando  hasta obtener 0.9 de accuracy en el lineal y 0.95 en la red /bloque 25-0.8905/0.371


# In[8]:


nn=[100,100]
regresor_DNN_lgt=tf.estimator.DNNClassifier(hidden_units=nn,feature_columns=lista_columnas,n_classes=10,optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.05))
regresor_DNN_lgt.train(set_train,max_steps=1000)
resultado=regresor_DNN_lgt.evaluate(set_eval)
print(resultado)


# In[ ]:




