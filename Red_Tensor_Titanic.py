#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from category_encoders import OneHotEncoder
from sklearn.base import BaseEstimator,TransformerMixin
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline,FeatureUnion
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(42)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[104]:


data_train=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/titanic_train.csv',index_col=False,sep=',')
data_train.head()


# In[105]:


data_train.pop('ii')
data_train.isna().sum()


# In[106]:


data_train=data_train.drop_duplicates()
data_train.duplicated().sum()


# In[107]:


label='survived'
X_train,y_train,X_eval,y_eval=train_test_split(data_train.drop(label,axis=1),data_train[label],train_size=0.80)
X_train.shape,X_eval.shape,y_train.shape,y_eval.shape


# In[108]:


col_numericas=X_train.select_dtypes(np.number).columns
col_cat=X_train.select_dtypes(np.object).columns

class ColumExtractor(TransformerMixin,BaseEstimator):
    def __init__(self,columns,output_type='dataframe'):
        self.columns=columns
        self.output_type=output_type
    
    def transform(self,X,**transform_params):
        if isinstance(X,list):
            X=pd.DataFrame.from_dict(X)
        elif self.output_type=='dataframe':
            return X[self.columns]
        
        raise Exception('output_type tiene que ser dataframe')
    
    def fit(self,X,y=None,**fit_params):
        return self


# In[109]:


pipe_num=Pipeline([
    ('selector',ColumExtractor(columns=col_numericas)),
    ('estandarizador',MinMaxScaler(feature_range=(-1,1)))
])
pipe_cat=Pipeline([
    ('selector',ColumExtractor(columns=col_cat)),
    ('codificador',OneHotEncoder()),
    ('estandarizador',MinMaxScaler(feature_range=(-1,1)))
])
pipe_union=FeatureUnion([
    ('numerico',pipe_num),
    ('categorico',pipe_cat)
])


# In[110]:


ds=pipe_union.fit_transform(X_train,X_eval)
X_eval=np.array(X_eval)
ds.shape


# In[120]:


# Red neuronal
gr=tf.Graph()
nn=[23,48,16,1]
lr=0.1
n_epoch=500
with  gr.as_default():
    X=tf.compat.v1.placeholder(dtype=tf.float32,shape=(None,ds.shape[1]),name='Input')
    y=tf.compat.v1.placeholder(dtype=tf.float32,shape=(None),name='Ouput')
    # Capa1
    W1=tf.Variable(tf.random.normal((nn[0],nn[1])),name='Peso_1')
    b1=tf.Variable(tf.random.normal((nn[1],)),name='Sesgo_1')
    func_1=tf.nn.relu(tf.add(tf.matmul(X,W1),b1))
    # Capa2
    W2=tf.Variable(tf.random.normal((nn[1],nn[2])),name='Peso_2')
    b2=tf.Variable(tf.random.normal((nn[2],)),name='Sesgo_2')
    func_2=tf.nn.relu(tf.add(tf.matmul(func_1,W2),b2))
    # Capa3
    W3=tf.Variable(tf.random.normal((nn[2],nn[3])),name='Peso_3')
    b3=tf.Variable(tf.random.normal((nn[3],)),name='Sesgo_3')
    py=tf.nn.sigmoid(tf.add(tf.matmul(func_2,W3),b3))[:,0]
    # Coste
    loss=tf.losses.mean_squared_error(y,py)
    # Optimizador
    optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

def generador_bath(X,y,bath_size=120,shuffle=False):
    #X_copy=np.array(X)
    #y_copy=np.array(y)

    if shuffle:
        data=np.column_stack((X,y))
        np.random.shuffle(data)
        X_copy=data[:,:-1]
        y_copy=data[:,-1].astype(int)
    
    for i in range(0,X.shape[0],bath_size):
        yield (X_copy[i:i+bath_size,:],y_copy[i:i+bath_size])


# In[121]:


predicciones=[]
with tf.compat.v1.Session(graph=gr) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(n_epoch):
        generador=generador_bath(ds,X_eval,bath_size=20,shuffle=True)
        for x_bath,y_bath in generador:
            feed={X:x_bath,y:y_bath}
            _,perdida,pred=sess.run([optimizer,loss,py], feed_dict=feed)
            if step % 25 == 0:
                acc=np.mean(np.round(pred)==y_bath)
            print('Step: ',step,'/',n_epoch,'--Loss: ',perdida,'--Exactitud: ',acc)
                    
        predicciones.append(np.round(pred))


# In[ ]:




