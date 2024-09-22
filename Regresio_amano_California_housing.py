#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[85]:


data_train=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/california_housing_train.csv',sep=',')
data_train.columns


# In[86]:


data_train.describe()


# In[87]:


data_train.shape


# In[88]:


data_train.isna().sum()


# In[89]:


etiqueta=data_train['median_house_value']
entrada=data_train['total_rooms']


# In[90]:


entrada=np.array(entrada)
entrada=entrada.reshape((17000,1))


# In[8]:


class RegresionLineal(object):
    def __init__(self,x_dim,learning_rate=0.01,random_seed=None):
        self.x_dim=x_dim
        self.learning_rate=learning_rate
        self.grafo=tf.Graph()
        with self.grafo.as_default():
            tf.compat.v1.set_random_seed(random_seed)
            self.crear()
            self.inicia=tf.compat.v1.global_variables_initializer()

    def crear(self):
        self.X=tf.compat.v1.placeholder(dtype=tf.float32,shape=(None,self.x_dim),name='x_input')
        self.y=tf.compat.v1.placeholder(dtype=tf.float32,shape=(None),name='y_input')
        w=tf.Variable(tf.zeros(shape=(1)),name='peso')
        b=tf.Variable(tf.zeros(shape=(1)),name='sesgo')
        self.lineal=tf.squeeze(w*self.X+b,name='lineal')
        raiz_error=tf.compat.v1.square(self.y-self.lineal,name='raiz_error')
        self.costo=tf.compat.v1.reduce_mean(raiz_error,name='costo')
        optimizador=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate,name='DecensoGradiente')
        self.optimizador=optimizador.minimize(self.costo)  
        self.variables=tf.compat.v1.get_static_value((w,b))
       


# In[9]:


lrn=RegresionLineal(x_dim=entrada.shape[1],learning_rate=0.00000001)


# In[10]:


def set_train(sesion,modelo,train,objetivo,numero_epoch=50):
    sesion.run(modelo.inicia)
    lista_coste=[]
    valores=[]
    for i in range(numero_epoch):
        _,costo,valor=sesion.run([modelo.optimizador,modelo.costo,modelo.variables],feed_dict={modelo.X:train, modelo.y:objetivo})
        lista_coste.append(costo)
        valores.append(valor)
    return lista_coste,valores

sess=tf.compat.v1.Session(graph=lrn.grafo)
training=set_train(sess,lrn,entrada,etiqueta)


# In[11]:


plt.plot(range(1,len(training[0])+1),training[0])
plt.tight_layout()
plt.xlabel('Epocas')
plt.ylabel('Costo entrenamineto')
plt.show();


# In[12]:


def prediccion(sess,modelo,X_test):
    y_pred=sess.run(modelo.lineal,feed_dict={modelo.X:X_test})
    return y_pred

predicciones=prediccion(sess,lrn,entrada)


# In[13]:


X_0=entrada.min()
X_1=entrada.max()
for i in training[1]:
    w,b=i
    y_0=w*X_0+b
    y_1=w*X_1+b
    plt.plot([X_0,X_1],[y_0,y_1],c='r')
    plt.scatter(entrada,etiqueta)
plt.show();


# ## Modelando al estilo Tensorflow 2.0

# In[102]:


def train_model(X,y,batch_size=10,shuffle=True):
    X=tf.convert_to_tensor(X,dtype=tf.float32)
    y=tf.convert_to_tensor(y,dtype=tf.float32)
    ds=tf.data.Dataset.from_tensor_slices((X,y))
    if shuffle:
        ds=ds.shuffle(buffer_size=len(X)).batch(batch_size=batch_size)
    else:
        ds=ds.batch(batch_size=batch_size)
    return ds


# In[103]:


lista_prediccion=[]
lista_costo=[]
w=tf.Variable(tf.zeros(shape=(1)),name='peso')
b=tf.Variable(tf.zeros(shape=(1)),name='sesgo')
def funcion(X,y):
    with tf.GradientTape(persistent=True) as tape:
        prediccion=tf.squeeze(w*X+b)
        loss=tf.reduce_mean(tf.square(y-prediccion))
    dw_loss=tape.gradient(loss,w)
    db_loss=tape.gradient(loss,b)
    optimizador=tf.keras.optimizers.SGD(learning_rate=0.000001)
    optimizador.apply_gradients(zip([dw_loss,db_loss],[w,b]))


# In[104]:


epocas=10
for stp in range(epocas):
    ds_batch=train_model(entrada,etiqueta)
    for x_batch,y_batch in ds_batch:
        funcion(x_batch,y_batch)


# In[ ]:




