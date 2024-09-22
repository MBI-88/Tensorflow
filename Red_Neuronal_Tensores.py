#!/usr/bin/env python
# coding: utf-8

# In[57]:


import matplotlib.pyplot as plt 
from matplotlib import animation
from sklearn.datasets import make_circles
import numpy as np 
import scipy as sc 
from IPython.core.display import display,Javascript
import  tensorflow as tf 

n=500
p=2
res=100
X_in,y_out= make_circles(n_samples=n,factor=0.45,noise=0.1)

x0=np.linspace(-1.5,1.5,res)
x1=np.linspace(-1.5,1.5,res)
px=np.array(np.meshgrid(x0,x1)).T.reshape(-1,2)
py=np.zeros((res,res))+0.5
fig=plt.figure(figsize=(8,8))
plt.pcolormesh(x0,x1,py,cmap='coolwarm',vmin=0,vmax=1)

plt.scatter(X_in[y_out==0,0],X_in[y_out==0,1],c='skyblue')
plt.scatter(X_in[y_out==1,0],X_in[y_out==1,1],c='y')
plt.tick_params(labelbottom=False,labelleft=False)


# In[58]:


# Puntos de entrada 
grafo= tf.Graph()
nn=[2,16,8,1]
lr=0.05
n_epoch=1000
with grafo.as_default():
    X=tf.compat.v1.placeholder(dtype=tf.float32,shape=(None,X_in.shape[1]),name='Input')
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
    


# In[59]:


predicciones=[]
with tf.compat.v1.Session(graph=grafo) as sess:
     sess.run(tf.compat.v1.global_variables_initializer()) 
     for i in range(n_epoch):
         _,perdida,pred=sess.run([optimizer,loss,py],feed_dict={X:X_in,y:y_out})
    
         if i % 25 == 0:
             acc=np.mean(np.round(pred)==y_out)
             print('Step: ',i,'/',n_epoch,'--Loss: ',perdida,'--Exactitud: ',acc)
             pred=sess.run(py,feed_dict={X:px}).reshape((res,res))
             predicciones.append(pred)


# In[60]:


# Animacion
ims=[]
for f in range(len(predicciones)):
    im=plt.pcolormesh(x0,x1,predicciones[f], cmap='coolwarm',animated=True)
    plt.scatter(X_in[y_out==0,0],X_in[y_out==0,1],c='skyblue')
    plt.scatter(X_in[y_out==1,0],X_in[y_out==1,1],c='y')
    plt.tick_params(labelbottom=False,labelleft=False)
    ims.append([im])

ani=animation.ArtistAnimation(fig,ims,interval=50,blit=True,repeat_delay=1000)
Javascript(ani.to_jshtml());


# ## Modelado a Tensorflow 2.0

# In[61]:


def input_fn(X,y,batch_size=32,shuffle=True,num_epo=None):
    X=tf.convert_to_tensor(X,dtype=tf.float32)
    y=tf.convert_to_tensor(y,dtype=tf.float32)
    ds=tf.data.Dataset.from_tensor_slices((X,y))
    if shuffle:
        ds=ds.shuffle(buffer_size=len(y)).batch(batch_size=batch_size).repeat(count=num_epo)
    else:
        ds=ds.batch(batch_size=batch_size)
    return ds


# In[71]:


# Red Neuronal
nn=[2,16,8,1]
lr=0.05
lista_predicciones=[]
class Neurona(object):
    def __init__(self,learning_rate=0.05):
        self.learning_rate=learning_rate
        # Capa1
        self.W1=tf.Variable(tf.random.normal((nn[0],nn[1])),name='Peso_1',trainable=True)
        self.b1=tf.Variable(tf.random.normal((nn[1],)),name='Sesgo_1',trainable=True)
        
        # Capa2
        self.W2=tf.Variable(tf.random.normal((nn[1],nn[2])),name='Peso_2',trainable=True)
        self.b2=tf.Variable(tf.random.normal((nn[2],)),name='Sesgo_2',trainable=True)
  
        # Capa3
        self.W3=tf.Variable(tf.random.normal((nn[2],nn[3])),name='Peso_3',trainable=True)
        self.b3=tf.Variable(tf.random.normal((nn[3],)),name='Sesgo_3',trainable=True)
        
    
    def predict(self,X):
        self.func_1=tf.nn.relu(tf.add(tf.matmul(X,self.W1),self.b1))
        self.func_2=tf.nn.relu(tf.add(tf.matmul(self.func_1,self.W2),self.b2))
        self.py=tf.nn.sigmoid(tf.add(tf.matmul(self.func_2,self.W3),self.b3))[:,0]
        return  self.py

    def losses(self,X,y):
        prediccion=self.predict(X)
        loss=tf.losses.mean_squared_error(y,prediccion)
        return loss

    def optimizador(self,X,y):
        with tf.GradientTape(persistent=True) as tape:
            coste=self.losses(X,y)
        dw1_coste=tape.gradient(coste,self.W1)
        db1_coste=tape.gradient(coste,self.b1)
        dw2_coste=tape.gradient(coste,self.W2)
        db2_coste=tape.gradient(coste,self.b2)
        dw3_coste=tape.gradient(coste,self.W3)
        db3_coste=tape.gradient(coste,self.b3)
        optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        optimizer.apply_gradients(zip([dw1_coste,db1_coste,dw2_coste,db2_coste,dw3_coste,db3_coste],[self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]))
        return coste

    
    def trian_model(self,X,y,step=10,num_epo=50,bloque=32,mesclado=True):
        step_x_epo=int(np.ceil(len(y)/bloque))
        data=input_fn(X,y,batch_size=bloque,shuffle=mesclado)
        for i,batch in enumerate(data):
            if i >= step_x_epo*num_epo:
                break
            x_batch,y_batch=batch
            resultado=self.optimizador(x_batch,y_batch)
            if i % step==0:
                pred=self.predict(x_batch)
                rmse=np.mean(tf.sqrt(tf.square(y_batch-pred)))
                print('Step: ',i,'/',num_epo,'--Loss (%): ',resultado.numpy()*100,'--RMSE (%): ',rmse*100)


# In[76]:


modelo=Neurona(learning_rate=0.05)


# In[77]:


modelo.trian_model(X_in,y_out,step=10,num_epo=50)


# In[78]:


def predecir(X):
    X=tf.convert_to_tensor(X,dtype=tf.float32)
    pred=modelo.predict(X)
    lista_predicciones.append(pred.numpy().reshape((res,res)))
    
predecir(px)


# In[79]:


ims=[]
for f in range(len(lista_predicciones)):
    im=plt.pcolormesh(x0,x1,lista_predicciones[f], cmap='coolwarm',animated=True)
    plt.scatter(X_in[y_out==0,0],X_in[y_out==0,1],c='skyblue')
    plt.scatter(X_in[y_out==1,0],X_in[y_out==1,1],c='y')
    plt.tick_params(labelbottom=False,labelleft=False)
    ims.append([im])

ani=animation.ArtistAnimation(fig,ims,interval=50,blit=True,repeat_delay=1000)
Javascript(ani.to_jshtml());


# In[67]:


len(y_out)


# In[ ]:




