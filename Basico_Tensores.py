#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy  as np 
# escalar 
entero=tf.Variable(5,tf.int32)
puntoflotante=tf.Variable(4.9988989,tf.float32)

# vector
cadena=tf.compat.v1.string_join(['Me',' llamo ','Maikel'])
enteros_v=tf.Variable([1,2,3,4,5],tf.int32)
flotantes=tf.Variable([1.1,2.2,3.3,5.213],tf.float32)

# matrices
matriz_enteros=tf.Variable([[1,2,4,5],[5,4,7,8]],tf.int32)
matriz_flotante=tf.Variable([[1.2,3.5,4.5,7.8],[4.2,3.5,4.8,4.2]],tf.float32)
matriz_cadena=tf.Variable([['Hola','Mundo'],['Ya','tendorflow']],tf.string)
# constantes
x=tf.constant(10)
y=tf.constant(5)


# In[2]:


# convirtiendo vectores  a tensores
with tf.compat.v1.Session() as sess:
    vector_1=[22,12,33,45,78]
    tensor=tf.convert_to_tensor(vector_1)
    print(sess.run(tensor))
    sess.close()


# In[3]:


# matrices de numpy a tensores 
with tf.compat.v1.Session() as sess:
    mt=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    matriz_tensor=tf.convert_to_tensor(mt)
    print(sess.run(matriz_tensor))
    sess.close()


# In[4]:


with tf.compat.v1.Session() as sess:
    mt_1=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    matriz_=tf.convert_to_tensor(mt_1)
    multi_tensor=matriz_tensor*matriz_
    print(sess.run(multi_tensor))
    sess.close()


# In[5]:


with tf.compat.v1.Session() as sess:
    mult_tensor=tf.matmul(matriz_,mt)
    print(sess.run(mult_tensor))
    sess.close()


# In[6]:


with tf.compat.v1.Session() as sess:
    xsuma=tf.reduce_sum(matriz_,axis=0)
    print(sess.run(xsuma))
    sess.close()


# In[7]:


matriz_.shape


# In[8]:


with tf.compat.v1.Session() as sess:
    media_por_colum=tf.reduce_mean(mult_tensor)
    print(sess.run(media_por_colum))
    sess.close()


# In[9]:


with tf.compat.v1.Session() as sess:
    reforma = tf.reshape(matriz_,[16])
    print(sess.run(reforma))
    sess.close()


# In[10]:


with tf.compat.v1.Session() as sess:
    matrix1=tf.constant([[3.,3.]])
    matrix2=tf.constant([[2.],[2.]])
    multiplicadion=tf.matmul(matrix1,matrix2)
    print(sess.run(multiplicadion))
    sess.close()


# In[11]:


f=tf.Graph()
with f.as_default():
    x = tf.compat.v1.placeholder(dtype=tf.float32,
                       shape=(None), name='x')
    w = tf.Variable(2.0, name='weight')
    b = tf.Variable(0.7, name='bias')
    z = w*x + b
    init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session(graph=f) as sess:
    sess.run(init)
    for m in [1.0,0.6,-1.8]:
        print('x={} --> z={}'.format(m,sess.run(z,feed_dict={x:m})))
    sess.close()


# In[12]:


# Conteo hasta 10
grafo=tf.Graph()
with grafo.as_default():
    state=tf.Variable(0)
    one=tf.constant(1)
    addtion=tf.add(state,one)
    update=tf.compat.v1.assign(state,addtion)
    inicial=tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session(graph=grafo) as sess:
    sess.run(inicial)
    print('The starting state is ',sess.run(state))
    for c in range(10):
        sess.run(update)
    print('The end state is ',sess.run(state))
    sess.close()


# In[13]:


# Trabajo con array
grafo=tf.Graph()
with grafo.as_default():
    x=tf.compat.v1.placeholder(dtype=tf.float32,shape=(None,2,3),name='input_x')
    x2=tf.reshape(x,shape=(-1,6),name='x2')
    xsum=tf.compat.v1.reduce_sum(x2,axis=0,name='col_sum')
    xmean=tf.compat.v1.reduce_mean(x2,axis=0,name='col_mean')
with tf.compat.v1.Session(graph=grafo) as sess:
    x_array=np.arange(18).reshape(3,2,3)
    print('input shape:',x_array)
    print('Reshape:\n',sess.run(x2,feed_dict={x:x_array}))
    print('Column Sums:\n',sess.run(xsum,feed_dict={x:x_array}))
    print('Column Mean:\n',sess.run(xmean,feed_dict={x:x_array}))
    sess.close()


# Intro a Pandas 

# In[3]:


from __future__ import print_function
import pandas as pd 
import numpy as np 

pd.__version__
city_name=pd.Series(['San Francisco','San Jose','Sacramento'])
population=pd.Series([852469,1015785,485199])
pd.DataFrame({'City_name':city_name,'Population':population})


# In[11]:


cities=pd.DataFrame({'City_name':city_name,'Population':population})
cities.hist('Population');


# In[12]:


cities.plot('City_name');


# In[24]:


population.apply(lambda val: val>1000000)
cities['Area square miles']=pd.Series([46.87,176.53,97.92])
cities['Population density']=cities['Population']/cities['Area square miles']
cities['Nombre santo']=(cities['Area square miles'] > 50) & cities['City_name'].apply(lambda name:name.startswith('San'))
cities.head()


# In[34]:


cities.plot('Area square miles');


# ## Tensorflow 2.0

# In[14]:


np.set_printoptions(precision=3)
a=np.array([1,2,3])
b=[4,5,6]
ta=tf.convert_to_tensor(a)
tb=tf.convert_to_tensor(b)
tf.print(ta,' ',tb)


# In[15]:


t_ones=tf.ones((2,3))
t_ones.shape


# In[16]:


t_ones.numpy()


# In[17]:


const=tf.constant([1.2,3.,np.pi],dtype=tf.float16)
tf.print(const)


# In[18]:


ta_new=tf.cast(ta,tf.float16)
tf.print(ta_new.dtype)


# In[19]:


tensor=tf.random.uniform(shape=(3,2))
traspose=tf.transpose(tensor)
tf.print(tensor,'-->',traspose)


# In[20]:


t=tf.zeros(shape=(30,))
t_reshape=tf.reshape(t,shape=(5,6))
print(t_reshape.shape)


# In[21]:


t=tf.zeros((1,2,1,4,1))
t_aqz=tf.squeeze(t,axis=(2,4))
print(t.shape,'-->',t_aqz.shape)


# In[22]:


t1=tf.random.uniform(shape=(5,2),minval=-1.0,maxval=1.0)
t2=tf.random.normal(shape=(5,2),mean=0.0,stddev=1.0)
t3=tf.multiply(t1,t2).numpy()
print(t3)


# In[23]:


t4=tf.linalg.matmul(t1,t2,transpose_b=True)
print(t4.numpy())


# In[24]:


t5=tf.linalg.matmul(t1,t2,transpose_a=True)
print(t5.numpy())


# In[25]:


norm_tensor=tf.norm(t1,ord=2,axis=1).numpy()
print(norm_tensor)


# In[26]:


tf.random.set_seed(1)
t=tf.random.uniform(shape=(6,))
print(t.numpy())


# In[27]:


t_split=tf.split(t,num_or_size_splits=3)
[item.numpy() for item in t_split]


# In[28]:


t=tf.random.uniform(shape=(5,))
print(t.numpy())
t_split=tf.split(t,num_or_size_splits=[3,2])
[item.numpy() for item in t_split]


# In[29]:


A=tf.ones(shape=(3,))
B=tf.zeros(shape=(2,))
C=tf.concat([A,B],axis=0)
tf.print(C)


# In[30]:


A=tf.ones(shape=(3,))
B=tf.zeros(shape=(3,))
C=tf.stack([A,B],axis=1)
tf.print(C)


# In[31]:


a=[1.2,3.4,7.5,4.1,5.0,1.0]
ds=tf.data.Dataset.from_tensor_slices(a)
tf.print(ds)
for item in ds:
    print(item)


# In[32]:


ds_batch=ds.batch(3)
for i, item in enumerate(ds_batch,1):
    print('batch {} '.format(i),item.numpy())


# In[33]:


t_x=tf.random.uniform(shape=(4,3),dtype=tf.float32)
t_y=tf.range(4)

ds_x=tf.data.Dataset.from_tensor_slices(t_x)
ds_y=tf.data.Dataset.from_tensor_slices(t_y)

ds_joint=tf.data.Dataset.zip((ds_x,ds_y)) # Se puede hacer con from tensor slices tambien
for ex in ds_joint:
    print('X : ',ex[0].numpy(),' y : ',ex[1].numpy())


# In[34]:


ds=ds_joint.shuffle(buffer_size=len(t_x))
for ex in ds:
     print('X : ',ex[0].numpy(),' y : ',ex[1].numpy())


# In[35]:


ds=ds_joint.batch(batch_size=3,drop_remainder=False)
batch_x,batch_y=next(iter(ds))
print('Batch_X :\n',batch_x.numpy())
print('\n')
print('Batch_y :\n',batch_y.numpy())


# In[36]:


ds=ds_joint.batch(3).repeat(count=2)
for i,(batch_x,batch_y) in enumerate(ds):
    print(i,batch_x.shape,batch_y.numpy())


# In[37]:


ds=ds_joint.shuffle(4).batch(2).repeat(3)
for i,(batch_x,batch_y) in enumerate(ds):
      print(i,batch_x.shape,batch_y.numpy()) # el mejor orden de mesclado


# In[38]:


ds=ds_joint.batch(2).shuffle(4).repeat(3)
for i,(batch_x,batch_y) in enumerate(ds):
      print(i,batch_x.shape,batch_y.numpy())


# In[ ]:




