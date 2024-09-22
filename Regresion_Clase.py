#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
import numpy as np 
tf.random.set_seed(1)
#X_train=np.arange(10).reshape((10,1))
X_train=np.array([0.0,1.1,2.1,3.1,4.0,5.0,6.0,7.0,8.0,9.0]).reshape((10,1))
y_train=np.array([1.0,1.3,3.1,2.0,5.0,6.3,6.6,7.4,8.0,9.0])

class TFLinreg(object):
    def __init__(self,x_dim,learning_rate=0.01,random_seed=None):
        self.x_dim=x_dim
        self.learning_rate=learning_rate
        self.g=tf.Graph()
        with self.g.as_default():
            tf.compat.v1.set_random_seed(random_seed)
            self.build()
            self.init_op=tf.compat.v1.global_variables_initializer()

    def build(self):
        self.X=tf.compat.v1.placeholder(dtype=tf.float32,shape=(None,self.x_dim),name='x_input')
        self.y=tf.compat.v1.placeholder(dtype=tf.float32,shape=(None),name='y_input')
        print(self.X)
        print(self.y)
        w=tf.Variable(tf.zeros(shape=(1)),name='Weigt')
        b=tf.Variable(tf.zeros(shape=(1)),name='bias')
        print(w)
        print(b)
        self.z_net=tf.squeeze(w*self.X+b,name='z_net')
        print(self.z_net)
        sqr_errors=tf.compat.v1.square(self.y-self.z_net,name='sqr_errors')
        print(sqr_errors)
        self.mean_cost=tf.compat.v1.reduce_mean(sqr_errors,name='mean_cost')
        optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate,name='GradientDescent')
        self.optimizer=optimizer.minimize(self.mean_cost)


# In[689]:


X_train


# In[690]:


lrmodel=TFLinreg(x_dim=X_train.shape[1],learning_rate=0.001)


# In[691]:


def train_linreg(sess,model,X_train,y_train,num_epoch=100):
    sess.run(model.init_op)
    training_cost=[]
    for i in range(num_epoch):
        _,cost=sess.run([model.optimizer,model.mean_cost],feed_dict={model.X:X_train,model.y:y_train})

        training_cost.append(cost)
    return training_cost

sess=tf.compat.v1.Session(graph=lrmodel.g)
training_cost=train_linreg(sess,lrmodel,X_train,y_train)


# In[692]:


import matplotlib.pyplot as plt 
plt.plot(range(1,len(training_cost)+1),training_cost)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Training cost')
plt.show();


# In[693]:


def predict_linreg(sess,model,X_test):
    y_pred=sess.run(model.z_net,feed_dict={model.X:X_test})
    return y_pred

plt.scatter(X_train,y_train,marker='s',s=50,label='Training Data')
plt.plot(range(X_train.shape[0]),predict_linreg(sess,lrmodel,X_train),color='gray',marker='o',markersize=6,linewidth=3,label='LinReg Model')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.show();


# ## Modelado a Tensorflow 2.0

# In[694]:


def train_model(X,y,batch_size=1,shuffle=True):
    X=tf.cast(X,dtype=tf.float32)
    y=tf.cast(y,dtype=tf.float32)
    ds=tf.data.Dataset.from_tensor_slices((X,y))
    if shuffle:
        ds=ds.shuffle(buffer_size=len(y)).repeat(count=None).batch(batch_size=batch_size)
    else:
        ds=ds.batch(batch_size=batch_size)
    return ds


# In[695]:


lista_costo=[]
w=tf.Variable(tf.zeros(shape=(1,)),name='peso',trainable=True)
b=tf.Variable(tf.zeros(shape=(1,)),name='sesgo',trainable=True)
def funcion(X,y):
    with tf.GradientTape(persistent=True) as tape:
        prediccion=tf.squeeze(w*X+b)
        loss=tf.reduce_mean(tf.square(y-prediccion))
        lista_costo.append(loss)
    dw_loss=tape.gradient(loss,w)
    db_loss=tape.gradient(loss,b)
    optimizador=tf.keras.optimizers.SGD(learning_rate=0.001)
    optimizador.apply_gradients(zip([dw_loss,db_loss],[w,b]))
    return loss


# In[696]:


num_epo=100
paso=10
batch_size=10
paso_x_epo=int(np.ceil(len(y_train)/batch_size))
ds_batch=train_model(X_train,y_train,batch_size=batch_size)
for i,batch in enumerate(ds_batch):
    if i >= paso_x_epo*num_epo:
        break
    x_batch,y_batch=batch
    coste=funcion(x_batch,y_batch)
    lista_costo.append(coste)
    if i % paso==0:
        print('Epoca {}  Perdidas {} '.format(i,coste))


# In[697]:


plt.plot(range(1,len(lista_costo)+1),lista_costo)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Training cost')
plt.show();


# In[698]:


def pred(X):
     prediccion=tf.squeeze(w*X+b)
     return prediccion

plt.scatter(X_train,y_train,marker='s',s=50,label='Training Data')
plt.plot(range(X_train.shape[0]),pred(X_train),color='red',marker='o',markersize=6,linewidth=3,label='LinReg Model')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.show();


# ## Implementacion con la API de Keras

# In[699]:


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.w=tf.Variable(tf.zeros(shape=(1,)),name='peso',trainable=True)
        self.b=tf.Variable(tf.zeros(shape=(1,)),name='sesgo',trainable=True)

    def call(self,x):
        return self.w*x+self.b
    
modelo=MyModel()
modelo.build(input_shape=(None,1))
modelo.summary()


# In[700]:


def lossis(y,pred):
    return tf.reduce_mean(tf.square(y-pred))
def train(modelo,X,y,learning_rate=0.01):
    with tf.GradientTape() as tape:
        coste=lossis(modelo(X),y)
    dw,db=tape.gradient(coste,[modelo.w,modelo.b])
    modelo.w.assign_sub(learning_rate*dw)
    modelo.b.assign_sub(learning_rate*db)
    


# In[710]:


num_epo=200
paso=100
batch_size=1
W,B=[],[]
paso_x_epo=int(np.ceil(len(y_train)/batch_size))
ds_batch=train_model(X_train,y_train,batch_size=batch_size)
for i,batch in enumerate(ds_batch):
    if i >= paso_x_epo*num_epo:
        break
    W.append(modelo.w.numpy())
    B.append(modelo.b.numpy())
    x_batch,y_batch=batch
    coste=lossis(y_batch,modelo(x_batch))
    train(modelo,x_batch,y_batch,learning_rate=0.001)
    lista_costo.append(coste)
    if i % paso==0:
        print('Epoca {}  Perdidas {} '.format(i,coste))


# In[711]:


plt.plot(range(1,len(lista_costo)+1),lista_costo)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Training cost')
plt.show();


# In[712]:


X_test=np.linspace(0,9,num=100).reshape(-1,1)
y_pred=modelo(tf.cast(X_test,tf.float32))
fig=plt.figure(figsize=(13,5))
ax=fig.add_subplot(1,2,1)
plt.plot(X_train,y_train,'o',markersize=10)
plt.plot(X_test,y_pred,'--',lw=3)
plt.legend(['Ejemplo Training','Reg Lineal'],fontsize=15)
ax.set_xlabel('x',size=15)
ax.set_ylabel('y',size=15)
ax.tick_params(axis='both',which='major',labelsize=15)
ax=fig.add_subplot(1,2,2)
plt.plot(W,lw=3)
plt.plot(B,lw=3)
plt.legend(['Peso','Sesgo'],fontsize=15)
ax.set_xlabel('Iteration',size=15)
ax.set_ylabel('Value',size=15)
ax.tick_params(axis='both',which='major',labelsize=15)
plt.show()


# ## Usando los metoddos de la API

# In[704]:


modelo_1=MyModel()
modelo_1.compile(optimizer='sgd',loss=lossis,metrics=['mae','mse'])


# In[713]:


modelo_1.fit(X_train,y_train,batch_size=1,epochs=200,steps_per_epoch=10)


# In[714]:


plt.scatter(X_train,y_train,marker='s',s=50,label='Training Data')
plt.plot(range(X_train.shape[0]),modelo_1.predict(X_train),color='red',marker='o',markersize=6,linewidth=3,label='LinReg Model')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.show();


# In[ ]:




