#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import clear_output
from matplotlib import pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn import metrics
import tensorflow as tf 
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
pd.options.display.max_rows=10
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(100)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data_train=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/titanic_train.csv',index_col=False,sep=',')
data_train.head()


# In[3]:


data_train.pop('ii')


# In[4]:


data_train.isna().sum()


# In[5]:


data_train=data_train.reindex(np.random.permutation(data_train.index))
data_train.head()


# In[6]:


data_train.describe()


# In[7]:


data_train.corr()


# In[8]:


data_train.duplicated().sum()


# In[9]:


data_train=data_train.drop_duplicates()


# In[10]:


data_train.duplicated().sum()


# In[12]:


plt.scatter(x=data_train['fare'],y=data_train['survived'],alpha=0.5)
plt.xlabel('Fare')
plt.ylabel('Survived')
plt.show()


# In[7]:


data_train.hist('age','class');


# In[8]:


data_train.alone.value_counts().plot(kind='barh');


# In[9]:


data_train.hist('age','alone');


# In[10]:


data_train.hist('age','sex');


# In[11]:


data_train.sex.value_counts().plot(kind='barh');


# In[12]:


data_train['class'].value_counts().plot(kind='barh');


# In[11]:


data_train.groupby('sex').survived.mean().plot(kind='barh');


# In[33]:


data_train.groupby('class').survived.mean().plot(kind='barh');


# In[11]:


X_train,y_train,X_eval,y_eval=train_test_split(data_train.drop('survived',axis=1),data_train['survived'],train_size=0.80)
X_train.shape,y_train.shape,X_eval.shape,y_eval.shape


# In[12]:


Col_categorica=X_train.select_dtypes(np.object).columns
Col_numerica=X_train.select_dtypes(np.number).columns

feature_column=[]
for feature_name in Col_categorica:
    vocabulario=X_train[feature_name].unique()
    feature_column.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulario))
for feature_name in Col_numerica:
    feature_column.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))


# In[13]:


def input_fn(X,y,batch_size=10,shuffle=True,num_epo=None):
    ds=tf.data.Dataset.from_tensor_slices((dict(X),y))
    if shuffle:
        ds=ds.shuffle(buffer_size=len(y_eval)).batch(batch_size=batch_size).repeat(num_epo)
    else:
        ds=ds.batch(batch_size=batch_size).repeat(num_epo)
    return ds
set_train=lambda: input_fn(X_train,X_eval,batch_size=140,num_epo=None)
set_eval=lambda: input_fn(y_train,y_eval,shuffle=False,num_epo=1)


# In[14]:


leng_regresor_cls=tf.estimator.LinearClassifier(feature_columns=feature_column,optimizer=tf.keras.optimizers.Ftrl(learning_rate=0.03,l2_regularization_strength=0.003))
leng_regresor_cls.train(set_train,max_steps=900)
resultados=leng_regresor_cls.evaluate(set_eval)
print(resultados)


# In[15]:


prediciones_dict=list(leng_regresor_cls.predict(set_eval))
data_short=pd.Series([pred['probabilities'][1] for pred  in prediciones_dict])
data_short.plot(kind='hist', bins=20, title='predicted probabilities');


# In[17]:


from sklearn.metrics import roc_curve

fps,tpr,_=roc_curve(y_eval,data_short)
plt.plot(fps,tpr)
plt.title('ROC curve')
plt.xlabel('falsos positivos')
plt.ylabel('verdaderos positivos')
plt.xlim(0,)
plt.ylim(0,)
plt.show();


# ## Mejoramiento del modelo

# In[18]:


def get_cuantiles(data,num_bucket):
    bondaries=np.arange(1.0,num_bucket)/num_bucket
    cuantiles=data.quantile(bondaries)
    return [cuantiles[p] for p in cuantiles.keys()]

age=tf.feature_column.numeric_column('age')
bucketize_age=tf.feature_column.bucketized_column(age,boundaries=get_cuantiles(X_train['age'],4))
fare=tf.feature_column.numeric_column('fare')
bucketize_fare=tf.feature_column.bucketized_column(fare,boundaries=get_cuantiles(X_train['fare'],4))
n_siblings_spouses=tf.feature_column.numeric_column('n_siblings_spouses')
bucketize_n_siblings_spouses=tf.feature_column.bucketized_column(n_siblings_spouses,boundaries=get_cuantiles(X_train['n_siblings_spouses'],2))


# In[20]:


age_x_fare=tf.feature_column.crossed_column([bucketize_age,bucketize_fare],hash_bucket_size=100)
columns_derivada=[age_x_fare]
leng_regresor_cls_1=tf.estimator.LinearClassifier(feature_columns=feature_column+columns_derivada,optimizer=tf.keras.optimizers.Ftrl(learning_rate=0.03,l2_regularization_strength=0.003))
leng_regresor_cls_1.train(set_train,max_steps=900)
resultados=leng_regresor_cls_1.evaluate(set_eval)
print(resultados)


# In[21]:


prediciones_dict=list(leng_regresor_cls_1.predict(set_eval))
data_short=pd.Series([pred['probabilities'][1] for pred  in prediciones_dict])
data_short.plot(kind='hist', bins=20, title='predicted probabilities');


# In[22]:


fps,tpr,_=roc_curve(y_eval,data_short)
plt.plot(fps,tpr)
plt.title('ROC curve')
plt.xlabel('falsos positivos')
plt.ylabel('verdaderos positivos')
plt.xlim(0,)
plt.ylim(0,)
plt.show();


# In[23]:


age_x_class=tf.feature_column.crossed_column(['age','sex'],hash_bucket_size=100)
columns_derivada_1=[age_x_class]
leng_regresor_cls_2=tf.estimator.LinearClassifier(feature_columns=feature_column+columns_derivada_1,optimizer=tf.keras.optimizers.Ftrl(learning_rate=0.03,l2_regularization_strength=0.003))
leng_regresor_cls_2.train(set_train,max_steps=900)
resultados=leng_regresor_cls_2.evaluate(set_eval)
print(resultados)


# In[29]:


fare_spouses=tf.feature_column.crossed_column([bucketize_fare,bucketize_n_siblings_spouses],hash_bucket_size=100)
columns_derivada_2=[fare_spouses]
leng_regresor_cls_3=tf.estimator.LinearClassifier(feature_columns=feature_column+columns_derivada_2+columns_derivada_1,optimizer=tf.keras.optimizers.Ftrl(learning_rate=0.05,l2_regularization_strength=0.003))
leng_regresor_cls_3.train(set_train,max_steps=900)
resultados=leng_regresor_cls_3.evaluate(set_eval)
print(resultados)


# In[30]:


data_test=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/titanic_eval.csv',index_col=False,sep=',')
data_test.head()


# In[31]:


data_test.pop('jj')
data_test.head()


# In[32]:


data_test=data_test.reindex(np.random.permutation(data_test.index))
data_test.head()


# In[35]:


y_muesta=data_test.drop('survived',axis=1)
set_test=lambda: input_fn(y_muesta,data_test['survived'],shuffle=False,num_epo=1)

resultados=leng_regresor_cls_1.evaluate(set_test)
print(resultados)


# In[36]:


prediciones_dict=list(leng_regresor_cls_1.predict(set_test))
data_short=pd.Series([pred['probabilities'][1] for pred  in prediciones_dict])
data_short.plot(kind='hist', bins=20, title='predicted probabilities');


# In[37]:


fps,tpr,_=roc_curve(data_test['survived'],data_short)
plt.plot(fps,tpr)
plt.title('ROC curve')
plt.xlabel('falsos positivos')
plt.ylabel('verdaderos positivos')
plt.xlim(0,)
plt.ylim(0,)
plt.show();


# In[38]:


resultados=leng_regresor_cls_3.evaluate(set_test)
print(resultados)


# In[39]:


prediciones_dict=list(leng_regresor_cls_3.predict(set_test))
data_short=pd.Series([pred['probabilities'][1] for pred  in prediciones_dict])
data_short.plot(kind='hist', bins=20, title='predicted probabilities');


# In[40]:


fps,tpr,_=roc_curve(data_test['survived'],data_short)
plt.plot(fps,tpr)
plt.title('ROC curve')
plt.xlabel('falsos positivos')
plt.ylabel('verdaderos positivos')
plt.xlim(0,)
plt.ylim(0,)
plt.show();


# In[41]:


resultados=leng_regresor_cls_2.evaluate(set_test)
print(resultados)


# In[42]:


prediciones_dict=list(leng_regresor_cls_2.predict(set_test))
data_short=pd.Series([pred['probabilities'][1] for pred  in prediciones_dict])
data_short.plot(kind='hist', bins=20, title='predicted probabilities');


# In[43]:


fps,tpr,_=roc_curve(data_test['survived'],data_short)
plt.plot(fps,tpr)
plt.title('ROC curve')
plt.xlabel('falsos positivos')
plt.ylabel('verdaderos positivos')
plt.xlim(0,)
plt.ylim(0,)
plt.show();


# In[ ]:




