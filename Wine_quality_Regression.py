#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
import collections
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer


# In[4]:


#loading development dataset
        
df = pd.read_csv("./competition_dataset/dev.tsv", sep = '\t')

#dropping duplicates

df.drop_duplicates(inplace = True)


# In[5]:


#data preprocessing 

#dealing with null values

mask = df['region_2'].isnull()
df['region_2'][mask] = df['region_1'][mask]

#filling null values of dataframe with value 'other'
df['designation'] = df['designation'].fillna('missing')
df['region_1'] = df['region_1'].fillna('missing')
df['region_2'] = df['region_2'].fillna('missing')

df.dropna(inplace=True)


# In[6]:


#hold out

X = df.drop(columns =  'quality')
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1423, test_size = 0.2) 

#saving the description and dropping them

tr_des = X_train['description']
te_des = X_test['description']

X_train = X_train.drop(columns = 'description')
X_test = X_test.drop(columns = 'description')


# In[7]:


#one hot encoding for validation phase

OneHot_enc = OneHotEncoder(handle_unknown = 'ignore')


OneHot_enc.fit(X_train)

X_train = OneHot_enc.transform(X_train)
X_test = OneHot_enc.transform(X_test)


# In[8]:


#lemmatisation

def Lemmatisation(des):
    
    lemm = []
    for w in des:
        split_text = w.split()
    
    #Lemmatisation
        lem = WordNetLemmatizer()
        split_text = [lem.lemmatize(word) for word in split_text] 
        split_text = " ".join(split_text)
        lemm.append(split_text)
        
    return lemm



lemm_train = Lemmatisation(tr_des)
lemm_test = Lemmatisation(te_des)


# In[9]:


#Textual feature extraction

tf_idf = TfidfVectorizer(stop_words = 'english',ngram_range=(1,1), min_df = 10, max_df=10000)
tf_idf.fit(lemm_train)
tr_des_ = tf_idf.transform(lemm_train)
te_des_ = tf_idf.transform(lemm_test)

#create the sets with all the feaures extracted used for mlp

X_train_ = sparse.hstack([tr_des_, X_train], format="csr")
X_test_ = sparse.hstack([te_des_, X_test], format="csr")


# In[ ]:


#validation of the random forest

rf = RandomForestRegressor(n_estimators=350, max_features=5)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_score = r2_score(y_test, rf_pred)

print('the r2 value obtained, with the Random Forest regressor, on the validation set is:', rf_score)


# In[ ]:


#validation of the multi-layer perceptron

mlp = MLPRegressor(hidden_layer_sizes= 300, learning_rate= 'adaptive', learning_rate_init= 0.01, solver= 'sgd', verbose= True, early_stopping = True)
        
mlp.fit(X_train_, y_train)
mlp_pred = mlp.predict(X_test_)
mlp_score = r2_score(y_test, mlp_pred)

print('the r2 value obtained, with the MLP regressor, on the validation set is:', mlp_score)


# In[ ]:


#Testing the final models on the evaluation set

#loading the evaluation set

ev = pd.read_csv("./competition_dataset/eval.tsv", sep = '\t')

#preprocessing the evaluation set

mask2 = ev['region_2'].isnull()
ev['region_2'][mask2] = ev['region_1'][mask2]
ev['designation'] = ev['designation'].fillna('missing')
ev['region_1'] = ev['region_1'].fillna('missing')
ev['region_2'] = ev['region_2'].fillna('missing')

#saving and dropping descriptions from the eval

ev_des = ev['description']
ev = ev.drop(columns = 'description')

#saving and dropping description from the development

dev_des = X['description']
dev = X.drop(columns = 'description')


# In[ ]:


#One hot encoding for the final model



Final_OneHot = OneHotEncoder(handle_unknown = 'ignore')


Final_OneHot.fit(dev)

dev = Final_OneHot.transform(dev)
ev = Final_OneHot.transform(ev)


# In[ ]:


#Lemmatisation for the final model

lemm_dev = Lemmatisation(dev_des)
lemm_ev = Lemmatisation(ev_des)


#Tf-idf for the final model

Final_tf_idf = TfidfVectorizer(stop_words = 'english',ngram_range=(1,1), min_df = 10, max_df=10000)
Final_tf_idf.fit(lemm_dev)
dev_des_ = Final_tf_idf.transform(lemm_dev)
ev_des_ = Final_tf_idf.transform(lemm_ev)

#create the sets with all the feaures extracted, used for mlp

dev_ = sparse.hstack([dev_des_, dev], format="csr")
ev_ = sparse.hstack([ev_des_, ev], format="csr")


# In[ ]:


#predicting the quality scores of the evaluation set with random forest

RF = RandomForestRegressor(n_estimators=350, max_features=5)
RF.fit(dev, y)
RF_pred = RF.predict(ev)


# In[ ]:


#predicting the quality scores of the evaluation set with MLP

MLP = MLPRegressor(hidden_layer_sizes= 300, learning_rate= 'adaptive', learning_rate_init= 0.01, solver= 'sgd', verbose= True)
        
MLP.fit(dev_, y)
MLP_pred = MLP.predict(ev_)

