#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


import streamlit as st


# In[3]:


df = pd.read_csv('heart.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.info()


# In[10]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True, cmap = 'terrain' )


# In[11]:


sns.pairplot(data=df)


# In[12]:


sns.pairplot(data=df, hue = 'target')


# In[13]:


# box and whiskers plot
df.plot(kind='box', subplots=True, layout=(5,5), figsize=(15,15))
plt.show()


# In[14]:


sns.catplot(data=df, x='sex', y='age',  hue='target', palette='husl')


# In[15]:


sns.barplot(data=df, x='sex', y='chol', hue='target', palette='spring')


# In[16]:


df['sex'].value_counts()


# In[17]:


df['target'].value_counts()


# In[18]:


df['thal'].value_counts()


# In[19]:


sns.countplot(x='sex', data=df, palette='husl', hue='target')


# In[20]:


sns.countplot(x='target',palette='BuGn', data=df)


# In[21]:


sns.countplot(x='ca',hue='target',data=df)


# In[22]:


df['ca'].value_counts()


# In[23]:


sns.countplot(x='thal',data=df, hue='target', palette='BuPu' )


# In[24]:


sns.countplot(x='thal', hue='sex',data=df, palette='terrain')


# In[25]:


df['cp'].value_counts()  # chest pain type


# In[26]:


sns.countplot(x='cp' ,hue='target', data=df, palette='rocket')


# In[27]:


sns.countplot(x='cp', hue='sex',data=df, palette='BrBG')


# In[28]:


sns.boxplot(x='sex', y='chol', hue='target', palette='seismic', data=df)


# In[29]:


sns.barplot(x='sex', y='cp', hue='target',data=df, palette='cividis')


# In[30]:


sns.barplot(x='sex', y='thal', data=df, hue='target', palette='nipy_spectral')


# In[31]:


sns.barplot(x='target', y='ca', hue='sex', data=df, palette='mako')


# In[32]:


sns.barplot(x='sex', y='oldpeak', hue='target', palette='rainbow', data=df)


# In[33]:


df['fbs'].value_counts()


# In[34]:


sns.barplot(x='fbs', y='chol', hue='target', data=df,palette='plasma' )


# In[35]:


sns.barplot(x='sex',y='target', hue='fbs',data=df)


# In[36]:


gen = pd.crosstab(df['sex'], df['target'])
print(gen)


# In[37]:


gen.plot(kind='bar', stacked=True, color=['green','yellow'], grid=False)


# In[38]:


temp=pd.crosstab(index=df['sex'],
            columns=[df['thal']], 
            margins=True)
temp


# In[39]:


temp.plot(kind="bar",stacked=True)
plt.show()


# In[40]:


temp=pd.crosstab(index=df['target'],
            columns=[df['thal']], 
            margins=True)
temp


# In[41]:


temp.plot(kind='bar', stacked=True)
plt.show()


# In[42]:


chest_pain = pd.crosstab(df['cp'], df['target'])
chest_pain


# In[43]:


chest_pain.plot(kind='bar', stacked=True, color=['purple','blue'], grid=False)


# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()  
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
df[columns_to_scale] = StandardScaler.fit_transform(df[columns_to_scale])


# In[45]:


df.head()


# In[46]:


X= df.drop(['target'], axis=1)
y= df['target']


# In[47]:


X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)


# In[48]:


print('X_train-', X_train.size)
print('X_test-',X_test.size)
print('y_train-', y_train.size)
print('y_test-', y_test.size)


# In[49]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

model1=lr.fit(X_train,y_train)
prediction1=model1.predict(X_test)


# In[50]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,prediction1)
cm


# In[51]:


sns.heatmap(cm, annot=True,cmap='BuPu')


# In[52]:


TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]
print('Testing Accuracy:',(TP+TN)/(TP+TN+FN+FP))


# In[53]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction1)


# In[54]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction1))


# In[55]:


from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()
model2=dtc.fit(X_train,y_train)
prediction2=model2.predict(X_test)
cm2= confusion_matrix(y_test,prediction2)


# In[56]:


cm2


# In[57]:


accuracy_score(y_test,prediction2)


# In[58]:


print(classification_report(y_test, prediction2))


# In[59]:


from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
model3 = rfc.fit(X_train, y_train)
prediction3 = model3.predict(X_test)
confusion_matrix(y_test, prediction3)


# In[60]:


accuracy_score(y_test, prediction3)


# In[61]:


print(classification_report(y_test, prediction3))


# In[62]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[63]:


from sklearn.svm import SVC

svm=SVC()
model4=svm.fit(X_train,y_train)
prediction4=model4.predict(X_test)
cm4= confusion_matrix(y_test,prediction4)


# In[64]:


cm4


# In[65]:


accuracy_score(y_test, prediction4)


# In[66]:


from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
model5 = NB.fit(X_train, y_train)
prediction5 = model5.predict(X_test)
cm5= confusion_matrix(y_test, prediction5)


# In[67]:


cm5


# In[68]:


accuracy_score(y_test, prediction5)


# In[69]:


print('cm4', cm4)
print('-----------')
print('cm5',cm5)


# In[70]:


from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier()
model6 = KNN.fit(X_train, y_train)
prediction6 = model6.predict(X_test)
cm6= confusion_matrix(y_test, prediction5)
cm6


# In[71]:


print('KNN :', accuracy_score(y_test, prediction6))
print('lr :', accuracy_score(y_test, prediction1))
print('dtc :', accuracy_score(y_test, prediction2))
print('rfc :', accuracy_score(y_test, prediction3))
print('NB: ', accuracy_score(y_test, prediction4))
print('SVC :', accuracy_score(y_test, prediction5))

