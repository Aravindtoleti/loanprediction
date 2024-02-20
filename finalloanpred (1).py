#!/usr/bin/env python
# coding: utf-8

# # IMPORT ALL THE NECESSARY LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# # IMPORT THE DATASET

# In[2]:


data = pd.read_csv('C://Users//ADITYA//Desktop//loan.csv')


# # UNDERSTAND THE DATASET

# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])


# In[7]:


data.info()


# # DEALING WITH THE MISSING VALUES

# In[8]:


data.isnull().sum()


# In[9]:


data.isnull().sum()*100 / len(data)


# In[10]:


data = data.drop('Loan_ID',axis=1)
data.head(1)


# In[11]:


columns = ['Gender','Dependents','LoanAmount','Loan_Amount_Term']
data = data.dropna(subset=columns)


# In[12]:


data.isnull().sum()*100 / len(data)


# In[13]:


data['Self_Employed'].mode()[0]


# In[14]:


data['Self_Employed'] =data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])


# In[15]:


data.isnull().sum()*100 / len(data)


# In[16]:


data['Self_Employed'].unique()


# In[17]:


data['Credit_History'].mode()[0]


# In[18]:


data['Credit_History'] =data['Credit_History'].fillna(data['Credit_History'].mode()[0])


# In[19]:


data.isnull().sum()*100 / len(data)


# # DATA VISUALIZATION

# In[20]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (100, 100))
sns.set(font_scale = 5)
plt.subplot(331)
sns.countplot(x='Gender',hue='Loan_Status', data=data)

plt.subplot(332)
sns.countplot(x='Married',hue='Loan_Status', data=data)

plt.subplot(333)
sns.countplot(x='Education',hue='Loan_Status', data=data)

plt.subplot(334)
sns.countplot(x='Self_Employed',hue='Loan_Status', data=data)

plt.subplot(335)
sns.countplot(x='Property_Area',hue='Loan_Status', data=data)


# # HANDLING CATEGORICAL COLUMNS

# In[21]:


data.sample(5)


# In[22]:


data['Dependents'] =data['Dependents'].replace(to_replace="3+",value='4')


# In[23]:


data['Dependents'].unique()


# In[24]:


data['Property_Area'].unique()


# In[25]:


data['Gender'] = data['Gender'].map({'Male':1,'Female':0}).astype('int')
data['Married'] = data['Married'].map({'Yes':1,'No':0}).astype('int')
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
data['Property_Area'] = data['Property_Area'].map({'Rural':0,'Urban':1,'Semiurban':2,'Semi-urban':3,'semiurban':4}).astype('int')
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0}).astype('int')


# In[26]:


data['Gender'].unique()


# In[27]:


data.head()


# # STORE FEATURE MATRIX IN X AND RESPONSE (TARGET) IN VECTOR Y

# In[28]:


X = data.drop('Loan_Status',axis=1)


# In[29]:


y = data['Loan_Status']


# In[30]:


y


# # FEATURE SCALING

# In[31]:


data.head()


# In[32]:


cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']


# In[33]:


from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X[cols]=st.fit_transform(X[cols])


# In[34]:


X


# # DIVIDE THE DATASET INTO TRAINING & TESTING SET

# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.25, random_state = 42)


# In[36]:


X_train.shape


# In[37]:


X_test.shape


# # BUILD THE ML MODEL & FIT THE MODEL ON THE TRAINING DATASET

# In[38]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


# In[39]:


y_pred = model.predict(X_test)


# # ACCURACY OF THE  MODEL

# In[43]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test , y_pred)


# # CONFUSION MATRIX

# In[40]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)
cm


# In[41]:


sns.heatmap(cm , annot=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




