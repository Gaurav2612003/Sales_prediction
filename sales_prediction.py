#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[49]:


df = pd.read_csv(r"C:\Users\91752\OneDrive - DIT University\Desktop\advertising.csv")


# In[52]:


df.head()


# In[53]:


df.shape


# In[54]:


df.info()


# In[55]:


df.isnull().sum()


# In[56]:


# no null value 
# we don't have to habdle missing value 


# In[57]:


#Pairplot to visualize relationships between variables
sns.pairplot(data=df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1)
plt.suptitle("Pairplot of Sales vs. Advertising Channels", y=1.02)
plt.show() 


# In[58]:


# Heatmap to visualize correlations
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# In[59]:


x = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']


# In[60]:


print( x, y)


# In[66]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[67]:


# Create and Train the Model
model = LinearRegression()
model.fit(x_train, y_train)


# In[68]:


# Make Predictions
y_pred = model.predict(x_test)


# In[69]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[77]:


# Visualization of Predicted vs. Actual Sales
plt.figure(figsize=(6,4), facecolor="lightblue")
ax = plt.scatter(y_test, y_pred)
ax.set_facecolor("black")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual Sales vs. Predicted Sales")
plt.show()


# In[78]:


new_data = pd.DataFrame({
    'TV': [100],
    'Radio': [25],
    'Newspaper': [10]
})

predicted_sales = model.predict(new_data)
print("\nPredicted Sales:", predicted_sales[0])

