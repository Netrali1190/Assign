#!/usr/bin/env python
# coding: utf-8
Assignmnet no 3 grpB
# In[1]:


import  matplotlib

print(matplotlib.__version__)

from matplotlib.figure import Figure


# In[7]:


import numpy as np
import matplotlib.pyplot as plt


# In[8]:


x = np.array([0,6])
y = np.array([0,200])
plt.plot(x,y)


# In[10]:


x= np.array([1,2])
y= np.array([5,10])
plt.plot(x,y)


# In[9]:


x= np.array([1,2,5,7,12])
y= np.array([5,10,7,9,11])
plt.plot(x,y,'o')


# In[10]:


plt.plot(y)


# In[11]:


x= np.array([1,2,5,7,12])
y= np.array([5,10,7,9,11])

plt.plot(x,y, marker ='_')
plt.show()


# # Format String fmt

# In[12]:


x= np.array([1,2,5,7,12])
y= np.array([5,10,7,9,11])

plt.plot(x,y, 'or')
plt.show()


# In[13]:


x=[10, 20, 30, 40]
y=[20, 25, 35, 55]

plt.plot(x,y)
plt.show


# In[14]:


plt.plot(x,y)

plt.title("Linear graph", fontsize =25, color="blue")

plt.xlabel("X Data")
plt.ylabel("Y Data")


# In[15]:


import numpy as np
x=np.array([1,3,5,8])
y=np.array([2,4,6,5])
plt.plot(x,y)


# In[16]:


get_ipython().system('pip install seaborn')


# In[1]:


import seaborn as sns
import pandas as pd


# In[2]:


tips = sns.load_dataset(r'C:/Users\Dell\Downloads\tips.csv')
tips.head()


# In[23]:


sns.scatterplot(data=tips, x="total_bill", y="tip")


# In[26]:


sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time")


# In[27]:


sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style="time")


# In[28]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot([0, 1, 2, 3, 4, 5])

plt.show()


# In[29]:


import matplotlib.pyplot as plt
import pandas as pd


# In[31]:


data_02 = pd.read_csv('tips.csv')
data_02.head()


# In[32]:


x=data_02['day']
y=data_02['total_bill']

#plotting the data

plt.bar(x,y)
plt.title('Tips Data set')
plt.xlabel('Day')
plt.ylabel('Total Bill')
plt.show


# In[33]:


plt.hist(x, bins=4, color='green', edgecolor='blue')
plt.ylabel('Frequency')
plt.xlabel('Total Bill')


# In[34]:


import matplotlib.pyplot as plt

cars = ['Awdi','BMW','FORD','Tesla','Jagwar']
data = [23, 10,35,15, 12]

plt.pie(data, labels=cars)
plt.show()


# # Catgorical scatterplot

# In[35]:


import seaborn as sns
tips = sns.load_dataset('tips')


# In[36]:


sns.catplot(x='day',y='total_bill',data=tips)


# In[37]:


sns.catplot(x='day',y='total_bill',data=tips, kind='box')

