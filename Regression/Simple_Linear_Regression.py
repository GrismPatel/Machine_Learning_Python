#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Reads the data files
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Diffrentiate training and test files
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)"""

#Feeds the Machine, it learns from the train data
from sklearn.linear_model import LinearRegression
a = LinearRegression()
a.fit(x_train,y_train)

#Now it predicts from the given train data on the test data
y_predict = a.predict(x_test)

#Now we draw graph for Traing Data
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,a.predict(x_train),color = 'black')
plt.title('Salary vs Experiance')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()

#Now we draw graph for test Data
plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,a.predict(x_train),color = 'black') 
"""We wont change this one because regression plot was based on training data and 
we are trying to check how training prediction fits our test data"""
plt.title('Salary vs Experiance')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()

# grismpatel@hotmail.com
# grismpatel94@gmail.com