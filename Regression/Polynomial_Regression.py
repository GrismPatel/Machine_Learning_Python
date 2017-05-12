import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values #we use :2 to make x as matrix
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
a = LinearRegression()
a.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
b = PolynomialFeatures(degree = 4)
x_poly = b.fit_transform(x)
poly = LinearRegression()
poly.fit(x_poly,y)

plt.scatter(x,y,color = 'red')
plt.plot(x,a.predict(x),color = 'black')
plt.title('Position vs Salaries (Linear Regression)')
plt.xlabel('Positions (Levels)')
plt.ylabel('Salaries')
plt.show()

plt.scatter(x,y,color = 'red')
plt.plot(x,poly.predict(b.fit_transform(x)),color = 'black')
plt.title('Position vs Salaries (Polynomial Regression)')
plt.xlabel('Positions (Levels)')
plt.ylabel('Salaries')
plt.show()

# grismpatel@hotmail.com
# grismpatel94@gmail.com