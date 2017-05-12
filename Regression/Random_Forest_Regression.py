import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
a = RandomForestRegressor(n_estimators = 300,random_state = 0)
a.fit(x,y)

y_predict = a.predict(6.5)

x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x,y,color = 'red')
plt.plot(x_grid, a.predict(x_grid),color = 'black')
plt.title('Position  vs Salaries')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

# grismpatel@hotmail.com
# grismpatel94@gmail.com