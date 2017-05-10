import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# Applying decission tree Regressor
from sklearn.tree import DecisionTreeRegressor
a = DecisionTreeRegressor(random_state = 0)
a.fit(x,y)

#predict the value
y_predict = a.predict(6.5)

#Plot the graph
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter (x,y,color = 'red')
plt.plot(x_grid,a.predict(x_grid),color = 'black')
plt.title('Positions vs Salaries')
plt.xlabel('Positions')
plt.ylabel('Salaries')

# grismpatel@hotmail.com
# grismpatel94@gmail.com