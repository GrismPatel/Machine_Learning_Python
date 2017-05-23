# K - Nearest Neighbours
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data 
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

# Split our data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)

# Now we scale our data
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# We use KNN 
from sklearn.neighbors import KNeighborsClassifier
a = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski', p = 2)
a.fit(x_train, y_train)

# We now predict
y_predict = a.predict(x_test)

#We will create confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)

#We will vizualize the training set
from matplotlib.colors import ListedColormap
x_set,y_set = x_train,y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min() -1,stop = x_set[:,0].max() +1,step = 0.01),
                              np.arange(start = x_set[:,1].min() -1,stop = x_set[:,0].max()+1,step = 0.01))
plt.contourf(x1,x2,a.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                                      alpha =  0.75,cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1],
                c = ListedColormap(('red','green'))(i),label = j)
plt.title ('KNN')
plt.xlabel ('Age')
plt.ylabel ('Estimated Salary')
plt.legend()
plt.show()

#We will vizualize the test set
from matplotlib.colors import ListedColormap
x_set,y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min() -1,stop = x_set[:,0].max() +1,step = 0.01),
                              np.arange(start = x_set[:,1].min() -1,stop = x_set[:,0].max()+1,step = 0.01))
plt.contourf(x1,x2,a.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                                      alpha =  0.75,cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1],
                c = ListedColormap(('red','green'))(i),label = j)
plt.title ('KNN')
plt.xlabel ('Age')
plt.ylabel ('Estimated Salary')
plt.legend()
plt.show()