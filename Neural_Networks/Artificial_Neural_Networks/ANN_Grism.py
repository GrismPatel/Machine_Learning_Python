# Part 1 - Data Preprocessing 
# Part 2 - Creating ANN model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values

# Encoding the data (changing the alphabets to numerical values)
# Example = France, Germany, Spain == 0, 1, 2 and Male, Female == 0, 1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

Label_Enc_Countries = LabelEncoder()
x[:, 1] = Label_Enc_Countries.fit_transform(x[:, 1])

Label_Enc_Gender = LabelEncoder()
x[:, 2] = Label_Enc_Gender.fit_transform(x[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

# Spliting the dataset into test and train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Part 2 - Creating ANN
# Importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Creating the object
classifier = Sequential ()

#Adding the input
#First Hidden Layer
classifier.add(Dense(output_dim = 6, 
                     init = 'uniform',
                     activation = 'relu', #Rectifier
                     input_dim = 11))

#Second Hidden Layer
classifier.add(Dense(output_dim = 6, 
                     init = 'uniform',
                     activation = 'relu', #Rectifier
                     ))

#Third Hidden Layer
classifier.add(Dense(output_dim = 1, 
                     init = 'uniform',
                     activation = 'sigmoid'))

#compiling the ANN (Backpropogation: Changing the weights)
classifier.compile(optimizer = 'adam', #Stochastic Gradient Descent
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

#Fitting the ANN to training set
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)


# Part 3 - Making Prediction and evaluating model
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
