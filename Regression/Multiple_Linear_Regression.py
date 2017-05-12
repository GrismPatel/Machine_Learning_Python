import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
x = onehotencoder.fit_transform(x).toarray()

x = x[:, 1:]

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LinearRegression
a = LinearRegression()
a.fit(x_train,y_train)

y_predict = a.predict(x_test)

import statsmodels.formula.api as sm
# We add a matrix of (50*1) so that; x0 can be added in; y = bx0 + bx1 + bx2 ... + bxn
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0,1,2,3,4,5]]
a_OLS = sm.OLS(endog = y,exog = x_opt).fit()
a_OLS.summary()

# we are using backward regression, so highest value of P is eliminated
x_opt = x[:, [0,1,3,4,5]]
a_OLS = sm.OLS(endog = y,exog = x_opt).fit()
a_OLS.summary()

x_opt = x[:, [0,3,4,5]]
a_OLS = sm.OLS(endog = y,exog = x_opt).fit()
a_OLS.summary()

# grismpatel@hotmail.com
# grismpatel94@gmail.com


x_opt = x[:, [0,3,5]]
a_OLS = sm.OLS(endog = y,exog = x_opt).fit()
a_OLS.summary()

x_opt = x[:, [0,3]]
a_OLS = sm.OLS(endog = y,exog = x_opt).fit()
a_OLS.summary()