import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv(r"C:\Users\konth\Downloads\emp_sal.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)
#compare actual and predicted
plt.scatter(x, y, color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title("linear regressuin model(linear regression)")
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
lin_model_pred=lin_reg.predict([[6]])
print(lin_model_pred)
#polynomial regressional model
#polynomial regression model defaultly haves 2 degrees
#in simple liniear regressin- 1 degree
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=7)
x_poly=poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)
plt.scatter(x, y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color="blue")
plt.title('polyn model(plonynomial regression)')
plt.xlabel("position level")
plt.ylabel('salary')
plt.show()
poly_model_pred=lin_reg_2.predict(poly_reg.fit_transform([[7]]))
print(poly_model_pred)
#SVR model
from sklearn.svm import SVR
svr_reg=SVR()
svr_reg.fit(x,y)
svr_pred=svr_reg.predict([[6]])
print(svr_pred)
svr_reg=SVR(kernel="poly",degree=5,gamma="scale")
#we can pass somemore values to kernal ...select SVR and click cltr+i
#KNN model 
from sklearn.neighbors import KNeighborsRegressor
# check KNeighborsRegressor by cltr+i
knn_reg=KNeighborsRegressor(n_neighbors=3)
#by default n_neighbors values is 5
knn_reg.fit(x,y)
knn_pred=knn_reg.predict([[6]])
print(knn_pred)


