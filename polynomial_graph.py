import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


data = pd.read_csv('car_price.csv')
 # Load your dataset here
X = data[["Age_Years"]]
y = data["Current_Price_USD"]
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
reg = LinearRegression()
reg.fit(X_poly, y)

X_sort = np.sort(X.values, axis=0)
X_sort_poly = poly_features.transform(X_sort)
y_pred = reg.predict(X_sort_poly)
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_sort, y_pred, color='red', label='Polynomial Regression Line')
plt.xlabel('Age of Car (Years)')
plt.ylabel('Current Price (USD)')
plt.title('Polynomial Regression of Car Prices')
plt.legend()
plt.show()