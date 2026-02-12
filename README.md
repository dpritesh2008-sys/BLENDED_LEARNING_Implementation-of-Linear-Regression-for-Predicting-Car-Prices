# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Linear Regression – Baseline model for price prediction.
2.Decision Tree Regressor – Captures non-linear patterns in data.
3.Random Forest Regressor – Improves accuracy using ensemble learning.
4.Gradient Boosting Regressor – Reduces errors step-by-step for better prediction.

## Program:
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Balasurya S
RegisterNumber: 212225100003
*/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df=pd.read_csv('CarPrice_Assignment (2).csv')
df.head()

x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price'] 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scalar=StandardScaler()
x_train_scaled=scalar.fit_transform(x_train)
x_test_scaled = scalar.transform(x_test)
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)

print('Name:BALASURYA S')
print('Reg. No:25000944')
print("MODEL COEFFICIENTS:")
for feature, coef in zip(x.columns,model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}: {model.intercept_:>10.2f}")
print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}:{mean_squared_error(y_test, y_pred):>10.2f}")
print(f"{'RMSE':>12}:{np.sqrt(mean_squared_error(y_test, y_pred)):>10.2f}")
print(f"{'R-squared':>12}:{r2_score(y_test, y_pred):>10.2f}")
print(f"{'MAE':>12}:{mean_absolute_error(y_test, y_pred):>10.2f}")

plt.figure(figsize=(10, 5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()


## Output:
<img width="868" height="468" alt="548117907-9c61ea94-ec6d-4627-9989-9fed295917b7" src="https://github.com/user-attachments/assets/6b2f5a94-5ff7-47e7-9a6d-841987347911" />



## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
