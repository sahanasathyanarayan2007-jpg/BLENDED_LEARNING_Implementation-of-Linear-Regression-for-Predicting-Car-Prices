# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import required Python libraries.
2.Load the car price dataset using pandas.
3.select independent variables and target variable.
4.Split the data into training and testing sets.
5.Apply standard scaling to the feature variables.
6.Train the Linear Regression model using training data.
7.Predict prices for the test dataset.
8.Evaluate the model using MSE, RMSE, R², and MAE.
9.Analyze residuals to check linear regression assumptions.
10.End.
```

## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: SAHANA.S
RegisterNumber:  25004522
*/
```
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df=pd.read_csv('CarPrice_Assignment.csv')
df.head
X=df[['enginesize','horsepower','citympg','highwaympg']]
Y=df['price']
df.head()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#feature scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
#train model
model=LinearRegression()
model.fit(X_train_scaled,Y_train)
#prediction
Y_pred=model.predict(X_test_scaled)
print("Name:SAHANA.S")
print("Reg. No:25004522")
print("MODEL COEFFICIENTS:")
for feature,coef in zip(X.columns,model.coef_):
    print(f"{feature:>12}: {coef:>10}")
print(f"{'Intercept':>12}: {model.intercept_:>10}")
print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(Y_test,Y_pred):>10}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(Y_test,Y_pred)):>10}")
print(f"{'R-squared':>12}: {r2_score(Y_test,Y_pred):10}")
print(f"{'MAE':>12}: {mean_absolute_error(Y_test,Y_pred):10}")
# linearity check
plt.figure(figsize=(10,5))
plt.scatter(Y_test,Y_pred,alpha=0.6)
plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Price")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
# Independence (Durbin-watson)
residuals=Y_test-Y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson statistic: {dw_test:.2f}",
      "\n(Values close to 2 indicates no autocorrelation)")
# Homoscedasticity
plt.figure(figsize=(10,5))
sns.residplot(x=Y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()
# Normality of residuals
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distrubution")
sm.qqplot(residuals,line='45',fit=True,ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```
## Output:
<img width="247" height="170" alt="image" src="https://github.com/user-attachments/assets/7984b0d9-61dd-4463-bdd2-38b2585cb000" />
<img width="257" height="88" alt="image" src="https://github.com/user-attachments/assets/7da70324-9e63-40a1-8971-c0241eb88293" />
<img width="291" height="42" alt="image" src="https://github.com/user-attachments/assets/7438db2c-1761-41ce-9292-3d74eb6ae9cf" />
<img width="262" height="37" alt="image" src="https://github.com/user-attachments/assets/09c1ff24-b810-454e-8698-ff26d9bb6885" />
<img width="1608" height="588" alt="image" src="https://github.com/user-attachments/assets/0e38d81f-b182-436e-8ee8-10b432c672d0" />
<img width="1287" height="597" alt="image" src="https://github.com/user-attachments/assets/7dac0036-7f75-4a48-98e7-8bbd19dbe7bf" />
<img width="1367" height="597" alt="image" src="https://github.com/user-attachments/assets/8f445ebd-c7d8-4dab-b783-93a408b7ab6e" />







## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
