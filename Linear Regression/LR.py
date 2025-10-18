import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

print("Importing our data...")
USAhousing = pd.read_csv('USA_Housing.csv')
print("... Done!")


print("Checking out the Data...")
print(USAhousing.head())
print(USAhousing.info())
print(USAhousing.describe())
print("... Done!")

print("Creating some simple plots...")
sns.pairplot(USAhousing)
plt.show()
sns.distplot(USAhousing['Price'])
plt.show()
sns.heatmap(USAhousing.corr(numeric_only=True),annot=True,fmt=".2f")
plt.show()
print("... Done!")


print("Defining our independent and dependent variables...")
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
print("... Done!")

print("Splitting our data to Train and Test sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("... Done!")

print("Creating and Training our model")
lm = LinearRegression()
lm.fit(X_train, y_train)

print("... Done!")

print("Checkin out coefficients")
print('Intercept: \n', lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)
print("... Done!")


print("Model evaluation")
y_hat = lm.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_hat))
print('MSE:', metrics.mean_squared_error(y_test, y_hat))
print("RMSE", metrics.root_mean_squared_error(y_test, y_hat))

sns.distplot((y_test-y_hat),bins=50)
plt.show()
print("... Done!")
