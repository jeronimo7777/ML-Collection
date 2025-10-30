import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("Importing data...")
ad_data = pd.read_csv('advertising.csv')
print("Done!!!")

print("Checking our data...")
print(ad_data.head())
print(ad_data.info())
print(ad_data.describe())
print("Done!!!")

print("Visualizing our data...")
sns.histplot(data=ad_data, x="Age", bins=30)
plt.xlabel("Age")

sns.jointplot(data=ad_data, x='Age',y='Area Income')
sns.jointplot(data=ad_data, x="Age", y= "Daily Time Spent on Site", color="red",kind="kde")
sns.jointplot(data=ad_data, x="Daily Time Spent on Site",y="Daily Internet Usage", color="green")
sns.pairplot(data=ad_data,hue="Clicked on Ad", palette="bwr")

plt.show()
print("Done!!!")

print("Splitting data into train and test set...")
X = ad_data[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Male"]]
y = ad_data["Clicked on Ad"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Done!!!")

print("Training our model")
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
print("Done!!!")

print("Predicting values for the testing data")
y_hat = logmodel.predict(X_test)
print("Done!!!")

print("Metrics...")
print("Confusion Matrix")
CM = confusion_matrix(y_test, y_hat)
print(CM)
print(f"\nAccuracy Score: {accuracy_score(y_test, y_hat):.4f}")
print(classification_report(y_test,y_hat))
print("Done!!!")
