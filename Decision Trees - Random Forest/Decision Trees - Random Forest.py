import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier


print("Importing our data...")
loans = pd.read_csv('loan_data.csv')
print("...Done")

print("Exploring the data...")
print(loans.head())
print(loans.describe())
print(loans.info)
print("...Done")


print("Visualizing our data")
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
plt.show()

sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
plt.show()

plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')
plt.show()
print("...Done")


print("Transforming 'purpose' column in dummy variable...")
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.info()
print("...Done")

print("Splitting our data in trainning set and test set...")
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
print("...Done")

print("Training a Decision Tree Model...")
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
print("...Done")


print("Evaluating the Decision Tree...")
predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print("...Done")


print("Training the Random Forest model...")
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
print("...Done")


print("Evaluating the Random Forest model...")
predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print("...Done")
