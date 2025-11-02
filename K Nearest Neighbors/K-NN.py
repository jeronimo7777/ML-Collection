import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

print("Importing our data...")
df = pd.read_csv("KNN_Project_Data", index_col=0)
print("Done!!!")

print("Checking our data...")
print(df.head())
print(df.info())
print(df.describe())
print("Done!!!")

print("Visualizing our data...")
sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')
plt.show()
print("Done!!!")

print("Standardizing the variables...")
scaler = StandardScaler()
scaler.fit(df.drop("TARGET CLASS", axis=1))
scaled_features = scaler.transform(df.drop("TARGET CLASS", axis=1))
print("Done!!!")

print("Convert the scaled features to a dataframe...")
df_feat = pd.DataFrame(scaled_features, columns= df.columns[:-1])
print(df_feat.head())
print("Done!!!")


print("Splitting data into train and test set...")
X = df_feat
y = df["TARGET CLASS"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3)
print("Done!!!")


print("Training our model")
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Done!!!")

print("Predicting values for the testing data")
predictions = knn.predict(X_test)
print("Done!!!")

print("Metrics...")
print("Confusion Matrix")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print("Done!!!")

print("Choosing a better K-Value")
error_rate = []

print("Creating a loop that trains various KNN models with different k values")
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
print("Done!!!")

print("Visualizing the error rate for each K value")
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
print("Done!!!")

print("For K=30")
knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=30')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
