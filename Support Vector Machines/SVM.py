import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import  train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


print("Importing our data...")
iris = sns.load_dataset('iris')
print(iris)
print("...Done")


print("Visualizing our data")
sns.pairplot(iris,hue='species',palette='Dark2')
plt.show()

setosa = iris[iris['species']=='setosa']
sns.kdeplot( setosa['sepal_width'],
                 cmap="plasma", fill=True, thresh=False)
plt.show()

setosa = iris[iris['species']=='setosa']
sns.kdeplot( setosa['sepal_width'],
                 cmap="plasma", fill=True, thresh=False)
plt.show()
print("...Done")


print("Splitting our data in training set and test set...")
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print("...Done")

print("Training our Model...")
svc_model = SVC()
svc_model.fit(X_train,y_train)
print("...Done")


print("Evaluating the Model...")
predictions = svc_model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print("...Done")


print("Defining the best values of C and Gamma...")
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
print("...Done")


print("Evaluating the the Model...")
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
print("...Done")
