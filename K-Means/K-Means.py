import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report



def converter(cluster):
    if cluster=="Yes":
        return 1
    else:
        return 0


print("Importing our data...")
df = pd.read_csv("College_Data",index_col=0)
print("Done!!!")

print("Checking the data...")
print(df.head())
print(df.info())
print(df.describe())
print("Done!!!")

print("Creating some data visualizations")
sns.set_style("whitegrid")
sns.lmplot(data=df,x="Room.Board",y= "Grad.Rate", hue="Private",
           palette="coolwarm",height=6,aspect=1,fit_reg=False)
plt.show()

sns.set_style("whitegrid")
sns.lmplot(data=df,x="Outstate",y="F.Undergrad", hue="Private",
           palette="coolwarm",height=6,aspect=1,fit_reg=False)
plt.show()

sns.set_style("darkgrid")
g = sns.FacetGrid(df,hue="Private",palette="coolwarm",height=6,aspect=2)
g = g.map(plt.hist,"Outstate",bins=20,alpha=0.7)
plt.show()

sns.set_style("darkgrid")
g = sns.FacetGrid(df,hue="Private",palette="coolwarm",height=6,aspect=2)
g = g.map(plt.hist,"Grad.Rate",bins=20,alpha=0.7)
plt.show()
print("Done!!!")


#Notice how there seems to be a private school with a graduation rate of higher than 100%#
print("Finding the name of that school...")
df[df["Grad.Rate"] > 100]
print("Done!!!")


print("Setting that school's graduation rate to 100 so it makes sense...")
df.loc["Cazenovia College", "Grad.Rate"] = 100
sns.set_style("darkgrid")
g = sns.FacetGrid(df,hue="Private",palette="coolwarm",height=6,aspect=2)
g = g.map(plt.hist,"Grad.Rate",bins=20,alpha=0.7)
plt.show()
print("Done!!!")

print("Training our model with 2 clusters...")
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop("Private",axis=1))
print("Done!!!")


print("Printing the cluster center vectors...")
print(kmeans.cluster_centers_)
print("Done!!!")

print("Creating a column which is  1 for a Private school, and  0 for a public school...")
# Check the function at the beginning!!!!
df["Cluster"] = df["Private"].apply(converter)
print(df.head())
print("Done!!!")

print("Evaluating our model...")
print(confusion_matrix(df["Cluster"],kmeans.labels_))
print(classification_report(df["Cluster"],kmeans.labels_))
print("Done!!!")