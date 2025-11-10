import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


print("Learning some info about the 'load_breast_cancer' dataset...")
cancer = load_breast_cancer()
print(cancer.keys())
print("Done!!!")


print("Loading our data...")
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print("Done!!!")

print("Checking the data...")
print(print(cancer['DESCR']))
print("Done!!!")

print("Scaling our data...")
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
print("Done!!!")

print("Dimensionality reduction...")
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)
print("Done!!!")

print("Visualizing the result")
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.show()
print("Done!!!")

print("Interpreting the components")
print(pca.components_)
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
print(df_comp)
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)
plt.show()
print("Done!!!")


