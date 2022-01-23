from re import I
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


Mall_Customers = pd.read_csv('Mall_Customers.csv')
print(Mall_Customers.head())

for col in Mall_Customers.columns:
    miss = Mall_Customers[col].isna().sum()
    miss_percent = miss/len(Mall_Customers.columns)
print(miss_percent)

X = Mall_Customers.iloc[:, [1,3,4]].values
X1 = Mall_Customers.iloc[:, [3,4]].values
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])

Male = np.zeros((1,2))
Female = np.zeros((1,2))
Temp = np.zeros((1,2))

for i in range(len(X)):
    if X[i,0] == 1:
        Temp[0,0] =  X1[i,0]
        Temp[0,1] =X1[i,1]
        Male = np.append(Male, Temp, axis=0)

    else:
        Temp[0,0] =  X1[i,0]
        Temp[0,1] =X1[i,1]
        Female = np.append(Female, Temp, axis=0)

Male = np.delete(Male, 0, 0)
Female = np.delete(Female, 0, 0)
I = range(1,11)
distortion_Male =[] 
distortion_Female =[] 

for i in I:
    KmeanModel_1 = KMeans(n_clusters= i, init = "k-means++", random_state= 42)
    KmeanModel_1.fit(Male)
    distortion_Male.append(KmeanModel_1.inertia_)

for i in I:
    KmeanModel_2 = KMeans(n_clusters= i, init = "k-means++", random_state= 42)
    KmeanModel_2.fit(Female)
    distortion_Female.append(KmeanModel_2.inertia_)

plt.subplot(1,2,1)
plt.plot(I, distortion_Male)
plt.xlabel('Number of clusters')
plt.ylabel('Distortion') 
plt.title("Male K-Mean-Cluster")

plt.subplot(1,2,2)
plt.plot(I, distortion_Female)
plt.xlabel('Number of clusters')
plt.ylabel('Distortion') 
plt.title("Female K-Mean-Cluster")



plt.show()


 

    

