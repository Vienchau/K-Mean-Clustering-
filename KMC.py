from re import I
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

#Data reading
Mall_Customers = pd.read_csv('Mall_Customers.csv')
print(Mall_Customers.head())

#Checking percent off data missing
for col in Mall_Customers.columns:
    miss = Mall_Customers[col].isna().sum()
    miss_percent = miss/len(Mall_Customers.columns)
print('Missing Data percent: %f ' %miss_percent)

#split Data, chose columns: Genre, Annual Income, Spending score. Split Male and Female to analyze
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

print('Number of Male: %i' %len(Male))
print('Number of Female: %i' %len(Female))

#Using Elbow to chosing K
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

plt.subplot(2,2,1)
plt.plot(I, distortion_Male)
plt.ylabel('Distortion') 
plt.title("Male Elbow K-Mean-Cluster")

plt.subplot(2,2,2)
plt.plot(I, distortion_Female)
plt.ylabel('Distortion') 
plt.title("Female Elbow K-Mean-Cluster")

#chosing K = 5 in K mean Clustering 
kmeans = KMeans(n_clusters=5, init = "k-means++", random_state = 42)
y_means_Male = kmeans.fit_predict(Male)
y_means_Female = kmeans.fit_predict(Female)


plt.subplot(2,2,3)
plt.scatter(Male[:, 0], Male[:, 1], c=y_means_Male, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=200, alpha=0.5);
plt.xlabel('Annual Income (k$)') 
plt.ylabel('Spending Score (1-100)') 
plt.title("Male K-Mean-Cluster")
plt.legend() 

plt.subplot(2,2,4)
plt.scatter(Female[:, 0], Female[:, 1], c=y_means_Female, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=200, alpha=0.5);
plt.xlabel('Annual Income (k$)') 
plt.ylabel('Spending Score (1-100)') 
plt.title("Female K-Mean-Cluster")
plt.legend() 


plt.show()


 

    

