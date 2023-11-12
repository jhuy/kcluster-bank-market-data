import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
# bank_marketing = fetch_ucirepo(id=222) 
file = pd.read_csv('bank-full.csv', delimiter=';')  
print (file)

# data (as pandas dataframes) 
#x = bank_marketing.data.features 
#y = bank_marketing.data.targets 
  
# metadata 
#print(bank_marketing.metadata) 
  
# variable information 
#print(bank_marketing.variables) 

# Variable Declartion
total_age = 0

age_input = []
balance_input = []
age_balance_input = []
age_balance_duration_input = []
duration_input = []

for input in file.iloc[:,0]:
    age_input.append(file.iloc[:,0][input])
    balance_input.append(file.iloc[:,5][input])
    duration_input.append(file.iloc[:,11][input])

    age_balance_input.append([file.iloc[:,0][input], file.iloc[:,5][input]])
    age_balance_duration_input.append([file.iloc[:,0][input], file.iloc[:,5][input], file.iloc[:,11][input]])

### 2D Clustering
#plt.scatter(age_input, balance_input)
#plt.show()

cluster_numbers = [2,3,4,5,6,7,8]
inertia = []
silhouette_scores = []

for k in cluster_numbers:
    kmeans = KMeans(n_clusters = k, random_state = 40, n_init=10).fit(age_balance_input)
    inertia.append(kmeans.inertia_)

    print ("Num of clusters: " + str(k))
    print (kmeans.cluster_centers_)

#plt.plot(cluster_numbers, inertia, marker='o')
#plt.show()

### 3D Clustering

fig2 = plt.figure()
ax = plt.axes(projection = "3d")
ax.scatter3D(age_input, balance_input, duration_input)
plt.show()

for k in cluster_numbers:
    kmeans = KMeans(n_clusters = k, random_state = 40, n_init=10).fit(age_balance_duration_input)
    inertia.append(kmeans.inertia_)

    print ("Num of clusters: " + str(k))
    print (kmeans.cluster_centers_)

fig3 = plt.figure()
ax = plt.axes(projection = "3d")
ax.scatter3D(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2])
plt.show()