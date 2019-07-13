from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import numpy as np
x = np.array([[5,3],[10,15],[15,12],[24,10],[30,45],[85,70],[71,80],[60,78],[55,52],[80,91]])
plt.scatter(x[:,0],x[:,1])

sos = []
for k in range(1,10):                           #Elbow method to find the number of cluster 
    km = KMeans(n_clusters = k)
    km.fit(x)
    a = km.inertia_
    sos.append(a)
klist = [k for k in range(1,10)]
plt.plot(klist,sos)

print(km.cluster_centers_)
print(km.inertia_)
print(km.labels_)
print(km.n_iter_)

plt.scatter(x[:,0],x[:,1],c = km.labels_)
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1])

km.predict(np.array([25,25]).reshape(1,-1))


# Iris Dataset
ir = load_iris()
d = ir.data
sos1 = []
for k in range(1,15):                           #Elbow method to find the number of cluster 
    km = KMeans(n_clusters = k)
    km.fit(d)
    a = km.inertia_
    sos1.append(a)
klist1 = [k for k in range(1,15)]
plt.plot(klist1,sos1)

print(km.cluster_centers_)
print(km.inertia_)
print(km.labels_)
print(km.n_iter_)

km.predict(np.array([1.5,4.8,3.6,2.4]).reshape(1,-1))


#mnist Dataset
mnist = fetch_mldata("mnist-original")
d = mnist.data
sos1 = []
for k in range(1,15):                           #Elbow method to find the number of cluster 
    km = KMeans(n_clusters = k)
    km.fit(d)
    a = km.inertia_
    sos1.append(a)
klist1 = [k for k in range(1,15)]
plt.plot(klist1,sos1)

print(km.cluster_centers_)
print(km.inertia_)
print(km.labels_)
print(km.n_iter_)