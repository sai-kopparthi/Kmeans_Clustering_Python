import numpy as np
import time
import pickle
import scipy.sparse.linalg

X = pickle.load(open('/home/sai/PycharmProjects/final/venv/hw3_data/data_sparse_E2006.pl','rb'),encoding='latin1')
print(X.shape)
n,p = X.shape

def cluster_mean(cluster):
    return np.mean(cluster,axis=0)

def euclidean_distance(a,b):
    c = np.linalg.norm(b)**2
    d = scipy.sparse.linalg.norm(a)**2
    e = a @ b.T
    f = 2*e
    h = c+d-f
    return h

def assign_cluster(a,centers):
    dists = np.array([euclidean_distance(a,x) for x in centers])
    return [np.argmin(dists), np.min(dists)]

def kmeans_plotting(points):
    n,p = points.shape
    init_centers = np.zeros((10,p))
    init_centers = points[0:10,:].todense()
    centers = init_centers
    start_time = time.time()
    for m in range(41):
        if(m%10 == 0):
            objective = 0
            a = []
            clusters = [[] for i in range(10)]
            for i in range(points.shape[0]):
                a = X[i]
                c, min_dist = assign_cluster(a, centers)
                objective = objective + min_dist
                clusters[c].append(i)
            print(objective)

            centers = []
            for i in range(10):
                centers.append(cluster_mean(X[clusters[i], :].todense()))
            centers = np.array(centers)
            print("Time after", m, "iteration.", time.time() - start_time)
        else:
            objective = 0
            a = []
            clusters = [[] for i in range(10)]
            for i in range(points.shape[0]):
                a = X[i]
                c, min_dist = assign_cluster(a, centers)
                objective = objective + min_dist
                clusters[c].append(i)
            print(objective)

            centers = []
            for i in range(10):
                centers.append(cluster_mean(X[clusters[i], :].todense()))
            centers = np.array(centers)



kmeans_plotting(X)