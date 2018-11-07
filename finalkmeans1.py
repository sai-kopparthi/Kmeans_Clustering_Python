import numpy as np
import time
import pickle

X = pickle.load(open('/home/sai/PycharmProjects/final/venv/hw3_data/data_dense.pl','rb'),encoding='latin1')
print(X.shape)

def euclidean_distance(a,b):
    return np.linalg.norm(a-b)

def assign_cluster(a,centers):
    dists = np.array([euclidean_distance(a,x) for x in centers])
    return [np.argmin(dists),np.min(dists)]

def cluster_mean(cluster):
    return np.mean(cluster,axis=0)

def kmeans_plotting(points):
    n,p = points.shape
    init_centers = np.zeros((10,p))
    init_centers[0:10,:] = points[0:10,:]
    centers = init_centers
    start_time = time.time()
    for m in range(41):
        if (m%10 == 0):
            objective = 0
            clusters = [[] for i in range(10)]
            for i in range(len(points)):
                a = points[i]
                c, min_dist = assign_cluster(a, centers)
                objective = objective + min_dist ** 2
                clusters[c].append(a)
            print(objective)

            centers = []
            for i in range(10):
                centers.append(cluster_mean(clusters[i]))
            centers = np.array(centers)
            print("Time after", m ,"iteration.", time.time() - start_time)
        else :
            objective = 0
            clusters = [[] for i in range(10)]
            for i in range(len(points)):
                a = points[i]
                c, min_dist = assign_cluster(a, centers)
                objective = objective + min_dist ** 2
                clusters[c].append(a)
            print(objective)

            centers = []
            for i in range(10):
                centers.append(cluster_mean(clusters[i]))
            centers = np.array(centers)


def main():
    kmeans_plotting(X)

if __name__ == "__main__":
    main()