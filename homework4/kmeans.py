import json
import random
import numpy as np


def cluster_points(X, mu):
    """
    Distribute data points into clusters.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - mu: A list of K cluster centers, each elements is a list of 2

    Returns:
    - clusters: A dict, keys are cluster index {1,2, ..., K} (int),
                value are the list of corresponding data points.
    """


    clusters = {}

    # you need to fill in your solution here
    
    X_arr=np.array(X)
    mu_arr=np.array(mu)
    assignment_arr=np.zeros(X_arr.shape[0])
    #print(X_arr.shape)
    #print(mu_arr.shape)
    for i in range(X_arr.shape[0]):
    	temp=np.zeros(mu_arr.shape[0])
    	for j in range(mu_arr.shape[0]):
    		diff =mu_arr[j]-X_arr[i]
    		#print(diff)
    		#print(np.square(diff[0]+diff[1]))
    		temp[j] = np.square(diff[0])+np.square(diff[1])

    	#print(temp.argmax(),temp.shape,assignment_arr.shape)
    	assignment_arr[i]=temp.argmin()+1
    	#print(assignment_arr[i])
    #print(assignment_arr)

    for i in range(mu_arr.shape[0]):
    	clusters[i+1] = list();

    for i in range(assignment_arr.shape[0]):
    	clusters[assignment_arr[i]].append(X[i])

    #print(clusters)

    return clusters


def reevaluate_centers(mu, clusters):
    """
    Update cluster centers.

    Inputs:
    - mu: A list of K cluster centers, each elements is a list of 2
    - clusters: A dict, keys are cluster index {1,2, ..., K} (int),
                value are the list of corresponding data points.

    Returns:
    - newmu: A list of K updated cluster centers, each elements is a list of 2
    """
    newmu = []

    for i in range(len(mu)):
    	if( len(clusters[i+1]) != 0):
    		temp=np.array(clusters[i+1])
    		temp=np.sum(temp,axis=0)/temp.shape[0]
    		newmu.append(temp.tolist())
    	else:
    		newmu.append(mu[i])

    #print(newmu)



    # you need to fill in your solution here

    return newmu


def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])


def find_centers(X, K):
    # Initialize to K random centers
    random.seed(100)
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)

    return(mu, clusters)


def kmeans_clustering():
    # load data
    with open('hw4_circle.json', 'r') as f:
        data_circle = json.load(f)
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    clusters_all = {}
    for K in [2, 3, 5]:
        key = 'blob_K=' + str(K)
        mu_all[key], clusters_all[key] = find_centers(data_blob, K)
        key = 'circle_K=' + str(K)
        mu_all[key], clusters_all[key] = find_centers(data_circle, K)

    return mu_all, clusters_all


def main():
    mu_all, clusters_all = kmeans_clustering()

    print('K-means Cluster Centers:')
    for key, value in mu_all.items():
        print('\n%s:'% key)
        print(np.array_str(np.array(value), precision=4))

    with open('kmeans.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'clusters': clusters_all}, f_json)


if __name__ == "__main__":
    main()