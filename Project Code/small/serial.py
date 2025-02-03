#Load the wine dataset from the folder
import numpy as np
import math
from sklearn.datasets import load_wine
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#check if two 1-d arrays have same values
def arrays_are_equal(arr1, arr2):
    # First, check if the arrays have the same length
    if len(arr1) != len(arr2):
        return False
    
    # Iterate through each element of the arrays
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False
    
    # If no mismatches were found, the arrays are equal
    return True

#check if two 2-d arrays have same values
def twoD_arrays_are_equal(arr1, arr2):
    # First, check if the arrays have the same length
    if len(arr1) != len(arr2):
        return False
    
    # Iterate through each element of the arrays
    for i in range(len(arr1)):
        if not arrays_are_equal(arr1[i], arr2[i]):
            return False
    
    # If no mismatches were found, the arrays are equal
    return True

# find the euclidean distance between the centroid and the data point
def euclidean_distance(x, y):
    distance = 0
    for i in range(len(x)):
        distance += math.pow(x[i] - y[i], 2)
    return math.sqrt(distance)

def calculate_centroids_sharding(dataset, k):
    # Step 1: Sum the attribute (column) values for each instance (row) of a dataset
    instance_sums = dataset.sum(axis=1)
    # Step 2: Sort the instances of the dataset by the newly created sum column
    sorted_indices = np.argsort(instance_sums)
    sorted_dataset = dataset[sorted_indices]
    # Step 3: Split the dataset horizontally into k equal-sized pieces, or shards
    shards = np.array_split(sorted_dataset, k)
    
    # Step 4: Compute centroids for each shard
    centroids = []
    for shard in shards:
        shard_mean = np.mean(shard, axis=0)
        centroids.append(shard_mean)
    
    # Step 5: Return the set of centroid instances
    return np.array(centroids)

# load the dataset
wine = load_wine()

#choose the number of clusters to form
k = 9

# scale the data using standard scaling
scaler = StandardScaler()
wine_scaled = scaler.fit_transform(wine.data)

#use PCA to retain columns that capture 80% of the variance
pca = PCA(0.8)
wine_scaled = pca.fit_transform(wine_scaled)

#start the execution time
start = time.time()

#initialize k centroids using sharding
centroids = calculate_centroids_sharding(wine_scaled.copy(), k)

# make a dictionary with key as points and value as the cluster number
clusters = {}
count = 0
while True:
    count += 1
    # assign the points to the clusters (assign point to nearest centroid)
    for i in range(len(wine_scaled)):
        distances = []
        for j in range(len(centroids)):
            distance = euclidean_distance(wine_scaled[i], centroids[j])
            distances.append((centroids[j], distance, j))
        centroid = min(distances, key=lambda x: x[1])[0]
        min_centroid_index = min(distances, key=lambda x: x[1])[2]
        clusters.update({i: min_centroid_index})

    # update the centroids. Take mean position of points in the cluster
    old_centroids = centroids.copy()
    for i in range(len(centroids)):
        close_points = []
        for j in clusters.keys():
            index = clusters[j]
            if arrays_are_equal(centroids[index], centroids[i]):
                close_points.append(wine_scaled[j])
        centroids[i] = np.mean(close_points, axis=0)

    #if centroids didn't change from previous iteration, then stopping criteria reached    
    if twoD_arrays_are_equal(old_centroids, centroids):
        break

end = time.time()

#print clusters
print(clusters)

#time taken
print("Time taken: ", (end-start))

#run kmeans clustering using sklearn for verification
kmeans = KMeans(n_clusters=k, random_state=0, init=centroids).fit(wine_scaled)
cluster_labels = kmeans.fit_predict(wine_scaled)

#make a dictionary with key as data point index and value as the cluster number
data_point_clusters = {i: cluster_labels[i] for i in range(len(cluster_labels))}

# Print sklearn clusters dictionary
print(data_point_clusters)

#compare our dictionary with sklearn dictionary to check if there is any mismatched key-value pairs
mismatched_pairs = {}
for key in data_point_clusters:
    if key in clusters and data_point_clusters[key] != clusters[key]:
        mismatched_pairs[key] = (data_point_clusters[key], clusters[key])

#if our code is correct then this dictionary would be empty
print("mismatched pairs\n", mismatched_pairs)

#Ratio of intra and inter cluster distances
intra_cluster_distance = 0
inter_cluster_distance = 0
for i in range(len(wine_scaled)):
    for j in range(i+1, len(wine_scaled)):
        if clusters[i] == clusters[j]:
            intra_cluster_distance += euclidean_distance(wine_scaled[i], wine_scaled[j])
        else:
            inter_cluster_distance += euclidean_distance(wine_scaled[i], wine_scaled[j])

print("Ratio of intra and inter cluster distances: ", intra_cluster_distance/inter_cluster_distance)