#import all the essential libraries
import numpy as np
import math
import time
import threading
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd

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

#for each chunk, assign cluster number to each data point based on minimum euclidean distance
def calculate_distance(dataset, x):
    global clusters
    for i in range(len(dataset)):
        distances = []
        for j in range(len(centroids)):
            distance = euclidean_distance(dataset[i], centroids[j])
            distances.append((centroids[j], distance, j))
        min_centroid_index = min(distances, key=lambda x: x[1])[2]
        offset = 0
        for dd in range(x):
            offset = offset + len(globals()['split' + str(dd)])
        clusters.update({i+offset: min_centroid_index})

sh = [] #array for storing initial centroids

#calculate mean centroid from each chunk in parallel
def worker_sharding(datachunk, i):
    global sh
    d_mean = np.mean(datachunk, axis=0)
    sh.insert(i, d_mean)

def calculate_centroids_sharding(dataset, k):
    global sh
    # Step 1: Sum the attribute (column) values for each instance (row) of a dataset
    instance_sums = dataset.sum(axis=1)
    # Step 2: Sort the instances of the dataset by the newly created sum column
    sorted_indices = np.argsort(instance_sums)
    sorted_dataset = dataset[sorted_indices]
    # Step 3: Split the dataset horizontally into k equal-sized pieces, or shards
    shards = np.array_split(sorted_dataset, k)

    thread_shard = []
    for i in range(k):
        thread3 = threading.Thread(target=worker_sharding, args=(shards[i], i))
        thread_shard.append(thread3)
        thread3.start()

    for thread3 in thread_shard:
        thread3.join()
    
    # Step 5: Return the set of centroid instances
    return np.array(sh)

#for each centroid, update its value by taking the average of data points assigned to it
def update_centroids(c, i):
    global clusters
    global centroids
    close_points = []
    for j in clusters.keys():
        index = clusters[j]
        if arrays_are_equal(c, centroids[index]):
            close_points.append(bank_scaled[j])
    centroids[i] = np.mean(close_points, axis=0)

# load the dataset
bank = pd.read_csv('BankChurners.csv')

#drop the last two columns as they are not required 
bank = bank.drop(bank.columns[-2:], axis=1)

#drop the first column as it is not required
bank = bank.drop(bank.columns[0], axis=1)

# one hot encoding
bank = pd.get_dummies(bank, ['Attrition_Flag', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'])

#choose the number of clusters to form
k = 71

#scale the dataset using standard scaling
scaler = StandardScaler()
bank_scaled = scaler.fit_transform(bank)

#use PCA to retain columns that capture 80% of the variance
pca = PCA(0.8)
bank_scaled = pca.fit_transform(bank_scaled)

#start the execution time
start = time.time()

#initialize k centroids using sharding
centroids = calculate_centroids_sharding(bank_scaled.copy(), k)

# choose number of threads
n = 2

#divide the dataset into n chunks, where n is the number of threads
bank_split = np.array_split(bank_scaled, n)
for i in range(n):
    globals()['split' + str(i)] = bank_split[i]

#for storing thread info
threads = []
threads2 = []

# make a dictionary with key as data point index and value as the cluster number
clusters = {}

#iteratively assign clusters and update centroids until a stopping criteria is reached
count = 0
while True:
    count += 1
    # assign the points to the clusters (assign point to nearest centroid) via multithreading
    for i in range(n):
        thread = threading.Thread(target=calculate_distance, args=(globals()['split' + str(i)], i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

     # update the centroids via multithreading. Choose mean of points in the cluster
    old_centroids = centroids.copy()
    for i in range(len(centroids)):
        thread1 = threading.Thread(target=update_centroids, args = (centroids[i], i))
        threads2.append(thread1)
        thread1.start()

    for thread1 in threads2:
        thread1.join()

    #if centroids didn't change from previous iteration, then stopping criteria reached
    if twoD_arrays_are_equal(old_centroids, centroids):
        break

end = time.time()

#print the clusters dictionary
print(clusters)

#execution time
print("Time taken: ", (end-start))

#run kmeans clustering using sklearn for verification
kmeans = KMeans(n_clusters=k, random_state=0, init=centroids).fit(bank_scaled)
cluster_labels = kmeans.fit_predict(bank_scaled)

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
for i in range(len(bank_scaled)):
    for j in range(i+1, len(bank_scaled)):
        if clusters[i] == clusters[j]:
            intra_cluster_distance += euclidean_distance(bank_scaled[i], bank_scaled[j])
        else:
            inter_cluster_distance += euclidean_distance(bank_scaled[i], bank_scaled[j])

print("Ratio of intra and inter cluster distances: ", intra_cluster_distance/inter_cluster_distance)