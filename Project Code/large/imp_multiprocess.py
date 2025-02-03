#import all the essential libraries
import numpy as np
import math
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import time
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
def euclidean_distance(x:list, y:list):
    distance = 0
    for i in range(len(x)):
        distance += math.pow(x[i] - y[i], 2)
    return math.sqrt(distance)

#assign cluster number to each data point based on minimum euclidean distance
def multiprocess(datapoint, centroids):
    distances = []
    for j in range(len(centroids)):
        distance = euclidean_distance(datapoint, centroids[j])
        distances.append((centroids[j], distance, j))
    min_centroid_index = min(distances, key=lambda x: x[1])[2]
    return min_centroid_index
    
#calculate mean centroid from each chunk in parallel
def worker_sharding(datachunk, i):
    return np.mean(datachunk, axis=0)
    
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

    pool_s = multiprocessing.Pool(processes=2)
    r = pool_s.starmap(worker_sharding, [(shards[i], i) for i in range(k)])
    pool_s.close()
    pool_s.join()
    for i in range(k):
        centroids.append(r[i])

    # Step 5: Return the set of centroid instances
    return np.array(centroids)

#for each centroid, update its value by taking the average of data points assigned to it
def multiprocess_update(cent, clusters, centroids, dataset):
    close_points = []
    for j in clusters.keys():
        index = clusters[j]
        if arrays_are_equal(centroids[index], cent):
            close_points.append(dataset[j])
    return np.mean(close_points, axis=0)

# Main function implementing the Farm skeleton
def farm(bank_scaled):
    global clusters
    global centroids
    count = 0
    while True:
        count += 1

        # assign each datapoint to the clusters (assign point to nearest centroid) in parallel
        args = [(bank_scaled[i], centroids) for i in range(len(bank_scaled))]
        pool = multiprocessing.Pool(processes=2)
        results = pool.starmap(multiprocess, args)
        pool.close()
        pool.join()

        #gather cluster numbers from each process and store in clusters dictionary
        for i in range(len(results)): 
            clusters.update({i: results[i]})

        #update the centroids in parallel. Choose mean of data points assigned to a cluster as the new centroid
        old_centroids = centroids.copy()
        args2 = [(centroids[i], clusters, centroids, bank_scaled) for i in range(len(centroids))]
        pool2 = multiprocessing.Pool(processes=2)
        result2 = pool2.starmap(multiprocess_update, args2)
        pool2.close()
        pool2.join()

        #gather new values of centroids from each process and store in centroids list
        for i in range(len(centroids)):
            centroids[i] = result2[i]

        #if centroids didn't change from previous iteration, then stopping criteria reached
        if twoD_arrays_are_equal(old_centroids, centroids):
            break
    return 0

if __name__ == '__main__':
    # load the dataset
    bank = pd.read_csv('BankChurners.csv')
    
    #drop the last two columns as they are not required 
    bank = bank.drop(bank.columns[-2:], axis=1)

    #drop the first column as it is not required
    bank = bank.drop(bank.columns[0], axis=1)

    # one hot encoding
    bank = pd.get_dummies(bank, ['Attrition_Flag', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'])

    #scale the dataset using standard scaling
    scaler = StandardScaler()
    bank_scaled = scaler.fit_transform(bank)

    #use PCA to retain columns that capture 80% of the variance
    pca = PCA(0.8)
    bank_scaled = pca.fit_transform(bank_scaled)

    #choose the number of clusters to form 
    k = 71

    #start the execution time
    start = time.time()

    #initialize k centroids using sharding
    centroids = calculate_centroids_sharding(bank_scaled.copy(), k)
    
    # make a dictionary with key as data point index and value as the cluster number
    clusters = {}
    farm(bank_scaled)
    
    end = time.time()

    #print the clusters dictionary
    print(clusters)

    #execution time 
    print("Time taken: ", (end - start))

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