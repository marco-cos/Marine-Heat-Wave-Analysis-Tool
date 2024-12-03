import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from code.FunctionLibrary import ProcessNetCDF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time as t

def SSTClustered(filelocation, n_clusters=8):
    starttime = t.time()

    # Load and process data
    data = ProcessNetCDF(filelocation, getcoords=False, gettime=True)
    sst, time = data["sst"], data["time"]

    # Convert time to datetime objects
    time = pd.to_datetime(time, origin='julian', unit='D')

    # Average SST over the area (time x lat x lon -> time)
    avgsst = np.mean(sst, axis=(1, 2))

    # Reshape SST data for clustering: (time, lat, lon) -> (time, lat*lon)
    time_lat_lon = sst.reshape(sst.shape[0], -1)

    # Normalize the data (important for clustering algorithms)
    scaler = StandardScaler()
    time_lat_lon_scaled = scaler.fit_transform(time_lat_lon)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(time_lat_lon_scaled)

    # Reshape clusters to match the time dimension
    cluster_data = clusters.reshape(sst.shape[0], 1)

    # Print out the clusters (for each time point)
    print("Cluster assignments for each time point:")
    print(cluster_data)

    # Visualize cluster analysis (plot with color-coded clusters)
    plt.figure(figsize=(10, 6))
    for cluster in range(n_clusters):
        # Select indices where the cluster is equal to the current cluster
        cluster_indices = np.where(clusters == cluster)[0]
        
        # Plot time vs. average SST, using a different color for each cluster
        plt.scatter(time[cluster_indices], avgsst[cluster_indices], label=f'Cluster {cluster + 1}')

    # Add labels and title
    plt.title(f'SST Average with Cluster Analysis: {n_clusters} Clusters')
    plt.xlabel('Time')
    plt.ylabel('Average SST')
    plt.legend(title="Clusters")
    plt.show()

    print("Execution time:", round(t.time() - starttime, 2), "seconds")
    
    return clusters

# Usage example
