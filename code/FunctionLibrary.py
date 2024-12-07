def ProcessNetCDF(filelocation, getcoords = True, gettime = False):
    #Extract SST, lon, lat arrays
    import netCDF4 as nc
    data = nc.Dataset(filelocation)

    toret =  {
        "sst": data.variables['sst'][:, 0, :, :],
        "lon": data.variables['lon'][:] if getcoords else False,
        "lat": data.variables['lat'][:] if getcoords else False,
        "time": data.variables['T'][:]
    }
    data.close()
    return toret

def LinePlot(xaxis, yaxis):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(xaxis, yaxis)
    plt.xlabel('Time')
    plt.ylabel('Sea Surface Temperature (C°)')
    plt.title('Area-Averaged Sea Surface Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()

def MapPlot(lon, lat, data, title, barlabel, levels=20):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    #Add in geographical features 
    axes = plt.axes(projection=ccrs.PlateCarree())
    axes.add_feature(cfeature.OCEAN)
    axes.add_feature(cfeature.COASTLINE)
    axes.add_feature(cfeature.BORDERS, linestyle=':')
    axes.add_feature(cfeature.LAKES, alpha=0.5)
    axes.add_feature(cfeature.RIVERS)
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',edgecolor='face',facecolor='beige')
    axes.add_feature(land, zorder=1)


    #Settings for map
    plt.contourf(lon, lat, data, levels, transform=ccrs.PlateCarree(), cmap='plasma')
    plt.title(title)
    plt.colorbar(label=barlabel)
    gridlines = axes.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', color='gray')
    gridlines.top_labels = False  
    gridlines.right_labels = False 
    plt.show()

def GetMHWFrequencyValues(filelocation):
    import sys
    from scipy.signal import detrend 
    from threading import Thread
    import numpy as np

    # Load the dataset
    data = ProcessNetCDF(filelocation)
    sst, lat, lon = data["sst"], data["lat"], data["lon"]

    print("Please enter your desired percentile to calculate MHW events:")
    selectedpercentile = int(input())

    if (selectedpercentile < 1 or selectedpercentile > 99):
        print("Invalid choice, terminating program")
        sys.exit()

    # Detrend SST and calculate the 90th percentile for each cell
    detrended_sst = detrend(sst, axis=0)
    percentile = np.percentile(detrended_sst, selectedpercentile, axis=0)
    MHWfrequency = np.zeros((detrended_sst.shape[1], detrended_sst.shape[2]), dtype=float)

        # Define a function to calculate MHW frequency for a range of latitude lines
    def calculate_mhw_for_chunk(latx_start, latx_end):
        for latx in range(latx_start, latx_end):
            mhw_counts = np.zeros(detrended_sst.shape[2], dtype=float)
            for long in range(detrended_sst.shape[2]):
                mhwdays = 0
                for time in range(detrended_sst.shape[0]):
                    if detrended_sst[time, latx, long] > percentile[latx, long]:
                        mhwdays += 1
                        if mhwdays == 5:
                            mhw_counts[long] += (1 / 12)  
                    else:
                        mhwdays = 0
            MHWfrequency[latx, :] = mhw_counts 

    # Split work into chunks and create threads for each chunk
    num_threads = 4  # Adjust based on system capabilities
    chunk_size = detrended_sst.shape[1] // num_threads
    threads = []

    for i in range(num_threads):
        latx_start = i * chunk_size
        latx_end = (i + 1) * chunk_size if i < num_threads - 1 else detrended_sst.shape[1]
        thread = Thread(target=calculate_mhw_for_chunk, args=(latx_start, latx_end))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    return {
        "values":MHWfrequency,
        "lon":lon,
        "lat":lat
    }

def GetMHWIntensityValues(filelocation):
    import sys
    from scipy.signal import detrend 
    from threading import Thread
    import numpy as np

   # Load the dataset
    data = ProcessNetCDF(filelocation)
    sst, lat, lon = data["sst"], data["lat"], data["lon"]

    print("Please enter your desired percentile to calculate MHW events:")
    selectedpercentile = int(input())

    if (selectedpercentile < 1 or selectedpercentile > 99):
        print("Invalid choice, terminating program")
        sys.exit()

    # Detrend SST and calculate the 90th percentile for each cell
    detrended_sst = detrend(sst, axis=0)
    percentile = np.percentile(detrended_sst, selectedpercentile, axis=0)
    MHWfrequency = np.zeros((detrended_sst.shape[1], detrended_sst.shape[2]), dtype=float)
    MHWduration = np.zeros((detrended_sst.shape[1], detrended_sst.shape[2]), dtype=float)
    MHWmeanintensity = np.zeros((detrended_sst.shape[1], detrended_sst.shape[2]), dtype=float)

    # Define a function to calculate MHW frequency for a range of latitude lines
    def calculate_mhw_for_chunk(latx_start, latx_end):
        for latx in range(latx_start, latx_end):
            mhw_counts = np.zeros(detrended_sst.shape[2], dtype=float)
            total_duration = np.zeros(detrended_sst.shape[2], dtype=float)  # Track total duration of events
            mean_intensity = np.zeros(detrended_sst.shape[2], dtype=float)

            for long in range(detrended_sst.shape[2]):            
                mhwdays = 0
                intensitysum = 0
                for time in range(detrended_sst.shape[0]):
                    if detrended_sst[time, latx, long] > percentile[latx, long]:
                        mhwdays += 1
                        if mhwdays == 5:
                            mhw_counts[long] += (1 / 12)  
                        if mhwdays >= 5:
                            total_duration[long] += 1
                            intensitysum+=(detrended_sst[time, latx, long] - percentile[latx, long])

                    else:
                        if (mhwdays >= 5):
                            current_count = mhw_counts[long]

                            mean_intensity[long] += (intensitysum / mhwdays - mean_intensity[long]) / current_count
                        mhwdays = 0
                        intensitysum=0
            MHWfrequency[latx, :] = mhw_counts 
            MHWduration[latx,:] = total_duration
            MHWmeanintensity[latx,:] = mean_intensity

    # Split work into chunks and create threads for each chunk
    num_threads = 4  # Adjust based on system capabilities
    chunk_size = detrended_sst.shape[1] // num_threads
    threads = []

    for i in range(num_threads):
        latx_start = i * chunk_size
        latx_end = (i + 1) * chunk_size if i < num_threads - 1 else detrended_sst.shape[1]
        thread = Thread(target=calculate_mhw_for_chunk, args=(latx_start, latx_end))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    return {
        "values":MHWmeanintensity,
        "lon":lon,
        "lat":lat
    }

def GetMHWDurationValues(filelocation):
    import sys
    import numpy as np
    from scipy.signal import detrend
    from threading import Thread

    # Load the dataset
    data = ProcessNetCDF(filelocation)
    sst, lat, lon = data["sst"], data["lat"], data["lon"]

    print("Please enter your desired percentile to calculate MHW events:")
    selectedpercentile = int(input())

    if (selectedpercentile < 1 or selectedpercentile > 99):
        print("Invalid choice, terminating program")
        sys.exit()

    # Detrend SST and calculate the 90th percentile for each cell
    detrended_sst = detrend(sst, axis=0)
    percentile = np.percentile(detrended_sst, selectedpercentile, axis=0)
    MHWfrequency = np.zeros((detrended_sst.shape[1], detrended_sst.shape[2]), dtype=float)
    MHWduration = np.zeros((detrended_sst.shape[1], detrended_sst.shape[2]), dtype=float)
    MHWmeanduration = np.zeros((detrended_sst.shape[1], detrended_sst.shape[2]), dtype=float)

    # Define a function to calculate MHW frequency for a range of latitude lines
    def calculate_mhw_for_chunk(latx_start, latx_end):
        for latx in range(latx_start, latx_end):
            mhw_counts = np.zeros(detrended_sst.shape[2], dtype=float)
            total_duration = np.zeros(detrended_sst.shape[2], dtype=float)  # Track total duration of events
            mean_duration = np.zeros(detrended_sst.shape[2], dtype=float)

            for long in range(detrended_sst.shape[2]):            
                mhwdays = 0
                for time in range(detrended_sst.shape[0]):
                    if detrended_sst[time, latx, long] > percentile[latx, long]:
                        mhwdays += 1
                        if mhwdays == 5:
                            mhw_counts[long] += (1 / 12)  
                        if mhwdays >= 5:
                            total_duration[long] += 1
                    else:
                        if (mhwdays >= 5):
                            current_count = mhw_counts[long]

                            mean_duration[long] += (mhwdays - mean_duration[long]) / current_count
                        mhwdays = 0
            MHWfrequency[latx, :] = mhw_counts 
            MHWduration[latx,:] = total_duration
            MHWmeanduration[latx,:] = mean_duration

    # Split work into chunks and create threads for each chunk
    num_threads = 4  # Adjust based on system capabilities
    chunk_size = detrended_sst.shape[1] // num_threads
    threads = []

    for i in range(num_threads):
        latx_start = i * chunk_size
        latx_end = (i + 1) * chunk_size if i < num_threads - 1 else detrended_sst.shape[1]
        thread = Thread(target=calculate_mhw_for_chunk, args=(latx_start, latx_end))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    return {
        "values":MHWmeanduration,
        "lon":lon,
        "lat":lat
    }


def PerformClusterAnalysis(variable,clusters):
    from sklearn.cluster import KMeans
    # Flatten the MHW frequency data for clustering
    flat_data = variable["values"].reshape(-1, 1)  # Each point is a single feature: its MHW frequency

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(flat_data)

    # Reshape the cluster labels to match the spatial dimensions
    cluster_labels_2d = cluster_labels.reshape(variable["values"].shape)

    # Plot the clustering results
    MapPlot(variable["lon"], variable["lat"], cluster_labels_2d, f"K-Means Clustering with {clusters} Clusters", "Cluster ID",clusters)

def SSTClusterAnalysis(filelocation, n_clusters=8, detrended=True):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from scipy.signal import detrend 


    # Load and process data
    data = ProcessNetCDF(filelocation, getcoords=False, gettime=True)
    sst, time = data["sst"], data["time"]

    # Convert time to datetime objects
    time = pd.to_datetime(time, origin='julian', unit='D')

    # Average SST over the area (time x lat x lon -> time)
    avgsst = np.mean(sst, axis=(1, 2))
    
    if detrended:
        avgsst = detrend(np.mean(sst,axis=(1,2)))

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
    return clusters