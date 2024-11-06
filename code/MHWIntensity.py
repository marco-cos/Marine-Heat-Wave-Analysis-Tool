def MHWIntensity(filelocation):
    import time as t
    starttime = t.time()
    import netCDF4 as nc
    import matplotlib.pyplot as plt
    import numpy as np
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from scipy.signal import detrend
    from threading import Thread

    # Load the dataset
    data = nc.Dataset(filelocation)
    sst = data.variables['sst'][:, 0, :, :]
    lon = data.variables['lon'][:]
    lat = data.variables['lat'][:]
    data.close()

    # Detrend SST and calculate the 90th percentile for each cell
    detrended_sst = detrend(sst, axis=0)
    percentile_90 = np.percentile(detrended_sst, 90, axis=0)
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
                    if detrended_sst[time, latx, long] > percentile_90[latx, long]:
                        mhwdays += 1
                        if mhwdays == 5:
                            mhw_counts[long] += (1 / 12)  
                        if mhwdays >= 5:
                            total_duration[long] += 1
                            intensitysum+=(detrended_sst[time, latx, long] - percentile_90[latx, long])

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

    # Create a meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Plot the MHW frequency on a map
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor='beige'))
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plot the MHW frequency data
    cmap = plt.get_cmap("plasma")
    vmin = np.min(MHWmeanintensity[np.nonzero(MHWmeanintensity)])  # Avoid zero values if they are not meaningful
    vmax = np.max(MHWmeanintensity)
    mesh = ax.pcolormesh(lon_grid, lat_grid, MHWmeanintensity, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    plt.colorbar(mesh, ax=ax, label='Amount over 90th Percentile')
    plt.title('Average MHW Intensity')

    # Add gridlines and labels
    gridlines = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', color='gray')
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {'size': 10, 'color': 'black'}
    gridlines.ylabel_style = {'size': 10, 'color': 'black'}

    print("Execution time:", round(t.time() - starttime, 2), "seconds")
    plt.show()
