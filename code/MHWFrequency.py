def MHWFrequency(filelocation):
    from code.FunctionLibrary import ProcessNetCDF
    from code.FunctionLibrary import MapPlot
    
    from scipy.signal import detrend 
    from threading import Thread
    import numpy as np

    import time as t
    starttime = t.time()    

    # Load the dataset
    data = ProcessNetCDF(filelocation)
    sst, lat, lon = data["sst"], data["lat"], data["lon"]

    # Detrend SST and calculate the 90th percentile for each cell
    # Detrend SST and calculate the 90th percentile for each cell
    detrended_sst = detrend(sst, axis=0)
    percentile_90 = np.percentile(detrended_sst, 90, axis=0)
    MHWfrequency = np.zeros((detrended_sst.shape[1], detrended_sst.shape[2]), dtype=float)

    # Define a function to calculate MHW frequency for a range of latitude lines
    def calculate_mhw_for_chunk(latx_start, latx_end):
        for latx in range(latx_start, latx_end):
            mhw_counts = np.zeros(detrended_sst.shape[2], dtype=float)
            for long in range(detrended_sst.shape[2]):
                mhwdays = 0
                for time in range(detrended_sst.shape[0]):
                    if detrended_sst[time, latx, long] > percentile_90[latx, long]:
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

    MapPlot(lon,lat,MHWfrequency, "Average Number of MHW Events per Year", "Number of MHW events")