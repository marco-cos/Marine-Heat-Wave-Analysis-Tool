
def SSTAvgGraph(filelocation):
    import time as t
    starttime = t.time()

    import netCDF4 as nc 
    import matplotlib.pyplot as plt
    import numpy
    from datetime import datetime
    import pandas as pd

    data = nc.Dataset(filelocation)

    #Get all time values in array, convert to datetime objects
    time = data.variables['T'][:]
    time = pd.to_datetime(time, origin='julian', unit='D')
    #Get SST accross all time, at zlev of 0, accross all lat and long
    sst = data.variables['sst'][:, 0, :, :] 

    #Average SST over area
    avgsst = numpy.mean(sst,axis=(1,2))
    data.close()

    #Print plot
    plt.figure(figsize=(10, 5))
    plt.plot(time, avgsst)
    plt.xlabel('Time')
    plt.ylabel('Sea Surface Temperature (C°)')
    plt.title('Area-Averaged Sea Surface Temperature')
    plt.legend()
    plt.grid(True)

    print("Execution time:", round(t.time() - starttime, 2), "seconds")
    plt.show()