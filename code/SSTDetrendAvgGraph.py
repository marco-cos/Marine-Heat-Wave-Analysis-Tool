def SSTDetrendAvgGraph(filelocation):
    from code.FunctionLibrary import ProcessNetCDF
    from code.FunctionLibrary import LinePlot

    import time as t
    starttime = t.time()
    import numpy
    import pandas as pd
    from scipy.signal import detrend 

    data = ProcessNetCDF(filelocation, getcoords=False, gettime=True)
    sst, time = data["sst"], data["time"]

    #Get all time values in array, convert to datetime objects
    time = pd.to_datetime(time, origin='julian', unit='D')
    #Get SST accross all time, at zlev of 0, accross all lat and long

    #Average SST over area, detrend
    avgsst = detrend(numpy.mean(sst,axis=(1,2)))

    #Print plot
    LinePlot(time, avgsst)

    print("Execution time:", round(t.time() - starttime, 2), "seconds")
