
def SSTAvgGraph(filelocation):
    from code.FunctionLibrary import ProcessNetCDF
    from code.FunctionLibrary import LinePlot

    import time as t
    starttime = t.time()

    import netCDF4 as nc 
    import matplotlib.pyplot as plt
    import numpy
    import pandas as pd

    data = ProcessNetCDF(filelocation, getcoords=False, gettime=True)
    sst, time = data["sst"], data["time"]

    #Covert time to datetime objects
    time = pd.to_datetime(time, origin='julian', unit='D')

    #Average SST over area
    avgsst = numpy.mean(sst,axis=(1,2))

    #Print plot
    LinePlot(time,avgsst)
    print("Execution time:", round(t.time() - starttime, 2), "seconds")
    