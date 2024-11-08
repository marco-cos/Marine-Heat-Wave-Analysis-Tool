def SSTMeanMap(filelocation):
    from code.FunctionLibrary import ProcessNetCDF
    from code.FunctionLibrary import MapPlot

    import time as t
    starttime = t.time()
    import netCDF4 as nc 
    
    import numpy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    data = nc.Dataset(filelocation)

    #Extract longitude, latitude, and SST variables
    data = ProcessNetCDF(filelocation)
    sst, lon, lat = data["sst"], data["lon"], data["lat"]

    #Average SST over time
    avgsst = numpy.mean(sst,axis=(0))

    print("Execution time:", round(t.time() - starttime, 2), "seconds")
    MapPlot(lon, lat, avgsst, "Average Sea Surface Temperature", "Temperature (°C)")
