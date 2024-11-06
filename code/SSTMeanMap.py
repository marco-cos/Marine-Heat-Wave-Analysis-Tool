def SSTMeanMap(filelocation):
    import time as t
    starttime = t.time()
    import netCDF4 as nc 
    import matplotlib.pyplot as plt
    import numpy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    data = nc.Dataset(filelocation)

    #Extract longitude, latitude, and SST variables
    lon = data.variables['lon'][:]
    lat = data.variables['lat'][:]
    sst = data.variables['sst'][:, 0, :, :] 

    #Average SST over time
    avgsst = numpy.mean(sst,axis=(0))
    data.close()

    axes = plt.axes(projection=ccrs.PlateCarree())

    #Add in geographical features 
    axes.add_feature(cfeature.LAND)
    axes.add_feature(cfeature.OCEAN)
    axes.add_feature(cfeature.COASTLINE)
    axes.add_feature(cfeature.BORDERS, linestyle=':')
    axes.add_feature(cfeature.LAKES, alpha=0.5)
    axes.add_feature(cfeature.RIVERS)

    #Set size of map
    axes.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    plt.contourf(lon, lat, avgsst, levels=20, transform=ccrs.PlateCarree(), cmap='plasma')
    plt.title('Average Sea Surface Temperature')
    plt.colorbar(label='Temperature (°C)')
    gridlines = axes.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', color='gray')
    gridlines.top_labels = False  
    gridlines.right_labels = False 

    print("Execution time:", round(t.time() - starttime, 2), "seconds")
    plt.show()