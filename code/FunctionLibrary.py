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

def MapPlot(lon, lat, data, title, barlabel):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    #Add in geographical features 
    axes = plt.axes(projection=ccrs.PlateCarree())
    axes.add_feature(cfeature.LAND)
    axes.add_feature(cfeature.OCEAN)
    axes.add_feature(cfeature.COASTLINE)
    axes.add_feature(cfeature.BORDERS, linestyle=':')
    axes.add_feature(cfeature.LAKES, alpha=0.5)
    axes.add_feature(cfeature.RIVERS)

    #Settings for map
    plt.contourf(lon, lat, data, levels=20, transform=ccrs.PlateCarree(), cmap='plasma')
    plt.title(title)
    plt.colorbar(label=barlabel)
    gridlines = axes.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', color='gray')
    gridlines.top_labels = False  
    gridlines.right_labels = False 
    plt.show()

