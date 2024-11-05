import netCDF4 as nc 
import matplotlib.pyplot as plt
import numpy
from datetime import datetime
import pandas as pd

data = nc.Dataset('../datasets/SST_Daily_Carib_1Jan2000_31Dec2023.nc')

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
plt.plot(time, avgsst, label='Average SST')
plt.xlabel('Time')
plt.ylabel('Average Sea Surface Temperature (SST)')
plt.title('Average Sea Surface Temperature Over Time')
plt.legend()
plt.grid(True)
plt.show()