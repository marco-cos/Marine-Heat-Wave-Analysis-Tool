from code.MHWDuration import MHWDuration
from code.MHWFrequency import MHWFrequency
from code.MHWIntensity import MHWIntensity
from code.SSTAvgGraph import SSTAvgGraph
from code.SSTDetrendAvgGraph import SSTDetrendAvgGraph
from code.SSTMeanMap import SSTMeanMap
import os
import sys

#Find available datasets
rawdatasetlist=os.listdir("datasets")
datasets={}
iter = 1
for file in rawdatasetlist:
    datasets[iter] = file
    iter+=1

amountofdatasets = len(datasets)
if amountofdatasets== 1:
    #If there is only one dataset, automatically select it
    selecteddataset = 1
elif amountofdatasets == 0:
    #If there are no datasets, throw error and terminate program
    print("No datasets found please add them to the datasets folder. Terminating program")
    sys.exit()
else:
    #If there are multiple datasets, allow user to select desired dataset
    print("Please enter the number associated with your desired dataset:")
    for key,value in datasets.items():
        print(f"{key}. {value}")
    selecteddataset = int(input())
    if selecteddataset not in datasets:
        print("Invalid choice, terminating program")
        sys.exit()

datasetlocation = (f"datasets/{datasets[selecteddataset]}")

actions = {
    1:("MHW Duration", MHWDuration),
    2:("MHW Frequency", MHWFrequency),
    3:("MHW Intensity", MHWIntensity),
    4:("SST Average Graph", SSTAvgGraph),
    5:("Detrended SST Average Graph", SSTDetrendAvgGraph),
    6:("SST Mean Map", SSTMeanMap)
}
print("Please enter the number associated with your desired action:")

for key,value in actions.items():
    print(f"{key}. {value[0]}")

selectedaction = int(input())

if selectedaction not in actions:
    print("Invalid choice, terminating program")
    sys.exit()

#Launch associated function
print(f"Beggining execution of {actions[selectedaction][0]}")
actions[selectedaction][1](datasetlocation)