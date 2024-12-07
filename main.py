from code.SSTAvgGraph import SSTAvgGraph
from code.SSTDetrendAvgGraph import SSTDetrendAvgGraph
from code.SSTMeanMap import SSTMeanMap
from code.FunctionLibrary import MapPlot
from code.FunctionLibrary import GetMHWFrequencyValues
from code.FunctionLibrary import GetMHWIntensityValues
from code.FunctionLibrary import GetMHWDurationValues
from code.FunctionLibrary import PerformClusterAnalysis
from code.FunctionLibrary import SSTClusterAnalysis

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


#Action-specific functons
def MHWDurationPlot(filelocation):
    MHWDurationRet = GetMHWDurationValues(filelocation)
    MapPlot(MHWDurationRet["lon"],MHWDurationRet["lat"],MHWDurationRet["values"], "Average Marine Heat Wave Duration", "Days")

def MHWFrequencyPlot(filelocation):
    MHWFrequencyRet = GetMHWFrequencyValues(filelocation)
    MapPlot(MHWFrequencyRet["lon"],MHWFrequencyRet["lat"],MHWFrequencyRet["values"], "Average Number of MHW Events per Year", "Number of MHW events")

def MHWIntensityPlot(filelocation):
    MHWIntensityRet = GetMHWIntensityValues(filelocation)
    MapPlot(MHWIntensityRet["lon"],MHWIntensityRet["lat"],MHWIntensityRet["values"],"Average MHW Intensity", "Amount over 90th Percentile")

def LaunchClusterAnalysis(filelocation):
    clustervariables = {
        1:("MHW Duration",GetMHWDurationValues),
        2:("MHW Frequency",GetMHWFrequencyValues),
        3:("MHW Intensity",GetMHWIntensityValues)
    }

    print("Please enter the number associated with the variable you would like to cluster:")
    
    for key,value in clustervariables.items():
        print(f"{key}. {value[0]}")
    
    selectedvariable = int(input())

    if selectedvariable not in clustervariables:
        print("Invalid choice, terminating program")
        sys.exit()
    
    print("How many clusters would you like?")
    clusters=int(input())

    if clusters < 1 or clusters > 50:
        print("Invalid choice, terminating program")
        sys.exit()

    returnedvariable = clustervariables[selectedvariable][1](filelocation)
    PerformClusterAnalysis(returnedvariable,clusters)

def LaunchSSTClusterAnalysis(filelocation):
    print("How many clusters would you like?")
    clusters=int(input())

    if clusters < 1 or clusters > 50:
        print("Invalid choice, terminating program")
        sys.exit()
    SSTClusterAnalysis(filelocation,clusters)
    
    

actions = {
    1:("MHW Duration", MHWDurationPlot),
    2:("MHW Frequency", MHWFrequencyPlot),
    3:("MHW Intensity", MHWIntensityPlot),
    4:("SST Average Graph", SSTAvgGraph),
    5:("Detrended SST Average Graph", SSTDetrendAvgGraph),
    6:("SST Mean Map", SSTMeanMap),
    7:("Detrended SST Cluster Analysis",LaunchSSTClusterAnalysis),
    8:("MHW Cluster Analysis",LaunchClusterAnalysis)
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