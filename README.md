# Marine Heat Wave Analysis Tool
This command-line tool allows users to analyze and visualize marine heat wave (MHW) events and sea surface temperature (SST) data. Using daily SST datasets, it calculates key MHW metrics like duration, frequency, and intensity, and creates visualizations such as line plots, maps, and cluster analyses. Designed to help visualize and understand trends in ocean temperatures while accounting for global SST rise.

## Included Functions
### SST Average Graph
Makes a line plot of the Area-Averaged sea surface temperature in celcius over time.
### Detrended SST Average Graph
Makes a line plot of the Area-Averaged sea surface temperature in celcius over time that is detrended to account for global sea surface temperature (SST) rise.
### Cluster Analysis for Detrended SST Average Graph
Like detrended SST average graph function, but with K-means cluster analysis performed. 
### SST Mean Map
Returns a map of the time-averaged sea surface temperature in celcius by location.
### Marine Heat Wave Duration
Returns a map showing the average duration in days of Marine Heat Wave (MHW) events, with an option to select what percentile is considered a MHW event (90th recommended). This uses detrended SST data to account for global SST rise. 
### Marine Heat Wave Frequency
Returns a map showing the average number of Marine Heat Wave (MHW) events in a year by location, with an option to select what percentile is considered a MHW event (90th recommended). This uses detrended SST data to account for global SST rise. 
### Marine Heat Wave Intensity
Returns a map showing the average itensity (amount over selected percentile) of Marine Heat Wave (MHW) by location, with an option to select what percentile is considered a MHW event (90th recommended). This uses detrended SST data to account for global SST rise. 
### Marine Heat Wave Cluster Analysis
Performs K-means clustering on your selected variable (MHW Duration, Frequency, or Intensity), with an option to select how many clusters you would like to use in your analysis. 

## Data
Designed to be used with data obtained from the [IRI/LDEO Climate Data Library](iridl.ldeo.columbia.edu). Included dataset contains daily sea surface temperature data from Jan 1 2000 - Dec 31 2023 in the Caribbean region.

## Required Third-Party Modules and Packages

1. **`netCDF4`**
   - Used to read and process NetCDF files containing sea surface temperature (SST) data.

2. **`matplotlib`**
   - Used to create line plots, maps, and visualizations of SST data, including Marine Heat Wave (MHW) metrics and clustering results.

3. **`scipy`**
   - Used for data detrending to remove long-term trends from SST data.

4. **`numpy`**
   - Used for numerical computations, including calculating percentiles, reshaping data arrays, and statistical analysis.

5. **`cartopy`**
   - Used for geographical plotting, adding map features like coastlines, borders, and projections to visualize SST and MHW data.

6. **`sklearn` (scikit-learn)**
   - Used for performing K-means clustering and data normalization during MHW cluster analysis.

7. **`pandas`**
   - Used to convert time data into a usable `datetime` format for plotting and analysis.

## Installation Instructions 
Before installing the repository, ensure that python3 and pip is installed. You may need to create a virtual environment using `python3 -m venv .venv` to install the required dependancies. To enter the virutal environment in Windows, use `venv\Scripts\activate`. For Mac and Linux, use `source .venv/bin/activate`.
1. Clone the repository:
   ```bash
   git clone https://github.com/marco-cos/Marine-Heat-Wave-Analysis-Tool
2. Navigate to the project directory::
   ```bash
   cd Marine-Heat-Wave-Analysis-Tool
3. Install required packages:
    ```bash
    pip install -r requirements.txt
4. Run the program:
    ```bash
    python3 main.py