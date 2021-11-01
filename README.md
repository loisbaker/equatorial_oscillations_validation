## equatorial_oscillations_validation

The code in this repository was written and used for a submitted publication:

Baker, L.E., Bell, M.J., Blaker, A.T. (submitted to GRL, 2021) TAO data support the existence of large high frequency variations in cross-equatorial overturning circulation

The software and data is released on Zenodo: [![DOI](https://zenodo.org/badge/421371473.svg)](https://zenodo.org/badge/latestdoi/421371473)
This code allows the comparison of high frequency (3-15 day) oscillations in dynamic height in the equatorial Pacific Ocean between observations from the TAO mooring array and a global numerical simulation. These oscillations correspond to equatorially trapped inertia-gravity and Rossby-gravity waves. 

The temporal/ meridional spectra of the zonally integrated dynamic height oscillations are found and shown to agree well between the observations and the model.

### Code

The code is found in the subdirectory *code* of this repository. The notebooks to perform the analysis and create the three figures in the article are foundin *analysis_notebooks*. The functions used in the analysis are found in *tools*. The file *requirements.txt* defines the required environment.

### Data

The mooring data can be freely downloaded thanks to the GTMBA Project Office of NOAA/PMEL at https://www.pmel.noaa.gov/tao/drupal/disdel/. The data should be saved in the subdirectory *./data/TAO* of this repository as per the instructions in *instructions.txt* found therein. 

The NEMO dynamic height and wind stress fields used in these calculations are attached to the release of this repository. The individual files should be saved in the subdirectory *./data/NEMO* of this repository as per the instructions in *instructions.txt* found therein. 

The file baroclinic_wave_speeds.mat contains a global map of the first and second baroclinic wavespeeds, calculated using WOCE neutral density data. This can be found in *./data/baroclinic_wave_speeds*


Please contact  l.baker18@imperial.ic.ac.uk with bugs, comments, or questions. 

### Authors

Lois Baker, Imperial College London, 2021
