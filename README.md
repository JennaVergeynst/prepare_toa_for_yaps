# Prepare TOA data for YAPS

Code to create a Time Of Arrival (TOA) matrix, departing from synchronised receiver data (for synchronisation, see [this page](https://github.com/JennaVergeynst/time_synchronization)).   
In the resulting matrix, each line contains the observations of one ping, and each ping has one line (if not observed, this is a line of NaT's).   
The TOA matrix can be fed to the [YAPS algorithm](https://github.com/baktoft/yaps), presented in Baktoft, Gjelland, Økland & Thygesen (2017): Positioning of aquatic animals based on time-of-arrival and random walk models using YAPS (Yet Another Positioning Solver).

Prepare_toa_data.py contains all required functions.   
Run Use_of_prepare_toa_data.py to create a TOA-matrix from the given example data.   
This will also create 4 figures that illustrate the process on the given data.   

For questions, feel free to open an issue or contact the author!

Cite as: Vergeynst, Jenna. (2019, November 13). JennaVergeynst/prepare_toa_for_yaps: Code to prepare TOA-data for YAPS. Zenodo. http://doi.org/10.5281/zenodo.3540682

## Note before you start
This code is in python, but it is also entirely [available in R](https://github.com/elipickh/ReceiverArrays), thanks to Eliezer Pickholtz.

The creation of TOA matrices is now included in the YAPS package, making this work possibly redundant. Please check the step-by-step guide available via the [YAPS page](https://github.com/baktoft/yaps).
