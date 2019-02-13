# Prepare TOA data for YAPS

Code to create a Time Of Arrival (TOA) matrix, departing from receiver data.   
In the resulting matrix, each line contains the observations of one ping, and each ping has one line (if not observed, this is a line of NaT's).   
The TOA matrix can be fed to the [YAPS algorithm] (https://github.com/baktoft/yaps), presented in Baktoft, Gjelland, Økland & Thygesen (2017): Positioning of aquatic animals based on time-of-arrival and random walk models using YAPS (Yet Another Positioning Solver).

Prepare_toa_data.py contains all required functions.   
Run Use_of_prepare_toa_data.py to create a TOA-matrix from the given example data.   
This will also create 4 figures that illustrate the process on the given data.   

For questions, feel free to open an issue or contact the author!