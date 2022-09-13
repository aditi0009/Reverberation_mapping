-> In this repository, the folder "raw_data" contains the raw data fits files
of the spectrum of each day of different RMIDS - 101,191,229,267,272,694,772 and 840.

-> The folder "QA_other" contains the spectral analysis of each observation of different
RMIDs using PyQSOFit.

-> The folder "results" contains the fitting results of each observation corresponding to the
 different emission lines in form of fits file.

-> NOTE : If one wants to determine the results of the RMID corresponding spectras, one can use test_suv.py
code using python2

-> Now, the file result.py analyzes the result in order to compute the following -

1. Hbeta emission flux over time
2. OIII emission over time
3. L_5100 Luminosity over time
4. Rescaled Emission fluxes with respect to OIII
5. Comparing a subset of data points to match with the range of the paper with the Emission
fluxes obtained in the paper - The Sloan Digital Sky Survey Reverberation Mapping Project:
Hα and Hβ Reverberation Measurements from First-year Spectroscopy and Photometry


NOTE: ALL THE CODES HERE ARE COMPATIBLE WITH PYTHON2 AND NOT PYTHON3.
IF ONE WANTS TO DO THE SPECTRAL ANALYSIS ON THEIR OWN - THEY CAN USE THE CODE FROM :
https://github.com/legolason/PyQSOFit.git
THE REQUIREMENTS FOR IT ARE GIVEN IN THE EXAMPLE.IPYNB
