#PATH="/Library/Frameworks/Python.framework/Versions/2.7/bin:${PATH}"
#export PATH
#%matplotlib inline
import glob, os,sys,timeit
import matplotlib
import numpy as np
from PyQSOFit_suv import QSOFit
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit
import scipy as sp
warnings.filterwarnings("ignore")


path='./'

path0='./data/772/'    # Path of data 
path1='./'                  # the path of the source code file and qsopar.fits
path2='./data/r_772/' # path of fitting results
path3='./data/QA_other/'   # path of figure
path4='./sfddata/'             # path of dusp reddening map

path5='./model/'   # path to store the models

def get_FWHM_kms(pp):
   """
   get_FWHM_kms([0, np.log(4861.), 0.0017]) = 1200
   """
   FWHM=(np.exp(pp[1]+pp[2])-np.exp(pp[1]))/np.exp(pp[1])*300000.*2.35
   return FWHM

def get_FWHM_kms_hb(p0):
   return (np.exp(np.log(4862.)+p0)-4862.)/4862*300000.*2.35



ngauss=2
voff_b = 0.0043 # 3000 km/s
voff_n = 0.00142 # 1000 km/s 

#get_FWHM_kms([0, np.log(4861.), 0.0017]) = 1200
### *** Check Voff limits. Its too much currently. 
### Check also SIgma calculation (test with Peterson 2004 method) due to very broad component. 
### Try somehow to remove the objects to very broad component from the fitting
### Full profile (without narrow component subtraction) is used for CIII, CIV and Lya:
### however for QA_l plot, all component with FWHM<1200 km/s is plotted in green 
#2.3e-4
#                         lc     name  min max     comp    ng    guess, minp, maxp, 
#                                                                (400 km for narrow) (minp 100 km/s - 900 km/s)
newdata = np.rec.array([(6564.61,'Ha',6400.,6800.,'Ha_br',  ngauss,       5e-3,0.0012761,0.05 ,voff_b,0,0,0,0.05),\
                        (6564.61,'Ha',6400.,6800.,'Ha_na',  1,   5.7e-4,1.42e-4, 0.001275,voff_n/2, 0,1,0,0.001),\
                        #(6564.61,'Ha',6400.,6800.,'Ha_na', 1,   5.7e-4,1.42e-4, 0.001275,voff_n/2, 0,0,0,0.001),\
                        (6549.85,'Ha',6400.,6800.,'NII6549',1,   5.7e-4,1.42e-4, 0.001275,voff_n/2, 1,1,1,0.001),\
                        (6585.28,'Ha',6400.,6800.,'NII6585',1,   5.7e-4,1.42e-4, 0.001275,voff_n/2, 1,1,1,0.003),\
                        (6718.29,'Ha',6400.,6800.,'SII6718',1,   5.7e-4,1.42e-4, 0.001275,voff_n/2, 0,1,0,0.001),\
                        (6732.67,'Ha',6400.,6800.,'SII6732',1,   5.7e-4,1.42e-4, 0.001275,voff_n/2, 0,1,0,0.001),\
                        (4862.68,'Hb',4640.,5100.,'Hb_br',    ngauss,5e-3,0.0012761,0.05,         voff_b,0,0,0,0.01),\
                        (4862.68,'Hb',4640.,5100.,'Hb_na',    1,    5.7e-4,1.42e-4,0.001275, voff_n,1,1,0,0.001),\
                        (4960.30,'Hb',4640.,5100.,'OIII4959c',1,    5.7e-4,1.42e-4,0.001275, voff_n,1,1,1,0.001),\
                        (5008.24,'Hb',4640.,5100.,'OIII5007c',1,    5.7e-4,1.42e-4,0.001275, voff_n,1,1,1,0.003),\
                        (4960.30,'Hb',4640.,5100.,'OIII4959w',1,    3e-3,2.3e-4,0.003,      voff_b,2,2,2,0.001),\
                        (5008.24,'Hb',4640.,5100.,'OIII5007w',1,    3e-3,2.3e-4,0.003,      voff_b,2,2,2,0.003),\
                        (4687.02,'Hb',4640.,5100.,'HeII4687_br',1,  5e-3,0.0012761,0.05,    voff_b,0,0,0,0.001),\
                        (4687.02,'Hb',4640.,5100.,'HeII4687_na',1,  5.7e-4,1.42e-4,0.001275,voff_n,1,1,0,0.001),\
                        #(5876,'He',5700.,6150.,'HeI5876_br',1,  5e-3,0.0012761,0.05,    voff_b,0,0,0,0.001),\
                        #(5876,'He',5700.,6150.,'HeI5876_na',1,  5.7e-4,1.42e-4,0.001275,voff_n,0,0,0,0.001),\
                        (4341.68,'Hg',4320.,4380.,'Hg_br',    1,5e-3,0.0012761,0.05, voff_b,0,0,0,0.01),\
                        (4341.68,'Hg',4320.,4380.,'Hg_na',    1,1e-3,2.3e-4,0.001275,voff_n,1,1,0,0.002),\
                        (4364.436,'Hg',4320.,4380.,'OIII4364',1,1e-3,2.3e-4,0.001275,voff_n,1,1,0,0.002),\
                        (2798.75,'MgII',2700.,2900.,'MgII_br',ngauss,5e-3,0.0012761,0.05, voff_b,0,0,0,0.05),\
                        (2798.75,'MgII',2700.,2900.,'MgII_na',1,1e-3,5e-4, 0.001275, voff_n,1,0,0,0.002),\
                        ],\
                     formats='float32,a20,float32,float32,a20,float32,float32,float32,float32,\
                     float32,float32,float32,float32,float32',\
                     names='lambda,compname,minwav,maxwav,linename,ngauss,inisig,minsig,maxsig,voff,vindex,windex,findex,fvalue')

#------header-----------------
hdr = fits.Header()
hdr['lambda'] = 'Vacuum Wavelength in Ang'
hdr['minwav'] = 'Lower complex fitting wavelength range'
hdr['maxwav'] = 'Upper complex fitting wavelength range'
hdr['ngauss'] = 'Number of Gaussians for the line'
hdr['inisig'] = 'Initial guess of linesigma [in lnlambda]'
hdr['minsig'] = 'Lower range of line sigma [lnlambda]'  
hdr['maxsig'] = 'Upper range of line sigma [lnlambda]'
hdr['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
hdr['vindex'] = 'Entries w/ same NONZERO vindex constrained to have same velocity'
hdr['windex'] = 'Entries w/ same NONZERO windex constrained to have same width'
hdr['findex'] = 'Entries w/ same NONZERO findex have constrained flux ratios'
hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'
#------save line info-----------
hdu = fits.BinTableHDU(data=newdata,header=hdr,name='data')
hdu.writeto(path+'qsopar.fits',overwrite=True)



def run_fit(spec_name):
  """
  This function takes one spectra (spec_name) as input and do the fitting.
  Store all the results, plots and models in different folder as mentioned
  """
  data = fits.open(spec_name)
  lam=10**data[1].data['loglam']        # OBS wavelength [A]
  flux=data[1].data['flux']             # OBS flux [erg/s/cm^2/A]
  err=1./np.sqrt(data[1].data['ivar'])  # 1 sigma error
  z=data[2].data['z'][0]                 # Redshift
  #Optional
  ra=data[0].header['plug_ra']          # RA 
  dec=data[0].header['plug_dec']        # DEC
  #plateid = data[0].header['plateid']   # SDSS plate ID
  #mjd = data[0].header['mjd']           # SDSS MJD
  #fiberid = data[0].header['fiberid']   # SDSS fiber ID
  #sps = str.split(str.split(spec_name, '.')[0], '-')  # some JD values are wrongly saved inside fits file: 
  #e.g. 0389-51794-0611 is saved instead of 51795
  #plateid, mjd, fiberid= int(sps[1]), int(sps[2]), int(sps[3])
  ra=data[0].header['plug_ra']
  dec=data[0].header['plug_dec']
  plateid = data[0].header['plateid']
  mjd = data[0].header['mjd']
  fiberid = data[0].header['fiberid']   

  #spec_name = str(plateid).zfill(4)+'_'+str(mjd)+'_'+str(fiberid).zfill(4)
  print ("spec_name:====", spec_name)
  # Do the fitting:
  if z<0.8: decomposition_host = True
  else: decomposition_host = False
  deredden, BC, poly, Fe_uv_op = True, True, False, True
  wave_range = None 
  print ("----------------------")
  print ("redshift:", z)
  print ("deredden:", deredden)
  print ("decomposition_host:", decomposition_host)
  print ("Fe_uv_op:", Fe_uv_op)
  print ("BC:", BC)
  print ("Poly:", poly)
  print ("----------------------")
  start = timeit.default_timer()
  # get data prepared 
  #q = QSOFit(lam, flux, err, z, ra = ra, dec = dec, plateid = plateid, mjd = mjd, fiberid = fiberid, path = path1)
  q = QSOFit(lam, flux, err, z, ra = ra, dec = dec, plateid = plateid, mjd = mjd, fiberid = fiberid, path = path1)
  #wave_range=[2190,3100]
  ### With MCMC ====
  q.Fit(nsmooth =1, and_or_mask = False, deredden = deredden, reject_badpix = False, wave_range=wave_range,\
      wave_mask = None, decomposition_host = decomposition_host, BC03= False, Mi = None, npca_gal = 10, npca_qso = 50,\
      Fe_uv_op = Fe_uv_op, poly = poly, BC = BC, rej_abs = True, initial_guess = None, MC = True, \
      n_trails = 50, linefit = True, tie_lambda = True, tie_width = True, tie_flux_1 = True, tie_flux_2 = True,\
      save_result = True, plot_fig = True, save_fig = True, plot_line_name = True, plot_legend = True, \
      dustmap_path = path4, save_fig_path = path3, save_fits_path = path2, save_fits_name = None, save_model_path=path5)


  end = timeit.default_timer()
  print ('Fitting finished in : '+str(np.round(end-start))+'s')
  # grey shade on the top is the continuum windiows used to fit.


############## Read all the spectra available in the data folder (path0) and fit them.

spec_name = glob.glob(path0 + '*.fits')
for j in range(len(glob.glob(path0 + '*.fits'))):
	run_fit(spec_name[j])
	print ("succesful")

