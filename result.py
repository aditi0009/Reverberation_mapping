
#export PATH
#%matplotlib inline
import glob, os,sys,timeit
import matplotlib
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit
import scipy as sp
import csv
warnings.filterwarnings("ignore")

path0='./r_191/'
fields = ["ra","dec","plateid","MJD","fiberid","sdss_id","redshift","SN_ratio_conti","min_wave","max_wave","Fe_uv_scale","Fe_uv_FWHM","Fe_uv_shift","Fe_op_scale","Fe_op_FWHM","Fe_op_shift","PL_norm","PL_slope","NORM_BC","Te_BC","TAU_BC","POLY_a","POLY_b","POLY_c","conti_rchi2","L1350","L3000","L4400","L5100","frac_host_4200","frac_host_5100","ssp_rchi2","fbc_3000","area_fe_uv","area_fe_op","ew_fe_uv","ew_fe_op","LINE_NPIX_HA","LINE_MED_SN_HA","NOISE_HA","LINE_NPIX_HB","LINE_MED_SN_HB","NOISE_HB","LINE_NPIX_HG","LINE_MED_SN_HG","NOISE_HG","LINE_NPIX_MGII","LINE_MED_SN_MGII",'NOISE_MGII','LINE_NPIX_CIII','LINE_MED_SN_CIII','NOISE_CIII','LINE_NPIX_CIV','LINE_MED_SN_CIV','NOISE_CIV','LINE_NPIX_LYA','LINE_MED_SN_LYA','NOISE_LYA','1_complex_name','1_line_status','1_line_min_chi2','1_line_red_chi2','1_niter','1_ndof','2_complex_name','2_line_status','2_line_min_chi2','2_line_red_chi2','2_niter','2_ndof','3_complex_name','3_line_status','3_line_min_chi2','3_line_red_chi2','3_niter','3_ndof','MgII_br_1_scale','MgII_br_1_centerwave','MgII_br_1_sigma','MgII_br_2_scale','MgII_br_2_centerwave','MgII_br_2_sigma','MgII_na_1_scale','MgII_na_1_centerwave','MgII_na_1_sigma','Hg_br_1_scale','Hg_br_1_centerwave','Hg_br_1_sigma','Hg_na_1_centerwave','Hg_na_1_sigma','OIII4364_1_scale','OIII4364_1_centerwave','OIII4364_1_sigma','Hb_br_1_scale','Hb_br_1_centerwave','Hb_br_1_sigma','Hb_br_2_scale','Hb_br_2_centerwave','Hb_br_2_sigma','Hb_na_1_scale','Hb_na_1_centerwave','Hb_na_1_sigma','OIII4959c_1_scale','OIII4959c_1_centerwave','OIII4959c_1_sigma','OIII5007c_1_scale','OIII5007c_1_centerwave','OIII5007c_1_sigma','OIII4959w_1_scale','OIII4959w_1_centerwave','OIII4959w_1_sigma','OIII5007w_1_scale','OIII5007w_1_centerwave','OIII5007w_1_sigma','HeII4687_br_1_scale','HeII4687_br_1_centerwave','HeII4687_br_1_sigma','HeII4687_na_1_scale','HeII4687_na_1_centerwave','HeII4687_na_1_sigma','MgII_whole_br_fwhm','MgII_whole_br_sigma_mad','MgII_whole_br_sigma','MgII_whole_br_ew','MgII_whole_br_peak','MgII_whole_br_area','MgII_whole_br_peak_flux','MgII_whole_br_MAD','MgII_whole_br_AI','Hg_whole_br_fwhm','Hg_whole_br_sigma_mad','Hg_whole_br_sigma','Hg_whole_br_ew','Hg_whole_br_peak','Hg_whole_br_area','Hg_whole_br_peak_flux','Hg_whole_br_MAD','Hg_whole_br_AI','Hb_whole_br_fwhm','Hb_whole_br_sigma_mad','Hb_whole_br_sigma','Hb_whole_br_ew','Hb_whole_br_peak','Hb_whole_br_area','Hb_whole_br_peak_flux','Hb_whole_br_MAD','Hb_whole_br_AI','MgII_br_1_fwhm','MgII_br_1_ew','MgII_br_1_peak','MgII_br_1_area','MgII_br_2_fwhm','MgII_br_2_ew','MgII_br_2_peak','MgII_br_2_area','MgII_na_1_fwhm','MgII_na_1_ew','MgII_na_1_peak','MgII_na_1_area','Hg_br_1_fwhm','Hg_br_1_ew','Hg_br_1_peak','Hg_br_1_area','Hg_na_1_fwhm','Hg_na_1_ew','Hg_na_1_peak','Hg_na_1_area','OIII4364_1_fwhm','OIII4364_1_ew','OIII4364_1_peak','OIII4364_1_area','Hb_br_1_fwhm','Hb_br_1_ew','Hb_br_1_peak','Hb_br_1_area','Hb_br_2_fwhm','Hb_br_2_ew','Hb_br_2_peak','Hb_br_2_area','Hb_na_1_fwhm','Hb_na_1_ew','Hb_na_1_peak','Hb_na_1_area','OIII4959c_1_fwhm','OIII4959c_1_ew','OIII4959c_1_peak','OIII4959c_1_area','OIII5007c_1_fwhm','OIII5007c_1_ew','OIII5007c_1_peak','OIII5007c_1_area','OIII4959w_1_fwhm','OIII4959w_1_ew','OIII4959w_1_peak','OIII4959w_1_area','OIII5007w_1_fwhm','OIII5007w_1_ew','OIII5007w_1_peak','OIII5007w_1_area','HeII4687_br_1_fwhm','HeII4687_br_1_ew','HeII4687_br_1_peak','HeII4687_br_1_area','HeII4687_na_1_fwhm','HeII4687_na_1_ew','HeII4687_na_1_peak','HeII4687_na_1_area']

#with open('r.csv','w') as f:
#	writer =  csv.writer(f)
#	writer.writerows(fields)
#array  = dict()
Hb_whole = []
mjd = []
L = []
Hb_err = []
mjd_prime=[]
mjd_final=[]
Hb_whole_final = []
L_final = []
L_err_1 = []
o1 = []
o2 = []
o_prime = []
L_rescaled = []
Hb_whole_rescaled = []
O_rescaled = []
def hb_whole(result_spec): 
	hdu = fits.open(result_spec)
	data = hdu[1].data
	return (data['Hb_whole_br_area'])
	#for i in range(len(glob.glob(path0 + '*.fits'))):
	#	array[i] = np.append(array,np.array(data))
		
def date(result_spec): 
	hdu = fits.open(result_spec)
	data = hdu[1].data
	
	return (data['MJD'])
	#for i in range(len(glob.glob(path0 + '*.fits'))):
	#	array[i] = np.append(array,np.array(data))
def Hb_whole_err(result_spec):
	hdu = fits.open(result_spec)
	data = hdu[1].data
	return (data['Hb_whole_br_area_err'])
	
def L_err(result_spec):
	hdu = fits.open(result_spec)
	data = hdu[1].data
	return (data['L5100_err'])
	
def lum(result_spec): 
	hdu = fits.open(result_spec)
	data = hdu[1].data
	return (data['L5100'])
	#for i in range(len(glob.glob(path0 + '*.fits'))):
	#	array[i] = np.append(array,np.array(data))


def Ox_c(result_spec): 
	hdu = fits.open(result_spec)
	data = hdu[1].data
	return (data['OIII5007c_1_area'])	
	
def Ox_w(result_spec): 
	hdu = fits.open(result_spec)
	data = hdu[1].data
	return (data['OIII5007w_1_area'])		
#with open('r_101.csv','w') as f:
#		writer =  csv.writer(f)
#		writer.writerow(fields)

def Ox_c_err(result_spec):
	hdu = fits.open(result_spec)
	data = hdu[1].data
	return (data['OIII5007c_1_area_err'])
	
def Ox_w_err(result_spec):
	hdu = fits.open(result_spec)
	data = hdu[1].data
	return (data['OIII5007w_1_area_err'])
o1_err = []
o2_err = []	
result_spec = glob.glob(path0 + '*.fits')
for i in range(len(glob.glob(path0 + '*.fits'))):
	Hb_whole.append(hb_whole(result_spec[i]))
	Hb_err.append(Hb_whole_err(result_spec[i]))
	mjd.append(date(result_spec[i]))
	L.append(lum(result_spec[i]))
	o1.append(Ox_c(result_spec[i]))
	o2.append(Ox_w(result_spec[i]))
	o1_err.append(Ox_c_err(result_spec[i]))
	o2_err.append(Ox_w_err(result_spec[i]))
	L_err_1.append(L_err(result_spec[i]))

#print(len(o1))
#print(len(o2))
#for j in range(len(o1) - 1):
#	o[j] = o1[j] + o2[j]
	
o1_final = []
o2_final = []
o1_err_final = []
o2_err_final = []
mjd_final = []
Hb_err_final = []
L_err_final = []
L_prime = np.asarray(Hb_whole, dtype=float)
Hb_whole_prime = np.asarray(Hb_whole, dtype=float)
o1_prime = np.asarray(o1,dtype=float)
o2_prime = np.asarray(o2,dtype=float)
mjd_prime = np.asarray(mjd, dtype=float)
Hb_err_prime = np.asarray(Hb_err,dtype=float)
L_err_prime = np.asarray(L_err_1,dtype = float)
o1_err_prime = np.asarray(o1_err,dtype = float)
o2_err_prime = np.asarray(o2_err,dtype = float)
for k in range(len(mjd_prime)):
	if mjd_prime[k]>56000 :
		mjd_final.append(mjd_prime[k])
		o1_final.append(o1_prime[k])
		o2_final.append(o2_prime[k])
		Hb_whole_final.append(Hb_whole_prime[k])
		L_final.append(L_prime[k])
		Hb_err_final.append(Hb_err_prime[k])
		L_err_final.append(L_err_prime[k])
		o1_err_final.append(o1_err_prime[k])
		o2_err_final.append(o2_err_prime[k])

o_err_prime = []

for j in range(len(o1_final)):
	o_prime.append(o1_final[j] + o2_final[j])
	o_err_prime.append(o1_err_final[j] + o2_err_final[j])
	
o_mean = np.mean(o_prime)
print("mean of OIII emission flux is",o_mean)
	
for j in range(len(o_prime)):
	Hb_whole_rescaled.append((Hb_whole_final[j]*o_mean)/o_prime[j])	
	
print("mean of h beta rescaled : ", np.mean(Hb_whole_rescaled))
for j in range(len(o_prime)):
	L_rescaled.append((L_final[j]*o_mean)/o_prime[j])	

for j in range(len(o_prime)):
	O_rescaled.append((o_prime[j]*o_mean)/o_prime[j])

df = pd.read_csv('test.csv')
mean = 360.155
l = 564.825801894676
m = df['MJD'].tolist()
F = df['Flux'].tolist()
print(F)

for i in range(len(m)):
	m[i] = m[i] + 50000
for i in range(len(F)):
	F[i] = F[i] - mean + l

#plt.scatter(m,F)
#plt.show()


print(mjd_final)
print(Hb_whole_prime)
print(L_prime)
print(o_prime)
print("L5100_err:",L_err_final)
print("Hb_err:",Hb_err_final)
print("size of mjd:",len(mjd_final))
print("size of Hb whole emission array:",len(Hb_whole_prime))
print("size of Luminosity at 5100:",len(L_prime))
print("size of OIII Emission:",len(o_prime))
plt.figure()
plt.scatter(mjd_final,Hb_whole_final,c = 'g')
plt.errorbar(mjd_final,Hb_whole_final,yerr = Hb_err_final,fmt="o",c = 'g')
plt.xlabel('MJD')
plt.ylabel('Hb_whole_br_area')
plt.title('Hb emission flux versus time')
plt.savefig('Hb broad emission flux versus time')
plt.show()

plt.figure()
plt.scatter(mjd_final,L_final, c='r')
plt.errorbar(mjd_final,L_final,yerr = L_err_final, fmt = "o",c='r')
plt.xlabel('MJD')
plt.ylabel('L_5100')
plt.title('Luminosity at 5100 $A_{0}$ versus time')
plt.savefig('Luminosity at 5100 $A_{0}$ versus time')
plt.show()

plt.figure()
plt.scatter(mjd_final,o_prime, c='b')
plt.errorbar(mjd_final,o_prime,yerr = o_err_prime,fmt = "o",c='b')
plt.xlabel('MJD')
plt.ylabel('OIII5007_area')
plt.title('OIII emission flux versus time')
plt.savefig('OIII emission flux versus time')
plt.show()

plt.figure()
plt.scatter(mjd_final,L_rescaled, c='black')
plt.errorbar(mjd_final,L_rescaled,yerr = L_err_final, fmt = "o",c='black')
plt.xlabel('MJD')
plt.ylabel('Luminosity at 5100 $A_{0}$ rescaled by OIII')
plt.title('Luminosity at 5100 $A_{0}$ rescaled by OIII versus time')
plt.savefig('Luminosity at 5100 $A_{0}$ rescaled by OIII versus time')
plt.show()

plt.figure()
plt.scatter(mjd_final,O_rescaled, c='cyan')
plt.xlabel('MJD')
plt.ylabel('OIII5007 rescaled')
plt.title('OIII emission flux rescaled by OIII versus time')
plt.savefig('OIII emission flux rescaled by OIII versus time(expected to be constant)')
plt.show()

plt.figure()
plt.scatter(mjd_final,Hb_whole_rescaled, c='purple')
plt.errorbar(mjd_final,Hb_whole_rescaled,yerr = Hb_err_final,fmt="o", c='purple')
#plt.scatter(m,F,c = 'green') 
plt.xlabel('MJD')
plt.ylabel('Hb_whole_br_area rescaled by OIII')
plt.title('Hb_whole_br_area rescaled by OIII versus time')
plt.savefig('Hb_whole_br_area rescaled by OIII versus time')
plt.show()

