#export PATH
#%matplotlib inline
import glob, os,sys,timeit
import matplotlib
matplotlib.use("Tkagg")
import numpy as np
from numpy import savetxt
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit
import scipy as sp
import csv
warnings.filterwarnings("ignore")

path0='./result/r_772/' 
fields = ["ra","dec","plateid","MJD","fiberid","sdss_id","redshift","SN_ratio_conti","min_wave","max_wave","Fe_uv_scale","Fe_uv_FWHM","Fe_uv_shift","Fe_op_scale","Fe_op_FWHM","Fe_op_shift","PL_norm","PL_slope","NORM_BC","Te_BC","TAU_BC","POLY_a","POLY_b","POLY_c","conti_rchi2","L1350","L3000","L4400","L5100","frac_host_4200","frac_host_5100","ssp_rchi2","fbc_3000","area_fe_uv","area_fe_op","ew_fe_uv","ew_fe_op","LINE_NPIX_HA","LINE_MED_SN_HA","NOISE_HA","LINE_NPIX_HB","LINE_MED_SN_HB","NOISE_HB","LINE_NPIX_HG","LINE_MED_SN_HG","NOISE_HG","LINE_NPIX_MGII","LINE_MED_SN_MGII",'NOISE_MGII','LINE_NPIX_CIII','LINE_MED_SN_CIII','NOISE_CIII','LINE_NPIX_CIV','LINE_MED_SN_CIV','NOISE_CIV','LINE_NPIX_LYA','LINE_MED_SN_LYA','NOISE_LYA','1_complex_name','1_line_status','1_line_min_chi2','1_line_red_chi2','1_niter','1_ndof','2_complex_name','2_line_status','2_line_min_chi2','2_line_red_chi2','2_niter','2_ndof','3_complex_name','3_line_status','3_line_min_chi2','3_line_red_chi2','3_niter','3_ndof','MgII_br_1_scale','MgII_br_1_centerwave','MgII_br_1_sigma','MgII_br_2_scale','MgII_br_2_centerwave','MgII_br_2_sigma','MgII_na_1_scale','MgII_na_1_centerwave','MgII_na_1_sigma','Hg_br_1_scale','Hg_br_1_centerwave','Hg_br_1_sigma','Hg_na_1_centerwave','Hg_na_1_sigma','OIII4364_1_scale','OIII4364_1_centerwave','OIII4364_1_sigma','Hb_br_1_scale','Hb_br_1_centerwave','Hb_br_1_sigma','Hb_br_2_scale','Hb_br_2_centerwave','Hb_br_2_sigma','Hb_na_1_scale','Hb_na_1_centerwave','Hb_na_1_sigma','OIII4959c_1_scale','OIII4959c_1_centerwave','OIII4959c_1_sigma','OIII5007c_1_scale','OIII5007c_1_centerwave','OIII5007c_1_sigma','OIII4959w_1_scale','OIII4959w_1_centerwave','OIII4959w_1_sigma','OIII5007w_1_scale','OIII5007w_1_centerwave','OIII5007w_1_sigma','HeII4687_br_1_scale','HeII4687_br_1_centerwave','HeII4687_br_1_sigma','HeII4687_na_1_scale','HeII4687_na_1_centerwave','HeII4687_na_1_sigma','MgII_whole_br_fwhm','MgII_whole_br_sigma_mad','MgII_whole_br_sigma','MgII_whole_br_ew','MgII_whole_br_peak','MgII_whole_br_area','MgII_whole_br_peak_flux','MgII_whole_br_MAD','MgII_whole_br_AI','Hg_whole_br_fwhm','Hg_whole_br_sigma_mad','Hg_whole_br_sigma','Hg_whole_br_ew','Hg_whole_br_peak','Hg_whole_br_area','Hg_whole_br_peak_flux','Hg_whole_br_MAD','Hg_whole_br_AI','Hb_whole_br_fwhm','Hb_whole_br_sigma_mad','Hb_whole_br_sigma','Hb_whole_br_ew','Hb_whole_br_peak','Hb_whole_br_area','Hb_whole_br_peak_flux','Hb_whole_br_MAD','Hb_whole_br_AI','MgII_br_1_fwhm','MgII_br_1_ew','MgII_br_1_peak','MgII_br_1_area','MgII_br_2_fwhm','MgII_br_2_ew','MgII_br_2_peak','MgII_br_2_area','MgII_na_1_fwhm','MgII_na_1_ew','MgII_na_1_peak','MgII_na_1_area','Hg_br_1_fwhm','Hg_br_1_ew','Hg_br_1_peak','Hg_br_1_area','Hg_na_1_fwhm','Hg_na_1_ew','Hg_na_1_peak','Hg_na_1_area','OIII4364_1_fwhm','OIII4364_1_ew','OIII4364_1_peak','OIII4364_1_area','Hb_br_1_fwhm','Hb_br_1_ew','Hb_br_1_peak','Hb_br_1_area','Hb_br_2_fwhm','Hb_br_2_ew','Hb_br_2_peak','Hb_br_2_area','Hb_na_1_fwhm','Hb_na_1_ew','Hb_na_1_peak','Hb_na_1_area','OIII4959c_1_fwhm','OIII4959c_1_ew','OIII4959c_1_peak','OIII4959c_1_area','OIII5007c_1_fwhm','OIII5007c_1_ew','OIII5007c_1_peak','OIII5007c_1_area','OIII4959w_1_fwhm','OIII4959w_1_ew','OIII4959w_1_peak','OIII4959w_1_area','OIII5007w_1_fwhm','OIII5007w_1_ew','OIII5007w_1_peak','OIII5007w_1_area','HeII4687_br_1_fwhm','HeII4687_br_1_ew','HeII4687_br_1_peak','HeII4687_br_1_area','HeII4687_na_1_fwhm','HeII4687_na_1_ew','HeII4687_na_1_peak','HeII4687_na_1_area']
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
#print("L:",L)	
o1_final = [] 
o2_final = []
o1_err_final = []
o2_err_final = []
mjd_final = []
Hb_err_final = []
L_err_final = []
o1_final_prime = []
o2_final_prime = []
o1_err_final_prime = []
o2_err_final_prime = []
L_final_prime = []
mjd_final_prime = []
Hb_err_final_prime = []
L_err_final_prime = []
Hb_whole_final_prime = []
L_prime = np.asarray(L, dtype=float)
Hb_whole_prime = np.asarray(Hb_whole, dtype=float)
o1_prime = np.asarray(o1,dtype=float)
o2_prime = np.asarray(o2,dtype=float)
mjd_prime = np.asarray(mjd, dtype=float)
Hb_err_prime = np.asarray(Hb_err,dtype=float)
L_err_prime = np.asarray(L_err_1,dtype = float)
o1_err_prime = np.asarray(o1_err,dtype = float)
o2_err_prime = np.asarray(o2_err,dtype = float)

for k in range(len(mjd_prime)):
	if mjd_prime[k]>56000 : #all time except the first point
		mjd_final.append(mjd_prime[k])
		o1_final.append(o1_prime[k])
		o2_final.append(o2_prime[k])
		Hb_whole_final.append(Hb_whole_prime[k])
		L_final.append(10**L_prime[k])
		Hb_err_final.append(Hb_err_prime[k])
		L_err_final.append(L_err_prime[k])
		o1_err_final.append(o1_err_prime[k])
		o2_err_final.append(o2_err_prime[k])
	if mjd_prime[k]>56500 and  mjd_prime[k]<57000: #around 56750
		mjd_final_prime.append(mjd_prime[k])
		o1_final_prime.append(o1_prime[k])
		o2_final_prime.append(o2_prime[k])
		Hb_whole_final_prime.append(Hb_whole_prime[k])
		L_final_prime.append(10**L_prime[k])
		Hb_err_final_prime.append(Hb_err_prime[k])
		L_err_final_prime.append(L_err_prime[k])
		o1_err_final_prime.append(o1_err_prime[k])
		o2_err_final_prime.append(o2_err_prime[k])

print("size of Hb_whole_prime", np.size(Hb_whole_final_prime))
print("size of Hb whole",np.size(Hb_whole_final))
o_err_prime = []

for j in range(len(o1_final)):
	o_prime.append(o1_final[j] + o2_final[j])
	o_err_prime.append(o1_err_final[j] + o2_err_final[j])

o_prime_1=[]
o_err_prime_1 = []
for j in range(len(o1_final_prime)):
	o_prime_1.append(o1_final_prime[j] + o2_final_prime[j])
	o_err_prime_1.append(o1_err_final_prime[j] + o2_err_final_prime[j])
		
o_mean = np.mean(o_prime)
o_mean_1 = np.mean(o_prime_1)
#print("mean of OIII emission flux is",o_mean)
	
for j in range(len(o_prime)):
	Hb_whole_rescaled.append((Hb_whole_final[j]*o_mean)/o_prime[j])	

Hb_whole_rescaled_1 = []
for j in range(len(o_prime_1)):
	Hb_whole_rescaled_1.append((Hb_whole_final_prime[j]*o_mean_1)/o_prime_1[j])	
	
#print("mean of h beta rescaled : ", np.mean(Hb_whole_rescaled))
for j in range(len(o_prime)):
	L_rescaled.append((L_final[j]*o_mean)/o_prime[j])	

L_rescaled_1= []
O_rescaled_1 = []
Hb_err_final_rescaled = []
L_err_final_rescaled = []
Hb_err_final_prime_rescaled = []
L_err_final_prime_rescaled = []


for j in range(len(o_prime_1)):
	L_rescaled_1.append((L_final_prime[j]*o_mean_1)/o_prime_1[j])

for j in range(len(o_prime)):
	O_rescaled.append((o_prime[j]*o_mean)/o_prime[j])

for j in range(len(o_prime)):
	Hb_err_final_rescaled.append((Hb_err_final[j]*o_mean)/o_prime[j])

for j in range(len(o_prime)):
	L_err_final_rescaled.append((L_err_final[j]*o_mean)/o_prime[j])


for j in range(len(o_prime_1)):
	O_rescaled_1.append((o_prime_1[j]*o_mean_1)/o_prime_1[j])


for j in range(len(o_prime_1)):
	Hb_err_final_prime_rescaled.append((Hb_err_final_prime[j]*o_mean_1)/o_prime_1[j])

for j in range(len(o_prime_1)):
	L_err_final_prime_rescaled.append((L_err_final_prime[j]*o_mean_1)/o_prime_1[j])



#reading L5100 data of the paper :
df = pd.read_csv('101_l5100_test.csv')

l = np.mean(L_rescaled_1)
m = df['mjd'].tolist()

#print("mean of the L5100 timeline of paper : ", mean)
F = df['flux'].tolist()
F_new = []
for k in range(len(F)):
        F_new.append((10**41)*F[k])
#print("mean of F before :",np.mean(F))
#print(F)
mean =  np.mean(F_new)
time = []
#print("size of m:", np.size(m))

for i in range(len(F_new)):
	F_new[i] = F_new[i] - mean + l
print("mean of F after:",np.mean(F))

F_final = []
#print("f before:",F)
for i in range(len(m)):
        if m[i]>56500 and m[i]<57000:
                time.append(m[i])
                F_final.append(F_new[i])

#print("size of time:",np.size(time))
#print("size of F:", np.size(F))
print("size of finl F:", np.size(F_final))
#print("f now:",F)
plt.scatter(time,F_final)
plt.show()


#Reading Hbeta data of the paper:

df = pd.read_csv('test.csv')

p = np.mean(Hb_whole_rescaled_1)
mjd_test = df['MJD'].tolist()
for j in range(len(mjd_test)) :
    mjd_test[j] = mjd_test[j] + 50000


Fl = df['Flux'].tolist()
#print(Fl)
t = []
Fl_final = []
mean_hb =  np.mean(Fl)
for i in range(len(Fl)):
	Fl[i] = Fl[i] - mean_hb + p


for i in range(len(mjd_test)):
        if mjd_test[i]>56500 and  mjd_test[i]<58000:
                t.append(mjd_test[i])
                Fl_final.append(Fl[i])
print("size of mjd:",np.size(mjd_test))
print("size of t:",np.size(t))
plt.scatter(t,Fl_final)
plt.show()
#figure.tight_layout()

#Reading sorted Hbeta light curve for our data:

df = pd.read_csv('101_Hb_AT_Re.csv')

time_hb = df['mjd'].tolist()

Flux_hb = df['flux'].tolist()

#plt.scatter(timw_hb,Flux_hb)
#plt.show()
#figure.tight_layout()

#Reading sorted continuum light curve for our data:

df = pd.read_csv('101_l_AT_Re.csv')

time_l = df['mjd'].tolist()

Flux_l = df['fkux'].tolist()

#plt.scatter(timw_hb,Flux_hb)
#plt.show()
#figure.tight_layout()

#Reading sorted continuum light curve of paper:

df = pd.read_csv('101_l5100_paper_final.csv')

time_l_paper = df['mjd'].tolist()

Flux_l_paper = df['flux'].tolist()

#reading sorted hbeta of paper:

df = pd.read_csv('101_hb_paper_sorted.csv')

time_hb_paper = df['mjd'].tolist()

Flux_hb_paper = df['flux'].tolist()

#reading sorted hbeta of our data around  56750:

df = pd.read_csv('101_hb_Re_sorted.csv')

time_hb_Re = df['mjd'].tolist()

Flux_hb_Re = df['flux'].tolist()

#reading sorted l5100 of our data around  56750:

df = pd.read_csv('101_l_Re_sorted.csv')

time_l_Re = df['mjd'].tolist()

Flux_l_Re = df['flux'].tolist()


#print(mjd_final)
#print("Hb:",Hb_whole_final)
b = np.ravel(Hb_whole_final) #without rescailing HB
b_prime = np.ravel(Hb_whole_final_prime) #around 56750 - without rescailing
#print("b:",b)
c = np.ravel(mjd_final) #all data time
#print("c :",c)
c_prime = np.ravel(mjd_final_prime) #around 56750 - some chunk of data
d = np.ravel(L_final) #without rescailing L5100
d_prime = np.ravel(L_final_prime) #around 56750 - without rescailing
p = np.ravel(Hb_whole_rescaled) #rescaled HB
p_prime = np.ravel(Hb_whole_rescaled_1) #rescaled Hb data around 56750
q = np.ravel(L_rescaled) #rescaled L5100
q_prime = np.ravel(L_rescaled_1) #around 56750
e_1 = np.ravel(Hb_err_final) #error in Hb
e_2 = np.ravel(L_err_final)  #error in L5100
e_1_rescaled = np.ravel(Hb_err_final_rescaled) #error in Hb - rescaled
e_2_rescaled = np.ravel(L_err_final_rescaled)  #error in L5100 - rescaled
e_1_prime = np.ravel(Hb_err_final_prime) #error in Hb around 56750
e_2_prime = np.ravel(L_err_final_prime)  #error in L around 56750
e_1_prime_rescaled = np.ravel(Hb_err_final_prime_rescaled) #error in Hb around 56750 - rescaled
e_2_prime_rescaled = np.ravel(L_err_final_prime_rescaled)  #error in L around 56750 - rescaled



#print("c:",c)
#print(L_prime)
#print(o_prime)
#print("L5100_err:",L_err_final)
#print("Hb_err:",Hb_err_final)
#print("size of mjd:",len(mjd_final))
#print("size of Hb whole emission array:",len(Hb_whole_prime))
#print("size of Luminosity at 5100:",len(L_prime))
#print("size of OIII Emission:",len(o_prime))

#np.savetxt("101_hb_AT_WR.txt", np.c_[c,b,e_1])
#np.savetxt("101_l_AT_WR.txt", np.c_[c,d,e_2])
#np.savetxt("101_hb_WR.txt", np.c_[c_prime,b_prime,e_1_prime])
#np.savetxt("101_l_WR.txt", np.c_[c_prime,d_prime,e_2_prime])
#np.savetxt("694_hb_AT_Re.txt",np.c_[c,p,e_1_rescaled])
#np.savetxt("694_l_AT_Re.txt",np.c_[c,q,e_2_rescaled])
#np.savetxt("694_hb_Re.txt",np.c_[c_prime,p_prime,e_1_prime_rescaled])
#np.savetxt("694_l_Re",np.c_[c_prime,q_prime,e_2_prime_rescaled])

#Hbeta Line flux :

plt.figure()
plt.scatter(mjd_final,Hb_whole_final,c = 'g')
plt.errorbar(mjd_final,Hb_whole_final,yerr = Hb_err_final,fmt="o",c = 'g')
plt.xlabel('MJD')
plt.ylabel('Hb_whole_br_area')
plt.title('Hb emission flux versus time')
plt.savefig('Hb broad emission flux versus time')
plt.show()

#L5100 continuum flux:

plt.figure()
plt.scatter(mjd_final,L_final, c='r')
plt.errorbar(mjd_final,L_final,yerr = L_err_final, fmt = "o",c='r')
plt.xlabel('MJD')
plt.ylabel('L_5100')
plt.title('Luminosity at 5100 $A_{0}$ versus time')
plt.savefig('Luminosity at 5100 $A_{0}$ versus time')
plt.show()

#OIII emission flux:

plt.figure()
plt.scatter(mjd_final,o_prime, c='b')
plt.errorbar(mjd_final,o_prime,yerr = o_err_prime,fmt = "o",c='b')
plt.xlabel('MJD')
plt.ylabel('OIII5007_area')
plt.title('OIII emission flux versus time')
plt.savefig('OIII emission flux versus time')
plt.show()

#L5100 Rescaled emission flux:

plt.figure()
plt.scatter(mjd_final,L_rescaled, c='black')
plt.errorbar(mjd_final,L_rescaled,yerr = L_err_final_rescaled, fmt = "o",c='black')
plt.xlabel('MJD')
plt.ylabel('Luminosity at 5100 $A_{0}$ rescaled by OIII')
plt.title('Luminosity at 5100 $A_{0}$ rescaled by OIII versus time')
plt.savefig('Luminosity at 5100 $A_{0}$ rescaled by OIII versus time')
plt.show()

#L5100 Rescaled around first chunk of data:

plt.figure()
plt.scatter(mjd_final_prime,L_rescaled_1, c='black')
plt.errorbar(mjd_final_prime,L_rescaled_1,yerr = L_err_final_prime_rescaled, fmt = "o",c='black')
plt.xlabel('MJD')
plt.ylabel('Luminosity at 5100 $A_{0}$ rescaled by OIII')
plt.title('Luminosity at 5100 $A_{0}$ rescaled by OIII versus time')
plt.savefig('Luminosity at 5100 $A_{0}$ rescaled by OIII versus time for first chunk of data')
plt.show()

#plt.figure()
#plt.scatter(mjd_final,O_rescaled, c='cyan')
#plt.xlabel('MJD')
#plt.ylabel('OIII5007 rescaled')
#plt.title('OIII emission flux rescaled by OIII versus time')
#plt.savefig('OIII emission flux rescaled by OIII versus time(expected to be constant)')
#plt.show()

#Hbeta Rescaled Emission Flux:

plt.figure()
plt.scatter(mjd_final,Hb_whole_rescaled, c='purple')
plt.errorbar(mjd_final,Hb_whole_rescaled,yerr = Hb_err_final_rescaled,fmt="o", c='purple')
#plt.scatter(m,F,c = 'green') 
plt.xlabel('MJD')
plt.ylabel('Hb_whole_br_area rescaled by OIII')
plt.title('Hb_whole_br_area rescaled by OIII versus time')
plt.savefig('Hb_whole_br_area rescaled by OIII versus time')
plt.show()

#Hbeta Rescaled around first chunk of data flux:

plt.figure()
plt.scatter(mjd_final_prime,Hb_whole_rescaled_1, c='purple')
plt.errorbar(mjd_final_prime,Hb_whole_rescaled_1,yerr = Hb_err_final_prime_rescaled,fmt="o", c='purple')
#plt.scatter(m,F,c = 'green') 
plt.xlabel('MJD')
plt.ylabel('Hb_whole_br_area rescaled by OIII')
plt.title('Hb_whole_br_area rescaled by OIII versus time')
plt.savefig('Hb_whole_br_area rescaled by OIII versus time for first chunk of data')
plt.show()

#Subplot of Hbeta and L5100 rescaled

plt.subplot(2,1,1)
plt.scatter(mjd_final_prime,Hb_whole_rescaled_1, c='purple')
#plt.errorbar(mjd_final,Hb_whole_rescaled,fmt="o", c='purple')
#plt.scatter(m,F,c = 'green') 
#plt.xlabel('MJD')
plt.ylabel('Hb rescaled by OIII')
#plt.title('Hb_whole_br_area rescaled by OIII versus time')


plt.subplot(2,1,2)
plt.scatter(mjd_final_prime,L_rescaled_1, c='black')
#plt.errorbar(mjd_final,L_rescaled, fmt = "o",c='black')
plt.xlabel('MJD')
plt.ylabel('L5100 rescaled by OIII')
#plt.title('Luminosity at 5100 $A_{0}$ rescaled by OIII versus time')
plt.savefig("subplot.png")

plt.show()

#overplotting paper L5100 AND our reprocessed L5100 in a lineplot 

plt.figure()
plt.plot(time_l_Re,Flux_l_Re, c='purple')
#plt.errorbar(mjd_final,Hb_whole_rescaled,fmt="o", c='purple')
#plt.scatter(m,F,c = 'green') 
#plt.xlabel('MJD')
#plt.title('Hb_whole_br_area rescaled by OIII versus time')
plt.plot(time,F_final, c='black')
#plt.errorbar(mjd_final,L_rescaled, fmt = "o",c='black')
plt.xlabel('MJD')
plt.ylabel('L5100')
#plt.title('Luminosity at 5100 $A_{0}$ rescaled by OIII versus time')
plt.legend("of our data"," of paper")
plt.savefig("L5100.png")

plt.show()

#Subplots of two L5100 datsets:

plt.subplot(2,1,1)
plt.plot(time_l_Re,Flux_l_Re, c='purple')
#plt.errorbar(mjd_final,Hb_whole_rescaled,fmt="o", c='purple')
#plt.scatter(m,F,c = 'green') 
#plt.xlabel('MJD')
plt.ylabel("L5100 - our data")
#plt.title('Hb_whole_br_area rescaled by OIII versus time')

plt.subplot(2,1,2)
plt.plot(time,F_final, c='black')
#plt.errorbar(mjd_final,L_rescaled, fmt = "o",c='black')
plt.xlabel('MJD')
plt.ylabel('L5100 - paper')
#plt.title('Luminosity at 5100 $A_{0}$ rescaled by OIII versus time')
#plt.legend("of our data"," of paper")
plt.savefig("L5100-subplot.png")

plt.show()

#Subplots of two Hbeta datsets:

plt.subplot(2,1,1)
plt.plot(time_hb_Re,Flux_hb_Re, c='purple')
#plt.errorbar(mjd_final,Hb_whole_rescaled,fmt="o", c='purple')
#plt.scatter(m,F,c = 'green') 
#plt.xlabel('MJD')
plt.ylabel("Hbeta - our data")
#plt.title('Hb_whole_br_area rescaled by OIII versus time')

plt.subplot(2,1,2)
plt.plot(t,Fl_final, c='orange')
#plt.errorbar(mjd_final,L_rescaled, fmt = "o",c='black')
plt.xlabel('MJD')
plt.ylabel('Hbeta - paper')
#plt.title('Luminosity at 5100 $A_{0}$ rescaled by OIII versus time')
#plt.legend("of our data"," of paper")
plt.savefig("Hbeta-subplot.png")

plt.show()

#overplotting paper Hbeta AND our reprocessed Hbeta in a lineplot

plt.figure()
plt.plot(time_hb_Re,Flux_hb_Re, c='purple')
#plt.errorbar(mjd_final,Hb_whole_rescaled,fmt="o", c='purple')
#plt.scatter(m,F,c = 'green') 
#plt.xlabel('MJD')
#plt.ylabel('Hbeta')
#plt.title('Hb_whole_br_area rescaled by OIII versus time')

plt.plot(t,Fl_final, c='orange')
#plt.errorbar(mjd_final,L_rescaled, fmt = "o",c='black')
plt.xlabel('MJD')
plt.ylabel('Hbeta flux')

#plt.title('Luminosity at 5100 $A_{0}$ rescaled by OIII versus time')
plt.legend(" of our data"," of paper")
plt.savefig("Hbeta.png")

plt.show()

#Plots of Hbeta and L5100 rescaled in the same plot to look for any obvious lag:

plt.subplot(2,1,1)
#plt.scatter(mjd_final_prime,Hb_whole_rescaled_1, 'g')
plt.plot(time_hb,Flux_hb, marker = '*')
#plt.errorbar(mjd_final,Hb_whole_rescaled,fmt="o", c='purple')
#plt.scatter(m,F,c = 'green') 
#plt.xlabel('MJD')
plt.ylabel('Hb ')
#plt.title('Hb_whole_br_area rescaled by OIII versus time')


plt.subplot(2,1,2)
plt.plot(time_l,Flux_l, c='black', marker = '*')
#plt.errorbar(mjd_final,L_rescaled, fmt = "o",c='black')
plt.xlabel('MJD')
plt.ylabel('L5100 ')
#plt.title('Luminosity at 5100 $A_{0}$ rescaled by OIII versus time')
plt.savefig("look.png")

plt.show()

