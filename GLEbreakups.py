############################################################################
# Project: Forest Dynamics at the 'Woods of the World'
# Title: Calculation of the fractional light recieved by each stand
# Author: James Gilmore
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 1.0.0
# Date: 22/12/15
# Status: Alpha
############################################################################

#Initialising the python script
from __future__ import absolute_import, division, print_function
from scipy.integrate import quad, dblquad
from array import array
import numpy as np
import warnings

#Define the equitorial coordinate system
zenith = lambda H, dec, lat: np.arcsin(np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(H))
azimuth = lambda lat, dec, zenith: np.arccos((np.sin(dec)-np.sin(zenith)*np.sin(lat))/(np.cos(zenith)*np.cos(lat)))
dec = lambda Jday: np.arcsin(np.sin(np.radians(-23.44))*np.cos((np.radians(360)/365.24)*(Jday+10)+(np.radians(360)/np.pi)*0.0167*np.sin((np.radians(360)/365.24)*(Jday-2))))
he = lambda Jday: 0.17*np.sin((4*np.pi*(Jday-80))/373)-0.129*np.sin((2*np.pi*(Jday-8))/355)
gam = lambda h: -15*np.floor(h)
H = lambda hl, gaml, gam, he: (np.pi/12)*(hl-(24*(gaml-gam)/360)+he-12)
td = lambda Jday: (2*np.pi*(np.floor(Jday)-1))/365

#Define the equations specific to the NBGW
SolElvNBGW = lambda h, day: zenith(H(h, np.radians(-4.15176), gam(0), he(day)), np.radians(dec(day)), np.radians(51.83756))*57.2957795

#### Be careful: Put hours in double format (i.e. 3am is 3.0 NOT 3)
SolAziNBGW = lambda h, day: np.piecewise(h, [h < 12.01854275271367, h >= 12.01854275271366], [lambda h: azimuth(np.radians(51.83756), np.radians(dec(day)), np.radians(SolElvNBGW(h,day)))*57.2957795, lambda h: 360-azimuth(np.radians(51.83756), np.radians(dec(day)), np.radians(SolElvNBGW(h,day)))*57.2957795])

#Dawn and Dusk
Dawn = lambda day: 11.9952 - 3.81972*np.arccos(1.27249*np.tan(0.0174533*np.arcsin(0.397789*np.cos(0.172029 + 0.0172029*day + 0.0334*np.sin(0.0172029*(-2. + day)))))) - 0.17*np.sin(0.03369*(-80. + day)) + 0.129*np.sin(0.0176991*(-8. + day))
Dusk = lambda day: 11.9952 + 3.81972*np.arccos(1.27249*np.tan(0.0174533*np.arcsin(0.397789*np.cos(0.172029 + 0.0172029*day + 0.0334*np.sin(0.0172029*(-2. + day)))))) - 0.17*np.sin(0.03369*(-80. + day)) + 0.129*np.sin(0.0176991*(-8. + day))

#Import the data of the tree set in the correct columnar form:
X, Y, HE, CH, CW = np.genfromtxt("GillyShadingTIFull.csv", dtype=float, delimiter=',', usecols=(0, 1, 2, 3, 4),  unpack=True, skiprows=1)

############################################################################
#The Gilly Light Equaton

#The Gille Shade Criterion Function (GSCF)

RM = lambda x: np.array([[np.cos(x), -np.sin(x)], [np.sin(x),  np.cos(x)]])

[X, Y]+np.dot(RM(np.radians(-(SolAziNBGW(12,173)-210)))/np.tan(SolElvNBGW(12,173)),[-CW/200,HE])



def KMatrixFun(N=0, O=0, h=12, day=173, iter=len(X)):
	KMatrix = KMatrixCrown = KMatrixHeight = np.zeros([iter, 2]) 
	
	for i in xrange(iter):
		KMatrixCrown[i] = ([X[i], Y[i]]+np.dot(RM(np.radians(-(SolAziNBGW(h,day)-210)))/np.tan(SolElvNBGW(h,day)),[-CW[i]/200,HE[i]])-[N, O])/([X[i], Y[i]]+np.dot(RM(np.radians(-(SolAziNBGW(h,day)-210)))/np.tan(SolElvNBGW(h,day)),[-CW[i]/200,HE[i]-CH[i]])-[N, O])
		for j in xrange(2):
			if KMatrixCrown[i, j] <= 0:
				KMatrixCrown[i, j] = 0
			else:
				KMatrixCrown[i, j] = 1
	
		KMatrixHeight[i] = ([X[i], Y[i]]+np.dot(RM(np.radians(-(SolAziNBGW(h,day)-210)))/np.tan(SolElvNBGW(h,day)),[0,HE[i]])-[N, O])/([X[i], Y[i]]+np.dot(RM(np.radians(-(SolAziNBGW(h,day)-210)))/np.tan(SolElvNBGW(h,day)),[0,HE[i]-CH[i]])-[N, O])
		for j in xrange(2):
			if KMatrixHeight[i, j] <= 0:
				KMatrixHeight[i, j] = 0
			else:
				KMatrixHeight[i, j] = 1
			
		KMatrix[i] = KMatrixHeight[i]+KMatrixCrown[i]
	
	return KMatrix
	
def ZetaUnitGrid(h=12, day=173, nmin=0, nmax=60, omin=0, omax=60, spacings=1):
	KMatrixTot = KMatrixTotTot = np.zeros([(nmax-nmin)/spacings,(omax-omin)/spacings])
	
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		fxn()
		for i in xrange(nmin,nmax,spacings):
			for j in xrange(omin,omax,spacings):
		
				KMatrixTot[i,j] = KMatrixFun(i, j, h, day)[i,0]+KMatrixFun(i, j, h, day)[i,1]
		
				for i in xrange(int((nmax-nmin)/spacings)):
					if KMatrixTot[i,j] <= 0:
						KMatrixTot[i,j] = 0
					else:
						KMatrixTot[i,j] = 1
	
				if ((nmax-nmin)/spacings - KMatrixTot.sum()) <= 0:
					KMatrixTotTot[i,j] = 1
				else:
					KMatrixTotTot[i,j] = 0
				
	return KMatrixTotTot


def GillyLightEquation():
	GillyTop = 0.0
	for day in xrange(0,365):
		hmax = (Dusk(day)-Dawn(day))/10
		for h in xrange(int(hmax)):
			GillyTop += ZetaUnitGrid(Dusk(day)+h*hmax,day,0,1,0,1,1)
	return GillyTop
	
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

#KMatrixTotTot = ZetaUnitGrid(12,173,0,2,0,2,1)
#area = quad(KMatrixTotTot[x,y,h,day,473], 0, 1, args=(x,y,h))


GillyTop = GillyLightEquation()

print(GillyTop)
#KMatrix = lambda X, Y, HE, CH, CW, Zenith, Azimuth: np.ceiling()

