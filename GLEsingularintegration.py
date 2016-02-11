############################################################################
# Project: Forest Dynamics at the 'Woods of the World'
# Title: Calculation of the fractional light recieved by each stand
# Author: James Gilmore
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 1.0.0
# Date: 22/12/15
# Status: Beta
############################################################################

#Initialising the python script
from __future__ import absolute_import, division, print_function
from scipy.integrate import quad, dblquad, simps
from array import array
import numpy as np
import time

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

#Clear Sky and Beam Fraction Coefficients

acoff = lambda w, mp: np.exp(-(0.465+0.134*w)*(0.179+0.421*np.exp(-0.721*mp))*mp)
mp = lambda A, z: ((288-0.0065*z)/288)**5.256/(np.sin(A)+0.15*(A+3.885)**-1.253)
w = lambda Tdpt: np.exp(-0.0592+0.06912*Tdpt)
kb = lambda kt: -10.627*kt**5 + 15.307*kt**4 - 5.205*kt**3 + 0.994*kt**2 - 0.059*kt + 0.002
BeamNBGW = lambda h, day: kb(acoff(w(21), mp(np.radians(SolElvNBGW(h, day)),0)))

print(BeamNBGW(12,173))

#Import the data of the tree set in the correct columnar form:
X, Y, HE, CH, CW = np.genfromtxt("GillyShadingTIFull.csv", dtype=float, delimiter=',', usecols=(0, 1, 2, 3, 4),  unpack=True, skiprows=1)

############################################################################
#The Gilly Light Equaton

#The Gilly Shade Criterion Function (GSCF)

RM = lambda x: np.array([[np.cos(x), -np.sin(x)], [np.sin(x),  np.cos(x)]])

def GLE(iter=len(X), h=12, day=173, nmin=0, nmax=60, omin=0, omax=60, spacings=1):
	ts = np.zeros(omax)
	KMatrix = KMatrixCrown = KMatrixHeight = np.zeros([iter, 2]) 
	KMatrixTot = np.zeros(iter)
	KMatrixTotTot = GillyTop = GillyBottom = GillyLight = np.zeros([nmax/spacings,omax/spacings])
	tstart = time.time()
	#ts[h] = time.time()
	#print("Time Left:", (ts[n]-tstart)/((n+o+h+day)/(omax+nmax+int(hmax)+2)))
	#print("Percentage Done: ", (h+(10*day))/(int(hmax)+(1000*2)))
	for o in xrange(nmin,nmax,spacings):
			for n in xrange(omin,omax,spacings):
				for i in xrange(iter):
					KMatrixCrown[i] = ([X[i], Y[i]]+np.dot(RM(np.radians(-(SolAziNBGW(h,day)-210)))/np.tan(SolElvNBGW(h,day)),[-CW[i]/200,HE[i]])-[n, o])/([X[i], Y[i]]+np.dot(RM(np.radians(-(SolAziNBGW(h,day)-210)))/np.tan(SolElvNBGW(h,day)),[-CW[i]/200,HE[i]-CH[i]])-[n, o])
					for j in xrange(2):
						if KMatrixCrown[i, j] <= 0:
							KMatrixCrown[i, j] = 0
						else:
							KMatrixCrown[i, j] = 1
				
					KMatrixHeight[i] = ([X[i], Y[i]]+np.dot(RM(np.radians(-(SolAziNBGW(h,day)-210)))/np.tan(SolElvNBGW(h,day)),[0,HE[i]])-[n, o])/([X[i], Y[i]]+np.dot(RM(np.radians(-(SolAziNBGW(h,day)-210)))/np.tan(SolElvNBGW(h,day)),[0,HE[i]-CH[i]])-[n, o])
					for j in xrange(2):
						if KMatrixHeight[i, j] <= 0:
							KMatrixHeight[i, j] = 0
						else:
							KMatrixHeight[i, j] = 1
						
					KMatrix[i] = KMatrixHeight[i]+KMatrixCrown[i]
					KMatrixTot[i] = KMatrix[i,0]+KMatrix[i,1]
				for i in xrange(iter):
					if KMatrixTot[i] <= 0:
						KMatrixTot[i] = 0
					else:
						KMatrixTot[i] = 1
				if (iter - KMatrixTot.sum()) <= 0:
					KMatrixTotTot[o,n] = 1
				else:
					KMatrixTotTot[o,n] = 0
				
				GillyTop[o,n] += KMatrixTotTot[o,n]
				GillyBottom[o,n] += 1
	
	for o in xrange(nmin,nmax,spacings):
		for n in xrange(omin,omax,spacings):
			GillyLight[o,n] = GillyTop[o,n]/GillyBottom[o,n]
	
	return GillyLight
np.set_printoptions(threshold='nan')	

GillyLight = GLE(473,12,163,0,5,0,5,1)
print(GillyLight)
#print(GLE(473, h, day, 0, 1, 0, 1, 1))
#GillyIntTop = quad(lambda h: GLE(473, h, 173, 0, 1, 0, 1, 1), Dawn(105), Dawn(105)+1)
GillyIntTop = simps(GLE(473, h, 173, 0, 1, 0, 1, 1))

print(GillyIntTop)