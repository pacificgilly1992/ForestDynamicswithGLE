############################################################################
# Project: Forest Dynamics at the 'Woods of the World'
# Title: Calculation of the fractional light recieved by each stand
# Author: James Gilmore
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 1.3.1
# Date: 11/02/16
# Status: Operational
############################################################################

#Initialising the python script
from __future__ import absolute_import, division, print_function
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
from array import array
import numpy as np
import time, os, csv, sys

############################################################################
#Pre-requiste: Define all of the equations needed for the GLE equation

#Define the equitorial coordinate system ##(All Equations Correct)##
zenith = lambda H, dec, lat: np.arcsin(np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(H)) #GOOD
azimuth = lambda lat, dec, zenith: np.arccos((np.sin(dec)-np.sin(zenith)*np.sin(lat))/(np.cos(zenith)*np.cos(lat))) #GOOD
dec = lambda Jday: np.arcsin(np.sin(np.radians(-23.44))*np.cos((np.radians(360)/365.24)*(Jday+10)+(np.radians(360)/np.pi)*0.0167*np.sin((np.radians(360)/365.24)*(Jday-2))))*57.2957795 #GOOD
he = lambda Jday: 0.17*np.sin((4*np.pi*(Jday-80))/373)-0.129*np.sin((2*np.pi*(Jday-8))/355) #GOOD
gam2 = lambda h: -15*np.floor(h) #GOOD
gam = lambda h: 15.0*np.floor(h/15.0) #GOOD
H = lambda hl, gaml, gam, he: (np.pi/12)*(hl-(24*(gaml-gam)/360)+he-12) #GOOD
td = lambda Jday: (2*np.pi*(np.floor(Jday)-1))/365 #GOOD

#Define the equations specific to the NBGW ##(All Equations Correct)##
#SolAziNBGW used a definition function rather than lambda method due to issues with older versions of Numpy not recognising the 
#functions h < 12.01854275271367 etc. Prehaps its meant to be in brackets?
SolElvNBGW = lambda h, day: zenith(H(h, np.radians(-4.15176), gam(0), he(day)), np.radians(dec(day)), np.radians(51.83756))*57.2957795
def SolAziNBGW(h,day):
	return azimuth(np.radians(51.83756), np.radians(dec(day)), np.radians(SolElvNBGW(h,day)))*57.2957795 if h < 12.01854275271367 else 360-azimuth(np.radians(51.83756), np.radians(dec(day)), np.radians(SolElvNBGW(h,day)))*57.2957795

#### Be careful for SolAziNBGW: Put hours in double format (i.e. 3am is 3.0 NOT 3)

#Define the Dawn and Dusk equations (The sunrise and sunset times can be solved for the condition when the angle from horizon reach 0deg using the Zenith equation)
Dawn = lambda day: 11.9952 - 3.81972*np.arccos(1.27249*np.tan(np.arcsin(0.397789*np.cos(0.172029 + 0.0172029*day + 0.0334*np.sin(0.0172029*(-2. + day)))))) - 0.17*np.sin(0.03369*(-80. + day)) + 0.129*np.sin(0.0176991*(-8. + day))
Dusk = lambda day: 11.9952 + 3.81972*np.arccos(1.27249*np.tan(np.arcsin(0.397789*np.cos(0.172029 + 0.0172029*day + 0.0334*np.sin(0.0172029*(-2. + day)))))) - 0.17*np.sin(0.03369*(-80. + day)) + 0.129*np.sin(0.0176991*(-8. + day))

#Clear Sky and Beam Fraction Coefficients ##(All Equations Correct)##
acoff = lambda w, mp: np.exp(-(0.465+0.134*w)*(0.179+0.421*np.exp(-0.721*mp))*mp)
mp = lambda A, z: ((288-0.0065*z)/288)**5.256/(np.sin(A)+0.15*(A+3.885)**-1.253)
w = lambda Tdpt: np.exp(-0.0592+0.06912*Tdpt) #good
kb = lambda kt: -10.627*kt**5 + 15.307*kt**4 - 5.205*kt**3 + 0.994*kt**2 - 0.059*kt + 0.002
BeamNBGW = lambda h, day: kb(acoff(w(21), mp(np.radians(SolElvNBGW(h, day)),0)))
ClearNBGW = lambda h, day: 24*(2.044*SolElvNBGW(h, day)+0.12964*SolElvNBGW(h, day)**2-1.941*10**(-3)*SolElvNBGW(h, day)**3+7.591*10**(-6)*SolElvNBGW(h, day)**4)*0.1314

#Rotational Matrix required for positioning of trees and shade
RM = lambda x: np.array([[np.cos(x), -np.sin(x)], [np.sin(x),  np.cos(x)]])

#Import the data of the tree set in the correct columnar form (X, Y, Height, Crown Height, Crown Width):
X, Y, HE, CH, CW = np.genfromtxt("analysis/GillyShadingTIFullminus.csv", dtype=float, delimiter=',', usecols=(0, 1, 2, 3, 4),  unpack=True, skiprows=1)

############################################################################
#The Gilly Light Equaton

"There are 3 stages to determining the amount of sunlight present at a location"
"for a given year:"
"(1) The first requires solving the Gilly Shade Criterion Function (GSCF) which"
"determines if any entities are casting a shadow on position n,o"
"(2) Combine the GSCF with the Beam Sky Radiation factor for each position in time"
"to determine the amount of avaliable sunlight when no shade is cast."
"(3) Solve for h*day interations in time and summerise each step and compare to a"
"point with full light avaliability to normalised the equation between 0 and 1."

def GLE(iter=len(X), tree_min=0, tree_max=len(X), daymin=105, daymax=263, hspa=10):
	
	############################################################################
	#Initalise the variables for this function

	it=0 					#iterator counter
	treeskip=0 				#Number of trees skipped
	AllTrees=0 				#Number of trees processed

	KMatrixCrown = np.zeros([iter, 2]) 	#Determines the shade using the crown of each tree
	KMatrixHeight = np.zeros([iter, 2]) 	#Determines the shade using the height of each tree
	KMatrix = np.zeros([iter, 2]) 		#Hold the combintion of KMatrixCrown and KMatrixHeight
	KMatrixTot = np.zeros(iter)		#Determines if each individual tree is causing shade on specific location (n, o)
	KMatrixTotTot = np.zeros(iter)		#Combines all tree shading information to determine if any tree at also is cast shadows at (n, o)
	GillyTop = np.zeros(iter)		#Holds the numerator of the GLE equation
	GillyBottom = np.zeros(iter)		#Holds the denomiator of the GLE equation
	GillyLight = np.zeros(iter)		#This will hold the actual fractional light score
	npos = np.zeros(iter)			#Used to define the position of of each tree
	opos = np.zeros(iter)			#Used to define the position of of each tree
	tstart = time.time()			#Starts the clock on this massive processing task
	ts = np.zeros(iter)			#Current time at the current iterator of the equation
	eps = sys.float_info.epsilon		#Used to minimise the divide by zero errors
	
	for i in xrange(iter):
		npos[i] = X[i]
		opos[i] = Y[i]
		
	############################################################################
	############################################################################
	#Determine amount of sunlight for each position over the whole year.
	
	for day in xrange(daymin, daymax):
		#Calculate equally spacing times for the particular day.
		hmax = (Dusk(day)-Dawn(day))/hspa
		h_day = np.arange(Dawn(day)+0.1, Dusk(day), hmax)		
		for h in xrange(len(h_day)):
			for n in xrange(tree_min, tree_max):
				#Update the progress of the GLE equation
				it+=1
				ts[n] = time.time()
				os.system('cls' if os.name=='nt' else 'clear')
				#print("Dawn(day)", Dawn(day))
				#print("Dusk(day)", Dusk(day))
				#print("iter", iter, " KMatrixTot.sum()", KMatrixTot.sum())
				#print(GillyTop)
				print("Trees Skipped Ratio: ", "%.3f" % ((treeskip/(AllTrees+eps))*100), "%")
				print("Day: ", day, " Hour: ", "%.2f" % h_day[h], " Tree: ", n)
				print("Time Progressed (s): ", "%.0f" % (ts[n]-tstart)),
				print("Time Left (s): ", "%.0f" % ((((tree_max-tree_min)*hspa*(daymax-daymin))-it)*((ts[n]-tstart)/it))),
				print("Percentage Completed: ", "%.3f" % ((1-(((tree_max-tree_min)*hspa*(daymax-daymin))-it)/((tree_max-tree_min)*hspa*(daymax-daymin)))*100), "%")
				#Calculate the GSCF for each entity (Tree or otherwise)
				for i in xrange(iter):
					AllTrees+=1
					#This 'if' statement greatly reduces load as it quickly determines if a tree is close to the position n,o and removes from calculation if it isn't
					if (np.abs(X[i]-npos[n])-(HE[i]/np.tan(SolElvNBGW(h_day[h],day))) and np.abs(Y[i]-opos[n])-(HE[i]/np.tan(SolElvNBGW(h_day[h],day)))) < 0:
						#Determines if the width of the shade overlaps with n,o and if so normalises it in binary form
						KMatrixCrown[i] = (([X[i], Y[i]]+np.dot(RM(-np.radians(SolAziNBGW(float(h_day[h]),day)-210)),[-CW[i]/200,HE[i]])/np.tan(SolElvNBGW(h_day[h],day)))-[npos[n], opos[n]])/(([X[i], Y[i]]+np.dot(RM(-np.radians(SolAziNBGW(float(h_day[h]),day)-210))/np.tan(SolElvNBGW(h_day[h],day)),[CW[i]/200,HE[i]]))-[npos[n], opos[n]])
						#print(KMatrixCrown[i])
						for j in xrange(2):
							if KMatrixCrown[i, j] <= 0: 
								KMatrixCrown[i, j] = 0   #Shaded
							else:
								KMatrixCrown[i, j] = 1   #NOT Shaded
						#Determines if the height of the shade overlaps with n,o
						KMatrixHeight[i] = (([X[i], Y[i]]+np.dot(RM(-np.radians(SolAziNBGW(float(h_day[h]),day)-210)),[0,HE[i]])/np.tan(SolElvNBGW(h_day[h],day)))-[npos[n], opos[n]])/(([X[i], Y[i]]+np.dot(RM(-np.radians(SolAziNBGW(float(h_day[h]),day)-210))/np.tan(SolElvNBGW(h_day[h],day)),[0,HE[i]-(CH[i]/100)]))-[npos[n], opos[n]])
						
						for j in xrange(2):
							if KMatrixHeight[i, j] <= 0:
								KMatrixHeight[i, j] = 0  #Shaded
							else:
								KMatrixHeight[i, j] = 1  #NOT Shaded
						#Sum the Height and Crown criterias and them sums the x and y directions
						KMatrix[i] = KMatrixHeight[i]+KMatrixCrown[i]
						KMatrixTot[i] = KMatrix[i,0]+KMatrix[i,1]  #0: Fully Shaded, 1: 1 directional shaded, 2: NOT Shaded
					else:
						treeskip+=1
						KMatrixTot[i] = 2
				#After all trees have been tested with the GSCF at position n,o we need to combine the x and y directions togeteher
				for i in xrange(iter):
					if KMatrixTot[i] == 0:
						KMatrixTot[i] = 0  #Shaded
					else:
						KMatrixTot[i] = 1  #NOT Shaded
				
				#As long as 1 tree had a light value of 0 then we say position n,o was beening shaded from the sun at that specific time.
				if (iter - KMatrixTot.sum()) <= 0: #############This probably should be 'iter - 2 - KMatrixTot.sum()' from comparison with Wolfram code??
					KMatrixTotTot[n] = (ClearNBGW(h_day[h],day)+BeamNBGW(h_day[h],day))*1.26
				else:
					KMatrixTotTot[n] = 0
				
				#Sum up for each time step h and day.
				GillyTop[n] += KMatrixTotTot[n]
				GillyBottom[n] += (ClearNBGW(h_day[h],day)+BeamNBGW(h_day[h],day))*1.26

	############################################################################
	#Finally normalise the GLE values with respects to full sunlight
	
	for n in xrange(tree_min, tree_max):
		GillyLight[n] = GillyTop[n]/GillyBottom[n]
			
	############################################################################
	tfinal = time.time()
	return GillyLight, daymax-daymin, hspa, tfinal, tstart
	
############################################################################
############################################################################
#Solve the GLE with a given input parameters

GillyLight, Days, TemporalSpacing, tfinal, tstart = GLE(474, 420, 474, 105, 263, 120)

#Save the data of 'GillyLight' to csv file.
ID = np.arange(475)
GillyLight = zip(ID, X, Y, HE, CH, CW, GillyLight)

#Output the data from the Gilly equation to the specified file
with open("rawdata/GLEOutputTreev5minus_8.csv", "wb") as output:
	writer = csv.writer(output, lineterminator='\n')
	writer.writerows(GillyLight)
	
#Output the retrospective log file aswell for information about the run of the model
datalog = [["Temporal Spacing", "No. of Days", "No. of Grids", "Time To Solve"],[TemporalSpacing, Days, len(GillyLight), tfinal-tstart]]
with open("rawdata/GLEOutputTreev5minus_8_log.ini", "wb") as output:
	writer = csv.writer(output, lineterminator='\n')
	writer.writerows(datalog)

#Print out post run information.
os.system('cls' if os.name=='nt' else 'clear')	
print("Gilly Light Equation Output Information\n")
print("---------------------------------------------------")
print("Gilly Light Equation has been solved successfully")
print("Some final information about this run:")
print("Days: ", Days)
print("Temporal Spacing: ", TemporalSpacing, "s")
#print("Spatial Spacing: ", SpatialSpacing, "m")
print("No. of Grids: ", len(GillyLight))
print("---------------------------------------------------")
print("Time taken to solve GLE: ", "%.0f" % (tfinal-tstart), "s")
