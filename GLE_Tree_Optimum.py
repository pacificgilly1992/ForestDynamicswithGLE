############################################################################
# Project: Forest Dynamics at the 'Woods of the World'
# Author: James Gilmore
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 1.3.4
# Date: 23/04/16
# Status: Operational
# Change: GitHub Test
############################################################################

#Initialising the python script
from __future__ import absolute_import, division, print_function
from multiprocessing import Queue, Pool, cpu_count, Manager
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
X, Y, HE, CH, CW, DE = np.genfromtxt("../Data/Tree/GillyShadingTIFullwithOverGrowth_minus.csv", dtype=float, delimiter=',', usecols=(0, 1, 2, 3, 4, 8),  unpack=True, skip_header=1)
SP = np.genfromtxt("../Data/Tree/GillyShadingTIFullwithOverGrowth_minus.csv", dtype=str, delimiter=',', usecols=(7),  unpack=True, skip_header=1)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def GLE_SingleCore(iter=len(X), tree_min=0, tree_max=len(X), daymin=105, daymax=263, hspa=10):
	
	############################################################################
	#Initalise the variables for this function

	it=0 								#iterator counter
	treeskip=0 							#Number of trees skipped
	AllTrees=0 							#Number of trees processed

	KMatrixCrown = np.zeros([iter, 2]) 	#Determines the shade using the crown of each tree
	KMatrixHeight = np.zeros([iter, 2]) #Determines the shade using the height of each tree
	KMatrix = np.zeros([iter, 2]) 		#Hold the combintion of KMatrixCrown and KMatrixHeight
	KMatrixTot = np.zeros(iter)			#Determines if each individual tree is causing shade on specific location (n, o)
	KMatrixTotTot = np.zeros(iter)		#Combines all tree shading information to determine if any tree at also is cast shadows at (n, o)
	GillyTop = np.zeros(iter)			#Holds the numerator of the GLE equation
	GillyBottom = np.zeros(iter)		#Holds the denomiator of the GLE equation
	GillyLight = np.zeros(iter)			#This will hold the actual fractional light score
	npos = np.zeros(iter)				#Used to define the position of of each tree
	opos = np.zeros(iter)				#Used to define the position of of each tree
	tstart = time.time()				#Starts the clock on this massive processing task
	ts = np.zeros(iter)					#Current time at the current iterator of the equation
	eps = sys.float_info.epsilon		#Used to minimise the divide by zero errors
	
	print(len(X))
	print(len(npos))
	
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
					KMatrixTotTot[n] = ClearNBGW(h_day[h],day)*BeamNBGW(h_day[h],day)
				else:
					KMatrixTotTot[n] = 0
				
				#Sum up for each time step h and day.
				GillyTop[n] += KMatrixTotTot[n]
				GillyBottom[n] += ClearNBGW(h_day[h],day)*BeamNBGW(h_day[h],day)

	############################################################################
	#Finally normalise the GLE values with respects to full sunlight
	
	for n in xrange(tree_min, tree_max):
		GillyLight[n] = GillyTop[n]/GillyBottom[n]
			
	############################################################################
	tfinal = time.time()
	return GillyLight, daymax-daymin, hspa, tfinal, tstart
 
def GSCF_MultiCore(params):
	############################################################################
	#The Gilly Light Equaton

	"""There are 3 stages to determining the amount of sunlight present at a location
	for a given year:
	(1) The first requires solving the Gilly Shade Criterion Function (GSCF) which
	determines if any entities are casting a shadow on position n,o
	(2) Combine the GSCF with the Beam Sky Radiation factor for each position in time
	to determine the amount of avaliable sunlight when no shade is cast.
	(3) Solve for h*day interations in time and summerise each step and compare to a
	point with full light avaliability to normalised the equation between 0 and 1."""

	############################################################################
	#Initalise the variables for this function

	day, hspa, tree_min, tree_max, daymin, daymax, iter, tstart, q = params
	
	it=0 								#iterator counter
	treeskip=0 							#Number of trees skipped
	AllTrees=0 							#Number of trees processed

	KMatrixCrown = np.zeros([iter, 2]) 	#Determines the shade using the crown of each tree
	KMatrixHeight = np.zeros([iter, 2]) #Determines the shade using the height of each tree
	KMatrix = np.zeros([iter, 2]) 		#Hold the combintion of KMatrixCrown and KMatrixHeight
	KMatrixTot = np.zeros(iter)			#Determines if each individual tree is causing shade on specific location (n, o)
	KMatrixTotTot = np.zeros(iter)		#Combines all tree shading information to determine if any tree at also is cast shadows at (n, o)
	GillyTop = np.zeros(iter)			#Holds the numerator of the GLE equation
	GillyBottom = np.zeros(iter)		#Holds the denomiator of the GLE equation
	GillyLight = np.zeros(iter)			#This will hold the actual fractional light score
	npos = np.zeros(iter)				#Used to define the position of of each tree
	opos = np.zeros(iter)				#Used to define the position of of each tree
	ts = np.zeros(iter)					#Current time at the current iterator of the equation
	eps = sys.float_info.epsilon		#Used to minimise the divide by zero errors
	
	for i in xrange(iter):
		npos[i] = X[i]
		opos[i] = Y[i]
	
	#Update Status
	ts[day] = time.time()
	it+=1
	
	#Calculate equally spacing times for the particular day.
	hmax = (Dusk(day)-Dawn(day))/hspa
	h_day = np.arange(Dawn(day)+0.1, Dusk(day), hmax)
	
	for h in xrange(len(h_day)):
		for n in xrange(tree_min, tree_max):
			q.put(n) #Call home to notify next iteraion has begun
			for i in xrange(iter):
				AllTrees+=1
				#This 'if' statement greatly reduces load as it quickly determines if a tree is close to the position n,o and removes from calculation if it isn't
				if (np.abs(X[i]-npos[n])-(HE[i]/np.tan(SolElvNBGW(h_day[h],day))) and np.abs(Y[i]-opos[n])-(HE[i]/np.tan(SolElvNBGW(h_day[h],day)))) < 0:
					#Determines if the width of the shade o	verlaps with n,o and if so normalises it in binary form
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
				KMatrixTotTot[n] = ClearNBGW(h_day[h],day)*BeamNBGW(h_day[h],day)
			else:
				KMatrixTotTot[n] = 0
			
			#Sum up for each time step h and day.
			GillyTop[n] += KMatrixTotTot[n]
			GillyBottom[n] += ClearNBGW(h_day[h],day)*BeamNBGW(h_day[h],day)	
	return GillyTop, GillyBottom, daymax-daymin, hmax, tstart

def GSCF_Normalise(GillyTop, GillyBottom, tree_min, tree_max, iter):
	"""Normalises the GLE values with respects to full sunlight"""
	
	GillyLight = np.zeros(iter)			#This will hold the actual fractional light score
	
	for n in xrange(tree_min, tree_max):
		GillyLight[n] = GillyTop[n]/GillyBottom[n]
	
	tfinal = time.time()
	return GillyLight, tfinal

def Updater(time_left, tstart, size, day_count, process_amount, event_speed, event_speed_max, tree_min, tree_max, hspa):
	"""Updates the console with information during the GSCF run"""
	
	os.system('cls' if os.name=='nt' else 'clear')
	if time_left <0:
		time_left=0
	print("Time Left (s): ", "%.0f" % time_left)
	print("Time Progressed (s): ", "%.0f" % (time.time()-tstart))
	print("Events Processed: ", size-process_amount)
	print("Percentage Completed: ", "%.3f" % ((1-(((tree_max-tree_min)*hspa*len(day_count))-(size+1))/((tree_max-tree_min)*hspa*len(day_count)))*100), "%")
	print("Processing Speed (Max): ", "%.0f" % event_speed," (", "%.0f" % event_speed_max, ")", " Events/Sec", sep='')
	
def GLE_Print(X, Y, HE, CH, CW, SP, DE, GillyLight, TemporalSpacing, Days, tstart, tfinal):
	"""Saves the data of 'GillyLight' to csv file."""
	
	ID = np.arange(475)
	GillyLight = zip(ID, X, Y, HE, CH, CW, SP, DE, GillyLight)

	#Output the data from the Gilly equation to the specified file
	with open("../Data/Raw/GLEOutputTreev26.csv", "wb") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(GillyLight)
		
	#Output the retrospective log file aswell for information about the run of the model
	datalog = [["Temporal Spacing", "No. of Days", "No. of Grids", "Time To Solve"],[TemporalSpacing, Days, len(GillyLight), tfinal-tstart]]
	with open("../Data/Raw/GLEOutputTreev26_log.ini", "wb") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(datalog)

	############################################################################
	#Print out post run information.
	os.system('cls' if os.name=='nt' else 'clear')	
	print("Gilly Light Equation Output Information\n")
	print("---------------------------------------------------")
	print("Gilly Light Equation has been solved successfully")
	print("Some final information about this run:")
	print("Days: ", Days)
	print("Temporal Spacing: ~", "%.0f" % (TemporalSpacing*3600), "s")
	#print("Spatial Spacing: ", SpatialSpacing, "m")
	print("No. of Grids: ", len(GillyLight))
	print("---------------------------------------------------")
	print("Time taken to solve GLE: ", "%.0f" % (tfinal-tstart), "s")
	
	return
	
def GLE_MultiCore_Initiliser(iter=len(X), tree_min=0, tree_max=len(X), daymin=105, daymax=263, hspa=10, process_amount=5):
	"""Initialises the system to run on multiple cores and then calls GLE_Multi
	which is a bespoke versions to run for parallel processing"""

	############################################################################
	# Define the parameters to test
	day_count = range(daymin,daymax)	#Defines day year range
	tstart = time.time()				#Starts the clock on this massive processing task
	eps = sys.float_info.epsilon		#Used to minimise the divide by zero errors
	event_speed_max = 0					#Defines the processing speed of the analysis
	day_done_old = 0
	
	p = Pool(process_amount)
	m = Manager()
	q = m.Queue()
	
	args = [(day, hspa, tree_min, tree_max, daymin, daymax, iter, tstart, q) for day in day_count]
	
	############################################################################
	#Solve the GLE with a given input parameters
	
	GSCF_Output = p.map_async(GSCF_MultiCore, args)
	
	# monitor loop
	while True:
		if GSCF_Output.ready():
			print("GSCF has been calculated. Thanks for sticking with us! Now final cleanup and sending it to the printers!")
			time.sleep(10)
			break
		else:
			size = q.qsize()
			time_left = ((((tree_max-tree_min)*hspa*len(day_count))-(size+1))*((time.time()-tstart)/(size+1)))
			event_speed = ((size+1)/(time.time()-tstart))
			if event_speed > event_speed_max:
				event_speed_max = event_speed
			Updater(time_left, tstart, size, day_count, process_amount, event_speed, event_speed_max, tree_min, tree_max, hspa)
			time.sleep(1)
			
	############################################################################
	#Distribute data to relevant variables
	
	GSCF_Output = GSCF_Output.get()
	GSCF_Output = np.asarray(GSCF_Output)
	GillyTop = GSCF_Output[:,0]
	GillyBottom = GSCF_Output[:,1]
	Days = GSCF_Output[:,2]
	TemporalSpacing = GSCF_Output[:,3]
	tstart = GSCF_Output[:,4]
	
	############################################################################	
	#Do the post processing to determine GLE value (-1 and 0 indicies are used to select correct case due to multiprocessing creating multiple arrays (at each timestep))
	
	GillyLight, tfinal = GSCF_Normalise(GillyTop[-1], GillyBottom[-1], tree_min, tree_max, iter)
	
	return GillyLight, TemporalSpacing, Days, tstart, tfinal
	
def lowpriority():
    """ Set the priority of the process to below-normal."""

    import sys
    try:
        sys.getwindowsversion()
    except:
        isWindows = False
    else:
        isWindows = True

    if isWindows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api,win32process,win32con

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        import os

        os.nice(1)

if __name__ == '__main__':

	#Give some initial infomation about the multiprocessing unit
	process_start = 0 #Condition hold any processing of GLE until all system checks have been make to help reduce errors
	os.system('cls' if os.name=='nt' else 'clear')
	print("Gilly Light Equation Setup\n")
	print("---------------------------------------------------")
	print("Starting GLE in multiple processes.")
	print("Number of CPUs Avaliable: ", cpu_count(), "\n")
	print("---------------------------------------------------")
	while process_start == 0:
		coretype = str(raw_input("Do you require a multicore (m) or singlecore (s) processing?"))
		if coretype == "m":
			process_amount = str(raw_input("Do you require all avaliable cores (type: all) or a specific number (type number)?"))
			if process_amount == "all":
				process_amount = 2*cpu_count()
				process_start = 2
			elif is_number(process_amount) == True:
				process_amount = int(process_amount)
				process_start = 2
			else:
				print("SUCK")
			process_priority = str(raw_input("Do you wish to run in below-normal priority (otherwise we'll set it to normal) (type: yes|no)?"))
			if process_priority == "yes":
				print("Running in low priority!!!")
				lowpriority()
		elif coretype == "s":
			process_start = 1
		else: 
			print("Command not recognised! Enter 's' for singlecore processor and 'm' for multicore processor without quote marks :)")
	print("---------------------------------------------------\n")
	print("Goodluck out there...this is gonna be a long journey!")
	time.sleep(5)
    
	if process_start == 1:
		GillyLight, Days, TemporalSpacing, tfinal, tstart = GLE_SingleCore(3554, 0, 60, 105, 263, 120)
		GLE_Print(X, Y, HE, CH, CW, SP, DE, GillyLight, TemporalSpacing, Days, tstart, tfinal)
	elif process_start == 2:
		GillyLight, TemporalSpacing, Days, tstart, tfinal = GLE_MultiCore_Initiliser(3554, 0, 474, 105, 263, 20, process_amount)
		print(type(TemporalSpacing))
		print(type(Days))
		print(type(tfinal))	
		GLE_Print(X, Y, HE, CH, CW, SP, DE, GillyLight, TemporalSpacing, Days, tstart, tfinal)
	else:
		sys.exit("Could not find a correct configuration for you. Please try again.")
