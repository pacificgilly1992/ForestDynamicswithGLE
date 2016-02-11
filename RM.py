#Initialising the python script
from __future__ import absolute_import, division, print_function
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
from array import array
import numpy as np
import time, os, csv, sys

RM = lambda x: np.array([[np.cos(x), -np.sin(x)], [np.sin(x),  np.cos(x)]])

print(np.dot([1,0],RM(-np.radians(90.1))))