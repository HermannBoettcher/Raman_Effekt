import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp


x0, y0 = np.loadtxt('CH2Cl2_pol1.txt', delimiter='\t', usecols=(0, 1), unpack=True)
x1, y1 = np.loadtxt('CHCl3_pol0.txt', delimiter='\t', usecols=(0, 1), unpack=True)
x2, y2 = np.loadtxt('CH2Cl2_pol0.txt', delimiter='\t', usecols=(0, 1), unpack=True)

plt.plot(x1,y1)
plt.plot(x2,y2)
plt.plot(x0,y0)
plt.show()