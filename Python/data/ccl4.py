import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp


x0, y0 = np.loadtxt('CCl4_pol0.txt', delimiter='\t', usecols=(0, 1), unpack=True)
x1, y1 = np.loadtxt('CCl4_pol1.txt', delimiter='\t', usecols=(0, 1), unpack=True)

plt.plot(x0,y0)
plt.show()