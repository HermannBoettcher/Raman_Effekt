import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp


x0, y0 = np.loadtxt('Schwefel_pol0.txt', delimiter='\t', usecols=(0, 1), unpack=True)
#x0=(x0*10**(2)+1/(532*10**(-9)))**(-1)*10**(9)
plt.plot(x0,y0)
plt.show()

h=6.626070040*10**(-34)
k= 1.38064852*10**(-23)
n1=51426
n2=6191
l=532*10**(-9)
c=299792458
nu=470*10**2

T=h*c*nu/(k*np.log((n1/n2)*((1/l-nu)**4/(1/l+nu)**4)))-273
deltaT = h*c*nu/(k*np.log((n1/n2)*((1/l-nu)**4/(1/l+nu)**4)))**2*(1/((n1/n2)*((1/l-nu)**4/(1/l+nu)**4)))*4*((1/l-nu)**3/(1/l+nu)**3)
print(T)