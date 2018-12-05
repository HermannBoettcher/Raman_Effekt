import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp


x0, y0 = np.loadtxt('Schwefel_pol0.txt', delimiter='\t', usecols=(0, 1), unpack=True)
#x0=(x0*10**(2)+1/(532*10**(-9)))**(-1)*10**(9)
plt.plot(x0,y0)


as_i = np.argwhere(x0<-500)
as_j = np.argwhere(x0>-450)
s_i = np.argwhere(x0<450)
s_j = np.argwhere(x0>500)


h=6.626070040*10**(-34)
k=1.38064852*10**(-23)
n_as=0
n_s=0
n1=51426
n2=6191
l=532*10**(-9)
c=299792458
nu=470*10**2
for k in range(as_j[0][0]-as_i[-1][0]):
    n_as += y0[k+as_i[-1][0]]

for k in range(s_j[0][0]-s_i[-1][0]):
    n_s += y0[k+s_i[-1][0]]
print(n_as, n_s)

n1=51426
n2=6191
l=532*10**(-9)
c=299792458
nu=470*10**2

T = h * c * nu / (k * np.log(661891.2000000001 / 6000 * ((1 / l + nu) / (1 / l - nu))**4))-273

#deltaT = h*c*nu/(k*np.log((n1/n2)*((1/l-nu)/(1/l+nu))**4))**2*(1/((n1/n2)*((1/l-nu)**4/(1/l+nu)**4)))*4*((1/l-nu)**3/(1/l+nu)**3)
print(T)