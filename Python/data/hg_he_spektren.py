import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

x, y = np.loadtxt('hg_spektrum.txt', delimiter='\t', usecols=(0, 1), unpack=True)

#plt.plot(x,y)
#plt.show()

def func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

guess = [545,58000, 1,
         576, 17000, 1,
         579, 15000, 1,]

popt, pcov = curve_fit(func, x, y, p0=guess)
fit = func(x, *popt)
print(popt)

for i in range(0, len(popt), 3):
    j=i+2
    k=i+1
    print("sgima" + str((i+3)/3) + " = ", popt[j], "+/-", pcov[j,j]**0.5)
    print("t" + str((i+3)/3) + " = ", popt[i], "+/-", pcov[i,i]**0.5)
    print("a" + str((i+3)/3) + " = ", popt[k], "+/-", pcov[k,k]**0.5)

#plt.plot(x,fit)
#plt.plot(x,y)
#plt.show()

data=np.array([x, y, fit])
np.savetxt('he_spektrum_fit.txt', np.transpose(data))


x, y = np.loadtxt('he_spektrum.txt', delimiter='\t', usecols=(0, 1), unpack=True)

#plt.plot(x,y)
#plt.show()

guess = [501,500, 1,
         587, 52000, 1,
         668, 7000, 0.5,]

popt, pcov = curve_fit(func, x, y, p0=guess)
fit = func(x, *popt)
print(popt)

for i in range(0, len(popt), 3):
    j=i+2
    k=i+1
    print("sgima" + str((i+3)/3) + " = ", popt[j], "+/-", pcov[j,j]**0.5)
    print("t" + str((i+3)/3) + " = ", popt[i], "+/-", pcov[i,i]**0.5)
    print("a" + str((i+3)/3) + " = ", popt[k], "+/-", pcov[k,k]**0.5)

#plt.plot(x,fit)
plt.plot(x,y)
plt.show()

data=np.array([x, y, fit])
np.savetxt('he_spektrum_fit.txt', np.transpose(data))