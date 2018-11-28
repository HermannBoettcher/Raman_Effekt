import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

max_list=[]
x, y = np.loadtxt('he_50.txt', delimiter='\t', usecols=(0, 1), unpack=True)
y=y[3000:]
max_list.append(max(y))
x, y = np.loadtxt('he_100.txt', delimiter='\t', usecols=(0, 1), unpack=True)
y=y[3000:]
max_list.append(max(y))
x, y = np.loadtxt('he_200.txt', delimiter='\t', usecols=(0, 1), unpack=True)
y=y[3000:]
max_list.append(max(y))
x, y = np.loadtxt('he_400.txt', delimiter='\t', usecols=(0, 1), unpack=True)
y=y[3000:]
max_list.append(max(y))
x, y = np.loadtxt('he_800.txt', delimiter='\t', usecols=(0, 1), unpack=True)
y=y[3000:]
max_list.append(max(y))
x, y = np.loadtxt('he_2000.txt', delimiter='\t', usecols=(0, 1), unpack=True)
y=y[3000:]
max_list.append(max(y))
x, y = np.loadtxt('he_4000.txt', delimiter='\t', usecols=(0, 1), unpack=True)
y=y[3000:]
max_list.append(max(y))

print(max_list)
y=[item for item in max_list]
x=[50.,100.,200.,400.,800.,2000.,4000.]
x=np.array(x)


def func(x, *a):
    y =  a * x
    return y

guess = [8]

popt, pcov = curve_fit(func, x, y, p0=guess)
fit = func(x, *popt)
print(popt)



plt.plot(x, y,'.')
plt.plot(x,popt*x)
plt.show()
print(pcov[0][0]**2)

