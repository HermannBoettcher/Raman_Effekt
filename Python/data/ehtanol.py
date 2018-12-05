import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

konz =[]
rel = []

n_e=0
n_w=0
x0, y0 = np.loadtxt('ethanol_wasser_2575_40000.txt', delimiter='\t', usecols=(0, 1), unpack=True)
w_i = np.argwhere(x0<3200)
w_j = np.argwhere(x0>3450)
e_i = np.argwhere(x0<2850)
e_j = np.argwhere(x0>3000)
for k in range(e_j[0][0]-e_i[-1][0]):
    n_e += y0[k+e_i[-1][0]]
for k in range(w_j[0][0]-w_i[-1][0]):
    n_w += y0[k+w_i[-1][0]]
konz.append(25.)
rel.append(n_e/n_w)

n_e=0
n_w=0
x0, y0 = np.loadtxt('ethanol_wasser_3070_40000.txt', delimiter='\t', usecols=(0, 1), unpack=True)
w_i = np.argwhere(x0<3200)
w_j = np.argwhere(x0>3450)
e_i = np.argwhere(x0<2850)
e_j = np.argwhere(x0>3000)
for k in range(e_j[0][0]-e_i[-1][0]):
    n_e += y0[k+e_i[-1][0]]
for k in range(w_j[0][0]-w_i[-1][0]):
    n_w += y0[k+w_i[-1][0]]
konz.append(30.)
rel.append(n_e/n_w)


n_e=0
n_w=0
x0, y0 = np.loadtxt('ethanol_wasser_3565_40000.txt', delimiter='\t', usecols=(0, 1), unpack=True)
w_i = np.argwhere(x0<3200)
w_j = np.argwhere(x0>3450)
e_i = np.argwhere(x0<2850)
e_j = np.argwhere(x0>3000)
for k in range(e_j[0][0]-e_i[-1][0]):
    n_e += y0[k+e_i[-1][0]]
for k in range(w_j[0][0]-w_i[-1][0]):
    n_w += y0[k+w_i[-1][0]]
konz.append(35.)
rel.append(n_e/n_w)


n_e=0
n_w=0
x0, y0 = np.loadtxt('ethanol_wasser_4060_40000.txt', delimiter='\t', usecols=(0, 1), unpack=True)
w_i = np.argwhere(x0<3200)
w_j = np.argwhere(x0>3450)
e_i = np.argwhere(x0<2850)
e_j = np.argwhere(x0>3000)
for k in range(e_j[0][0]-e_i[-1][0]):
    n_e += y0[k+e_i[-1][0]]
for k in range(w_j[0][0]-w_i[-1][0]):
    n_w += y0[k+w_i[-1][0]]
konz.append(40.)
rel.append(n_e/n_w)


n_e=0
n_w=0
x0, y0 = np.loadtxt('ethanol_wasser_4555_40000.txt', delimiter='\t', usecols=(0, 1), unpack=True)
w_i = np.argwhere(x0<3200)
w_j = np.argwhere(x0>3450)
e_i = np.argwhere(x0<2850)
e_j = np.argwhere(x0>3000)
for k in range(e_j[0][0]-e_i[-1][0]):
    n_e += y0[k+e_i[-1][0]]
for k in range(w_j[0][0]-w_i[-1][0]):
    n_w += y0[k+w_i[-1][0]]
konz.append(45.)
rel.append(n_e/n_w)


n_e=0
n_w=0
x0, y0 = np.loadtxt('ethanol_wasser_5050_40000.txt', delimiter='\t', usecols=(0, 1), unpack=True)
w_i = np.argwhere(x0<3200)
w_j = np.argwhere(x0>3450)
e_i = np.argwhere(x0<2850)
e_j = np.argwhere(x0>3000)
for k in range(e_j[0][0]-e_i[-1][0]):
    n_e += y0[k+e_i[-1][0]]
for k in range(w_j[0][0]-w_i[-1][0]):
    n_w += y0[k+w_i[-1][0]]
konz.append(50.)
rel.append(n_e/n_w)



n_e=0
n_w=0
x0, y0 = np.loadtxt('ethanol_wasser_2575_40000.txt', delimiter='\t', usecols=(0, 1), unpack=True)
w_i = np.argwhere(x0<3200)
w_j = np.argwhere(x0>3450)
e_i = np.argwhere(x0<2850)
e_j = np.argwhere(x0>3000)
for k in range(e_j[0][0]-e_i[-1][0]):
    n_e += y0[k+e_i[-1][0]]
for k in range(w_j[0][0]-w_i[-1][0]):
    n_w += y0[k+w_i[-1][0]]
konz.append(55.)
rel.append(n_e/n_w)

rel.pop()
rel.pop()
konz.pop()
konz.pop()

def func(c, *params):
    a = params[0]
    b = params[1]
    y = a + b *c
    return y

guess=[8,8]


popt, pcov = curve_fit(func, konz, rel, p0=guess)
print(popt)
print(konz)
fit = func(konz, *popt)


np.savetxt('ethanol_data.txt', np.transpose(np.array([konz,rel])), fmt='%.18e', delimiter='\t', newline='\n', )
plt.plot(konz,rel,'.')
plt.plot(konz, fit(konz))
plt.show()