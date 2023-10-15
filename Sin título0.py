#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 20:44:42 2023

@author: mcamara
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt



L=1.
A=0.3
N=21
z = np.linspace(0,L,N)
"""
self._L = L
self._A = A
self._N = N
self._z = np.linspace(0,L,N)
"""

h = L/(N-1)
h_arr = h*np.ones(N)
"""
self._h = h
"""

# FDM backward, 1st deriv
d0 = np.diag(1/h_arr, k = 0)
d1 = np.diag(-1/h_arr[1:], k = -1)
d = d0 + d1
d[0,:] = 0
"""
self._d = d
"""

# FDM foward, 1st deriv
d0_fo = np.diag(-1/h_arr, k = 0)
d1_fo = np.diag(1/h_arr[1:], k = 1)
d_fo = d0_fo + d1_fo
"""
self._d_fo  = d_fo
"""
# FDM centered, 2nd deriv
dd0 = np.diag(1/h_arr[1:]**2, k = -1)
dd1 = np.diag(-2/h_arr**2, k = 0)
dd2 = np.diag(1/h_arr[1:]**2, k = 1)
dd = dd0 + dd1 + dd2
dd[0,:]  = 0
dd[-1,:] = 0
"""
self._dd = dd
"""

#Solid Info
epsi = 0.3
dp = 0.01
rho_s = 1000
cp_s  = 1000
k_s   = 0.1

#Liquid Info

rho_l = 1000
def rhol(Tl):
    rho=1000*np.ones_like(Tl)
    return rho 

mu_l  = 1.E-3
def mul(Tl):
    mul=1.E-3*np.ones_like(Tl)
    return mul

cp_l  = 1000
def cpl(Tl):
    cpl=1000*np.ones_like(Tl)
    return cpl
k_l   = 0.1
def kl(Tl):
    kl=0.1*np.ones_like(Tl)
    return kl


#thermal transfer
hwall =3
Awall=10.                             
Asolid=1.
hsolid=30

#Boundary conditions
u=0.03
Tin=353.15
Tamb=298.15

#Initial conditions
Tl_init=298.15
Ts_init=298.15

Tl=Tl_init*np.ones([N,])
Ts=Ts_init*np.ones([N,])
y0_tmp =  [Tl] + [Ts]
y0 = np.concatenate(y0_tmp) 
    
    
def Graph(self, every_n_sec, index, 
          loc = [1,1], yaxis_label = None, 
          file_name = None, 
          figsize = [7,5], dpi = 85, y = None,):
    N = self._N
    one_sec = self._n_sec
    n_show = one_sec*every_n_sec
    if y == None:
        y = self._y
    lstyle = ['-','--','-.',(0,(3,3,1,3,1,3)),':']
    fig, ax = plt.subplots(figsize = figsize, dpi = 90)
    cc= 0
    lcolor = 'k'
    for j in range(0,len(self._y), n_show):
        if j <= 1:
            lcolor = 'r'
        elif j >= len(self._y)-n_show:
            lcolor = 'b'
        else:
            lcolor = 'k'
        ax.plot(self._z,self._y[j, index*N:(index+1)*N],
        color = lcolor, linestyle = lstyle[cc%len(lstyle)],
        label = 't = {}'.format(self._t[j]))
        cc = cc + 1
    fig.legend(fontsize = 14,bbox_to_anchor = loc)
    ax.set_xlabel('z-domain (m)', fontsize = 15)
    if yaxis_label == None:
        ylab = 'Variable index = {}'.format(index)
        ax.set_ylabel(ylab, fontsize = 15)
    else:
        ax.set_ylabel(yaxis_label, fontsize = 15)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.grid(linestyle = ':')
    if file_name != None:
        fig.savefig(file_name, bbox_inches='tight')
    
    return fig, ax
    
def solve(tmax=10000,n_sec=10,CPU_Time=True):
    t_max_int = np.int32(np.floor(t_max))
    n_t = t_max_int*n_sec+ 1
    t = np.linspace(0,t_max_int, n_t)
    
    def solveEnergy(y,t):
        Tl=y[ :  N ]
        Ts=y[N : 2*N]
        
        # rho_l = 1000*np.ones([N,])
        # mu_l  = 1.E-3*np.ones([N,])
        # cp_l  = 1000*np.ones([N,])
        # k_l   = 0.1*np.ones([N,])
        rho_l = rhol(Tl)
        mu_l = mul(Tl)
        cp_l = cpl(Tl)
        k_l = kl(Tl)
        
        rho_s = 1000*np.ones([N,])
        cp_s  = 1000*np.ones([N,])
        k_s   = 0.1*np.ones([N,])
        
        keff=0.3
        
        #solve
        
        dTl = d@Tl
        ddTl = dd@Tl
        dTs = d@Ts
        ddTs = dd@Ts
        
        
        dTldt= (- u*dTl +
                keff*ddTl/(epsi*rho_l*cp_l) +
                hsolid*Asolid*(Ts-Tl)/(epsi*rho_l*cp_l)+
                hwall*Awall*(Tamb-Tl)/(epsi*rho_l*cp_l)                    
                )
        
        dTsdt= (hsolid*Asolid*(Tl-Ts)/((1-epsi)*rho_s*cp_s))
        
        dTldt[0] = (u*(Tin - Tl[0])/h +
                    hsolid*Asolid*(Ts[0]-Tl[0])/(epsi*rho_l[0]*cp_l[0])+
                    hwall*Awall*(Tamb-Tl[0])/(epsi*rho_l[0]*cp_l[0])
                    )
        dTldt[-1]=(u*(Tl[-2] - Tl[-1])/h +
                    hsolid*Asolid*(Ts[-1]-Tl[-1])/(epsi*rho_l[-1]*cp_l[-1])+
                    hwall*Awall*(Tamb-Tl[-1])/(epsi*rho_l[0]*cp_l[-1])
                    )
        
        dydt_tmp = [dTldt] + [dTsdt]
        dydt = np.concatenate(dydt_tmp)
        return dydt
    
    y_result = odeint(solveEnergy,y0,t)

    return y_result,t    

#RUN
t_max=10000
t_max_int = np.int32(np.floor(t_max))
n_sec = 1000
n_t = t_max_int*n_sec+ 1
t = np.linspace(0,t_max_int, n_t)

T_result,t_result = solve(tmax=1000,n_sec=10,CPU_Time=False)


    
    
    
    
