#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:57:42 2023

@author: mcamara
"""
import numpy as np

class moltent_tank:
    
    def __init__(self,L,A,N):
        self._L = L
        self._A = A
        self._N = N
        self._z = np.linspace(0,L,N)

        h = L/(N-1)
        h_arr = h*np.ones(N)
        self._h = h
 
        # FDM backward, 1st deriv
        d0 = np.diag(1/h_arr, k = 0)
        d1 = np.diag(-1/h_arr[1:], k = -1)
        d = d0 + d1
        d[0,:] = 0
        self._d = d
        
        # FDM foward, 1st deriv
        d0_fo = np.diag(-1/h_arr, k = 0)
        d1_fo = np.diag(1/h_arr[1:], k = 1)
        d_fo = d0_fo + d1_fo
        self._d_fo  = d_fo
 
        # FDM centered, 2nd deriv
        dd0 = np.diag(1/h_arr[1:]**2, k = -1)
        dd1 = np.diag(-2/h_arr**2, k = 0)
        dd2 = np.diag(1/h_arr[1:]**2, k = 1)
        dd = dd0 + dd1 + dd2
        dd[0,:]  = 0
        dd[-1,:] = 0
        self._dd = dd
        
        self._required = {
            'Design tank':True,
            'Solid properties':False,
            'Fluid properties': False,
            'Thermal transfer' : False,
            'Boundary condtions' : False,
            'Flow direction' : '?????',
            'Initial conditions' : False}

    def __str__(self):
        str_return = '[[Current information included here]] \n'
        for kk in self._required.keys():
            str_return = str_return + '{0:16s}'.format(kk)
            if type(self._required[kk]) == type('  '):
                str_return = str_return+ ': ' + self._required[kk] + '\n'
            elif self._required[kk]:
                str_return = str_return + ': True\n'
            else:
                str_return = str_return + ': False\n'
        return str_return
    
    def solid_info(self, epsi = 0.3,
                   dp = 0.01,
                   rho_s = 1000,
                   cp_s  = 1000,
                   k_s   = 0.1):
        self._epsi=epsi
        self._dp=dp
        self._rho_s=rho_s
        self._cp_s=cp_s
        self._k_s=k_s
        self._required['Solid properties'] = True
    
    def liquid_info(self, mode,rho_f ,
                   mu_f  ,
                   cp_f  ,
                   k_f   ,
                   ):
        
        if mode == "constante":
            self._liquid_properties="constante"
            self._rho_f=rho_f
            self._cp_f=cp_f
            self._k_f=k_f
            self._mu_f=mu_f
        else :
            self._liquid_properties="noconstante"
            self._frho_f=rho_f
            self._fcp_f=cp_f
            self._fk_f=k_f
            self._fmu_f=mu_f
            
        self._required['Fluid properties'] = True
        
    def thermal_transfer_info(self,hwall =0.3,Awall=1.,
                              Asolid=1.,hsolid=0.3):
        self._hwall=hwall
        self._Awall=Awall
        self._hsolid=hsolid
        self._Asolid=Asolid
        self._required['Thermal transfer'] = True
        
    def boundaryC_info(self,direction='?????',u_inlet=0.1,T_inlet=303.,T_amb=298.):
        self._direction=direction
        self._u_inlet=u_inlet
        self._T_inlet=T_inlet
        self._T_amb=T_amb
        if direction == 'Forward':
            self._required['Flow direction'] = 'Foward'
            self._required['Boundary condtions'] = True
        elif direction == 'Backward':
            A_flip = np.zeros([self._N,self._N])
            for ii in range(self._N):
                A_flip[ii, -1-ii] = 1
            self._required['Flow direction'] = 'Backward'
            self._A_flip = A_flip
            self._required['Flow direction'] = 'Backward'
            self._required['Boundary condtions'] = True
        else:
            self._required['Flow direction'] = '??????'
            self._required['Boundary condtions'] = True
            
    def initialC_info(self,Tf_init=298,Ts_init=298.):
        self._Tf_init=Tf_init
        self._Ts_init=Ts_init
        self._required['Initial conditions'] = True

        
    def solve(self, t_max, n_sec = 5, CPUtime_print = False):
        t_max_int = np.int32(np.floor(t_max))
        self._n_sec = n_sec
        n_t = t_max_int*n_sec+ 1
        t_dom = np.linspace(0,t_max_int, n_t)
        if t_max_int < t_max:
            t_dom = np.concatenate((t_dom, [t_max]))
        
        
        u=self._u_inlet
        epsi = self._epsi
        
        rho_f = np.zeros(self._N)
        mu_f = np.zeros(self._N)
        cp_f = np.zeros(self._N)
        k_f = np.zeros(self._N)
        
        rho_s = np.zeros(self._N)
        cp_s = np.zeros(self._N)
        k_s = np.zeros(self._N)
        
        keff=(k_s+k_f)/2
        
        hwall=self._hwall
        Awall=self._Awall
        hsolid= self._hsolid
        Asolid= self._Asolid
        
        Tamb=self._T_amb
        Tin =self._T_inlet
        
        
        def solve_energy(y,t):
            Tf = y[self._N : 2*self._N ]
            Ts = y[2*self._N : 2*self._N + 2*self._N ]
            dTf = self._d@Tf
            ddTf = self._dd@Tf
            dTs = self._d@Ts
            ddTs = self._dd@Ts
            
            dTfdt= (- u*dTf +
                    keff*ddTf/(epsi*rho_f*cp_f) +
                    hsolid*Asolid*(Ts-Tf)/(epsi*rho_f*cp_f)+
                    hwall*Awall*(Tamb-Tf)/(epsi*rho_f*cp_f)                    
                    )
            
            dTfdt[0]=u*(self._T_inlet - Tf[0])/h+
            
            
            """
            # Temperature (gas)
            dTgdt = -v*dTg + h_heat*a_surf/epsi*(Ts - Tg)/Cov_Cpg
            dTgdt[0] = h_heat*a_surf/epsi*(Ts[0] - Tg[0])/Cov_Cpg[0]
            dTgdt[0] = dTgdt[0] + v_in*(self._T_in - Tg[0])/h
            dTgdt[-1] = h_heat*a_surf/epsi*(Ts[-1] - Tg[-1])/Cov_Cpg[-1]
            dTgdt[-1] = dTgdt[-1] + (v[-1]+v_out)/2*(Tg[-2]-Tg[-1])/h
            
            # Temperature (solid)
           dTsdt = (self._k_cond*ddTs+ h_heat*a_surf/(1-epsi)*(Tg-Ts))/self._rho_s/Cps
           
           dydt_tmp = [dTgdt] + [dTsdt]
           dydt = np.concatenate(dydt_tmp)
           return dydt
           """
ms = moltent_tank(L=1,A=0.1,N=21)
print(ms)
ms.solid_info(epsi = 0.3,dp = 0.01,rho_s = 1000,cp_s  = 1000,k_s   = 0.1)
print(ms)
ms.liquid_info(rho_f = 1000,mu_f  = 1.E-3,cp_f  = 1000,k_f   = 0.1)
print(ms)
ms.thermal_transfer_info(constant=True, hwall = 0.3,hsolid=0.3)
print(ms)
ms.boundaryC_info(direction='Forward',u_inlet=0.1,T_inlet=303.)
print(ms)
ms.initialC_info(Tf_init=298,Ts_init=298.)
print(ms)



        

        

        
        

        