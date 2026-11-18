#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 19:44:29 2026

@author: cristinamiergonzalez
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import destroy, qeye, sigmaz, tensor, jmat
from eigenshuffle import eigenshuffle_eigh

cm = 1/2.54 # converts inch to cm

#### operator and Hamiltonian definition

a, I, Z = destroy(2), qeye(2), sigmaz()
S = 0.5                 # impurity spin
dS = int(2*S + 1)
IS = qeye(dS)            # identity on the impurity factor

# Fermion operators for three sites (ordering: 1up,1dn,2up,2dn,3up,3dn,impurity)
c1_up = tensor(a, I, I, I, I, I, IS)
c1_dn = tensor(Z, a, I, I, I, I, IS)
c2_up = tensor(Z, Z, a, I, I, I, IS)
c2_dn = tensor(Z, Z, Z, a, I, I, IS)
c3_up = tensor(Z, Z, Z, Z, a, I, IS)
c3_dn = tensor(Z, Z, Z, Z, Z, a, IS)

sites_up = [c1_up, c2_up, c3_up]
sites_dn = [c1_dn, c2_dn, c3_dn]

n_up = [c.dag()*c for c in sites_up]
n_dn = [c.dag()*c for c in sites_dn]

# site-local electron spin operators (needed for the Kondo term)
sz = [0.5*(n_up[i] - n_dn[i]) for i in range(3)]
sp = [sites_up[i].dag()*sites_dn[i] for i in range(3)]
sm = [op.dag() for op in sp]

# impurity operators, acting on the 7th (last) factor
Sz = tensor(I, I, I, I, I, I, jmat(S, 'z'))
Sp = tensor(I, I, I, I, I, I, jmat(S, '+'))
Sm = tensor(I, I, I, I, I, I, jmat(S, '-'))


def H_3sites(Delta, J, t=None):
    """
    Delta: SC order parameter
    t: optional 2x2-ish nearest-neighbor hopping list [t12, t23] (chain),
       or None for no hopping.
    """
    H_pair  = sum(Delta[i]*(sites_up[i].dag()*sites_dn[i].dag()
                            + sites_dn[i]*sites_up[i]) for i in range(3)) #SC pairing
    
    H_K = sum(J[i]*(Sz*sz[i] + 0.5*(Sp*sm[i] + Sm*sp[i])) for i in range(3)) #Kondo
    
    H_hop = 0
    if t is not None:
        pairs = [(0,1), (1,2)]           # chain: 1-2, 2-3
        for (i, j), tij in zip(pairs, t):
            H_hop += tij*(sites_up[i].dag()*sites_up[j] + sites_up[j].dag()*sites_up[i]
                         + sites_dn[i].dag()*sites_dn[j] + sites_dn[j].dag()*sites_dn[i]) #Hopping hamiltonian 
            
    return (H_pair + H_K + H_hop)


#%% Hamiltonian parameters

Delta = 1
t12 = 0.5
t23 = 0.5
Jmax1 = 1.0
Jmax2 = 1.0
Jmax3 = 1.0

#%% solve Hamiltonian for each J separetly

N = 40
J1 = np.linspace(0,Jmax1,N)
J2 = np.linspace(0,Jmax2,N)
J3 = np.linspace(0,Jmax3,N)

H_stack = np.zeros([3*N+1,128,128])
H_stack[0,:,:] = H_3sites(Delta=[Delta,Delta,Delta], J=[0,0,0], t=[t12,t23]).full() #initial Hamiltonian


for i in range(N):
    H_stack[i,:,:] = H_3sites(Delta=[Delta, Delta, Delta], J = [J1[i],0,0], t=[t12, t23]).full()
    
for j in range(N):
    H_stack[j+N+1,:,:] = H_3sites(Delta=[Delta, Delta, Delta], J = [J1[-1],J2[j],0], t=[t12, t23]).full()
    
for k in range(N):
    H_stack[k+2*N+1,:,:] = H_3sites(Delta=[Delta, Delta, Delta], J = [J1[-1],J2[-1],J3[k]], t=[t12, t23]).full()    
    
    
# get eigenstates
E, V = eigenshuffle_eigh(H_stack) # solve all Hamiltonians while keeping eigenenergies sorted


#%% plots

fig,axs = plt.subplots(figsize=(15*cm, 20*cm))

### plot eigenstates
for k in range(128):
    #color = "C0" if parities[0, k] > 0 else "C1"
    axs.plot(E[:,k], 'o', ms = 2.0, c='k')




