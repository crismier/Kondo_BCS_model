# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:29:25 2026

@author: cmier
"""

import numpy as np
from qutip import destroy, qeye, sigmaz, tensor, jmat
import matplotlib.pyplot as plt
from eigenshuffle import eigenshuffle_eigh

cm = 1/2.54 # converts inch to cm

# impurity spin
S = 5/2
dimS = int(2*S + 1)
dimtot = 2*2*dimS

# 2x2 building blocks (electron modes only)
a = destroy(2)
I = qeye(2)
Z = sigmaz()

# Identity on the impurity factor 
IS = qeye(dimS)

# Fermion operators (ordering: up mode ⊗ down mode ⊗ impurity)
c_up = tensor(a, I, IS)
c_dn = tensor(Z, a, IS)# the Z string enforces anticommutation

# Impurity spin-S operators 
Sz = tensor(I, I, jmat(S, 'z'))
Sp = tensor(I, I, jmat(S, '+'))
Sm = tensor(I, I, jmat(S, '-'))# Sm = Sp.dag() also works and gives the same result

# Electron operators
n_up = c_up.dag() * c_up
n_dn = c_dn.dag() * c_dn
sz = 0.5 * (n_up - n_dn)
sp = c_up.dag() * c_dn
sm = sp.dag()

# define Hamiltonian
def H_tot(Delta, J, B, Dz):
    H_pair  = Delta * (c_up.dag() * c_dn.dag() + c_dn * c_up) #SC
    H_K     = J * (Sz * sz + 0.5 * (Sp * sm + Sm * sp)) # Kondo
    H_Zeeman = B*(Sz) # Zeeman
    H_SZ2 = Dz*Sz*Sz
    H =  H_pair + H_K + H_Zeeman + H_SZ2
    return(H)

#%% Hamiltonian parameters

Delta = 1
Jmax = 1
Bmax = 0.2
Dmax = 1.0

#%% solve Hamiltonian for Delta = 1, J = 0 --> Jmax; B = 0 --> Bmax; 

N = 40
J = np.linspace(0,Jmax,N)
B = np.linspace(0,Bmax,N)
D = np.linspace(0,Dmax,N)

H_stack = np.zeros([2*N+1,dimtot,dimtot])
H_stack[0,:,:] = H_tot(Delta,0,0,0).full()

# Stack of Hamiltonians, shape (N, 8, 8)
for i in range(N):
    H_stack[i+1,:,:] = H_tot(Delta,J[i],0,0).full()
    
for j in range(N):
    H_stack[j+N+1,:,:] = H_tot(Delta,J[-1],0,D[j]).full()
    
E, V = eigenshuffle_eigh(H_stack) # solve all Hamiltonians while keeping eigenenergies sorted

#calculate excitation energies
YSR_energy_1 = np.zeros(2*N+1)

for i in range(2*N+1):    
    YSR_energy_1[i] = abs(E[i,5] - E[i,0])

#%% plots

fig,axs = plt.subplots(2,1, figsize=(15*cm, 20*cm))
### plot eigenstates
axs[0].plot(E, 'o', ms = 1.0)
axs[0].axvline(1, c='k', lw=0.5)
axs[0].axvline(1+N, c='k', lw=0.5)
axs[0].set_ylabel('Energy')
axs[0].set_xlim([0, 2*N+1])

axs[1].plot(YSR_energy_1, c='C0')
axs[1].axvline(1, c='k', lw=0.5)
axs[1].axvline(1+N, c='k', lw=0.5)
axs[1].set_ylabel('Excitation energy')
axs[1].set_xlim([0, 2*N+1])



