# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:36:54 2026

@author: cmier
"""

import numpy as np
from qutip import destroy, qeye, sigmaz, tensor
import matplotlib.pyplot as plt
from eigenshuffle import eigenshuffle_eigh

cm = 1/2.54 # converts inch to cm

# 2x2 matrices used to create the other operators
a = destroy(2)      # [[0,1],[0,0]]
I = qeye(2)
Z = sigmaz()

# Fermion operators (ordering: up mode ⊗ down mode ⊗ impurity)
c_up = tensor(a, I, I)
c_dn = tensor(Z, a, I)      # the Z string enforces anticommutation

# Impurity spin-1/2 operators
Sz = tensor(I, I, sigmaz() / 2)
Sp = tensor(I, I, destroy(2))   # S+ (raises ⇓ -> ⇑ in the basis (⇑, ⇓))
Sm = Sp.dag()

# Electron operators
n_up = c_up.dag() * c_up
n_dn = c_dn.dag() * c_dn
sz = 0.5 * (n_up - n_dn)
sp = c_up.dag() * c_dn
sm = sp.dag()

# define Hamiltonian
def H_tot(Delta, J, B):
    H_pair  = Delta * (c_up.dag() * c_dn.dag() + c_dn * c_up) #SC
    H_K     = J * (Sz * sz + 0.5 * (Sp * sm + Sm * sp)) # Kondo
    H_Zeeman = B*(Sz + sz) # Zeeman
    H =  H_pair + H_K + H_Zeeman
    return(H)

#%% Hamiltonian parameters
Delta = 1
Jmax = 1.5
Bmax = 0.5

#%% solve Hamiltonian for Delta = 1, J = 0 --> Jmax; B = 0 --> Bmax; 

N = 40
J = np.linspace(0,Jmax,N)
B = np.linspace(0,Bmax,N)

H_stack = np.zeros([2*N+1,8,8])
H_stack[0,:,:] = H_tot(Delta,0,0).full()

# Stack of Hamiltonians, shape (N, 8, 8)
for i in range(N):
    H_stack[i+1,:,:] = H_tot(Delta,J[i],0).full()
    
for j in range(N):
    H_stack[j+N+1,:,:] = H_tot(Delta,J[-1],B[j]).full()
 
# solve eigenstates
E, V = eigenshuffle_eigh(H_stack) # solve all Hamiltonians while keeping eigenenergies sorted

# calculate parities
parity = (1j*np.pi*(n_up+n_dn)).expm()
P = parity.full().real 

parities = np.einsum('ijk,jl,ilk->ik', V.conj(), P, V)
parities = np.real(parities)  

# calculate excitation energies
excitation = E - E.min(axis=1, keepdims=True)    
  
# calculate excitation energies changing parity
#E0 = E.min(axis=1)                     # true ground-state energy at each J
#gs_idx = E.argmin(axis=1)              # which branch is the ground state, at each J
#p_gs = parities[np.arange(2*N + 1), gs_idx]   # ground-state parity at each J

# mask: branches with parity opposite to the ground state, at each J
#opposite_mask = np.sign(parities) != np.sign(p_gs)[:, None]
#excitations = np.where(opposite_mask, E - E0[:, None], np.nan)
  

#%% plots

fig,axs = plt.subplots(2,1, figsize=(15*cm, 20*cm))

### plot eigenstates
for k in range(8):
    color = "C0" if parities[0, k] > 0 else "C1"
    axs[0].plot(E[:,k], 'o', ms = 1.0, c=color)

axs[0].axvline(1, c='k', lw=0.5)
axs[0].axvline(1+N, c='k', lw=0.5)
axs[0].set_ylabel('Energy')
axs[0].set_xlim([0, 2*N+1])

axs[1].plot(excitation, c='C0')
axs[1].axvline(1, c='k', lw=0.5)
axs[1].axvline(1+N, c='k', lw=0.5)
axs[1].set_ylabel('Excitation energy')
axs[1].set_xlim([0, 2*N+1])








