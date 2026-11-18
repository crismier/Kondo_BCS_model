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


#%% spectra functions

def lorentz(x, e, eta):
    return 1/((x-e)**2 + eta**2)

def spectrum(H, creators, energy, kT, eta):
    E, V = np.linalg.eigh(H.full())

    # Boltzmann populations of the initial states
    p = np.exp(-(E - E[0])/kT)
    p /= p.sum()

    # W[i,j] = p_i * sum_sigma |<i|cd|j>|^2
    W = np.zeros((len(E), len(E)))
    for cd in creators:
        M = V.conj().T @ cd.full() @ V
        W += np.abs(M)**2
    W *= p[:, None]

    dE = E[None, :] - E[:, None]          # dE[i,j] = E_j - E_i
    mask = W.ravel() > 1e-12
    w, de = W.ravel()[mask], dE.ravel()[mask]

    dos = (w[None, :] * lorentz(energy[:, None], de[None, :], eta)).sum(axis=1)
    return dos, W

def spectrum_full(H, annihilators, energy, kT, eta):
    E, V = np.linalg.eigh(H.full())
    p = np.exp(-(E - E[0])/kT); p /= p.sum()
    dE = E[:, None] - E[None, :]                    # dE[m,n] = E_m - E_n
    dos = np.zeros_like(energy)
    for c in annihilators:
        for op, sign in ((c.dag(), +1), (c, -1)):   # addition at +dE, removal at -dE
            M = V.conj().T @ op.full() @ V          # M[m,n] = <m|op|n>
            W = np.abs(M)**2 * p[None, :]           # weight of initial state n
            mask = W > 1e-12
            dos += (W[mask][None, :] *
                    lorentz(energy[:, None], sign*dE[mask][None, :], eta)).sum(axis=1)
    return dos


#%% Hamiltonian parameters
Delta = 1
Jmax = 1.5
Bmax = 0.2

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

# parity operator
parity = (1j*np.pi*(n_up+n_dn)).expm()
P = parity.full().real 

#parities = np.einsum('ijk,jl,ilk->ik', V.conj(), P, V)
# calculate parities
parities = np.zeros((2*N+1, 8), dtype=complex)

for i in range(2*N+1):
    for k in range(8):
        v = V[i][:, k]
        parities[i, k] = v.conj() @ P @ v
        
parities = np.real(parities)  

# calculate excitation energies (not taking into account parity)
excitation = E - E.min(axis=1, keepdims=True)    
  
# calculate excitation energies changing parity
E0 = E.min(axis=1) #ground-state energy at each index
gs_idx = E.argmin(axis=1)
p_gs = parities[np.arange(2*N + 1), gs_idx] #ground-state parity

# states with parity opposite to the ground state
opposite_mask = np.sign(parities) != np.sign(p_gs)[:, None]
excitations = np.where(opposite_mask, E - E0[:, None], np.nan)


#%% plots

%matplotlib inline

fig,axs = plt.subplots(2,1, figsize=(15*cm, 20*cm))

### plot eigenstates
for k in range(8):
    color = "C0" if parities[0, k] > 0 else "C1"
    axs[0].plot(E[:,k], 'o', ms = 1.0, c=color)

axs[0].axvline(1, c='k', lw=0.5)
axs[0].axvline(1+N, c='k', lw=0.5)
axs[0].set_ylabel('Energy')
axs[0].set_xlim([0, 2*N+1])

# define labels for J and B sweeps
# Ticks: start, a few along the J sweep, the corner, a few along the B sweep
ticks = np.array([0, N//4, N//2, 3*N//4, N, N + N//4, N + N//2, N + 3*N//4, 2*N])
labels = []
for i in ticks:
    if i <= N:   # J sweep (B = 0)
        labels.append(f"{J[i-1]:.2f}")
    else:        # B sweep (J = Jmax)
        labels.append(f"{B[i-2*N-1]:.2f}")
labels[0] = 0

axs[0].set_xticks(ticks)
axs[0].set_xticklabels(labels)
axs[0].text(0.25, Delta + 0.05*Delta, r"$J$", transform=axs[0].transAxes, ha='center')
axs[0].text(0.75, Delta + 0.05*Delta, r"$B$", transform=axs[0].transAxes, ha='center')

### plot excitation energies
axs[1].plot(excitation, c='k')
axs[1].plot(excitations, c='r')
axs[1].axvline(1, c='k', lw=0.5)
axs[1].axvline(1+N, c='k', lw=0.5)
axs[1].set_ylabel('Excitation energy')
axs[1].set_xlim([0, 2*N+1])

axs[1].set_xticks(ticks)
axs[1].set_xticklabels(labels)


#%% spectrum calculation

%matplotlib inline

# single-site setup
creators = [c_up.dag(), c_dn.dag()]### creation and anihiliation operators
Delta = 1.0
energy = np.linspace(-0.9*Delta, 0.9*Delta, 300)
kT, eta = 0.001*Delta, 0.2*Delta


dos_SC = spectrum_full(H_tot(Delta, 0, 0.0), [c_up, c_dn], energy, kT, eta)
dos = spectrum_full(H_tot(Delta, 0.5*Delta, 0.0), [c_up, c_dn], energy, kT, eta)

#plt.plot(energy, dos_SC)
#plt.plot(energy, dos)

##### sweep J coupling
Js = np.linspace(0, 4*Delta, 101)
dos_J = np.array([spectrum_full(H_tot(Delta, Jv, 0.0), creators, energy, kT, eta)
                  for Jv in Js])

plt.pcolormesh(Js/Delta, energy/Delta, dos_J.T, rasterized=True)
plt.axvline(4/3, ls="--", c="w")        # QPT at J = 4*Delta/3
plt.xlabel(r"$J_K/\Delta$"); plt.ylabel(r"$E/\Delta$")
plt.colorbar(label="DOS")
plt.show()










