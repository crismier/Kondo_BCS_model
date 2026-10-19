#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 19:44:29 2026

@author: cristinamiergonzalez
"""

import numpy as np
from qutip import destroy, qeye, sigmaz, tensor

a, I, Z = destroy(2), qeye(2), sigmaz()

# Fermion operators for three sites (ordering: 1up,1dn,2up,2dn,3up,3dn)
c1_up = tensor(a, I, I, I, I, I)
c1_dn = tensor(Z, a, I, I, I, I)
c2_up = tensor(Z, Z, a, I, I, I)
c2_dn = tensor(Z, Z, Z, a, I, I)
c3_up = tensor(Z, Z, Z, Z, a, I)
c3_dn = tensor(Z, Z, Z, Z, Z, a)

sites_up = [c1_up, c2_up, c3_up]
sites_dn = [c1_dn, c2_dn, c3_dn]

n_up = [c.dag()*c for c in sites_up]
n_dn = [c.dag()*c for c in sites_dn]

def H_3sites(xi, Delta, t=None):
    """
    xi, Delta: length-3 lists/arrays, on-site energy and pairing per site.
    t: optional 2x2-ish nearest-neighbor hopping list [t12, t23] (chain),
       or None for no hopping.
    """
    H_level = sum(xi[i]*(n_up[i] + n_dn[i]) for i in range(3))
    H_pair  = sum(Delta[i]*(sites_up[i].dag()*sites_dn[i].dag()
                            + sites_dn[i]*sites_up[i]) for i in range(3))
    H_hop = 0
    if t is not None:
        pairs = [(0,1), (1,2)]           # chain: 1-2, 2-3
        for (i, j), tij in zip(pairs, t):
            H_hop += tij*(sites_up[i].dag()*sites_up[j] + sites_up[j].dag()*sites_up[i]
                         + sites_dn[i].dag()*sites_dn[j] + sites_dn[j].dag()*sites_dn[i])
    return H_level + H_pair + H_hop


#%%

H = H_3sites(xi=[0.1, 0.0, -0.1], Delta=[1.0, 0.8, 1.0], t=[0.3, 0.3])
print(H.shape)   # (64, 64)
assert H.isherm