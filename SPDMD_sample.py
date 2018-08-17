#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 23:47:07 2018

@author: Dohi
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag
from numpy.linalg import inv, eig, pinv, norm, solve, cholesky
from scipy.linalg import svd, svdvals
from scipy.sparse import csc_matrix as sparse
from scipy.sparse import vstack as spvstack
from scipy.sparse import hstack as sphstack
from scipy.sparse.linalg import spsolve

def admm_for_dmd(P, q, s, gamma_vec, rho=1, maxiter=10000, eps_abs=1e-6, eps_rel=1e-4):

    # blank return value
    answer = type('ADMMAnswer', (object,), {})()

    # check input vars
    P = np.squeeze(P)
    q = np.squeeze(q)[:,np.newaxis]
    gamma_vec = np.squeeze(gamma_vec)
    if P.ndim != 2:
        raise ValueError('invalid P')
    if q.ndim != 2:
        raise ValueError('invalid q')
    if gamma_vec.ndim != 1:
        raise ValueError('invalid gamma_vec')

    # number of optimization variables
    n = len(q)

    # identity matrix
    I = np.eye(n)

    # allocate memory for gamma-dependent output variables
    answer.gamma = gamma_vec
    answer.Nz    = np.zeros([len(gamma_vec),]) # number of non-zero amplitudes
    answer.Jsp   = np.zeros([len(gamma_vec),]) # square of Frobenius norm (before polishing)
    answer.Jpol  = np.zeros([len(gamma_vec),]) # square of Frobenius norm (after polishing)
    answer.Ploss = np.zeros([len(gamma_vec),]) # optimal performance loss (after polishing)
    answer.xsp   = np.zeros([n, len(gamma_vec)], dtype='complex') # vector of amplitudes (before polishing)
    answer.xpol  = np.zeros([n, len(gamma_vec)], dtype='complex') # vector of amplitudes (after polishing)

    # Cholesky factorization of matrix P + (rho/2)*I
    Prho = P + (rho/2) * I
    Plow = cholesky(Prho)
    Plow_star = Plow.conj().T

    # sparse P (for KKT system)
    Psparse = sparse(P)

    for i,gamma in enumerate(gamma_vec):

        # initial conditions
        y = np.zeros([n, 1], dtype='complex') # Lagrange multiplier
        z = np.zeros([n, 1], dtype='complex') # copy of x

        # Use ADMM to solve the gamma-parameterized problem  
        for step in range(maxiter):

            # x-minimization step
            u = z - (1/rho) * y
            # x = solve((P + (rho/2) * I), (q + rho * u))
            xnew = solve(Plow_star, solve(Plow, q + (rho/2) * u))

            # z-minimization step       
            a = (gamma/rho) * np.ones([n, 1])
            v = xnew + (1/rho) * y
            # soft-thresholding of v
            znew = multiply(multiply(np.divide(1 - a, np.abs(v)), v), (np.abs(v) > a))

            # primal and dual residuals
            res_prim = norm(xnew - znew, 2)
            res_dual = rho * norm(znew - z, 2)

            # Lagrange multiplier update step
            y += rho * (xnew - znew)

            # stopping criteria
            eps_prim = np.sqrt(n) * eps_abs + eps_rel * np.max([norm(xnew, 2), norm(znew, 2)])
            eps_dual = np.sqrt(n) * eps_abs + eps_rel * norm(y, 2)

            if (res_prim < eps_prim) and (res_dual < eps_dual):
                break
            else:
                z = znew        

        # record output data
        answer.xsp[:,i] = z.squeeze() # vector of amplitudes
        answer.Nz[i] = np.count_nonzero(answer.xsp[:,i]) # number of non-zero amplitudes
        answer.Jsp[i] = (
            np.real(dot(dot(z.conj().T, P), z))
            - 2 * np.real(dot(q.conj().T, z))
            + s) # Frobenius norm (before polishing)

        # polishing of the nonzero amplitudes
        # form the constraint matrix E for E^T x = 0
        ind_zero = np.flatnonzero(np.abs(z) < 1e-12) # find indices of zero elements of z
        m = len(ind_zero) # number of zero elements

        if m > 0:

            # form KKT system for the optimality conditions
            E = I[:,ind_zero]
            E = sparse(E, dtype='complex')
            KKT = spvstack([
                sphstack([Psparse, E], format='csc'),
                sphstack([E.conj().T, sparse((m, m), dtype='complex')], format='csc'),
                ], format='csc')            
            rhs = np.vstack([q, np.zeros([m, 1], dtype='complex')]) # stack vertically

            # solve KKT system
            sol = spsolve(KKT, rhs)
        else:
            sol = solve(P, q)

        # vector of polished (optimal) amplitudes
        xpol = sol[:n]

        # record output datas
        answer.xpol[:,i] = xpol.squeeze()

        # polished (optimal) least-squares residual
        answer.Jpol[i] = (
            np.real(dot(dot(xpol.conj().T, P), xpol))
            - 2 * np.real(dot(q.conj().T, xpol))
            + s)

        # polished (optimal) performance loss 
        answer.Ploss[i] = 100 * np.sqrt(answer.Jpol[i]/s)

        print(i)

    return answer 

# define time and space domains
x = np.linspace(-10, 10, 60)
t = np.linspace(0, 20, 80)
Xm,Tm = np.meshgrid(x, t)

# create data
D1 = 5 * (1/np.cosh(Xm/2)) * np.tanh(Xm/2) * np.exp((0.8j)*Tm) # strong primary mode
D2 = 0.2 * np.sin(2 * Xm) * np.exp(2j * Tm)                    # weak secondary mode
D3 = 0.1 * np.random.randn(*Xm.shape)                          # noise
D = (D1 + D2 + D3).T
    
    
def dmd(X, Y, truncate=None):
    U2,Sig2,Vh2 = svd(X, False) # SVD of input matrix
    r = len(Sig2) if truncate is None else truncate # rank truncation
    U = U2[:,:r]
    Sig = diag(Sig2)[:r,:r]
    V = Vh2.conj().T[:,:r]
    Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig)) # build A tilde
    mu,W = eig(Atil)
    Phi = dot(dot(dot(Y, V), inv(Sig)), W) # build DMD modes
    return mu, Phi

# extract input-output matrices
X = D[:,:-1]
Y = D[:,1:]

# do dmd
r = 30 # new rank
mu,Phi = dmd(X, Y, r)

# compute time evolution (verbose way)
b = dot(pinv(Phi), X[:,0])
Psi = np.zeros([r, len(t)], dtype='complex')
dt = t[2] - t[1]
for i,_t in enumerate(t):
    Psi[:,i] = multiply(np.power(mu, _t/dt), b)
    
# compute time evolution (concise way)
b = dot(pinv(Phi), X[:,0])
Vand = np.vander(mu, len(t), True)
Psi = (Vand.T * b).T # equivalently, Psi = dot(diag(b), Vand)

D_dmd = dot(Phi, Psi)

# vars for the objective function
U,sv,Vh = svd(D, False)
Vand = np.vander(mu, len(t), True)
P = multiply(dot(Phi.conj().T, Phi), np.conj(dot(Vand, Vand.conj().T)))
q = np.conj(diag(dot(dot(Vand, (dot(dot(U, diag(sv)), Vh)).conj().T), Phi)))
s = norm(diag(sv), ord='fro')**2
        
# the optimal solution
b_opt = solve(P, q)

# find optimum solutions
gamma_vec = np.logspace(np.log10(0.05), np.log10(200), 150)
answer = admm_for_dmd(P, q, s, gamma_vec)

b = b_opt
Psi = (Vand.T*b).T
D_dmd = dot(Phi, Psi)




    

