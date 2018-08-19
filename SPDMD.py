#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 13:35:34 2018

@author: Dohi
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag, power
from numpy.linalg import inv, eig, pinv, norm, solve, cholesky
from scipy.linalg import svd, svdvals
from scipy.sparse import csc_matrix as sparse
from scipy.sparse import vstack as spvstack
from scipy.sparse import hstack as sphstack
from scipy.sparse.linalg import spsolve
from itertools import islice



class SPDMD(object):
    
    def __init__(self,data,w,r,ncol,fs):
        self.data = data
        self.w = w
        self.r = r
        self.ncol = ncol
        self.fs = fs
        self.fill=True
    
    def window(self,seq, n):
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result
    
    def calc(self,data):
        data = np.array(
            tuple(x[:self.ncol] for x in self.window(data, self.w))[:self.w]
        )
        dt = 1/self.fs
        X,Y= data[:,:-1],data[:,1:]
        U,Sig,Vh = svd(X, False)
        U,Sig,V = U[:,:self.r],np.diag(Sig)[:self.r,:self.r],Vh.conj().T[:,:self.r]
        # freq
        mu,W = eig(dot(dot(dot(U.conj().T, Y), V), inv(Sig)))
        freq = abs((np.log(mu)/(2.0*np.pi*1.0/self.fs)).imag).real
        eta  = np.log(abs(mu))*self.fs
        #Psi
        t = np.linspace(0,1.0/self.fs*len(data.T),len(data.T))
        Phi = dot(dot(dot(Y, V), inv(Sig)), W)
        
        U,sv,Vh = svd(data, False)
        Vand = np.vander(mu, len(t), True)
        P = multiply(dot(Phi.conj().T, Phi), np.conj(dot(Vand, Vand.conj().T)))
        q = np.conj(diag(dot(dot(Vand, (dot(dot(U, diag(sv)), Vh)).conj().T), Phi)))
        s = norm(diag(sv), ord='fro')**2
#change here         
        gamma_vec = np.logspace(np.log10(0.05), np.log10(1000), 2)
        answer = self.admm_for_dmd(P, q, s, gamma_vec)
#change here
        b = answer.xpol[:,0]
        Psi = (Vand.T*b).T

        for i,_t in enumerate(t):
            Psi[:,i] = multiply(power(mu, _t/dt), b)
        
        Sig,b,mu,eta = np.diag(Sig),abs(b).real,abs(mu).real,eta.real
        contri=Sig/sum(Sig)*100
        total = pd.DataFrame(np.vstack([freq,b,mu,contri,eta]).T,columns=['freq','b','eig','contribution','eta']).sort_values(by='contribution',ascending=False)     
        self.total = total
        
        #totalの表の中から周波数０以外で、もっともbの高いモードをとってくる
        output = total
        output = output[output.freq>0]
#        output['eta'] = abs(output['eta'])
        output = output.sort_values(by='b',ascending=False)
        output1 = output.iloc[0]['eta']
        output2 = output.iloc[0]['freq']
        return (output1,output2)
    
    
    def admm_for_dmd(self,P, q, s, gamma_vec, rho=1, maxiter=10000, eps_abs=1e-6, eps_rel=1e-4):
    
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
        return answer 

    def main(self):
        start_at = self.w + self.ncol
        end_at = len(self.data) + 1
        res = []
        for t in range(start_at, end_at):
            res = res + [self.calc(self.data[t - self.w - self.ncol:t])]
            print("start_at:",t - self.w - self.ncol)
            #print("end_at:",t)
        eta = [round(eta, 5) for eta, freq in res]
        freq = [round(freq, 5) for eta, freq in res]
        if self.fill == True:
            eta = [np.nan] * (start_at - 1) + eta
            freq = [np.nan] * (start_at - 1) + freq
        sst = pd.DataFrame(data={'time':range(len(self.data)),
                                 'original':self.data,
                                 }).set_index('time')
        
        sst['s={}'.format("decay")] = pd.Series(eta)
        sst['s={}'.format("freq")] = pd.Series(freq)
        sst.plot(subplots=True, figsize=(10, sst.shape[1] * 3))
        return res,self.total



