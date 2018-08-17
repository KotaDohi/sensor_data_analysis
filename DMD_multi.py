#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:35:30 2018

@author: Dohi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import dot,multiply,power
from numpy.linalg import inv,pinv
from scipy.linalg import svd,eig
from itertools import islice


class DMD(object):
    
    def __init__(self,data,data_sp,r,ncol):
        self.data = data
        self.data_sp = data_sp
        self.r = r
        self.ncol = ncol
        self.fs = 1
        self.fill=True
    
    
    def calc(self,data):
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
        b = dot(pinv(Phi), X[:,0])
        Psi = np.zeros([self.r, len(t)], dtype='complex')
        for i,_t in enumerate(t):
            Psi[:,i] = multiply(power(mu, _t/dt), b)
        
        Sig,b,mu,eta = np.diag(Sig),abs(b).real,abs(mu).real,eta.real
        contri=Sig/sum(Sig)*100
        total = pd.DataFrame(np.vstack([freq,b,mu,contri,eta]).T,columns=['freq','b','eig','contribution','eta']).sort_values(by='contribution',ascending=False)     
        self.total = total
        #print(total['eta'][0])
        return (total['eta'][0],total['freq'][0])

    def main(self):
        start_at = self.ncol
        end_at = len(self.data.T) + 1
        res = []
        for t in range(start_at, end_at):
            res = res + [self.calc(self.data[:,t-self.ncol:t])]
        eta = [round(eta, 14) for eta, freq in res]
        freq = [round(freq, 14) for eta, freq in res]
        if self.fill == True:
            eta = [np.nan] * (start_at - 1) + eta
            freq = [np.nan] * (start_at - 1) + freq
        sst = pd.DataFrame(data={'time':range(len(self.data.T)),
                                 'original':self.data_sp,
                                 }).set_index('time')
        
        sst['s={}'.format("decay")] = pd.Series(eta)
        sst['s={}'.format("freq")] = pd.Series(freq)
        sst.plot(subplots=True, figsize=(10, sst.shape[1] * 3))
        return res,self.total