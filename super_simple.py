#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 21:59:11 2018

@author: Dohi
"""
import numpy as np
import matplotlib.pyplot as plt

T = 1.0
dt = 0.001
t = np.linspace(0,T-dt,int(T/dt))
fs = 10.0
noise_array = np.ones(len(t))
noise = [np.random.rand() for i in range(len(t))]
y = np.sin(fs*t*np.pi*2)*np.exp(-2*t)+noise
plt.plot(t,y)

import DMD_single
w = 780
r = 10
ncol = int(0.25*w)
#a = DMD_single.DMD(y,w,r,ncol,1.0/dt)
#res,total = a.main()


import SPDMD
a = SPDMD.SPDMD(y,w,r,ncol,1.0/dt)
res,total = a.main()

print(total)
print(res)


