#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:42:07 2018

@author: Dohi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:33:43 2017

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
        b = dot(pinv(Phi), X[:,0])
        Psi = np.zeros([self.r, len(t)], dtype='complex')
        for i,_t in enumerate(t):
            Psi[:,i] = multiply(power(mu, _t/dt), b)
        
        Sig,b,mu,eta = np.diag(Sig),abs(b).real,abs(mu).real,eta.real
        contri=Sig/sum(Sig)*100
        total = pd.DataFrame(np.vstack([freq,b,mu,contri,eta]).T,columns=['freq','b','eig','contribution','eta']).sort_values(by='contribution',ascending=False)     
        self.total = total
        
        #totalの表の中から周波数０以外で、もっともbの高いモードをとってくる
        output = total
#        output = output[output.freq>0]
        output['eta'] = abs(output['eta'])
        output = output.sort_values(by='eta',ascending=False)
        output1 = output.iloc[0]['eta']
        output2 = output.iloc[0]['freq']
        return (output1,output2)

    def main(self):
        start_at = self.w + self.ncol
        end_at = len(self.data) + 1
        res = []
        for t in range(start_at, end_at):
            res = res + [self.calc(self.data[t - self.w - self.ncol:t])]
            print("start_at:",t - self.w - self.ncol)
            #print("end_at:",t)
        eta = [round(eta ,5) for eta, freq in res]
        freq = [round(freq,5) for eta, freq in res]
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

"""
#    filename(input)
DMDA = Analysis()
filename = '../Sounddata2/normal_1200rpm_1'
filename1 = filename+'.csv'
#filename = '/Users/Dohi/Desktop/DMD/DMDprogram/Sounddata/bomb.csv'
Datam= np.array(pd.read_csv(filename1,index_col=0))
print(Datam.shape) 

filename2 = filename+'flist.csv'
f_list= (np.array(pd.read_csv(filename2,index_col=0)))[:len(Datam)]



#パラメータ(T:窓幅、T>rこれはTの間隔でデータを切り出してDMDを適用している。）
T = 10
d = len(Datam.T)/T
print(d)
r = int(len(Datam.T)/d-1)
mx = 8
L = len(Datam)

my = int(len(Datam)/mx)
fs = 44100/1024




#実行
#LでDMD窓幅を変更
j=0
jlist = []
sub = np.linspace(0,len(Datam.T),int(d+1))

if len(sub)>2:
    sub1,sub2 = sub[0:-1],sub[1:]
    for (n0,n1) in zip(sub1,sub2):
        n0 = int(n0)
        n1 = int(n1)
        total,j,jlist = DMDA.main(Datam[:,n0:n1],fs,r,mx,my,f_list,j,jlist)
else:
    total,j,jlist = DMDA.main(Datam[:,:sub[1]],fs,r,mx,my,f_list,j,jlist)


print("時間")
print(np.round(1/fs*len(Datam.T)/d,2),"s")
j=j/len(sub)
print("全体のサンプル数:",len(sub))
print("減衰成分が支配的な割合")
print(np.round(j*100,2),"%")
    

    
    
#異常音の方が明らかに減衰率大のモードの寄与が大きい。


print("減衰率の平均")
print(np.average(jlist))
print("減衰率の分散")
print(np.std(jlist))
#plt.figure()
#plt.hist(jlist,bins=50,range=(0,5),rwidth=0.8)
"""