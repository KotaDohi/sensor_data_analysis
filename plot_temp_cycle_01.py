# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 11:47:50 2018

@author: tyair
"""

# 3PLAT の「燃焼ゾーン温度サイクル」と呼ばれる不具合事象を解析
# サイクルが顕著に表れる 7CT3369C.PV, R7CF3369D.PV を1週間分ずつプロット

import numpy as np
import os
import os.path
#import xlrd
import pandas as pd
import sys
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.relativedelta import relativedelta

# プロットしたグラフをpngで保存
#save_fig = True
save_fig = False

############## 環境異存の設定(ディレクトリ等) ###ここから  ######

# オリジナルデータの置き場所 (環境依存)
dir_dat = os.path.join('../data/')
# 作業ディレクトリ(結果を保存するディレクトリ) (環境依存)
dir_cwd=os.path.join('../result/')
#結果を保存するディレクトリ (環境依存)
dir_res=os.path.join(dir_cwd,'res_3plat_plot')
if not os.path.exists(dir_res):
    print('%s does not exist. Creating..' % dir_res)
    os.mkdir(dir_res)

############## 環境異存の設定(ディレクトリ等) ###ここまで  ######

# データを pandas の dataframe オブジェクト df_merged に読み込む
if not 'df_merged' in globals():
    fn_orig = os.path.join(dir_dat,'3plat_merged_3min.h5')
    print('Reading original data')
    df_merged = pd.read_hdf(fn_orig,'df_merged')
    fn_new = os.path.join(dir_dat,'3plat_merged_3min_2.h5')
    print('Reading new data')
    df_merged_new = pd.read_hdf(fn_new,'df_merged_new')
    # マージ
    print('Merging data')
    df_merged = df_merged.join(df_merged_new,how='outer')
else:
    print('Original data has already been read')

#プロットする変数
lst_tags = ['7CT3369C.PV','7CT3369D.PV']
#lst_tags = df_merged.columns[200:300]
ntags = len(lst_tags)

# プロット期間の最初の日時
dt_beg = dt.datetime(2017,8,29)
#dt_beg = df_merged.index[0]

trial = 0

while True:
#for i in range(10):
    # データの最終日時に達したら終了
    if dt_beg > df_merged.index[-1]:
        print('Reached the end')
        break
    # プロットの最後の日時(1週間後)
    dt_end = dt_beg + relativedelta(weeks=1)
    # データが存在する場合
    if df_merged[dt_beg:dt_end][lst_tags].notnull().sum().all():
        # プロット
        plt.figure(num=1,figsize=(16,10))
        plt.clf()
        f,axs = plt.subplots(ntags,sharex=True,num=1)
        for j in range(ntags):
            axs[j].plot(df_merged[dt_beg:dt_end][lst_tags[j]])
            axs[j].set_ylabel(lst_tags[j],fontsize=14)
            axs[j].tick_params(labelsize=14)
        trial+=1
        print("trials",trial)
        if save_fig:
            fn_png = '_'.join(lst_tags) + dt_beg.to_pydatetime().strftime('%Y%m%d') + '.png'
            plt.savefig(os.path.join(dir_res,fn_png))
        #plt.pause(1)
    # 次の開始時刻
    dt_beg = dt_end

data = df_merged[dt.datetime(2017,8,28):dt.datetime(2017,9,5)]['7CT3369D.PV']
data = data.fillna(method='ffill')
data = data.fillna(method='bfill')
data = list(data)

import SSA 
a = SSA.SSA(data,w=5,lag=10,ncol_h=20,ncol_t=20,ns_h=1,ns_t=1)
a.main()

import DMD_single
a = DMD_single.DMD(data,w=100,r=5,ncol=20,fs=1)
res,total = a.main()

#import SPDMD
#a = SPDMD.SPDMD(data,w=100,r=10,ncol=20)
#a.main()

