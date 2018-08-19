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
lst_tags = ['7CT3369D.PV','7CAC3383.PV']

ntags = len(lst_tags)

# プロット期間の最初の日時
week = 1
dt_begin = dt.datetime(2017,8,29)

for week in range(week):
    dt_beg = dt_begin - relativedelta(weeks=week)
    
    # プロットの最後の日時(1週間後)
    dt_end = dt_beg + relativedelta(weeks=1)
    plt.figure(num=1,figsize=(16,10))
    plt.clf()
    f,axs = plt.subplots(ntags,sharex=True,num=1)
    for j in range(ntags):
        axs[j].plot(df_merged[dt_beg:dt_end][lst_tags[j]])
        axs[j].set_ylabel(lst_tags[j],fontsize=14)
        axs[j].tick_params(labelsize=14)
    

