#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description:
------------
This script loads all saved markers before unblinding to determine the outliers.
In some cases participants had inadvertently not been instructed not to talk and move
during tsDCS administration, leading to abnormal signal fluctuations in these participants' autonomic measures.
Here we plot heart rate, breathing variability and skin conductance fluctations and mark the excluded participants.

Authors:
--------
Ulrike Horn

Contact:
--------
uhorn@cbs.mpg.de

Date:
-----
10th July 2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# combine all markers
df_scr = pd.read_csv('/data/pt_02582/tsDCS_results/SCR/AUC.csv', sep=',', header=0)
df_card = pd.read_csv('/data/pt_02582/tsDCS_results/ECG/HR_HRV.csv', sep=',', header=0)
df_resp = pd.read_csv('/data/pt_02582/tsDCS_results/Resp/BPM.csv', sep=',', header=0)

df = df_scr.merge(df_card, left_on=['Subject', 'Session'],
                  right_on=['Subject', 'Session']).merge(df_resp, left_on=['Subject', 'Session'], right_on=['Subject', 'Session'])

fig, axes = plt.subplots(3, 1, figsize=(18, 8))
stops = np.arange(start=0, step=3, stop=60)
for i in stops:
    axes[0].axvline(x=i, lw=1, color='darkgray')
stops = np.arange(start=1, step=3, stop=60)
for i in stops:
    axes[0].axvline(x=i, lw=1, ls='--', color='darkgray')
stops = np.arange(start=2, step=3, stop=60)
for i in stops:
    axes[0].axvline(x=i, lw=1, ls=':', color='darkgray')
axes[0].plot(df['HR'], label='heart rate')
axes[0].legend(loc='upper right')
axes[0].set_xticks(np.arange(start=0, stop=60, step=1))
axes[0].set_xticklabels(['01', '', '', '02', '', '', '03', '', '', '04', '', '', '05', '', '', '06', '', '',
                         '07', '', '', '08', '', '', '09', '', '', '10', '', '', '11', '', '', '12', '', '',
                         '13', '', '', '14', '', '', '15', '', '', '16', '', '', '17', '', '', '18', '', '',
                         '19', '', '', '20', '', ''])
axes[0].annotate("", xy=(1, 100), xytext=(1, 110), arrowprops=dict(arrowstyle="->"))  # sub 01 sess 2
axes[0].annotate("", xy=(22, 100), xytext=(22, 110), arrowprops=dict(arrowstyle="->"))  # sub 08 sess 2
axes[0].annotate("", xy=(24, 100), xytext=(24, 110), arrowprops=dict(arrowstyle="->"))  # sub 09 sess 1
axes[0].annotate("", xy=(25, 100), xytext=(25, 110), arrowprops=dict(arrowstyle="->"))  # sub 09 sess 2
axes[0].annotate("", xy=(26, 100), xytext=(26, 110), arrowprops=dict(arrowstyle="->"))  # sub 09 sess 3
axes[0].annotate("", xy=(29, 100), xytext=(29, 110), arrowprops=dict(arrowstyle="->"))  # sub 10 sess 3
axes[0].annotate("", xy=(36, 100), xytext=(36, 110), arrowprops=dict(arrowstyle="->"))  # sub 13 sess 1
axes[0].annotate("", xy=(37, 100), xytext=(37, 110), arrowprops=dict(arrowstyle="->"))  # sub 13 sess 2
axes[0].annotate("", xy=(38, 100), xytext=(38, 110), arrowprops=dict(arrowstyle="->"))  # sub 13 sess 3
axes[0].annotate("", xy=(39, 100), xytext=(39, 110), arrowprops=dict(arrowstyle="->"))  # sub 14 sess 1
axes[0].annotate("", xy=(40, 100), xytext=(40, 110), arrowprops=dict(arrowstyle="->"))  # sub 14 sess 2
axes[0].annotate("", xy=(41, 100), xytext=(41, 110), arrowprops=dict(arrowstyle="->"))  # sub 14 sess 3
axes[0].annotate("", xy=(47, 100), xytext=(47, 110), arrowprops=dict(arrowstyle="->"))  # sub 16 sess 3
axes[0].annotate("", xy=(50, 100), xytext=(50, 110), arrowprops=dict(arrowstyle="->"))  # sub 17 sess 3
axes[0].annotate("", xy=(54, 100), xytext=(54, 110), arrowprops=dict(arrowstyle="->"))  # sub 19 sess 1

# -----------------------------------------------
stops = np.arange(start=0, step=3, stop=60)
for i in stops:
    axes[1].axvline(x=i, lw=1, color='darkgray')
stops = np.arange(start=1, step=3, stop=60)
for i in stops:
    axes[1].axvline(x=i, lw=1, ls='--', color='darkgray')
stops = np.arange(start=2, step=3, stop=60)
for i in stops:
    axes[1].axvline(x=i, lw=1, ls=':', color='darkgray')
axes[1].plot(df['BDev'], label='breathing variability')
axes[1].legend(loc='upper right')
axes[1].set_xticks(np.arange(start=0, stop=60, step=1))
axes[1].set_xticklabels(['01', '', '', '02', '', '', '03', '', '', '04', '', '', '05', '', '', '06', '', '',
                         '07', '', '', '08', '', '', '09', '', '', '10', '', '', '11', '', '', '12', '', '',
                         '13', '', '', '14', '', '', '15', '', '', '16', '', '', '17', '', '', '18', '', '',
                         '19', '', '', '20', '', ''])
lower = 2.2
upper = 2.6
axes[1].annotate("", xy=(1, lower), xytext=(1, upper), arrowprops=dict(arrowstyle="->"))  # sub 01 sess 2
axes[1].annotate("", xy=(22, lower), xytext=(22, upper), arrowprops=dict(arrowstyle="->"))  # sub 08 sess 2
axes[1].annotate("", xy=(24, lower), xytext=(24, upper), arrowprops=dict(arrowstyle="->"))  # sub 09 sess 1
axes[1].annotate("", xy=(25, lower), xytext=(25, upper), arrowprops=dict(arrowstyle="->"))  # sub 09 sess 2
axes[1].annotate("", xy=(26, lower), xytext=(26, upper), arrowprops=dict(arrowstyle="->"))  # sub 09 sess 3
axes[1].annotate("", xy=(29, lower), xytext=(29, upper), arrowprops=dict(arrowstyle="->"))  # sub 10 sess 3
axes[1].annotate("", xy=(36, lower), xytext=(36, upper), arrowprops=dict(arrowstyle="->"))  # sub 13 sess 1
axes[1].annotate("", xy=(37, lower), xytext=(37, upper), arrowprops=dict(arrowstyle="->"))  # sub 13 sess 2
axes[1].annotate("", xy=(38, lower), xytext=(38, upper), arrowprops=dict(arrowstyle="->"))  # sub 13 sess 3
axes[1].annotate("", xy=(39, lower), xytext=(39, upper), arrowprops=dict(arrowstyle="->"))  # sub 14 sess 1
axes[1].annotate("", xy=(40, lower), xytext=(40, upper), arrowprops=dict(arrowstyle="->"))  # sub 14 sess 2
axes[1].annotate("", xy=(41, lower), xytext=(41, upper), arrowprops=dict(arrowstyle="->"))  # sub 14 sess 3
axes[1].annotate("", xy=(47, lower), xytext=(47, upper), arrowprops=dict(arrowstyle="->"))  # sub 16 sess 3
axes[1].annotate("", xy=(50, lower), xytext=(50, upper), arrowprops=dict(arrowstyle="->"))  # sub 17 sess 3
axes[1].annotate("", xy=(54, lower), xytext=(54, upper), arrowprops=dict(arrowstyle="->"))  # sub 19 sess 1
# -----------------------------------------------
stops = np.arange(start=0, step=3, stop=60)
for i in stops:
    axes[2].axvline(x=i, lw=1, color='darkgray')
stops = np.arange(start=1, step=3, stop=60)
for i in stops:
    axes[2].axvline(x=i, lw=1, ls='--', color='darkgray')
stops = np.arange(start=2, step=3, stop=60)
for i in stops:
    axes[2].axvline(x=i, lw=1, ls=':', color='darkgray')
axes[2].plot(df['AUC'], label='SCF AUC')
axes[2].legend(loc='upper right')
axes[2].set_xticks(np.arange(start=0, stop=60, step=1))
axes[2].set_xticklabels(['01', '', '', '02', '', '', '03', '', '', '04', '', '', '05', '', '', '06', '', '',
                         '07', '', '', '08', '', '', '09', '', '', '10', '', '', '11', '', '', '12', '', '',
                         '13', '', '', '14', '', '', '15', '', '', '16', '', '', '17', '', '', '18', '', '',
                         '19', '', '', '20', '', ''])
lower = 65000
upper = 80000
axes[2].annotate("", xy=(1, lower), xytext=(1, upper), arrowprops=dict(arrowstyle="->"))  # sub 01 sess 2
axes[2].annotate("", xy=(22, lower), xytext=(22, upper), arrowprops=dict(arrowstyle="->"))  # sub 08 sess 2
axes[2].annotate("", xy=(24, lower), xytext=(24, upper), arrowprops=dict(arrowstyle="->"))  # sub 09 sess 1
axes[2].annotate("", xy=(25, lower), xytext=(25, upper), arrowprops=dict(arrowstyle="->"))  # sub 09 sess 2
axes[2].annotate("", xy=(26, lower), xytext=(26, upper), arrowprops=dict(arrowstyle="->"))  # sub 09 sess 3
axes[2].annotate("", xy=(29, lower), xytext=(29, upper), arrowprops=dict(arrowstyle="->"))  # sub 10 sess 3
axes[2].annotate("", xy=(36, lower), xytext=(36, upper), arrowprops=dict(arrowstyle="->"))  # sub 13 sess 1
axes[2].annotate("", xy=(37, lower), xytext=(37, upper), arrowprops=dict(arrowstyle="->"))  # sub 13 sess 2
axes[2].annotate("", xy=(38, lower), xytext=(38, upper), arrowprops=dict(arrowstyle="->"))  # sub 13 sess 3
axes[2].annotate("", xy=(39, lower), xytext=(39, upper), arrowprops=dict(arrowstyle="->"))  # sub 14 sess 1
axes[2].annotate("", xy=(40, lower), xytext=(40, upper), arrowprops=dict(arrowstyle="->"))  # sub 14 sess 2
axes[2].annotate("", xy=(41, lower), xytext=(41, upper), arrowprops=dict(arrowstyle="->"))  # sub 14 sess 3
axes[2].annotate("", xy=(47, lower), xytext=(47, upper), arrowprops=dict(arrowstyle="->"))  # sub 16 sess 3
axes[2].annotate("", xy=(50, lower), xytext=(50, upper), arrowprops=dict(arrowstyle="->"))  # sub 17 sess 3
axes[2].annotate("", xy=(54, lower), xytext=(54, upper), arrowprops=dict(arrowstyle="->"))  # sub 19 sess 1

fig.suptitle("Overview subjects to exclude")
plt.show()
