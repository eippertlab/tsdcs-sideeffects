#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description:
------------
This script analyzes the spontaneous skin conductance fluctuations.
First we cut the data according to the information saved in the previous step.
The data is then down-sampled to 100 Hz and filtered with a first-order
Butterworth bandpass filter (0.0159 & 5 Hz).
SCF are then quantified via an area under the curve (AUC) approach:
first interpolating over all local minima of the time series and
second determining the area between this baseline signal and the time series.
For each subject and session the AUC value for the whole interval as well as
for the 4 quartiles are saved in a table.
In the unblinding step the session-condition mapping is read and
outliers are excluded. The resulting table AUC_unblind.csv can be used
for statistical analysis and further plotting.
The within-subject plot (fig 4A) is created.

Authors:
--------
Ulrike Horn

Contact:
--------
uhorn@cbs.mpg.de

Date:
-----
20th November 2023
"""

import mne
import numpy as np
import pandas as pd
import os
import glob
import json
import scipy.interpolate
from scipy import signal, stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
from numpy import trapz

raw_path = '/data/pt_02582/tsDCS_BIDS/'
output_path = '/data/pt_02582/tsDCS_processed/'
result_path = '/data/pt_02582/tsDCS_results/SCR/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

np.random.seed(1990)  # for jitter (reproducible figures)

new_sr = 100  # Hz that data will be downsampled to

tasks = ['tsDCS']

subjects = ['SR01', 'SR02', 'SR03', 'SR04', 'SR05', 'SR06', 'SR07', 'SR08', 'SR09', 'SR10',
            'SR11', 'SR12', 'SR13', 'SR14', 'SR15', 'SR16', 'SR17', 'SR18', 'SR19', 'SR20']

sessions = [1, 2, 3]

preprocess = True
save_markers = True
unblinding = True
within_sub_plot = True

color_blue = [0, 128/255, 255/255]
color_red = [227/255, 0, 15/255]
color_orange = [239/255, 138/255, 16/255]

auc_all = []
auc_1 = []
auc_2 = []
auc_3 = []
auc_4 = []
for isub in subjects:
    sub = 'sub-' + isub
    print('subject {}'.format(isub))

    for ses in sessions:
        for task in tasks:
            physio_files = glob.glob(raw_path + sub + os.sep + 'ses-' + str(ses) + os.sep + 'beh' + os.sep + '*' + task + '*_emg.vhdr')
            physio_files = np.array(sorted(physio_files))
            if len(physio_files) > 1:
                print('Warning! More than one file found!')

            sub_out_path = output_path + sub + os.sep + 'ses-' + str(ses) + os.sep + 'SCR'
            if not os.path.exists(sub_out_path):
                os.makedirs(sub_out_path)

            if preprocess:
                p = physio_files[0]
                # import raw data with MNE
                raw = mne.io.read_raw_brainvision(p, preload=True)
                sr = raw.info['sfreq']
                # pick skin conductance channel (if name is not correct check with print(raw.info) what the name is)
                raw.pick_channels(ch_names=["GSR_MR_100_xx"])
                raw.set_channel_types(mapping={"GSR_MR_100_xx": 'eeg'})  # convert channel type from MISC to eeg

                # get events file
                tmp = p.split('_emg')
                event_file = tmp[0] + '_events.tsv'
                events = pd.read_table(event_file, sep='\t')

                # cut data from manually inserted event
                onset = events.loc[events['value'] == 'manual', 'onset'].values[0]
                duration = events.loc[events['value'] == 'manual', 'duration'].values[0]
                if onset + duration >= len(raw) / sr:
                    raw.crop(tmin=onset, tmax=None, include_tmax=True)
                else:
                    raw.crop(tmin=onset, tmax=onset + duration, include_tmax=True)

                # downsampling to 100 Hz
                gsr_downsampled = raw.resample(new_sr, npad="auto")

                # 1st order butterworth filter (iir) band-pass with 0.0159 and 5 Hz
                l_freq = 0.0159  # for iir filter the lower cutoff frequency
                h_freq = 5  # for iir filter the upper cutoff frequency
                iir_params = dict(order=1, ftype='butter', output='sos')
                gsr_filtered = gsr_downsampled.filter(l_freq=l_freq, h_freq=h_freq, picks=["GSR_MR_100_xx"],
                                                      method="iir",
                                                      iir_params=iir_params)
                filt = gsr_filtered.get_data()[0]
                my_data = filt * 1e6  # convert unit: V -> µS

                # find local minima
                all_mins = []
                win_size = 20
                num_win = len(my_data)/new_sr/win_size
                for win in range(0, int(np.ceil(num_win))):
                    tmp = my_data[win*win_size*new_sr:(win+1)*win_size*new_sr]
                    # prefer real minima (with smaller values left and right)
                    m, _ = scipy.signal.find_peaks(-tmp, distance=win_size*new_sr)
                    if len(m) > 0:
                        my_min = m[0] + win*win_size*new_sr
                    else:
                        my_min = np.argmin(tmp) + win * win_size * new_sr
                    all_mins.append(my_min)
                minima = np.unique(all_mins)
                # add first and last data point to interpolate in between
                x = np.concatenate(([0], minima, [len(my_data)-1]))
                f = scipy.interpolate.interp1d(x, my_data[x], kind='slinear', fill_value='extrapolate')
                x_new = np.arange(0, len(my_data), 1)
                y_new = f(x_new)

                # area between the two weirdly shaped curves
                z = my_data - y_new
                dx = x_new[1:] - x_new[:-1]
                cross_test = np.sign(z[:-1] * z[1:])
                x_intersect = x_new[:-1] - dx / (z[1:] - z[:-1]) * z[:-1]
                dx_intersect = - dx / (z[1:] - z[:-1]) * z[:-1]
                areas_pos = abs(z[:-1] + z[1:]) * 0.5 * dx  # signs of both z are same
                areas_neg = 0.5 * dx_intersect * abs(z[:-1]) + 0.5 * (dx - dx_intersect) * abs(z[1:])
                areas = np.where(cross_test < 0, areas_neg, areas_pos)
                total_area = np.sum(areas)
                auc_all.append(total_area)

                # same for 5 minute blocks
                len_quart = int(np.floor(len(areas)/4))
                area_1 = np.sum(areas[0:len_quart])
                area_2 = np.sum(areas[len_quart:2 * len_quart])
                area_3 = np.sum(areas[2 * len_quart:3 * len_quart])
                area_4 = np.sum(areas[3 * len_quart:])
                auc_1.append(area_1)
                auc_2.append(area_2)
                auc_3.append(area_3)
                auc_4.append(area_4)

if save_markers:
    rep_subjects = subjects * 3
    rep_subjects_prefix = ['sub-' + subject for subject in rep_subjects]
    markers = pd.DataFrame({'AUC': auc_all, 'AUC_Q1': auc_1, 'AUC_Q2': auc_2, 'AUC_Q3': auc_3, 'AUC_Q4': auc_4,
                            'Subject': rep_subjects_prefix,
                            'Session': np.resize(np.arange(1, 4), len(subjects)*3)})
    markers.to_csv(result_path + 'AUC.csv', sep=',', index=False)

if unblinding:
    def adjust_box_widths(g, fac):
        """
        Adjust the withs of a seaborn-generated boxplot.
        """

        # iterating through Axes instances
        for ax in g.axes:

            # iterating through axes artists:
            for c in ax.get_children():

                # searching for PathPatches
                if isinstance(c, PathPatch):
                    # getting current width of box:
                    p = c.get_path()
                    verts = p.vertices
                    verts_sub = verts[:-1]
                    xmin = np.min(verts_sub[:, 0])
                    xmax = np.max(verts_sub[:, 0])
                    xmid = 0.5 * (xmin + xmax)
                    xhalf = 0.5 * (xmax - xmin)

                    # setting new width of box
                    xmin_new = xmid - fac * xhalf
                    xmax_new = xmid + fac * xhalf
                    verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                    verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                    # setting new width of median line
                    for l in ax.lines:
                        if np.all(l.get_xdata() == [xmin, xmax]):
                            l.set_xdata([xmin_new, xmax_new])

    df = pd.read_csv(result_path + 'AUC.csv', sep=',', header=0)
    rand_df = pd.read_csv('/data/pt_02582/tsDCS_BIDS/participants.tsv', sep='\t')

    def replace_cond(row):
        this_sub = row['Subject']
        this_rand = rand_df[rand_df['participant_id'] == this_sub]
        if row['Session'] == 1:
            val = this_rand['condition_ses-1'].values[0]
        elif row['Session'] == 2:
            val = this_rand['condition_ses-2'].values[0]
        elif row['Session'] == 3:
            val = this_rand['condition_ses-3'].values[0]
        else:
            print('something weird happened')
        return val

    df['Condition'] = df.apply(replace_cond, axis=1)

    # exclude the subjects' sessions that we decided on:
    df.loc[(df['Subject'] == 'sub-SR01') & (df['Session'] == 2), ['AUC', 'AUC_Q1', 'AUC_Q2', 'AUC_Q3', 'AUC_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR08') & (df['Session'] == 2), ['AUC', 'AUC_Q1', 'AUC_Q2', 'AUC_Q3', 'AUC_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR09'), ['AUC', 'AUC_Q1', 'AUC_Q2', 'AUC_Q3', 'AUC_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR10') & (df['Session'] == 3), ['AUC', 'AUC_Q1', 'AUC_Q2', 'AUC_Q3', 'AUC_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR13'), ['AUC', 'AUC_Q1', 'AUC_Q2', 'AUC_Q3', 'AUC_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR14'), ['AUC', 'AUC_Q1', 'AUC_Q2', 'AUC_Q3', 'AUC_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR16') & (df['Session'] == 3), ['AUC', 'AUC_Q1', 'AUC_Q2', 'AUC_Q3', 'AUC_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR17') & (df['Session'] == 3), ['AUC', 'AUC_Q1', 'AUC_Q2', 'AUC_Q3', 'AUC_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR19') & (df['Session'] == 1), ['AUC', 'AUC_Q1', 'AUC_Q2', 'AUC_Q3', 'AUC_Q4']] = np.nan

    # make a plot
    sns.set_context("talk")
    hue_order = ['A', 'C', 'B']
    fig, axes = plt.subplots(1, 1)
    fig.suptitle('Skin conductance AUC', fontsize=16)
    sns.boxplot(df, x='Condition', y='AUC', order=hue_order,
                boxprops={'alpha': 0.4}, showfliers=False)
    sns.stripplot(data=df, x="Condition", y="AUC", order=hue_order, hue='Condition', hue_order=hue_order,
                  dodge=False)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles=handles, labels=['A (Anodal)', 'C (Cathodal)', 'B (Sham)'], loc='upper left')
    plt.show()

    # create tables for the stats
    df = df.drop(['Session'], axis=1)
    wide_df = df.pivot(index='Subject', columns='Condition', values=['AUC', 'AUC_Q1', 'AUC_Q2', 'AUC_Q3', 'AUC_Q4'])
    wide_df.columns = ['_'.join(col).strip() for col in wide_df.columns.values]
    wide_df.reset_index('Subject', inplace=True)
    wide_df.to_csv(result_path + 'AUC_unblind.csv', sep=',', index=None)

if within_sub_plot:
    df = pd.read_csv(result_path + 'AUC_unblind.csv', sep=',')

    # cathodal comparison
    cathodal = df.drop(labels=['AUC_A', 'AUC_Q1_A', 'AUC_Q2_A', 'AUC_Q3_A', 'AUC_Q4_A'], axis=1)
    cathodal.dropna(axis=0, inplace=True)
    cathodal.reset_index(inplace=True, drop=True)
    cathodal.drop('Subject', axis=1, inplace=True)

    sns.set_context("poster")
    sns.set_style("white")
    jitter = 0.05
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=(cathodal.values.shape[0], 2)),
                               columns=['Sham', 'Cathodal'])
    df_x_jitter += np.arange(2)
    fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    fig.suptitle("Skin conductance fluctuations", fontsize=30)
    b = sns.boxplot(ax=ax[0], x=np.repeat(0, len(cathodal)), y=cathodal['AUC_B'], color=color_blue, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in b.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    c = sns.boxplot(ax=ax[0], x=np.repeat(1, len(cathodal)), y=cathodal['AUC_C'], color=color_red, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in c.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    df_sham = pd.DataFrame({'jitter': df_x_jitter['Sham'], 'auc': cathodal['AUC_B']})
    df_cath = pd.DataFrame({'jitter': df_x_jitter['Cathodal'], 'auc': cathodal['AUC_C']})
    sns.scatterplot(ax=ax[0], data=df_sham, x='jitter', y='auc', color=color_blue, zorder=100, edgecolor="black")
    sns.scatterplot(ax=ax[0], data=df_cath, x='jitter', y='auc', color=color_red, zorder=100, edgecolor="black")
    ax[0].set_xticks(range(2))
    ax[0].set_xticklabels(['Sham', 'Cathodal'])
    ax[0].tick_params(axis='x', which='major', labelsize=25)
    ax[0].set_xlim(-1.0, 2)
    sns.despine()
    ax[0].set_xlabel('')
    ax[0].set_ylabel('AUC', fontsize=25)
    plt.subplots_adjust(bottom=0.2, left=0.15)
    ax[0].set_title('Cathodal', fontsize=25)
    for idx in cathodal.index:
        ax[0].plot(df_x_jitter.loc[idx, ['Sham', 'Cathodal']], cathodal.loc[idx, ['AUC_B', 'AUC_C']], color='grey',
                   linewidth=0.5, linestyle='--', zorder=-1)

    # anodal comparison
    anodal = df.drop(labels=['AUC_C', 'AUC_Q1_C', 'AUC_Q2_C', 'AUC_Q3_C', 'AUC_Q4_C'], axis=1)
    anodal.dropna(axis=0, inplace=True)
    anodal.reset_index(inplace=True, drop=True)
    anodal.drop('Subject', axis=1, inplace=True)

    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=(anodal.values.shape[0], 2)),
                               columns=['Sham', 'Anodal'])
    df_x_jitter += np.arange(2)
    b = sns.boxplot(ax=ax[1], x=np.repeat(0, len(anodal)), y=anodal['AUC_B'], color=color_blue, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in b.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    c = sns.boxplot(ax=ax[1], x=np.repeat(1, len(anodal)), y=anodal['AUC_A'], color=color_orange, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in c.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    df_sham = pd.DataFrame({'jitter': df_x_jitter['Sham'], 'auc': anodal['AUC_B']})
    df_anod = pd.DataFrame({'jitter': df_x_jitter['Anodal'], 'auc': anodal['AUC_A']})
    sns.scatterplot(ax=ax[1], data=df_sham, x='jitter', y='auc', color=color_blue, zorder=100, edgecolor="black")
    sns.scatterplot(ax=ax[1], data=df_anod, x='jitter', y='auc', color=color_orange, zorder=100, edgecolor="black")
    ax[1].set_xticks(range(2))
    ax[1].set_xticklabels(['Sham', 'Anodal'])
    ax[1].tick_params(axis='x', which='major', labelsize=25)
    ax[1].set_xlim(-1.0, 2)
    sns.despine()
    ax[1].set_xlabel('')
    ax[1].set_ylabel('AUC', fontsize=25)
    plt.subplots_adjust(bottom=0.2, left=0.15)
    ax[1].set_title('Anodal', fontsize=25)
    for idx in anodal.index:
        ax[1].plot(df_x_jitter.loc[idx, ['Sham', 'Anodal']], anodal.loc[idx, ['AUC_B', 'AUC_A']], color='grey',
                   linewidth=0.5, linestyle='--', zorder=-1)

    plt.savefig(result_path + 'SCR_AUC_within_sub.png')
    plt.show()
