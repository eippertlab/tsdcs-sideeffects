#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Description:
------------
This script analyzes the respiratory activity.
First we cut the data according to the information saved using a_tsdcs_UE_analysis_prep.py.
The time points that mark the beginning of a new breathing cycle are automatically detected
as signal minima represent maximum inhalation.
We then extract the breathing rate (breaths per minute) and breathing rate variability
(standard deviation of the interval between consecutive breaths) over the whole interval
and within the 4 quartiles.
In the unblinding step the session-condition mapping is read and
outliers are excluded. The resulting table BR_BDev_unblind.csv can be used
for statistical analysis and further plotting.
The within-subject plots (fig 4D and 4E) are created.
Additionally, an interaction plot (fig 4F) is created.

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
from scipy import signal, stats
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

raw_path = '/data/pt_02582/tsDCS_BIDS/'
output_path = '/data/pt_02582/tsDCS_processed/'
result_path = '/data/pt_02582/tsDCS_results/Resp/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

if not os.path.exists(result_path):
    os.makedirs(result_path)

new_sr = 100  # Hz that data will be downsampled to

tasks = ['tsDCS']

subjects = ['SR01', 'SR02', 'SR03', 'SR04', 'SR05', 'SR06', 'SR07', 'SR08', 'SR09', 'SR10',
            'SR11', 'SR12', 'SR13', 'SR14', 'SR15', 'SR16', 'SR17', 'SR18', 'SR19', 'SR20']

sessions = [1, 2, 3]

preprocess = True
save_markers = True
unblinding = True
within_sub_plot = True
interaction_plot = True

np.random.seed(1990)  # for jitter

color_blue = [0, 128/255, 255/255]
color_red = [227/255, 0, 15/255]
color_orange = [239/255, 138/255, 16/255]

new_rc_params = {"font.family": 'Arial', "font.size": 12, "font.serif": [],
                 "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)

bpm_all = []
dev_all = []
bpm_Q1 = []
dev_Q1 = []
bpm_Q2 = []
dev_Q2 = []
bpm_Q3 = []
dev_Q3 = []
bpm_Q4 = []
dev_Q4 = []

for isub in subjects:
    sub = 'sub-' + isub
    print('subject {}'.format(isub))

    for ses in sessions:
        for task in tasks:
            physio_files = glob.glob(raw_path + sub + os.sep + 'ses-' + str(ses) + os.sep + 'beh' + os.sep + '*' + task + '*_emg.vhdr')
            physio_files = np.array(sorted(physio_files))
            if len(physio_files) > 1:
                print('Warning! More than one file found!')

            sub_out_path = output_path + sub + os.sep + 'ses-' + str(ses) + os.sep + 'Resp'
            if not os.path.exists(sub_out_path):
                os.makedirs(sub_out_path)

            if preprocess:
                p = physio_files[0]
                # import raw data with MNE
                raw = mne.io.read_raw_brainvision(p, preload=True)
                sr = raw.info['sfreq']
                # pick respiration channel
                raw.pick_channels(ch_names=["RES"])

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

                raw_plot = raw.get_data()  # convert object to numpy array of shape (n_channels, n_times)
                raw_plot = np.squeeze(raw_plot)  # get rid of dimension "n_channels" -> (rows: time points)

                # inhalation starts = min peaks, so invert data
                raw_plot = 1 - raw_plot
                median_height = np.median(raw_plot)
                inhale_peaks, properties = signal.find_peaks(raw_plot, height=median_height, distance=1.0 * sr, width=0.5 * sr)

                # how many breaths per minute?
                bpm = len(inhale_peaks) / (len(raw_plot) / sr / 60)
                bpm_all.append(bpm)
                # how much deviation is in the peaks
                dev = np.std(np.diff(inhale_peaks))/sr
                dev_all.append(dev)

                # same for 5 minute blocks
                len_quart = int(np.floor(len(raw_plot)/4))
                inhale_peaks_1 = np.delete(inhale_peaks, inhale_peaks > len_quart)
                bpm_1 = len(inhale_peaks_1) / ((len(raw_plot)/4) / sr / 60)
                bpm_Q1.append(bpm_1)
                dev_1 = np.std(np.diff(inhale_peaks_1)) / sr
                dev_Q1.append(dev_1)

                inhale_peaks_2 = np.delete(inhale_peaks, (inhale_peaks < len_quart) | (inhale_peaks > 2 * len_quart))
                bpm_2 = len(inhale_peaks_2) / ((len(raw_plot) / 4) / sr / 60)
                bpm_Q2.append(bpm_2)
                dev_2 = np.std(np.diff(inhale_peaks_2)) / sr
                dev_Q2.append(dev_2)

                inhale_peaks_3 = np.delete(inhale_peaks, (inhale_peaks < 2 * len_quart) | (inhale_peaks > 3 * len_quart))
                bpm_3 = len(inhale_peaks_3) / ((len(raw_plot) / 4) / sr / 60)
                bpm_Q3.append(bpm_3)
                dev_3 = np.std(np.diff(inhale_peaks_3)) / sr
                dev_Q3.append(dev_3)

                inhale_peaks_4 = np.delete(inhale_peaks, inhale_peaks < 3 * len_quart)
                bpm_4 = len(inhale_peaks_4) / ((len(raw_plot) / 4) / sr / 60)
                bpm_Q4.append(bpm_4)
                dev_4 = np.std(np.diff(inhale_peaks_4)) / sr
                dev_Q4.append(dev_4)

if save_markers:
    rep_subjects = subjects * 3
    rep_subjects_prefix = ['sub-' + subject for subject in rep_subjects]

    markers = pd.DataFrame({'BPM': bpm_all, 'BDev': dev_all,
                            'BPM_Q1': bpm_Q1, 'BPM_Q2': bpm_Q2, 'BPM_Q3': bpm_Q3, 'BPM_Q4': bpm_Q4,
                            'BDev_Q1': dev_Q1, 'BDev_Q2': dev_Q2, 'BDev_Q3': dev_Q3, 'BDev_Q4': dev_Q4,
                            'Subject': rep_subjects_prefix,
                            'Session': np.resize(np.arange(1, 4), len(subjects)*3)})
    markers.to_csv(result_path + 'BPM.csv', sep=',', index=False)

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

    df = pd.read_csv(result_path + 'BPM.csv', sep=',', header=0)
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
    df.loc[(df['Subject'] == 'sub-SR01') & (df['Session'] == 2), ['BPM', 'BPM_Q1', 'BPM_Q2', 'BPM_Q3', 'BPM_Q4',
                                                                  'BDev', 'BDev_Q1', 'BDev_Q2', 'BDev_Q3', 'BDev_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR08') & (df['Session'] == 2), ['BPM', 'BPM_Q1', 'BPM_Q2', 'BPM_Q3', 'BPM_Q4',
                                                                  'BDev', 'BDev_Q1', 'BDev_Q2', 'BDev_Q3', 'BDev_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR09'), ['BPM', 'BPM_Q1', 'BPM_Q2', 'BPM_Q3', 'BPM_Q4',
                                           'BDev', 'BDev_Q1', 'BDev_Q2', 'BDev_Q3', 'BDev_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR10') & (df['Session'] == 3), ['BPM', 'BPM_Q1', 'BPM_Q2', 'BPM_Q3', 'BPM_Q4',
                                                                  'BDev', 'BDev_Q1', 'BDev_Q2', 'BDev_Q3', 'BDev_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR13'), ['BPM', 'BPM_Q1', 'BPM_Q2', 'BPM_Q3', 'BPM_Q4',
                                           'BDev', 'BDev_Q1', 'BDev_Q2', 'BDev_Q3', 'BDev_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR14'), ['BPM', 'BPM_Q1', 'BPM_Q2', 'BPM_Q3', 'BPM_Q4',
                                           'BDev', 'BDev_Q1', 'BDev_Q2', 'BDev_Q3', 'BDev_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR16') & (df['Session'] == 3), ['BPM', 'BPM_Q1', 'BPM_Q2', 'BPM_Q3', 'BPM_Q4',
                                                                  'BDev', 'BDev_Q1', 'BDev_Q2', 'BDev_Q3', 'BDev_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR17') & (df['Session'] == 3), ['BPM', 'BPM_Q1', 'BPM_Q2', 'BPM_Q3', 'BPM_Q4',
                                                                  'BDev', 'BDev_Q1', 'BDev_Q2', 'BDev_Q3', 'BDev_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR19') & (df['Session'] == 1), ['BPM', 'BPM_Q1', 'BPM_Q2', 'BPM_Q3', 'BPM_Q4',
                                                                  'BDev', 'BDev_Q1', 'BDev_Q2', 'BDev_Q3', 'BDev_Q4']] = np.nan

    # make a plot
    sns.set_context("talk")
    hue_order = ['A', 'C', 'B']
    fig, axes = plt.subplots(2, 1)
    fig.suptitle('Breathing rate and breathing rate variability', fontsize=16)
    sns.boxplot(df, x='Condition', y='BPM', order=hue_order,
                ax=axes[0], boxprops={'alpha': 0.4}, showfliers=False)
    sns.stripplot(data=df, x="Condition", y="BPM", order=hue_order, hue='Condition', hue_order=hue_order,
                  dodge=False, ax=axes[0])
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles, labels=['A (Anodal)', 'C (Cathodal)', 'B (Sham)'], loc='upper right')
    sns.boxplot(df, x='Condition', y='BDev', order=hue_order,
                ax=axes[1], boxprops={'alpha': 0.4}, showfliers=False)
    sns.stripplot(data=df, x="Condition", y="BDev", order=hue_order, hue='Condition', hue_order=hue_order,
                  dodge=False, ax=axes[1], legend=False)
    handles, labels = axes[0].get_legend_handles_labels()
    plt.show()

    # create tables for the stats
    df = df.drop(['Session'], axis=1)
    wide_df = df.pivot(index='Subject', columns='Condition', values=['BPM', 'BDev', 'BPM_Q1',
                                                                     'BPM_Q2', 'BPM_Q3', 'BPM_Q4',
                                                                     'BDev_Q1', 'BDev_Q2', 'BDev_Q3', 'BDev_Q4'])
    wide_df.columns = ['_'.join(col).strip() for col in wide_df.columns.values]
    wide_df.reset_index('Subject', inplace=True)
    wide_df.to_csv(result_path + 'BPM_BDev_unblind.csv', sep=',', index=None)

if within_sub_plot:
    df = pd.read_csv(result_path + 'BPM_BDev_unblind.csv', sep=',')

    # 1. Breathing rate BPM
    # cathodal comparison
    cathodal = df.drop(labels=['BPM_A', 'BDev_A', 'BDev_B', 'BDev_C',
                               'BPM_Q1_A', 'BPM_Q2_A', 'BPM_Q3_A', 'BPM_Q4_A',
                               'BDev_Q1_A', 'BDev_Q2_A', 'BDev_Q3_A', 'BDev_Q4_A'
                               ], axis=1)
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
    fig.suptitle("Breathing rate", fontsize=30)
    b = sns.boxplot(ax=ax[0], x=np.repeat(0, len(cathodal)), y=cathodal['BPM_B'], color=color_blue, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in b.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    c = sns.boxplot(ax=ax[0], x=np.repeat(1, len(cathodal)), y=cathodal['BPM_C'], color=color_red, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in c.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    df_sham = pd.DataFrame({'jitter': df_x_jitter['Sham'], 'bpm': cathodal['BPM_B']})
    df_cath = pd.DataFrame({'jitter': df_x_jitter['Cathodal'], 'bpm': cathodal['BPM_C']})
    sns.scatterplot(ax=ax[0], data=df_sham, x='jitter', y='bpm', color=color_blue, zorder=100, edgecolor="black")
    sns.scatterplot(ax=ax[0], data=df_cath, x='jitter', y='bpm', color=color_red, zorder=100, edgecolor="black")
    ax[0].set_xticks(range(2))
    ax[0].set_xticklabels(['Sham', 'Cathodal'])
    ax[0].tick_params(axis='x', which='major', labelsize=25)
    ax[0].set_xlim(-1.0, 2)
    sns.despine()
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Breaths per minute', fontsize=25)
    plt.subplots_adjust(bottom=0.2, left=0.15)
    ax[0].set_title('Cathodal', fontsize=25)
    for idx in cathodal.index:
        ax[0].plot(df_x_jitter.loc[idx, ['Sham', 'Cathodal']], cathodal.loc[idx, ['BPM_B', 'BPM_C']], color='grey',
                   linewidth=0.5, linestyle='--', zorder=-1)

    # anodal comparison
    anodal = df.drop(labels=['BPM_C', 'BDev_A', 'BDev_B', 'BDev_C',
                             'BPM_Q1_C', 'BPM_Q2_C', 'BPM_Q3_C', 'BPM_Q4_C',
                             'BDev_Q1_C', 'BDev_Q2_C', 'BDev_Q3_C', 'BDev_Q4_C'
                             ], axis=1)
    anodal.dropna(axis=0, inplace=True)
    anodal.reset_index(inplace=True, drop=True)
    anodal.drop('Subject', axis=1, inplace=True)

    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=(anodal.values.shape[0], 2)),
                               columns=['Sham', 'Anodal'])
    df_x_jitter += np.arange(2)
    b = sns.boxplot(ax=ax[1], x=np.repeat(0, len(anodal)), y=anodal['BPM_B'], color=color_blue, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in b.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    c = sns.boxplot(ax=ax[1], x=np.repeat(1, len(anodal)), y=anodal['BPM_A'], color=color_orange, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in c.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    df_sham = pd.DataFrame({'jitter': df_x_jitter['Sham'], 'bpm': anodal['BPM_B']})
    df_anod = pd.DataFrame({'jitter': df_x_jitter['Anodal'], 'bpm': anodal['BPM_A']})
    sns.scatterplot(ax=ax[1], data=df_sham, x='jitter', y='bpm', color=color_blue, zorder=100, edgecolor="black")
    sns.scatterplot(ax=ax[1], data=df_anod, x='jitter', y='bpm', color=color_orange, zorder=100, edgecolor="black")
    ax[1].set_xticks(range(2))
    ax[1].set_xticklabels(['Sham', 'Anodal'])
    ax[1].tick_params(axis='x', which='major', labelsize=25)
    ax[1].set_xlim(-1.0, 2)
    sns.despine()
    ax[1].set_xlabel('')
    ax[1].set_ylabel('Breaths per minute', fontsize=25)
    plt.subplots_adjust(bottom=0.2, left=0.15)
    ax[1].set_title('Anodal', fontsize=25)
    for idx in anodal.index:
        ax[1].plot(df_x_jitter.loc[idx, ['Sham', 'Anodal']], anodal.loc[idx, ['BPM_B', 'BPM_A']], color='grey',
                   linewidth=0.5, linestyle='--', zorder=-1)

    plt.savefig(result_path + 'BR_within_sub.png')
    plt.show()

    # 2. Breathing rate variability HRV
    # cathodal comparison
    cathodal = df.drop(labels=['BDev_A', 'BPM_A', 'BPM_B', 'BPM_C',
                               'BPM_Q1_A', 'BPM_Q2_A', 'BPM_Q3_A', 'BPM_Q4_A',
                               'BDev_Q1_A', 'BDev_Q2_A', 'BDev_Q3_A', 'BDev_Q4_A'
                               ], axis=1)
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
    fig.suptitle("Breathing rate variability", fontsize=30)
    b = sns.boxplot(ax=ax[0], x=np.repeat(0, len(cathodal)), y=cathodal['BDev_B'], color=color_blue, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in b.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    c = sns.boxplot(ax=ax[0], x=np.repeat(1, len(cathodal)), y=cathodal['BDev_C'], color=color_red, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in c.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    df_sham = pd.DataFrame({'jitter': df_x_jitter['Sham'], 'bdev': cathodal['BDev_B']})
    df_cath = pd.DataFrame({'jitter': df_x_jitter['Cathodal'], 'bdev': cathodal['BDev_C']})
    sns.scatterplot(ax=ax[0], data=df_sham, x='jitter', y='bdev', color=color_blue, zorder=100, edgecolor="black")
    sns.scatterplot(ax=ax[0], data=df_cath, x='jitter', y='bdev', color=color_red, zorder=100, edgecolor="black")
    ax[0].set_xticks(range(2))
    ax[0].set_xticklabels(['Sham', 'Cathodal'])
    ax[0].tick_params(axis='x', which='major', labelsize=25)
    ax[0].set_xlim(-1.0, 2)
    sns.despine()
    ax[0].set_xlabel('')
    ax[0].set_ylabel('SD of breath intervals', fontsize=25)
    plt.subplots_adjust(bottom=0.2, left=0.15)
    ax[0].set_title('Cathodal', fontsize=25)
    for idx in cathodal.index:
        ax[0].plot(df_x_jitter.loc[idx, ['Sham', 'Cathodal']], cathodal.loc[idx, ['BDev_B', 'BDev_C']], color='grey',
                   linewidth=0.5, linestyle='--', zorder=-1)

    # anodal comparison
    anodal = df.drop(labels=['BDev_C', 'BPM_A', 'BPM_B', 'BPM_C',
                             'BPM_Q1_C', 'BPM_Q2_C', 'BPM_Q3_C', 'BPM_Q4_C',
                             'BDev_Q1_C', 'BDev_Q2_C', 'BDev_Q3_C', 'BDev_Q4_C'
                             ], axis=1)
    anodal.dropna(axis=0, inplace=True)
    anodal.reset_index(inplace=True, drop=True)
    anodal.drop('Subject', axis=1, inplace=True)

    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=(anodal.values.shape[0], 2)),
                               columns=['Sham', 'Anodal'])
    df_x_jitter += np.arange(2)
    b = sns.boxplot(ax=ax[1], x=np.repeat(0, len(anodal)), y=anodal['BDev_B'], color=color_blue, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in b.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    c = sns.boxplot(ax=ax[1], x=np.repeat(1, len(anodal)), y=anodal['BDev_A'], color=color_orange, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in c.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    df_sham = pd.DataFrame({'jitter': df_x_jitter['Sham'], 'bdev': anodal['BDev_B']})
    df_anod = pd.DataFrame({'jitter': df_x_jitter['Anodal'], 'bdev': anodal['BDev_A']})
    sns.scatterplot(ax=ax[1], data=df_sham, x='jitter', y='bdev', color=color_blue, zorder=100, edgecolor="black")
    sns.scatterplot(ax=ax[1], data=df_anod, x='jitter', y='bdev', color=color_orange, zorder=100, edgecolor="black")
    ax[1].set_xticks(range(2))
    ax[1].set_xticklabels(['Sham', 'Anodal'])
    ax[1].tick_params(axis='x', which='major', labelsize=25)
    ax[1].set_xlim(-1.0, 2)
    sns.despine()
    ax[1].set_xlabel('')
    ax[1].set_ylabel('SD of breath intervals', fontsize=25)
    plt.subplots_adjust(bottom=0.2, left=0.15)
    ax[1].set_title('Anodal', fontsize=25)
    for idx in anodal.index:
        ax[1].plot(df_x_jitter.loc[idx, ['Sham', 'Anodal']], anodal.loc[idx, ['BDev_B', 'BDev_A']], color='grey',
                   linewidth=0.5, linestyle='--', zorder=-1)

    plt.savefig(result_path + 'BRV_within_sub.png')
    plt.show()

if interaction_plot:
    df = pd.read_csv(result_path + 'BPM_BDev_unblind.csv', sep=',')
    # breathing rate
    br = df[df.columns.drop(list(df.filter(regex='BDev_')))]
    br = br[br.columns.drop(['BPM_A', 'BPM_B', 'BPM_C'])]
    df_q_long = pd.wide_to_long(br, stubnames=['BPM_Q1', 'BPM_Q2', 'BPM_Q3', 'BPM_Q4'], i='Subject', j='condition',
                                sep='_', suffix=r'\w+').reset_index()
    df_q_longer = pd.wide_to_long(df_q_long, stubnames=['BPM'], i=['Subject', 'condition'], j='Quartile',
                                  sep='_Q', suffix=r'\w+').reset_index()
    sns.set_context("poster")
    sns.set_style("white")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle("Breathing rate over time", fontsize=30)
    sns.lineplot(data=df_q_longer, x='Quartile', y='BPM', hue='condition', palette=[color_blue, color_red, color_orange],
                 hue_order=['B', 'C', 'A'], errorbar='se')
    handles, labels = ax.get_legend_handles_labels()
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.ylim(14, 19)
    sns.despine()
    plt.subplots_adjust(bottom=0.2, left=0.15)
    ax.set_ylabel('Quartile', fontsize=25)
    ax.set_ylabel('Breaths per minute', fontsize=25)
    plt.legend(handles=handles, labels=['Sham', 'Cathodal', 'Anodal'])
    plt.savefig(result_path + 'BR_Quartile_interaction.png')
    plt.show()
