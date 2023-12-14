#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Description:
------------
This script analyzes the electrocardiographic activity.
The ECG data has already been preprocessed in the script UE_analysis_1_prep.py
Therefore, we load the data and saved R-peaks,
as well as the events file with the information on where to cut the data.
The user has the possibility to visually inspect and manually correct them again if necessary.
We then extract the average heart rate (number of heart beats in one minute)
and the heart rate variability (root-mean-square of successive differences) over the whole interval
and within the 4 quartiles.
In the unblinding step the session-condition mapping is read and
outliers are excluded. The resulting table HR_HRV_unblind.csv can be used
for statistical analysis and further plotting.
The within-subject plots (fig 4B and 4C) are created.

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
import random
from scipy import signal, stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from UE_analysis_helper_class_GUI_hb import r_peak_GUI

raw_path = '/data/pt_02582/tsDCS_BIDS/'
output_path = '/data/pt_02582/tsDCS_processed/'
result_path = '/data/pt_02582/tsDCS_results/ECG/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

if not os.path.exists(result_path):
    os.makedirs(result_path)

tasks = ['tsDCS']

subjects = ['SR01', 'SR02', 'SR03', 'SR04', 'SR05', 'SR06', 'SR07', 'SR08', 'SR09', 'SR10',
            'SR11', 'SR12', 'SR13', 'SR14', 'SR15', 'SR16', 'SR17', 'SR18', 'SR19', 'SR20']

sessions = [1, 2, 3]

preprocess = True
manual = False  # did some correction in the prep script already, can just close/skip it here
extract_markers = True
unblinding = True
within_sub_plot = True

color_blue = [0, 128/255, 255/255]
color_red = [227/255, 0, 15/255]
color_orange = [239/255, 138/255, 16/255]

# how many seconds you want to display in the manual correction GUI
time_window = 20

if extract_markers:
    hr_all = []
    hrv_all = []
    hr_Q1 = []
    hr_Q2 = []
    hr_Q3 = []
    hr_Q4 = []
    hrv_Q1 = []
    hrv_Q2 = []
    hrv_Q3 = []
    hrv_Q4 = []
for isub in subjects:
    sub = 'sub-' + isub
    print('subject {}'.format(isub))

    for ses in sessions:
        for task in tasks:
            physio_files = glob.glob(raw_path + sub + os.sep + 'ses-' + str(ses) + os.sep + 'beh' + os.sep + '*' + task + '*_emg.vhdr')
            physio_files = np.array(sorted(physio_files))

            sub_out_path = output_path + sub + os.sep + 'ses-' + str(ses) + os.sep + 'ECG'
            if not os.path.exists(sub_out_path):
                os.makedirs(sub_out_path)

            if preprocess:
                for i, p in enumerate(physio_files):
                    # import raw data with MNE
                    raw = mne.io.read_raw_brainvision(p, preload=True)
                    channel = 'ECG'
                    raw.pick_channels(ch_names=[channel])
                    sr = raw.info['sfreq']

                    # get events file
                    tmp = p.split('_emg')
                    event_file = tmp[0] + '_events.tsv'
                    events = pd.read_table(event_file, sep='\t')

                    # read saved cleaned R-peaks
                    save_name = sub_out_path + os.sep + 'cleaned_ecg_events_' + task + '.npy'
                    r_peaks = np.load(save_name)

                    # cut data and R peaks from manually inserted event
                    onset = events.loc[events['value'] == 'manual', 'onset'].values[0]
                    duration = events.loc[events['value'] == 'manual', 'duration'].values[0]
                    if onset+duration >= len(raw)/sr:
                        raw.crop(tmin=onset, tmax=None, include_tmax=True)
                    else:
                        raw.crop(tmin=onset, tmax=onset+duration, include_tmax=True)
                    r_peaks = np.delete(r_peaks, r_peaks/sr < onset)
                    r_peaks = np.delete(r_peaks, r_peaks / sr > onset + duration)
                    r_peaks = r_peaks - onset * sr
                    r_peaks = r_peaks.astype(int)
                    save_name = sub_out_path + os.sep + 'cleaned_ecg_events_cropped_' + task
                    np.save(save_name, r_peaks)

                    # get data into numpy
                    raw_data = raw.get_data()  # (n_epochs, n_channels, n_times)
                    raw_data = np.squeeze(raw_data)  # (n_epochs, n_times)

                    # manual correction
                    if manual:
                        interval_size = int(sr * time_window)
                        save_name = sub_out_path + os.sep + 'cleaned_ecg_events_cropped_manual_' + task
                        my_GUI = r_peak_GUI(raw_data, r_peaks, interval_size, sr, save_name)

            if extract_markers:
                for i, p in enumerate(physio_files):
                    # import raw data with MNE
                    raw = mne.io.read_raw_brainvision(p, preload=True)
                    channel = 'ECG'
                    raw.pick_channels(ch_names=[channel])
                    sr = raw.info['sfreq']

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
                    len_quart = int(np.floor(len(raw_plot)/4))

                    save_name = sub_out_path + os.sep + 'cleaned_ecg_events_cropped_manual_' + task + '.npy'
                    # save_name = sub_out_path + os.sep + 'cleaned_ecg_events_cropped_' + task + '.npy'
                    r_peaks = np.load(save_name)
                    hb = np.array([r / sr * 1000 for r in r_peaks])  # in ms
                    ibi = np.diff(hb)
                    heart_rate = 60000/ibi
                    print('The average heart rate is {}'.format(np.mean(heart_rate)))
                    hr_all.append(np.mean(heart_rate))
                    RMSSD = np.sqrt(np.mean(np.square(np.diff(ibi))))
                    print('The heart rate variability in ms is {}'.format(np.mean(RMSSD)))
                    hrv_all.append(RMSSD)

                    # do the same thing within 5 minute intervals
                    r_peaks_1 = np.delete(r_peaks, r_peaks > len_quart)
                    hb_1 = np.array([r / sr * 1000 for r in r_peaks_1])  # in ms
                    ibi_1 = np.diff(hb_1)
                    heart_rate_1 = 60000 / ibi_1
                    hr_Q1.append(np.mean(heart_rate_1))
                    RMSSD_1 = np.sqrt(np.mean(np.square(np.diff(ibi_1))))
                    hrv_Q1.append(RMSSD_1)

                    r_peaks_2 = np.delete(r_peaks, (r_peaks < len_quart) | (r_peaks > 2 * len_quart))
                    hb_2 = np.array([r / sr * 1000 for r in r_peaks_2])  # in ms
                    ibi_2 = np.diff(hb_2)
                    heart_rate_2 = 60000 / ibi_2
                    hr_Q2.append(np.mean(heart_rate_2))
                    RMSSD_2 = np.sqrt(np.mean(np.square(np.diff(ibi_2))))
                    hrv_Q2.append(RMSSD_2)

                    r_peaks_3 = np.delete(r_peaks, (r_peaks < 2 * len_quart) | (r_peaks > 3 * len_quart))
                    hb_3 = np.array([r / sr * 1000 for r in r_peaks_3])  # in ms
                    ibi_3 = np.diff(hb_3)
                    heart_rate_3 = 60000 / ibi_3
                    hr_Q3.append(np.mean(heart_rate_3))
                    RMSSD_3 = np.sqrt(np.mean(np.square(np.diff(ibi_3))))
                    hrv_Q3.append(RMSSD_3)

                    r_peaks_4 = np.delete(r_peaks, (r_peaks < 3 * len_quart))
                    hb_4 = np.array([r / sr * 1000 for r in r_peaks_4])  # in ms
                    ibi_4 = np.diff(hb_4)
                    heart_rate_4 = 60000 / ibi_4
                    hr_Q4.append(np.mean(heart_rate_4))
                    RMSSD_4 = np.sqrt(np.mean(np.square(np.diff(ibi_4))))
                    hrv_Q4.append(RMSSD_4)


if extract_markers:

    rep_subjects = subjects * 3
    rep_subjects_prefix = ['sub-' + subject for subject in rep_subjects]
    markers = pd.DataFrame({'HR': hr_all, 'HRV': hrv_all,
                            'HR_Q1': hr_Q1, 'HR_Q2': hr_Q2, 'HR_Q3': hr_Q3, 'HR_Q4': hr_Q4,
                            'HRV_Q1': hrv_Q1, 'HRV_Q2': hrv_Q2, 'HRV_Q3': hrv_Q3, 'HRV_Q4': hrv_Q4,
                            'Subject': rep_subjects_prefix,
                            'Session': np.resize(np.arange(1, 4), len(subjects)*3)})
    markers.to_csv(result_path + 'HR_HRV.csv', sep=',', index=False)

    ax = sns.lineplot(data=markers, x="Session", y="HR", hue="Subject", style="Subject", markers=True)
    ax.set_xticks([1, 2, 3])
    ax.set_ylim([0, 130])
    ax.legend(loc='upper right')
    plt.show()

    ax1 = sns.lineplot(data=markers, x="Session", y="HRV", hue="Subject", style="Subject", markers=True)
    ax1.set_xticks([1, 2, 3])
    ax1.set_ylim([0, 130])
    ax1.legend(loc='upper right')
    plt.show()

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

    df = pd.read_csv(result_path + 'HR_HRV.csv', sep=',', header=0)
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
    df.loc[(df['Subject'] == 'sub-SR01') & (df['Session'] == 2), ['HR', 'HR_Q1', 'HR_Q2', 'HR_Q3', 'HR_Q4',
                                                                  'HRV', 'HRV_Q1', 'HRV_Q2', 'HRV_Q3', 'HRV_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR08') & (df['Session'] == 2), ['HR', 'HR_Q1', 'HR_Q2', 'HR_Q3', 'HR_Q4',
                                                                  'HRV', 'HRV_Q1', 'HRV_Q2', 'HRV_Q3', 'HRV_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR09'), ['HR', 'HR_Q1', 'HR_Q2', 'HR_Q3', 'HR_Q4',
                                           'HRV', 'HRV_Q1', 'HRV_Q2', 'HRV_Q3', 'HRV_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR10') & (df['Session'] == 3), ['HR', 'HR_Q1', 'HR_Q2', 'HR_Q3', 'HR_Q4',
                                                                  'HRV', 'HRV_Q1', 'HRV_Q2', 'HRV_Q3', 'HRV_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR13'), ['HR', 'HR_Q1', 'HR_Q2', 'HR_Q3', 'HR_Q4',
                                           'HRV', 'HRV_Q1', 'HRV_Q2', 'HRV_Q3', 'HRV_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR14'), ['HR', 'HR_Q1', 'HR_Q2', 'HR_Q3', 'HR_Q4',
                                           'HRV', 'HRV_Q1', 'HRV_Q2', 'HRV_Q3', 'HRV_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR16') & (df['Session'] == 3), ['HR', 'HR_Q1', 'HR_Q2', 'HR_Q3', 'HR_Q4',
                                                                  'HRV', 'HRV_Q1', 'HRV_Q2', 'HRV_Q3', 'HRV_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR17') & (df['Session'] == 3), ['HR', 'HR_Q1', 'HR_Q2', 'HR_Q3', 'HR_Q4',
                                                                  'HRV', 'HRV_Q1', 'HRV_Q2', 'HRV_Q3', 'HRV_Q4']] = np.nan
    df.loc[(df['Subject'] == 'sub-SR19') & (df['Session'] == 1), ['HR', 'HR_Q1', 'HR_Q2', 'HR_Q3', 'HR_Q4',
                                                                  'HRV', 'HRV_Q1', 'HRV_Q2', 'HRV_Q3', 'HRV_Q4']] = np.nan

    # make a plot
    sns.set_context("talk")
    hue_order = ['A', 'C', 'B']
    fig, axes = plt.subplots(2, 1)
    fig.suptitle('Heart rate and heart rate variability', fontsize=16)
    sns.boxplot(df, x='Condition', y='HR', order=hue_order,
                ax=axes[0], boxprops={'alpha': 0.4}, showfliers=False)
    sns.stripplot(data=df, x="Condition", y="HR", order=hue_order, hue='Condition', hue_order=hue_order,
                  dodge=False, ax=axes[0])
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles, labels=['A (Anodal)', 'C (Cathodal)', 'B (Sham)'], loc='upper left')
    sns.boxplot(df, x='Condition', y='HRV', order=hue_order,
                ax=axes[1], boxprops={'alpha': 0.4}, showfliers=False)
    sns.stripplot(data=df, x="Condition", y="HRV", order=hue_order, hue='Condition', hue_order=hue_order,
                  dodge=False, ax=axes[1])
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles=handles, labels=['A (Anodal)', 'C (Cathodal)', 'B (Sham)'], loc='upper right')
    plt.show()

    # create tables for the stats
    df = df.drop(['Session'], axis=1)
    wide_df = df.pivot(index='Subject', columns='Condition', values=['HR', 'HRV', 'HR_Q1', 'HR_Q2', 'HR_Q3', 'HR_Q4',
                                                                     'HRV_Q1', 'HRV_Q2', 'HRV_Q3', 'HRV_Q4'])
    wide_df.columns = ['_'.join(col).strip() for col in wide_df.columns.values]
    wide_df.reset_index('Subject', inplace=True)
    wide_df.to_csv(result_path + 'HR_HRV_unblind.csv', sep=',', index=None)

if within_sub_plot:
    df = pd.read_csv(result_path + 'HR_HRV_unblind.csv', sep=',')

    # 1. Heart rate HR
    # cathodal comparison
    cathodal = df.drop(labels=['HR_A', 'HRV_A', 'HRV_B', 'HRV_C',
                               'HR_Q1_A', 'HR_Q2_A', 'HR_Q3_A', 'HR_Q4_A',
                               'HRV_Q1_A', 'HRV_Q2_A', 'HRV_Q3_A', 'HRV_Q4_A'], axis=1)
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
    fig.suptitle("Heart rate", fontsize=30)
    b = sns.boxplot(ax=ax[0], x=np.repeat(0, len(cathodal)), y=cathodal['HR_B'], color=color_blue, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in b.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    c = sns.boxplot(ax=ax[0], x=np.repeat(1, len(cathodal)), y=cathodal['HR_C'], color=color_red, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in c.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    df_sham = pd.DataFrame({'jitter': df_x_jitter['Sham'], 'hr': cathodal['HR_B']})
    df_cath = pd.DataFrame({'jitter': df_x_jitter['Cathodal'], 'hr': cathodal['HR_C']})
    sns.scatterplot(ax=ax[0], data=df_sham, x='jitter', y='hr', color=color_blue, zorder=100, edgecolor="black")
    sns.scatterplot(ax=ax[0], data=df_cath, x='jitter', y='hr', color=color_red, zorder=100, edgecolor="black")
    ax[0].set_xticks(range(2))
    ax[0].set_xticklabels(['Sham', 'Cathodal'])
    ax[0].tick_params(axis='x', which='major', labelsize=25)
    ax[0].set_xlim(-1.0, 2)
    sns.despine()
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Beats per minute', fontsize=25)
    # ax[0].set(xlabel='Condition', ylabel='Beats per minute')
    plt.subplots_adjust(bottom=0.2, left=0.15)
    ax[0].set_title('Cathodal', fontsize=25)
    for idx in cathodal.index:
        ax[0].plot(df_x_jitter.loc[idx, ['Sham', 'Cathodal']], cathodal.loc[idx, ['HR_B', 'HR_C']], color='grey',
                   linewidth=0.5, linestyle='--', zorder=-1)

    # anodal comparison
    anodal = df.drop(labels=['HR_C', 'HRV_A', 'HRV_B', 'HRV_C',
                             'HR_Q1_C', 'HR_Q2_C', 'HR_Q3_C', 'HR_Q4_C',
                             'HRV_Q1_C', 'HRV_Q2_C', 'HRV_Q3_C', 'HRV_Q4_C'
                             ], axis=1)
    anodal.dropna(axis=0, inplace=True)
    anodal.reset_index(inplace=True, drop=True)
    anodal.drop('Subject', axis=1, inplace=True)

    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=(anodal.values.shape[0], 2)),
                               columns=['Sham', 'Anodal'])
    df_x_jitter += np.arange(2)
    b = sns.boxplot(ax=ax[1], x=np.repeat(0, len(anodal)), y=anodal['HR_B'], color=color_blue, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in b.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    c = sns.boxplot(ax=ax[1], x=np.repeat(1, len(anodal)), y=anodal['HR_A'], color=color_orange, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in c.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    df_sham = pd.DataFrame({'jitter': df_x_jitter['Sham'], 'hr': anodal['HR_B']})
    df_anod = pd.DataFrame({'jitter': df_x_jitter['Anodal'], 'hr': anodal['HR_A']})
    sns.scatterplot(ax=ax[1], data=df_sham, x='jitter', y='hr', color=color_blue, zorder=100, edgecolor="black")
    sns.scatterplot(ax=ax[1], data=df_anod, x='jitter', y='hr', color=color_orange, zorder=100, edgecolor="black")
    ax[1].set_xticks(range(2))
    ax[1].set_xticklabels(['Sham', 'Anodal'])
    ax[1].tick_params(axis='x', which='major', labelsize=25)
    ax[1].set_xlim(-1.0, 2)
    sns.despine()
    ax[1].set_xlabel('')
    ax[1].set_ylabel('Beats per minute', fontsize=25)
    # ax[1].set(xlabel='Condition', ylabel='Beats per minute')
    plt.subplots_adjust(bottom=0.2, left=0.15)
    ax[1].set_title('Anodal', fontsize=25)
    for idx in anodal.index:
        ax[1].plot(df_x_jitter.loc[idx, ['Sham', 'Anodal']], anodal.loc[idx, ['HR_B', 'HR_A']], color='grey',
                   linewidth=0.5, linestyle='--', zorder=-1)

    plt.savefig(result_path + 'HR_within_sub.png')
    plt.show()

    # 2. Heart rate variability HRV
    # cathodal comparison
    cathodal = df.drop(labels=['HRV_A', 'HR_A', 'HR_B', 'HR_C',
                               'HR_Q1_A', 'HR_Q2_A', 'HR_Q3_A', 'HR_Q4_A',
                               'HRV_Q1_A', 'HRV_Q2_A', 'HRV_Q3_A', 'HRV_Q4_A'
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
    fig.suptitle("Heart rate variability", fontsize=30)
    b = sns.boxplot(ax=ax[0], x=np.repeat(0, len(cathodal)), y=cathodal['HRV_B'], color=color_blue, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in b.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    c = sns.boxplot(ax=ax[0], x=np.repeat(1, len(cathodal)), y=cathodal['HRV_C'], color=color_red, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in c.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    df_sham = pd.DataFrame({'jitter': df_x_jitter['Sham'], 'hrv': cathodal['HRV_B']})
    df_cath = pd.DataFrame({'jitter': df_x_jitter['Cathodal'], 'hrv': cathodal['HRV_C']})
    sns.scatterplot(ax=ax[0], data=df_sham, x='jitter', y='hrv', color=color_blue, zorder=100, edgecolor="black")
    sns.scatterplot(ax=ax[0], data=df_cath, x='jitter', y='hrv', color=color_red, zorder=100, edgecolor="black")
    ax[0].set_xticks(range(2))
    ax[0].set_xticklabels(['Sham', 'Cathodal'])
    ax[0].tick_params(axis='x', which='major', labelsize=25)
    ax[0].set_xlim(-1.0, 2)
    sns.despine()
    ax[0].set_xlabel('')
    ax[0].set_ylabel('RMSSD', fontsize=25)
    # ax[0].set(xlabel='Condition', ylabel='RMSSD')
    plt.subplots_adjust(bottom=0.2, left=0.15)
    ax[0].set_title('Cathodal', fontsize=25)
    for idx in cathodal.index:
        ax[0].plot(df_x_jitter.loc[idx, ['Sham', 'Cathodal']], cathodal.loc[idx, ['HRV_B', 'HRV_C']], color='grey',
                   linewidth=0.5, linestyle='--', zorder=-1)

    # anodal comparison
    anodal = df.drop(labels=['HRV_C', 'HR_A', 'HR_B', 'HR_C',
                             'HR_Q1_C', 'HR_Q2_C', 'HR_Q3_C', 'HR_Q4_C',
                             'HRV_Q1_C', 'HRV_Q2_C', 'HRV_Q3_C', 'HRV_Q4_C'
                             ], axis=1)
    anodal.dropna(axis=0, inplace=True)
    anodal.reset_index(inplace=True, drop=True)
    anodal.drop('Subject', axis=1, inplace=True)

    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=(anodal.values.shape[0], 2)),
                               columns=['Sham', 'Anodal'])
    df_x_jitter += np.arange(2)
    b = sns.boxplot(ax=ax[1], x=np.repeat(0, len(anodal)), y=anodal['HRV_B'], color=color_blue, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in b.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    c = sns.boxplot(ax=ax[1], x=np.repeat(1, len(anodal)), y=anodal['HRV_A'], color=color_orange, width=0.5,
                    native_scale=True, showfliers=False)
    for patch in c.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.7))
    df_sham = pd.DataFrame({'jitter': df_x_jitter['Sham'], 'hrv': anodal['HRV_B']})
    df_anod = pd.DataFrame({'jitter': df_x_jitter['Anodal'], 'hrv': anodal['HRV_A']})
    sns.scatterplot(ax=ax[1], data=df_sham, x='jitter', y='hrv', color=color_blue, zorder=100, edgecolor="black")
    sns.scatterplot(ax=ax[1], data=df_anod, x='jitter', y='hrv', color=color_orange, zorder=100, edgecolor="black")
    ax[1].set_xticks(range(2))
    ax[1].set_xticklabels(['Sham', 'Anodal'])
    ax[1].tick_params(axis='x', which='major', labelsize=25)
    ax[1].set_xlim(-1.0, 2)
    sns.despine()
    ax[1].set_xlabel('')
    ax[1].set_ylabel('RMSSD', fontsize=25)
    # ax[1].set(xlabel='Condition', ylabel='RMSSD')
    plt.subplots_adjust(bottom=0.2, left=0.15)
    ax[1].set_title('Anodal', fontsize=25)
    for idx in anodal.index:
        ax[1].plot(df_x_jitter.loc[idx, ['Sham', 'Anodal']], anodal.loc[idx, ['HRV_B', 'HRV_A']], color='grey',
                   linewidth=0.5, linestyle='--', zorder=-1)

    plt.savefig(result_path + 'HRV_within_sub.png')
    plt.show()
