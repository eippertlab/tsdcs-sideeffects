#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Description:
------------
This script determines the analysis time window for the continuously recorded autonomic measures 
(i.e., the period of tsDCS administration).
We made use of the prominent artefacts in the ECG trace that accompany the
tsDCS fade-in (strong signal shifts). The largest artifact-free interval starting immediately after the
fade-in period was selected for further processing of all data types but was limited to maximally 20
minutes.
To not be unblinded by observing specific patterns related to active or sham stimulation,
the sessions are shuffled before loading their data.
In the process R peaks are automatically detected using a Pan-Tompkins algorithm.
The user corrects missing or wrong R-peaks.
Artifacts are then defined as inter-beat-intervals that are more than 3 times larger than the median interval.
The script then finds the longest artifact-free section of data.
As a result of running this script the events.tsv files are modified to include a start time and duration 
of this interval to later analyze all data accordingly.

Author:
--------
Ulrike Horn

Contact:
--------
uhorn@cbs.mpg.de

Date:
-----
23rd May 2023
"""

import mne
import numpy as np
import pandas as pd
import os
import glob
import json
import random
from ecgdetectors import Detectors
from scipy.interpolate import RegularGridInterpolator
from scipy import signal, stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
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

# because you could recognize the condition, they are presented in random order
random.shuffle(sessions)
mne.set_log_level(verbose='WARNING')

preprocess = True  # load data and find R-peaks
manual = True  # during preprocessing: manually correct R-peaks
load_previous = True  # during manual correction: load previously corrected R-peaks if you want to continue with a subject
find_stim = True  # find interval of stimulation as the longest interval between IBI outliers

# how many seconds you want to display in the manual correction GUI
time_window = 20

for isub in subjects:
    sub = 'sub-' + isub
    print('subject {}'.format(isub))
    for ses in sessions:
        for task in tasks:
            physio_files = glob.glob(raw_path + sub + os.sep + 'ses-' + str(ses) + os.sep + 'beh' + os.sep + '*' + task + '*_emg.vhdr')
            physio_files = np.array(sorted(physio_files))

            curr_output_path = output_path + sub + os.sep + 'ses-' + str(ses) + os.sep + 'ECG'
            if not os.path.exists(curr_output_path):
                os.makedirs(curr_output_path)

            if preprocess:
                for i, p in enumerate(physio_files):
                    # import raw data with MNE
                    raw = mne.io.read_raw_brainvision(p, preload=True)
                    channel = 'ECG'
                    raw.pick_channels(ch_names=[channel])
                    sr = raw.info['sfreq']

                    # get raw data
                    raw_data = raw.get_data()  # (n_epochs, n_channels, n_times)
                    raw_data = np.squeeze(raw_data)  # (n_epochs, n_times)

                    # find R peaks
                    detectors = Detectors(sr)
                    r_peaks = detectors.pan_tompkins_detector(raw_data)

                    # strong artifacts disturb the algorithm
                    if (isub == 'SR09') & (ses == 1):
                        r_peaks = signal.find_peaks(raw_data, distance=0.8*sr)[0]

                    # find maximum peaks as sometimes these are not at the real peak yet
                    range = [-0.2 * sr, 0.2 * sr]
                    for ii, peak in enumerate(r_peaks):
                        bgn = int(peak + range[0])
                        enn = int(peak + range[1])
                        if enn <= len(raw_data) and bgn > 0:
                            temp = raw_data[bgn:enn]
                            ind = np.where(temp == np.max(temp))[0][0]
                            r_peaks[ii] = bgn + ind
                    # due to this step you sometimes assign the same maximum twice
                    # delete duplicates
                    r_peaks = np.unique(r_peaks)

                    # manual correction
                    if manual:
                        interval_size = int(sr * time_window)
                        save_name = curr_output_path + os.sep + 'cleaned_ecg_events_' + task
                        # load previously corrected data
                        if load_previous:
                            if os.path.isfile(save_name + '.npy'):
                                r_peaks = np.load(save_name + '.npy')
                        else:
                            print('No previously cleaned data found')
                        my_GUI = r_peak_GUI(raw_data, r_peaks, interval_size, sr, save_name)

            # find interval of stimulation as the longest interval between IBI outliers
            if find_stim:
                for i, p in enumerate(physio_files):
                    # get events file
                    tmp = p.split('_emg')
                    event_file = tmp[0] + '_events.tsv'
                    events = pd.read_table(event_file, sep='\t')

                    # import raw data with MNE
                    raw = mne.io.read_raw_brainvision(p, preload=True)
                    channel = 'ECG'
                    raw.pick_channels(ch_names=[channel])
                    sr = raw.info['sfreq']

                    # get raw data
                    raw_data = raw.get_data()  # (n_epochs, n_channels, n_times)
                    raw_data = np.squeeze(raw_data)  # (n_epochs, n_times)

                    # read saved cleaned R-peaks
                    save_name = curr_output_path + os.sep + 'cleaned_ecg_events_' + task + '.npy'
                    r_peaks = np.load(save_name)

                    # calculate inter-beat-interval
                    hb = np.array([r / sr for r in r_peaks])
                    ibi = np.diff(hb)
                    weird_ibi = np.where(ibi > 3 * np.median(ibi))[0]

                    # weird ibis are counted as artifacts
                    artifact = np.zeros(len(raw_data))
                    for w in weird_ibi:
                        artifact[r_peaks[w]:r_peaks[w+1]] = 1

                    # find the longest artifact-free section
                    if sum(artifact) > 0:
                        count = 0
                        maximum = 0
                        for j in range(len(artifact)):
                            if artifact[j] == 0:
                                count = count + 1
                                if count > maximum:
                                    maximum = count
                                    last_max_index = j
                            else:
                                count = 0
                        stim_start = last_max_index + 1 - maximum
                        stim_end = last_max_index
                    else:
                        stim_start = 0
                        stim_end = len(raw_data)
                        print('all data seems to have no artifacts?!')

                    # some artifacts did not lead to weird IBIs -> manually introduce events
                    if (isub == 'SR09') & (ses == 3):
                        stim_start = r_peaks[96]
                    if stim_start == 0:
                        if (isub == 'SR07') & (ses == 3):
                            stim_start = r_peaks[22]
                        elif (isub == 'SR12') & (ses == 1):
                            stim_start = r_peaks[75]
                        elif (isub == 'SR12') & (ses == 2):
                            stim_start = r_peaks[23]
                        elif (isub == 'SR12') & (ses == 3):
                            stim_start = r_peaks[107]
                        elif (isub == 'SR16') & (ses == 2):
                            stim_start = r_peaks[29]
                        elif (isub == 'SR17') & (ses == 1):
                            stim_start = r_peaks[113]
                        elif (isub == 'SR17') & (ses == 3):
                            stim_start = r_peaks[39]
                        elif (isub == 'SR20') & (ses == 1):
                            stim_start = r_peaks[29]
                        else:
                            print('warning! interval starts at 0')
                    print('start: ' + str(stim_start))
                    print('end: ' + str(stim_end))

                    nice_interval = (stim_end-stim_start)/sr/60
                    if nice_interval >= 10:
                        if stim_start/sr in events['onset'].values:
                            print('added this event before')
                        else:
                            if nice_interval >= 20:
                                duration = 20*60
                            else:
                                duration = nice_interval*60
                            addon = pd.DataFrame([[stim_start/sr, duration, stim_start, 'Stimulus', 'manual', 1]],
                                                 columns=['onset', 'duration', 'sample', 'type', 'value', 'trial'])
                            events = pd.concat([events, addon])
                            events.to_csv(event_file, sep='\t', index=False)

                    else:
                        print('Subject {} session {} does not have enough data that is artifact free - check again!'.format(sub, ses))
