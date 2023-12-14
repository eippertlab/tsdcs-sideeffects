#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description:
------------
This script creates the different parts of figure 3 by
loading the summarized data from the questionnaire
and plotting it grouped by the different questions and conditions.

Authors:
--------
Ulrike Horn

Contact:
--------
uhorn@cbs.mpg.de

Date:
-----
20th Nov 2023
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

output_path = '/data/pt_02582/tsDCS_BIDS/derivatives/'
result_path = '/data/pt_02582/tsDCS_results/Questionnaires/'

color_blue = [0, 128/255, 255/255]
color_red = [227/255, 0, 15/255]
color_orange = [239/255, 138/255, 16/255]

new_rc_params = {"font.family": 'Arial', "font.size": 12, "font.serif": [],
                 "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)

df = pd.read_csv(output_path + 'questionnaire_additional_questions_summarized.csv')
df = df[df.columns.drop('Count_General')]

sns.set_context("poster")
sns.set_style("white")

# Question 3
df_4a = df[df.Question == '3']
df_4a_long = pd.wide_to_long(df_4a, ['Count'], i='Reports', j='Condition', sep='_', suffix=r'\w+').reset_index()

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
fig.suptitle("Starting time of adverse effects")
sns.barplot(data=df_4a_long, x='Reports', y='Count', hue='Condition', err_kws={'linewidth': 0}, alpha=0.7, #linewidth=0.5, edgecolor='black',
            hue_order=['Anodal', 'Cathodal', 'Sham'], palette=[color_orange, color_red, color_blue], ax=ax)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.bar_label(ax.containers[2])
plt.ylim(0, 16)
sns.despine(right=False)
plt.yticks([])
plt.xlabel('')
plt.subplots_adjust(bottom=0.2, left=0.05)
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.xaxis.grid(True, which='minor', color='gray', lw=2)
plt.legend(framealpha=1, edgecolor='gray')
plt.savefig(result_path + 'Fig3a.png')
plt.show()

# Question 4
df_4b = df[df.Question == '4']
df_4b_long = pd.wide_to_long(df_4b, ['Count'], i='Reports', j='Condition', sep='_', suffix=r'\w+').reset_index()
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
fig.suptitle("Duration of adverse effects")
sns.barplot(data=df_4b_long, x='Reports', y='Count', hue='Condition', err_kws={'linewidth': 0}, alpha=0.7,
            hue_order=['Anodal', 'Cathodal', 'Sham'], palette=[color_orange, color_red, color_blue], ax=ax)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.bar_label(ax.containers[2])
plt.ylim(0, 16)
sns.despine(right=False)
plt.yticks([])
plt.xlabel('')
plt.subplots_adjust(bottom=0.2, left=0.05)
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.xaxis.grid(True, which='minor', color='gray', lw=2)
plt.legend(framealpha=1, edgecolor='gray')
plt.savefig(result_path + 'Fig3b.png')
plt.show()

# Question 5
df_4c = df[df.Question == '5']
df_4c_long = pd.wide_to_long(df_4c, ['Count'], i='Reports', j='Condition', sep='_', suffix=r'\w+').reset_index()

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
fig.suptitle("Location of adverse effects")
sns.barplot(data=df_4c_long, x='Reports', y='Count', hue='Condition', err_kws={'linewidth': 0}, alpha=0.7,
            hue_order=['Anodal', 'Cathodal', 'Sham'], palette=[color_orange, color_red, color_blue], ax=ax)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.bar_label(ax.containers[2])
plt.ylim(0, 16)
sns.despine(right=False)
plt.yticks([])
plt.xlabel('')
plt.subplots_adjust(bottom=0.2, left=0.05)
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.xaxis.grid(True, which='minor', color='gray', lw=2)
plt.legend(framealpha=1, edgecolor='gray')
plt.savefig(result_path + 'Fig3c.png')
plt.show()

# Question 5 skin redness
df_4d = df[df.Question == 'skin_redness']
df_4d_long = pd.wide_to_long(df_4d, ['Count'], i='Reports', j='Condition', sep='_', suffix=r'\w+').reset_index()

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
fig.suptitle("Location of skin redness (experimenter-reported)")
sns.barplot(data=df_4d_long, x='Reports', y='Count', hue='Condition', err_kws={'linewidth': 0}, alpha=0.7,
            hue_order=['Anodal', 'Cathodal', 'Sham'], palette=[color_orange, color_red, color_blue], ax=ax)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.bar_label(ax.containers[2])
plt.ylim(0, 16)
sns.despine(right=False)
plt.yticks([])
plt.xlabel('')
plt.subplots_adjust(bottom=0.2, left=0.05)
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.xaxis.grid(True, which='minor', color='gray', lw=2)
plt.legend(framealpha=1, edgecolor='gray')
plt.savefig(result_path + 'Fig3d.png')
plt.show()
