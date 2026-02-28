import glob
import os
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Event_array_modifiers import find_events
from Save_and_Report import create_report
from scipy.signal import find_peaks

from SS_output_to_EM_input import load_sim_output



# loading function
def match_EM_with_SS_output(data, dataset, csv_file):
    # find local path
    sim_df = pd.read_csv(csv_file)
    tag = data.patient_tag[0]
    if dataset == 'mgh':
        sim_path = [p for p in sim_df.SS_path if tag in p]
    elif dataset == 'redeker':
        sim_path = [p for p in sim_df.SS_path if str(tag) in p]
    elif dataset == 'rt':
        sim_path = [p.split('.hf5')[0] for p in sim_df.patient_num if tag in p]
    elif dataset == 'bdsp':
        sim_path = [p for p in sim_df.subjectID if tag in p]
    assert len(sim_path) == 1, f'.hf5 file not found for patient {tag}'

    return sim_path[0], sim_df


def add_arousals(data, version, dataset, hf5_folder, csv_file):
    # find matching SS output path
    sim_path, sim_df = match_EM_with_SS_output(data, dataset, csv_file)
    path = hf5_folder + sim_path + '.hf5'

    # retrieve arousals
    if not 'Simulation' in version:
        arousals, _ = load_sim_output(path, ['arousal'])
        assert len(arousals) > 0.99 * \
            len(data), 'Something is going wrong when adding arousals!'
        # assert len(data) == len(arousals), 'Arousal recording does not match length'
        data.loc[:len(arousals)-1, 'Arousals'] = arousals.values

    # add group
    group = sim_df.loc[np.where(sim_df.SS_path == sim_path), 'SS group'].values[0]

    return data, group


def convert_ss_seg_scores_into_arrays(data):
    # run over all segments from (N)REM
    for stage in ['nrem', 'rem']:
        starts = data[f'{stage}_starts'].dropna().values.astype(int)
        ends = data[f'{stage}_ends'].dropna().values.astype(int)
        seg_scores = data[f'{stage}_SS_score'].values
        # convert scores into array
        for st, end, score in zip(starts, ends, seg_scores):
            data.loc[st-1:end-2, 'SS_score'] = score
        # remove singe scores
        data.drop(columns=[f'{stage}_SS_score'])

    return data


def extract_patient_metrics(hdr, dataset, csv_file):
    if dataset == 'rt':
        return hdr, {}

    # find local path
    sim_df = pd.read_csv(csv_file)
    tag = hdr['patient_tag']

    # set metric map per dataset
    if dataset in ['mgh', 'bdsp']:
        ind = np.where(sim_df.SS_path == tag)[0][0]
        metric_map = {
            'Sex': 'Sex',
            'age': 'Age',
            'ahi_3%': 'AHI',
            'Obs_i': 'OAI',
            'cai_3%': 'CAI',
            'Mix_i': 'MAI',
            'Hyp_i': 'HI',
        }
    elif dataset == 'redeker':
        ind = np.where(sim_df.ID.astype(str) == str(tag)[:4])[0][0]
        metric_map = {
            'Sex': 'Sex',
            'age': 'Age',
            'ahi_3%': 'AHI',
            'cai': 'CAI',
        }
        for i in range(len(sim_df)):
            sim_df.loc[i, 'Sex'] = {1: 'Male', 2: 'Female'}[sim_df.loc[i, 'Sex']]

    # extract relevant metrics
    for metric in metric_map.keys():
        if type(sim_df.loc[ind, metric]) == str:
            hdr[metric_map[metric]] = sim_df.loc[ind, metric]
        else:
            if metric == 'age':
                hdr[metric_map[metric]] = sim_df.loc[ind, metric].astype(int)
            else:
                hdr[metric_map[metric]] = sim_df.loc[ind, metric].round(1)

    return hdr, metric_map

# post-processing


def post_process_estimated_arousals(data, arousal_dur):
    # set Vd
    tag = '1' if data.loc[0, 'Vo_est_scaled1'].round(5) == 0 else '2'
    data['Vd_est_scaled'] = data['Vo_est_scaled'+tag] - data['Arousal'+tag]
    data['Vd_est'] = data['Vo_est'+tag] - data['Arousal'+tag]

    # define scale
    locs = data['Arousal'+tag] == 0
    data['Aest_loc'] = data['Arousal'+tag] > 0

    # compute new arousal heights
    Vo_diff = data['Vd_est_scaled'] - data['Vd_est']
    Vo_diff[locs] = 0
    data['Arousal_unscaled'] = data['Arousal'+tag] - Vo_diff
    data.loc[data['Arousal_unscaled'] < 0, 'Arousal_unscaled'] = 0

    # set corrected Vo
    data['Vo_est_corrected'] = data['Vd_est_scaled'] + data['Arousal_unscaled']
    # plt.plot(data['Vo_est_scaled'], 'k', alpha=0.8)
    # plt.plot(data['Vo_est'], 'b', alpha=0.5)
    # plt.plot(data['Vd_est_scaled'], 'k--')
    # plt.plot(data['Vd_est'], 'b--')
    # plt.plot(data['Vo_est_corrected'], 'r')
    # plt.show()

    return data


def post_process_EM_output(data, thresh=0.8):
    # set run variables
    starts_nrem = data['nrem_starts'].dropna().values.astype(int)
    ends_nrem = data['nrem_ends'].dropna().values.astype(int)
    LGs_nrem = np.round(data['LG_nrem'].values[:len(starts_nrem)], 2)
    starts_rem = data['rem_starts'].dropna().values.astype(int)
    ends_rem = data['rem_ends'].dropna().values.astype(int)
    LGs_rem = np.round(data['LG_rem'].values[:len(starts_rem)], 2)
    starts = np.concatenate([starts_nrem, starts_rem])
    ends = np.concatenate([ends_nrem, ends_rem])
    LGs = np.concatenate([LGs_nrem, LGs_rem])
    loc = np.argsort(starts)
    starts = starts[loc]
    ends = ends[loc]
    LGs = LGs[loc]

    # extract all consecutive LG values
    LGs_corrected = np.array(LGs)
    win = 5
    cnt = win//2
    len_ = len(LGs[win//2:-win//2])
    while cnt < len_:
        # set original LG
        LG = LGs_corrected[cnt]
        # skip all NaNs
        window = LGs_corrected[cnt-win//2:cnt+1+win//2]
        if all(np.isnan(window)):
            LGs_corrected[cnt] = np.nan
            cnt += 1
            continue
        # extract neighbor values
        inds = np.array([cnt-2, cnt-1, cnt+1, cnt+2])
        neighbors = LGs_corrected[inds]
        median = np.nanmedian(neighbors)
        mean = np.nanmean(neighbors)
        # smooth output if mean&median jump > threshold (and mean&median was below 0)
        if (LG-median) > thresh and (LG-mean) > thresh and np.logical_and(median < 1, mean < 1):
            val = round(np.mean(neighbors), 2)
            # print(LG, neighbors, val)
            LGs_corrected[cnt] = val
            cnt = win//2
            continue
        else:
            LGs_corrected[cnt] = LG
        cnt += 1

    # print([c for c in zip(LGs, LGs_corrected)])

    # insert into DF
    for st, end, LG, LG_c in zip(starts, ends, LGs, LGs_corrected):
        tag = 'nrem'
        loc = np.where(data['nrem_starts'] == st)[0]
        if len(loc) == 0:
            tag = 'rem'
            loc = np.where(data['rem_starts'] == st)[0]
        data.loc[loc[0], f'LG_{tag}_corrected'] = LG_c

    return data

# plotting function


def plot_EM_output_per_segment(data_og, hdr, metric_map, start, end, stage, arousal_dur, out_folder, Ut_smooth, hf5_folder, csv_file):
    # set segment
    data = data_og.loc[start-1:end-2].reset_index(drop=True).copy()
    fs = hdr['Fs']
    fz = 14

    # find matching SS output path, and extract SS data
    sim_path, _ = match_EM_with_SS_output(data_og, dataset, csv_file)
    path = hf5_folder + sim_path + '.hf5'
    SS_df, SS_hdr = load_sim_output(path, ['apnea'])
    assert len(SS_df) == len(data_og), 'matching SS output does not match EM data'
    SS_df = SS_df.loc[start-1:end-2].reset_index(drop=True).copy()
    # min_events = len(find_events(SS_df.flow_reductions>0))>=4
    min_events = len(find_events(SS_df.apnea > 0)) >= 4

    # Extract parameters, and set out_path
    loc = np.where(data_og[f'{stage}_starts'] == start)[0][0]
    SS_seg = round(data.loc[0, 'SS_score'], 2)
    Error = round(data_og.loc[loc, 'rmse_Vo'], 2)
    Vmax = round(data_og.loc[loc, 'Vmax'], 2)
    LG = round(data_og.loc[loc, f'LG_{stage}_corrected'], 2)
    G = data_og.loc[loc, f'G_{stage}']
    Delay = data_og.loc[loc, f'D_{stage}']
    L = data_og.loc[loc, f'L_{stage}']
    Alpha = data_og.loc[loc, f'Alpha_{stage}']
    ex = '' if Error < 1.8 else 'Excl. '
    ex = '' if min_events else 'Exc. '
    tag = f'{stage} {ex}LG({LG}) g({G}) d({Delay}) a({Alpha}) -{SS_seg}- {start}-{end}.png'
    out_path = out_folder + tag
    if os.path.exists(out_path):
        return

    # post-process V_est and Arousals
    data = post_process_estimated_arousals(data, arousal_dur*fs)

    # set signal arrays
    ABD = data.ABD.values
    Vo = data.Ventilation_ABD.values
    Vd_est_scaled = data.Vd_est_scaled.values
    Vo_est_scaled = data.Vo_est_corrected.values
    a_locs = data.Aest_loc.values.astype(float)
    di_abd = data.d_i_ABD.values if Ut_smooth == 'non-smooth' else data.d_i_ABD_smooth.values
    Disturbance = di_abd + Alpha*(1-di_abd)
    spo2 = data.SpO2.values.astype(float)

    # label arrays
    y_tech = data.Apnea.values
    y_algo = data.Apnea_algo.values
    if 'Arousals' in data.columns:
        arousals = data.Arousals.values
    else:
        arousals = np.zeros(len(data))

    ##########
    # Compute CO2 levels using Euler integration:
    # Differential equation: dx/dt = L - v_o(t-tau) * x(t)
    dt = 1.0 / fs
    delay_steps = int(Delay * fs)
    initial_CO2 = 1.0  # initial CO2 level (arbitrary units)
    CO2 = np.zeros(len(Vo_est_scaled))
    CO2[0] = initial_CO2
    for i in range(1, len(Vo_est_scaled)):
        if i - delay_steps >= 0:
            v_delayed = Vo_est_scaled[i - delay_steps]
        else:
            v_delayed = Vo_est_scaled[0]
        dCO2 = L - v_delayed * CO2[i-1]
        CO2[i] = CO2[i-1] + dCO2 * dt
    ##########

    # setup figure
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(111)
    label_txt_dic = {'fontsize': fz, 'ha': 'right', 'va': 'center'}

    ## PLOT SIGNALS ##
    # plot effort trace
    factor = 4
    maxi = np.nanmax(ABD) - np.nanmin(ABD)
    ABD_n = ABD/maxi * factor*2
    ax.plot(ABD_n, c='k', lw=0.5, alpha=0.75)
    max_y = np.median(ABD_n[find_peaks(ABD_n, distance=fs)[0]])
    min_y = -np.median(ABD_n[find_peaks(-ABD_n, distance=fs)[0]])
    ax.text(-10*fs, 0, 'Abdominal effort', label_txt_dic)

    # plot arousals
    arousals[arousals != 1] = np.nan
    a_locs[a_locs != 1] = np.nan
    offset = max_y + 8.5
    ax.axhline(offset, c='k', lw=0.5, linestyle='dashed')
    ax.axhline(offset-1, c='k', lw=0.5, linestyle='dashed')
    ax.plot(arousals*offset, c='k', lw=4)
    ax.plot(a_locs*offset-1, c='k', lw=4)
    ax.text(-10*fs, offset, 'Manual arousals', label_txt_dic)
    ax.text(-10*fs, offset-1, 'Estimated arousals', label_txt_dic)

    # PLOT LABELS
    if True:
        label_color = [None, 'b', 'g', 'c', 'm', 'r', None, 'b']
        offset = max_y + 6.5

        # create hline
        ax.axhline(offset, c='k', lw=0.5, linestyle='dashed')
        ax.text(-10*fs, offset, 'Manual resp. events', label_txt_dic)

        # run over all events
        loc = 0
        for i, j in groupby(y_tech):
            len_j = len(list(j))
            if np.isfinite(i) and label_color[int(i)] is not None:
                # add scored events
                ax.plot([loc, loc+len_j], [offset]*2, c=label_color[int(i)], lw=3)
            # update loc
            loc += len_j

    # plot Ut
    offset = max_y + 3.75
    factor = 2
    ax.text(-2*fs, offset+factor, '1', fontsize=fz-5, ha='right', va='center')  # max
    ax.text(-2*fs, offset, '0', fontsize=fz-5, ha='right', va='center')  # min
    ax.plot(di_abd*factor + offset, c='k', lw=1, alpha=0.25)
    ax.plot(Disturbance*factor + offset, c='k', lw=2, alpha=0.5)
    ut_txt_dic = {'fontsize': fz, 'ha': 'right', 'va': 'top'}
    ax.text(-10*fs, offset+1, 'Disturbance ($U$)', label_txt_dic)
    # add shade
    ax.fill_between([0, len(data)], offset, offset+factor, fc='k', alpha=0.1)

    # plot spo2
    offset = min_y - 8.5
    factor = 5
    spo2[np.less(spo2, 80, where=np.isfinite(spo2))] = np.nan
    if any(np.isfinite(spo2)):
        spo2_n = (spo2-np.nanmin(spo2)) / (np.nanmax(spo2)-np.nanmin(spo2)) * factor
        ax.plot(spo2_n + offset, c='y', lw=1)
    ax.text(-10*fs, offset + factor/2, 'SpO$_{2}$', label_txt_dic)  # title
    maxi, mini = ('NaN', 'NaN') if np.all(np.isnan(spo2)) else (
        int(np.nanmax(spo2)), int(np.nanmin(spo2)))
    ax.text(-2*fs, offset + factor, f'{maxi}%',
            fontsize=fz-5, ha='right', va='top')  # max
    ax.text(-2*fs, offset, f'{mini}%', fontsize=fz-5, ha='right', va='bottom')  # min

    # format Ventilation* traces
    offset_V = min_y - 17
    factor = 5
    maxi = max(np.nanmax(Vo), np.nanmax(Vo_est_scaled))
    mini = 0
    Vo_n = (Vo-mini) / (maxi-mini) * factor
    Vo_est_scaled_n = (Vo_est_scaled-mini) / (maxi-mini) * factor
    Vd_est_scaled_n = (Vd_est_scaled-mini) / (maxi-mini) * factor
    Vo_est_scaled_n -= np.nanquantile(Vo_est_scaled_n, 0.002)
    Vd_est_scaled_n -= np.nanquantile(Vd_est_scaled_n, 0.002)
    Vo_est_scaled_n[Vo_est_scaled_n < 0] = 0
    Vd_est_scaled_n[Vd_est_scaled_n < 0] = 0
    # correct V_ar
    Var_est_scaled_n = np.array(Vo_est_scaled_n)
    same = Vd_est_scaled_n == Var_est_scaled_n
    # Var_est_scaled_n[same] = np.nan
    Vd_est_scaled_n[~same] = np.nan
    for st, end in find_events(np.isfinite(Var_est_scaled_n)):
        Var_est_scaled_n[st] = Vd_est_scaled_n[st]
        Var_est_scaled_n[end:end+2] = Vd_est_scaled_n[end:end+2]

    # plot Vo & Vo_est
    ax.plot(Vo_n + offset_V, c='k', lw=2, alpha=1)
    ax.plot(Var_est_scaled_n + offset_V, c='b', linestyle='solid', lw=2)
    ax.plot(Vd_est_scaled_n + offset_V, c='b', linestyle='solid', lw=2)
    ax.text(-10*fs, offset_V + factor/2, 'Ventilation', label_txt_dic)
    ax.axhline(offset_V, c='k', lw=0.5, linestyle='dashed')  # box
    ax.axhline(offset_V+factor, c='k', lw=0.5, linestyle='dashed')
    # add modeled arousal locations
    yl = [offset_V]*2
    yu = [max(np.nanmax(Vo_n), np.nanmax(Vo_est_scaled_n))]*2
    for st, end in find_events(data['Aest_loc'] == 1):
        ax.fill_between([st, end], yl, yu+offset_V, color='k', alpha=0.1, ec='none')

    ##########
    # PLOT CO₂ LEVELS (x)
    # Normalize CO₂ for plotting
    offset_CO2 = min_y - 12  # adjust offset to avoid overlap with ventilation trace
    factor_CO2 = 3

    # Define the duration (in seconds) to mask at the start, e.g., first 30 seconds
    mask_duration = Delay*2.5
    mask_steps = int(mask_duration * fs)  # convert to number of samples

    # Normalize the CO2 array using values after the initial mask period
    CO2_n = (CO2 - np.nanmin(CO2[mask_steps:])) / \
        (np.nanmax(CO2[mask_steps:]) - np.nanmin(CO2[mask_steps:])) * factor_CO2

    # Create non-delayed (instantaneous) CO2 trace and mask its initial part
    CO2_non_delayed = np.copy(CO2_n)
    CO2_non_delayed[:mask_steps] = np.nan

    # Create delayed CO2 trace by shifting the non-delayed trace by delay_steps (τ)
    CO2_delayed = np.roll(CO2_n, delay_steps)
    # mask the beginning where the shift wraps around
    CO2_delayed[:delay_steps+mask_steps] = np.nan

    # Plot both traces: non-delayed in dashed black, delayed in solid red
    ax.plot(CO2_non_delayed + offset_CO2, c='k', lw=1,
            linestyle='dashed', alpha=0.1, label='CO₂ (non-shifted)')
    ax.plot(CO2_delayed + offset_CO2, c='b', lw=1.5,
            linestyle='dashed', alpha=0.75, label='CO₂ (shifted by τ)')

    # Label the CO₂ plot and add a horizontal guide line
    ax.text(-10*fs, offset_CO2 + factor_CO2/2, 'Modeled CO₂ (x)', label_txt_dic)
    # ax.axhline(offset_CO2 + factor_CO2, c='k', lw=0.5, linestyle='dashed')

    # Identify location of highest CO₂ level in the delayed trace (only consider finite values)
    first_CO2_index = np.where(np.isfinite(CO2_delayed))[0][0]
    max_CO2_value = CO2_delayed[first_CO2_index] + 0.25

    # Determine the x-coordinate for the left end of the arrow (width equals delay_steps)
    left_index = first_CO2_index - delay_steps

    # Draw a horizontal double-headed arrow (no text in the annotation)
    ax.annotate(
        '',
        xy=(first_CO2_index, max_CO2_value + offset_CO2),
        xytext=(left_index, max_CO2_value + offset_CO2),
        arrowprops=dict(arrowstyle='<->', color='k', lw=1)
    )

    # Compute the midpoint of the arrow
    mid_x = (first_CO2_index + left_index) / 2
    mid_y = max_CO2_value + offset_CO2 + 0.25
    va = 'bottom'
    if mid_y > 4:
        mid_y = max_CO2_value + offset_CO2 - 0.25
        va = 'top'
    # Add the τ label at the midpoint below the arrow
    ax.text(mid_x, mid_y, '$\\tau$', fontsize=fz-2, ha='center', va=va)

    yu = [offset_CO2]*2
    yl = [offset_CO2+factor_CO2]*2
    ax.fill_between([0, left_index], yl, yu, color='k', alpha=0.1, ec='none')
    ax.text(fs, offset_CO2+0.25, 'Calibrating..', fontsize=fz -
            4, fontstyle='italic', ha='left', va="bottom")
    ##########

    # plot layout setup
    ax.set_xlim([-5, len(data)+5])
    ax.axis('off')

    ### construct legend box ###
    len_x = len(data)

    # add global metrics + SS
    metric_map['SS'] = 'SS'
    hdr['SS'] = SS_seg
    dx = len_x//30
    y = max_y + 10
    for i, metric in enumerate(metric_map.values()):
        x = 30 + dx*i*2
        ax.text(x+dx/2, y, metric, fontsize=fz, ha='center', va='bottom')
        ax.text(x+dx/2, y-0.25, hdr[metric], fontsize=fz-2, ha='center', va='top')

    # add event legend
    event_types = ['RERA', 'Hypopnea', 'Mixed\napnea',
                   'Central\napnea', 'Obstructive\napnea']
    label_colors = ['r', 'm', 'c', 'g', 'b']
    dx = len_x//25
    y = max_y + 10
    for i, (color, e_type) in enumerate(zip(label_colors, event_types)):
        x = len_x-20 - dx*i*2
        ax.plot([x, x-dx], [y-0.5]*2, c=color, lw=3)
        ax.text(x-dx/2, y, e_type, fontsize=fz-2, ha='center', va='bottom')

    # add Ventilation legend
    lines = ['Observed', 'Modeled']
    line_colors = ['k', 'b']
    line_styles = ['solid', 'solid']
    dx = len_x//22.5
    offset = offset_V - 1
    for i, (line, c, ls) in enumerate(zip(lines, line_colors, line_styles)):
        x = dx*i*1.5
        ax.plot([x, x+dx], [offset+0.25]*2, c=c, lw=2, linestyle=ls)
        ax.text(x+dx/2, offset, line, fontsize=fz-2, ha='center', va='top')

    # add LG & Delay estimation
    scores = [r"$\bf{LG}$", '$\\gamma$', '$\\tau$',
              '$v_{max}$', '$L$', '$\\alpha$', 'RMSE']
    values = [r"$\bf{" + str(LG) + "}$", G, f'{Delay} sec', Vmax, L, Alpha, Error]
    for i, (tag, val) in enumerate(zip(scores, values)):
        minus = np.floor(len(scores)/2)
        x = len_x/2 + ((i-minus)*2)*dx
        ax.text(x, offset-0.5, tag, fontsize=fz, ha='center', va='bottom')
        ax.text(x, offset-0.75, val, fontsize=fz-2, ha='center', va='top')

    # add duration marker
    sec = 30
    ax.plot([len_x-sec*fs, len_x], [offset+0.25]*2, color='k', lw=1.5)  # hz
    ax.plot([len_x-sec*fs]*2, [offset, offset+0.5], color='k', lw=1.5)  # left va
    ax.plot([len_x]*2, [offset, offset+0.5], color='k', lw=1.5)         # right va
    ax.text(len_x-sec/2*fs, offset, f'{sec} sec\n({stage})',
            color='k', fontsize=fz-2, ha='center', va='top')

    # set title
    # num = hdr['Study_num']
    # plt.title(f'ID: {num}\nversion: {version}')
    # plt.tight_layout()
    # plt.show()
    # import pdb; pdb.set_trace()
    # save the figure

    plt.savefig(out_path, dpi=900)
    plt.close()


def plot_full_night(EM_data, EM_hdr, figure_path, hf5_folder, csv_file, plot_all_tagged=False):
    # find matching SS output path
    sim_path, _ = match_EM_with_SS_output(EM_data, dataset, csv_file)
    path = hf5_folder + sim_path + '.hf5'

    # extract SS data
    SS_data, hdr = load_sim_output(path)
    assert len(SS_data) == len(EM_data), 'matching SS output does not match EM data'

    # cut excessive start/end Wake
    EM_data, SS_data = remove_excessive_wake(EM_data, SS_data, hdr['newFs'])

    # create report
    SS_data = SS_data.rename(
        columns={'self similarity': 'T_sim', 'sleep_stages': 'stage'})
    _, summary_report = create_report(SS_data, hdr)

    # set signal variables
    signal = SS_data.abd.values.astype(float)
    sleep_stages = SS_data.stage.values.astype(float)
    y_algo = SS_data.flow_reductions.values.astype(float)
    below_u = np.array(EM_data.d_i_ABD_smooth < 1).astype(int)
    tagged_breaths = SS_data.tagged.values.astype(float)
    ss_conv_score = SS_data.ss_conv_score.values.astype(float)
    selfsim = SS_data.T_sim.values.astype(int)

    # define the ids each row
    fs = hdr['newFs']
    block = 60*60*fs
    row_ids = [np.arange(i*block, (i+1)*block) for i in range(len(signal)//block+1)]
    row_ids.reverse()
    row_ids[0] = np.arange(row_ids[0][0], len(SS_data))
    nrow = len(row_ids)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    row_height = 16

    # set sleep array
    sleep = np.array(signal)
    sleep[np.isnan(sleep_stages)] = np.nan
    sleep[sleep_stages == 5] = np.nan
    # set wake array
    wake = np.zeros(signal.shape)
    wake[np.isnan(sleep_stages)] += signal[np.isnan(sleep_stages)]
    wake[sleep_stages == 5] += signal[sleep_stages == 5]
    wake[wake == 0] = np.nan
    # set rem array
    rem = np.array(signal)
    rem[sleep_stages != 4] = np.nan

    # PLOT SIGNALS
    for ri in range(nrow):
        # set autoscale factor
        factor = 1

        # plot signal
        ax.plot(sleep[row_ids[ri]] + ri*row_height, c='k', lw=.3, alpha=0.75)
        ax.plot(wake[row_ids[ri]] + ri*row_height, c='r', lw=.3, alpha=0.5)
        ax.plot(rem[row_ids[ri]] + ri*row_height, c='b', lw=.3, alpha=0.5)

        # set max_y
        if ri == nrow-1:
            max_y = np.nanmax([sleep[row_ids[ri]], wake[row_ids[ri]],
                              rem[row_ids[ri]]]) + ri*row_height

    # PLOT LABELS
    for yi in range(3):
        if yi == 0:
            labels = y_algo                 # plot resp eventsl
            label_color = [None, 'b', 'b', 'b', 'm']
        # if yi==0:
        #     labels = below_u                 # plot below eupnea
        #     label_color = [None, 'm']
        if yi == 1:
            labels = tagged_breaths         # '*' for HLG breathing oscillations
            label_color = [None, 'k', 'r']
        if yi == 2:
            labels = selfsim         # '*' for HLG breathing oscillations
            label_color = [None, 'b']

        # run over each plot row
        for ri in range(nrow):
            # group all labels and plot them
            loc = 0
            for i, j in groupby(labels[row_ids[ri]]):
                len_j = len(list(j))
                if not np.isnan(i) and label_color[int(i)] is not None:
                    if yi == 0:
                        # add scored events
                        shift = 3.5 if i == 1 else 4
                        ax.plot([loc, loc+len_j], [ri*row_height-shift] *
                                2, c=label_color[int(i)], lw=1.5, alpha=1)
                    if yi == 1:
                        # add tags
                        tag = 'o' if i == 1 else '\''
                        c_score = np.round(ss_conv_score[row_ids[ri]][loc], 2)
                        c, sz = ('b', 6) if c_score >= hdr['SS_threshold'] else ('k', 8)
                        if c_score >= hdr['SS_threshold'] or plot_all_tagged:
                            offset = -5
                            ax.text(loc, ri*row_height+offset, tag, c=c,
                                    ha='center', va='center', fontsize=sz)
                            # ax.text(loc, ri*row_height+offset+2, c_score, c=c, ha='center', va='top', fontsize=6)
                    if yi == 2:
                        # add SS bar
                        ymin = ri*row_height-0.4*row_height
                        ymax = ri*row_height+0.4*row_height
                        # ax.fill_between([loc, loc+len_j], ymin, ymax, color='k', alpha=0.1, ec=None)
                        # ax.plot([loc, loc+len_j], [ri*row_height+offset+2]*2, c=label_color[int(i)], lw=2)

                loc += len_j

    # PLOT EM segments
    add_LG_hooks(EM_data, SS_data, EM_hdr, row_ids, nrow, row_height, fs, ax)

    # plot layout setup
    ax.set_xlim([0, max([len(x) for x in row_ids])])
    ax.axis('off')

    ### construct legend box ###
    len_x = len(row_ids[-1])
    fz = 11
    offset = row_height*(nrow-1) + 17
    dx = len_x//10

    # add summary report
    for i, key in enumerate(summary_report.keys()):
        tag = key.replace('detected ', '') + ':\n' + str(summary_report[key].values[0])
        ax.text((i)*dx, offset, tag, fontsize=7, ha='left', va='bottom')

    # add line legend (bottom right)
    y = -10
    line_types = ['NREM', 'REM', 'Wake']
    line_colors = ['k', 'b', 'r']
    for i, (color, e_type) in enumerate(zip(line_colors, line_types)):
        x = 60*fs + 200*fs*i
        ax.plot([x, x+50*fs], [y]*2, c=color, lw=0.8)
        ax.text(x+25*fs, y-3, e_type, fontsize=fz, c=color, ha='center', va='top')

    # lines types
    event_types = ['Apnea', 'Hypopnea']
    label_colors = ['b', 'm']
    for i, (color, e_type) in enumerate(zip(label_colors, event_types)):
        x = 200*fs*(len(line_types)+0.5) + 300*fs*(i+1)
        ax.plot([x, x+100*fs], [y]*2, c=color, lw=2)
        ax.text(x+50*fs, y-3, e_type, fontsize=fz, ha='center', va='top')

    # add <duration> min marking
    duration = 5
    ax.plot([len_x-60*fs*duration, len_x], [y]*2,
            color='k', lw=1)             # <duration>
    ax.plot([len_x-60*fs*duration]*2, [y-0.5, y+0.5], color='k', lw=1)  # left va
    ax.plot([len_x]*2, [y-0.5, y+0.5], color='k', lw=1)                 # right va
    ax.text(len_x-60*fs*(duration/2), y+1,
            f'{duration} min', color='k', fontsize=fz, ha='center', va='bottom')
    ax.text(len_x-60*fs*(duration/2), y-1, f'(abd RIP)',
            color='k', fontsize=8, ha='center', va='top')

    # add (*) detected HLG oscillation
    tag = 'Detected SS\nbreathing oscillation'
    ax.text(len_x-60*fs*(duration/2)-2*dx, y-3, tag,
            color='k', fontsize=fz-1, ha='center', va='top')
    ax.text(len_x-60*fs*(duration/2)-2*dx, y, 'o',
            c='b', fontsize=fz, ha='center', va='bottom')

    # add |---| LG estimation
    tag = 'Estimated LG'
    duration = 8*60*fs
    left = len_x - 4.75*dx
    right = left + duration
    ax.text(left+duration/2, y-3, tag, color='k', fontsize=fz-1, ha='center', va='top')
    ax.plot([left, right], [y]*2, color='k', lw=0.5)
    ax.plot([left]*2, [y-1, y], color='k', lw=0.5)  # left hook
    ax.plot([right]*2, [y-1, y], color='k', lw=0.5)  # right hook

    # save Figure
    plt.tight_layout()
    plt.savefig(fname=figure_path + '.pdf', format='pdf', dpi=1200)
    # plt.savefig(fname=figure_path, format='png', dpi=1200)
    plt.close()


def add_LG_hooks(data, SS_data, hdr, row_ids, nrow, row_height, fs, ax):
    len_x = len(row_ids[-1])
    for stage in ['nrem', 'rem']:
        # set run variables
        starts = data[f'{stage}_starts'].dropna().values.astype(int)
        ends = data[f'{stage}_ends'].dropna().values.astype(int)
        if len(starts) == 0:
            continue
        LGs = np.round(data[f'LG_{stage}'].values[:len(starts)], 2)
        LGs_c = np.round(data[f'LG_{stage}_corrected'].values[:len(starts)], 2)
        for i, (st, end, LG, LG_c) in enumerate(zip(starts, ends, LGs, LGs_c)):
            # identify start of segment
            x_st, y_st = find_row_location(st, row_ids)
            up_st = y_st*row_height + 0.425*row_height
            # identify end of segment
            x_end, y_end = find_row_location(end, row_ids)
            up_end = y_end*row_height + 0.425*row_height
            # set plotting parameters
            yy = up_st if i % 2 == 0 else up_st + 0.075*row_height
            yy_ = up_end if i % 2 == 0 else up_end + 0.075*row_height
            hook = yy - 0.05*row_height
            hook_ = yy_ - 0.05*row_height
            shift = yy - 0.025*row_height if i % 2 == 0 else yy + 0.01*row_height
            shift_ = yy_ - 0.025*row_height if i % 2 == 0 else yy_ + 0.01*row_height
            va = 'top' if i % 2 == 0 else 'bottom'
            # if <4 apneas, show LG estimation is ignored (faded)
            LG_alpha = 1 if len(find_events(
                SS_data.loc[st:end, 'flow_reductions'] > 0)) >= 4 else 0.3
            hook_alpha = 0.5 if LG_alpha == 1 else 0.3
            # check whether end of line correction is needed
            tag = LG if LG == LG_c or np.isnan(LG) else f'{LG} --> {LG_c}'
            if y_st == y_end:
                # plot hooks
                ax.plot([x_st+15*fs, x_end-15*fs], [yy, yy],
                        'k', lw=0.5, alpha=hook_alpha)
                ax.plot([x_st+15*fs, x_st+15*fs], [yy, hook],
                        'k', lw=0.5, alpha=hook_alpha)
                ax.plot([x_end-15*fs, x_end-15*fs], [yy, hook],
                        'k', lw=0.5, alpha=hook_alpha)
                # add LG
                x = (x_st+x_end)/2
                ax.text(x, shift, tag, fontsize=6, ha='center', va=va, alpha=LG_alpha)
            else:
                # plot hooks, and add correction in case of line split
                ax.plot([x_st+15*fs, len_x], [yy, yy], 'k', lw=0.5, alpha=hook_alpha)
                ax.plot([x_st+15*fs, x_st+15*fs], [yy, hook],
                        'k', lw=0.5, alpha=hook_alpha)
                ax.plot([0, x_end-15*fs], [yy_, yy_], 'k', lw=0.5, alpha=hook_alpha)
                ax.plot([x_end-15*fs, x_end-15*fs], [yy_, hook_],
                        'k', lw=0.5, alpha=hook_alpha)
                # add LG
                half_win = (4*60*fs)
                if x_st+half_win <= len_x:
                    x = x_st + half_win
                    ax.text(x, shift, tag, fontsize=6,
                            ha='center', va=va, alpha=LG_alpha)
                else:
                    x = x_end - half_win
                    ax.text(x, shift_, tag, fontsize=6,
                            ha='center', va=va, alpha=LG_alpha)


def find_row_location(loc, row_ids):
    # run over all row ids
    for i, row in enumerate(row_ids):
        match = np.where(loc == row)[0]
        if len(match) == 0:
            continue
        # return matching x,y location
        return match[0], i

    raise Exception('No matching index found..!')


def remove_excessive_wake(EM_data, SS_data, Fs):
    # cut trailing NaN values
    SS = EM_data['Stage'].values
    end = np.where(np.isfinite(SS))[0][-1]
    EM_data = EM_data.loc[:end, :].reset_index(drop=True)
    SS_data = SS_data.loc[:end, :].reset_index(drop=True)

    # cut >1hr of trailing Wake
    thresh = int(3600*Fs / 2)
    start = np.where(SS < 5)[0][0]
    end = np.where(SS < 5)[0][-1]
    if SS[-1] == 5 and end < len(SS)-thresh:
        EM_data = EM_data.loc[:end+thresh, :].reset_index(drop=True)
        SS_data = SS_data.loc[:end+thresh, :].reset_index(drop=True)

    return EM_data, SS_data


if __name__ == '__main__':
    dataset = 'mgh'
    if dataset == 'mgh':
        selection = 'HLG_OSA'
    elif dataset == 'redeker':
        selection = 'Heart_Failure'
    elif dataset == 'rt':
        selection = 'Altitude'
    elif dataset == 'bdsp':
        selection = 'CPAP_failure'
    version = f'{dataset}_{selection}'
    Ut_smooth = 'non-smooth'  # smooth   non-smooth

    # SS output folder
    arousal_dur = 5
    if dataset == 'mgh':
        hf5_folder = 'Paper example files/'
    elif dataset == 'bdsp':
        hf5_folder = 'SS paper files/'
    csv_files = 'csv_files/'
    # set input folder TODO: set path to this dropbox folder
    input_folder = f'Drobbox/Final Code Revision/{Ut_smooth}/{version}/'
    input_files = glob.glob(input_folder + '*.csv')
    input_files.sort(reverse=True)

    # set Figure output folder
    output_folder = f'EM_algo_segment_figures/{Ut_smooth}/{version}/'
    os.makedirs(output_folder, exist_ok=True)

    # run over all fies
    print(f'>> {version} <<')
    for input_file in np.sort(input_files):
        try:
            # extract data
            data = pd.read_csv(input_file)

            # convert SS scores into columns
            data = convert_ss_seg_scores_into_arrays(data)

            # extract header
            num = input_file.split(
                '/Study')[-1].split('\\Study')[-1].split('.csv')[0].split('(')[0]
            csv_tag = version.split(f'{dataset}_')[-1].split('_V')[0]
            csv_file = f'csv_files/{dataset}_table1 100_{csv_tag}_cases.csv'
            if 'CPAP' in csv_file:
                csv_file = csv_file.replace('100_', '200_')
            hdr = {}
            if 'Simulation' in version:
                study_tag = 'Study ' + num.split('_')[0]
                hdr['LG_sim'] = num.split('_')[1]
                hdr['TAU_sim'] = num.split('_')[2]
                hdr['SS group'] = ''
            else:
                # add arousals
                if dataset == 'mgh':
                    data, hdr['SS group'] = add_arousals(
                        data, version, dataset, hf5_folder, csv_file)
                study_tag = f'Study {num}'.replace('  ', ' ')
            for col in ['patient_tag', 'Fs', 'original_Fs']:
                hdr[col] = data.loc[0, col]

            # extact global patient metrics
            hdr, metric_map = extract_patient_metrics(hdr, dataset, csv_file)

            #############
            patient_tag = hdr['patient_tag']
            hdr['Study_num'] = study_tag
            print(f'\nCreating figures for {study_tag}')
            #############

            # post-process EM output
            data = post_process_EM_output(data)

            # create full-night figure
            figure_path = f'{output_folder}{study_tag} {patient_tag}'
            plot_full_night(data, hdr, figure_path, hf5_folder, csv_file)

            # run over all segments
            for stage in ['nrem', 'rem']:
                # set run variables
                if 'Simulation' in version:
                    if stage == 'rem':
                        continue
                    starts = [int(data.loc[0, 'ind0'] * hdr['Fs'] / hdr['original_Fs'])]
                    ends = [int(data.loc[len(data)-1, 'ind1'] *
                                hdr['Fs'] / hdr['original_Fs'])]
                    data = data.set_index(pd.Index(list(range(starts[0], ends[0]))))
                else:
                    starts = data[f'{stage}_starts'].dropna().values.astype(int)
                    ends = data[f'{stage}_ends'].dropna().values.astype(int)

                # set output folder
                sub_out_folder = output_folder + f'{study_tag}/'
                os.makedirs(sub_out_folder, exist_ok=True)

                # create plot per segment
                for i, (st, end) in enumerate(zip(starts, ends)):
                    print(f'segment {i+1}/{len(starts)}..', end='\r')
                    try:
                        plot_EM_output_per_segment(
                            data, hdr, metric_map, st, end, stage, arousal_dur, sub_out_folder, Ut_smooth, hf5_folder, csv_file)
                    except Exception as e:
                        print(f'Error processing segment {i+1}/{len(starts)}: {e}')
                        continue

        except Exception as e:
            print(f'Error processing {input_file.split("/")[-1]}: {e}')
            continue
