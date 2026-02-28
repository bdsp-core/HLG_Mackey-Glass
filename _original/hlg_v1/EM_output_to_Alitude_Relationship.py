import glob, h5py, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

# import local functions
from Event_array_modifiers import find_events
from EM_output_to_Figures import post_process_EM_output
from Convert_SS_seg_scores import convert_ss_seg_scores_into_arrays

# reading functions
def sort_dic_keys(dicts):
    sorted_dics = []
    # run over all dictionaries
    for dic in dicts:
        sorted_dic = {}
        # sort keys
        keys = list(dic.keys())
        keys.sort()
        for key in keys:
            sorted_dic[key] = dic[key]
        sorted_dics.append(sorted_dic)

    return sorted_dics

def extract_EM_output(input_files, interm_folder, version, hf5_folder):
    # set empty dictionaries
    LG_data, G_data, D_data, GxD_data, SS_data, valid_data = {}, {}, {}, {}, {}, {}
    SS_percentages = []

    # run over all fies
    for i, input_file in enumerate(input_files):
        num = input_file.split('/Study ')[-1].split('.csv')[0]
        print(f'Extracting Study {num} ({i+1}/{len(input_files)}) ..', end='\r')
        
        # extract data
        data = pd.read_csv(input_file)
        group = data.loc[0, 'patient_tag'].replace('a','A')

        # convert SS scores into columns
        data = convert_ss_seg_scores_into_arrays(data)

        # post-process EM output
        data = post_process_EM_output(data)

        # extract other header fields
        hdr = {'Study_num': f'Study {num}'}
        for col in ['patient_tag', 'Fs', 'original_Fs']:
            hdr[col] = data.loc[0, col]
            data = data.drop(columns=col)

        # extract SS% score
        SS, resp = load_SS_percentage(hf5_folder, hdr['patient_tag'])
        SS_percentages.append(SS)

        # run over all segments
        Errors, Vmaxs, LGs, Gs, Ds, Ls, SSs_seg, valid_seg = [], [], [], [], [], [], [], []
        for stage in ['nrem', 'rem']:
            # set run variables
            starts = data[f'{stage}_starts'].dropna().values.astype(int)
            ends = data[f'{stage}_ends'].dropna().values.astype(int)
            for start, end in zip(starts, ends):
                # Extract parameters, and set out_path
                loc = np.where(data[f'{stage}_starts']==start)[0][0]
                Errors.append(round(data.loc[loc, 'rmse_Vo'], 2))
                Ls.append(data.loc[loc, f'L_{stage}'])
                Vmaxs.append(round(data.loc[loc, 'Vmax'], 2))
                LGs.append(data.loc[loc, f'LG_{stage}_corrected'])
                Gs.append(data.loc[loc, f'G_{stage}'])
                Ds.append(data.loc[loc, f'D_{stage}'])
                valid_seg.append(len(find_events(resp[start:end]>0))>=5)
                SSs_seg.append(data.loc[start, 'SS_score'])
        
        # save estimated LG values per patient self-similarity group
        inds = np.array(Errors) < error_thresh
        LG_data[group] = np.array(LGs)[inds]
        G_data[group] = np.array(Gs)[inds]
        D_data[group] = np.array(Ds)[inds]
        GxD_data[group] = np.array(Gs)[inds]*np.array(Ds)[inds]
        SS_data[group] = np.array(SSs_seg)[inds]
        valid_data[group] = np.array(valid_seg)[inds]
            
    # sort dictionary
    sorted_dics = sort_dic_keys([LG_data, G_data, D_data, GxD_data, SS_data, valid_data])
    names = ['LG_data', 'G_data', 'D_data', 'GxD_data', 'SS_data', 'valid_data']
    # save per group estimations
    for group, SS in zip(LG_data.keys(), SS_percentages):
        df = pd.DataFrame([], dtype=float)
        for dic, name in zip(sorted_dics, names):
            df[name] = dic[group]
        os.makedirs(interm_folder, exist_ok=True)
        df.to_csv(f'{interm_folder}/{group}-{SS}.csv', header=df.columns, index=None, mode='w+')
 
def load_SS_percentage(hf5_folder, ID):
    # init DF
    cols = ['patient_tag', 'self similarity', 'sleep_stages', 'flow_reductions']
    data = pd.DataFrame([], columns=cols)

    # find path
    path = [p for p in glob.glob(hf5_folder+ '*.hf5') if f'{ID}.hf5' in p]
    assert len(path)==1, f'No matching SS output file found in {hf5_folder}'
    
    # extract data
    f = h5py.File(path[0], 'r')
    for key in cols:
        data[key] = f[key][:]
    f.close()

    # compute SS
    patient_asleep = np.logical_and(data['sleep_stages']>0, data['sleep_stages']<5)
    SS = np.round((np.sum(data['self similarity']==1) / (sum(patient_asleep))) * 100, 1) 

    return SS, data.flow_reductions.values

# analysis functions 
def add_statistical_significance(data, ref_data, pos, ax):
    # compute significance
    U, p = stats.mannwhitneyu(ref_data, data, alternative='two-sided')
    U, p = stats.ttest_ind(ref_data, data)
    print(pos, p)
    if p < 0.001:
        sig_symbol = '***'
    elif p < 0.01:
        sig_symbol = '**'
    elif p < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'no significance'

    # plot *stars*
    level = 0.13
    offset = - 0.07*pos
    ax.plot([1, pos], [level+offset]*2, color='k', lw=1)   # <duration>
    ax.plot([1, 1], [level+offset, level+0.015+offset], color='k', lw=1)     # left va
    ax.plot([pos, pos], [level+offset, level+0.015+offset], color='k', lw=1) # right va
    ax.text((1+pos)/2, level+offset-0.0075, sig_symbol, va='top', ha='center')

def func(x, a, b, c):
    # return a * np.exp(b * x)
    return a*x*x + b*x + c

def predband(x, xd, yd, p, func, conf=0.95):
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters

    # func = function name
    alpha = 1.0 - conf      # significance
    N = xd.size             # data sample size
    var_n = len(p)          # number of parameters

    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)

    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - func(xd, *p)) ** 2))

    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)

    # Predicted values (best-fit model)
    yp = func(x, *p)

    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))

    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy

    return lpb, upb

# histogram functions
def create_histogram_bars(LG_all, valids, percentile):
    # histogram parameters
    LG = LG_all[valids]
    max_edge = 1.4
    edges = np.arange(0, max_edge, 0.1)

    # init bins with invalids
    bins = np.zeros(len(edges)+2)
    bins[0] = sum(valids==False)

    # fill bins
    for e, edge in enumerate(edges[:-1]):
        lo, up = edge, edges[e+1]
        count = sum(np.logical_and(LG>=lo, LG<up))
        bins[e+1] = count

    # add everything above max
    bins[-1] = sum(LG>=max_edge)
    
    # normalize
    for i in range(len(bins)):
        bins[i] = bins[i] / len(valids) * 100

    # compute percentile bin
    LG_all[~valids] = 0
    pct = np.quantile(LG_all, percentile)

    return bins, pct

def plot_histogram_bins(bins, pct, axes, row, col, tag, height, c, fz):
    # set plotting hyperparameters
    ax = axes[row, col]
    n_bins = len(bins)
    width = 0.1
    x_range = np.arange(0, n_bins/10, width) - 0.1

    # plot histogram
    ax.bar(x_range[1:], bins[1:], color='k', width=0.85*width, align='edge')
    # highlight first bar
    ax.bar(x_range[0], bins[0], color='grey', width=0.85*width, align='edge', alpha=0.75)

    # plot pct
    ax.plot(pct, 90, marker='v', ms=6, color=c, alpha=0.75)
    txt = round(pct, 1)
    if txt==0: txt = 0
    ax.text(pct, 70, txt, ha='center', va='top', fontsize=fz-3)#, fontweight='bold')
    
    # layout
    if row!=7:
        ax.set_xticks([])
        if row==0:
            ax.set_title(height, fontsize=fz)
    else:
        xx = [0, 0.5, 1, x_range[-1]]
        xran = [0, 0.5, 1.0, x_range[-1]]
        ax.set_xticks(xx)
        ax.set_xticklabels(xran, fontsize=fz-3)
        ax.set_xlabel('LG', fontsize=fz)
    yy = [0, 50, 100]
    ax.set_yticks(yy)
    ax.set_yticklabels([], fontsize=fz-3)
    if col==0:
        ax.text(-0.25, -5, '0%', ha='right', va='bottom', fontsize=10)
        ax.text(-0.25, 105, '100%', ha='right', va='top', fontsize=10)
        ax.set_ylabel(f'{tag}          ', rotation='horizontal', ha='right', fontsize=fz)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 100)



if __name__ == '__main__':
    version = 'Altitude'
    error_thresh = 1.8
    dataset = 'rt' 
    Ut_smooth = 'non-smooth'  # smooth   non-smooth
    # TODO: first run EM_output_to_Group_Analysis.py to create the intermediate results folder
    base_folder = f'./interm_Results/{Ut_smooth}/{dataset}_{version}/'
    
    ##################################
    #### ALTITUDE SPAGHETTI PLOTS ####
    ##################################
    if True:
        # setup figure
        rows = 15
        columns = 4
        percentile = 0.95
        fig, axes = plt.subplots(rows, columns, figsize=(16, 14))

        # add spagetthi plots
        ax1 = plt.subplot2grid((rows, columns), (9, 0), colspan=2, rowspan=6, fig=fig)
        ax2 = plt.subplot2grid((rows, columns), (9, 2), colspan=2, rowspan=6, fig=fig)
        axes1 = [ax1, ax2]
        xs, ys, zs = [], [], []
        colors = [k for k in mcolors.TABLEAU_COLORS.keys()]
        heights = ['Sea level', '5,000 ft', '8,000 ft', '13,000 ft']

        # boxplot properties
        med_prop = {'color':'k', 'linestyle':'solid', 'linewidth':1.5}
        men_prop = {'color':'r', 'linestyle':'dashed', 'linewidth':1.5}
        boxdic = {'showmeans':False, 'meanline':True, 'showfliers':False, 'widths':0.8, 'medianprops':med_prop, 'meanprops':men_prop}
        fz = 14

        # run over all patients
        cnt = -1
        for p in range(1,12):
            csvs = np.sort(glob.glob(base_folder + f'/P40-{p}-*.csv'))
            if len(csvs)==0: continue
            cnt += 1
            option1, option2, option3, option4 = [], [], [], []
            for col, csv in enumerate(csvs):
                # extract SS percentage from path
                option2.append(float(csv.split('-')[-1].split('.csv')[0]))

                # extract intermediate results from sheet
                Alt_data = pd.read_csv(csv)
                if len(Alt_data)==0: continue
                LG = Alt_data['LG_data'].values
                G = Alt_data['G_data'].values
                D = Alt_data['D_data'].values
                valid_data = Alt_data['valid_data'].values
                SS = Alt_data['SS_data'].values
                
                # only include valid estimations
                no_lower_bound = np.logical_and(G==0.1, D==5)
                no_upper_bound = np.logical_and(G==0.1, D==50)
                no_edges = ~np.logical_or(no_lower_bound, no_upper_bound)
                valids = np.logical_and(no_edges, valid_data==True)
                val = LG[valids]

                # compute burden max
                val = np.concatenate([LG[valids], np.zeros(sum(valid_data==False))])
                option1.append(np.nanquantile(val, percentile))

                ##################
                ### HISTOGRAMS ###
                ##################
                color = colors[cnt]
                tag = f'#{cnt+1}'    # f'P40-{p}'
                height = heights[col]
                bins, pct = create_histogram_bars(LG, valids, percentile)
                plot_histogram_bins(bins, pct, axes, cnt, col, tag, height, color, fz)

            # 1. Plot 95th quantile LG
            tag = f'Patient {cnt+1}'
            ax1.plot(range(1,5), option1, c=color, marker='v', ms=6, alpha=0.75, lw=1, label=tag)

            # 2. Plot SS%
            ax2.plot(range(1,5), option2, c=color, marker='o', ms=6, alpha=0.75, lw=1, label=tag)

            # save for LR
            if cnt==0:
                total_1, total_2 = option1, option2
            else:
                total_1 = np.vstack([total_1, option1])
                total_2 = np.vstack([total_2, option2])

        ########################
        # run over all axes/data
        for a, (ax, total) in enumerate(zip(axes1, [total_1, total_2])):
            # add Mean/Median
            ax.plot(range(1,5), np.nanmean(total, 0), c='k', ls='dashed', ms=8, marker=None, lw=1.5, label='Mean')
            ax.plot(range(1,5), np.nanmedian(total, 0), c='k', ls='dotted', ms=8, marker=None, lw=1.5, label='Median')

            # layout
            ax.spines['top'].set_visible(False)
            ax.set_xticks(range(1,5))
            ax.set_xticklabels([heights[i] for i in range(4)], fontsize=fz)

            marge = 0.05*np.nanmax(total)
            ax.set_ylim(np.nanmin(total)-marge, np.nanmax(total)+marge)
            
        # ax1
        ax1.set_ylabel('LG\n($95^{th}$ percentile)\n', fontsize=fz)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.tick_params(axis="y", labelsize=fz-3)
        ax1.set_ylim(-0.1, 1.7)
        ax1.legend(frameon=False, loc=2, ncols=3, fontsize=fz-2)

        
        # ax2
        ax2.set_ylabel('\nSS %', fontsize=fz)
        ax2.spines['left'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.tick_params(axis="y", labelsize=fz-3)
        
        # adjust empty axes
        for axs in axes[8:]:
            for ax in axs:
                ax.axis('off')




        plt.show()
        # out_path = 'Altitude_figure.png'
        # plt.savefig(fname=out_path, format='png', dpi=1200)
