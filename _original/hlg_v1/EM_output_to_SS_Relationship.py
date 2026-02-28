import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns
import uncertainties.unumpy as unp
import uncertainties as unc

# import local functions
from Event_array_modifiers import find_events
from EM_output_to_Figures import add_arousals, match_EM_with_SS_output, post_process_EM_output
from Convert_SS_seg_scores import convert_ss_seg_scores_into_arrays
from SS_output_to_EM_input import load_sim_output
from Recreate_LG_array import create_total_LG_array

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

def extract_EM_output_old(input_files, interm_folder, version):
    # set empty dictionaries
    LG_data, G_data, D_data, GxD_data, SS_data, Valid_data, Stage_data = {}, {}, {}, {}, {}, {}, {}
    for i in np.arange(0, 10, 2):
        for param in ['SS', 'LG', 'G', 'D', 'VV', 'St']:
            SS_data[f'seg_{i/10}-{(i+2)/10}_{param}'] = []

    # run over all fies
    for i, input_file in enumerate(input_files):
        num = input_file.split('/Study')[-1].split('.csv')[0]
        print(f'Extracting Study {num} ({i+1}/{len(input_files)}) ..', end='\r')
        # extract data
        data = pd.read_csv(input_file)

        # convert SS scores into columns
        data = convert_ss_seg_scores_into_arrays(data)

        # post-process EM output
        data = post_process_EM_output(data)

        # add arousals
        hdr = {'Study_num': f'Study {num}'}
        _, hdr['SS group'] = add_arousals(data, version, 'mgh', hf5_folder)

        # find matching SS output path, and extract SS data
        sim_path, _ = match_EM_with_SS_output(data, version, dataset)
        path = hf5_folder + sim_path + '.hf5'
        SS_df, SS_hdr = load_sim_output(path, ['flow_reductions'])
        assert len(SS_df)>0.99*len(data), 'matching SS output does not match EM data'

        # recreate LG array for "LG hypnogram"
        total_LG = create_total_LG_array(data)
        total_LG = pd.DataFrame(total_LG, columns=['LG_hypno'])
        hypno_folder = interm_folder + '/hypnograms/'
        os.makedirs(hypno_folder, exist_ok=True)
        out_path = f'{hypno_folder}Study {num}.csv'
        total_LG.to_csv(out_path, header=total_LG.columns, index=None, mode='w+')

        # extract other header fields
        for col in ['patient_tag', 'Fs', 'original_Fs']:
            hdr[col] = data.loc[0, col]
            data = data.drop(columns=col)

        # run over all segments
        Errors, Vmaxs, LGs, Gs, Ds, Ls, SSs_seg, valid_seg, Stages = [], [], [], [], [], [], [], [], []
        for stage in ['nrem', 'rem']:
            # set run variables
            starts = data[f'{stage}_starts'].dropna().values.astype(int)
            ends = data[f'{stage}_ends'].dropna().values.astype(int)
            group = hdr['SS group']
            for start, end in zip(starts, ends):
                # Extract parameters, and set out_path
                loc = np.where(data[f'{stage}_starts']==start)[0][0]
                Errors.append(round(data.loc[loc, 'rmse_Vo'], 2))
                Ls.append(data.loc[loc, f'L_{stage}'])
                Vmaxs.append(round(data.loc[loc, 'Vmax'], 2))
                LGs.append(data.loc[loc, f'LG_{stage}_corrected'])
                Gs.append(data.loc[loc, f'G_{stage}'])
                Ds.append(data.loc[loc, f'D_{stage}'])
                # valid_seg.append(len(find_events(data.d_i_ABD_smooth<1))>=4)
                valid_seg.append(len(find_events(SS_df.loc[start:end, 'flow_reductions']>0))>=5)
                SSs_seg.append(data.loc[start, 'SS_score'])
                Stages.append(stage)
        
        # save estimated LG values per patient self-similarity group
        inds = np.array(Errors) < error_thresh
        LGs = np.array(LGs)[inds]
        Gs = np.array(Gs)[inds]
        Ds = np.array(Ds)[inds]
        SSs = np.array(SSs_seg)[inds]
        valids = np.array(valid_seg)[inds]
        Stages = np.array(Stages)[inds]
        if not group in LG_data.keys():
            LG_data[group] = LGs
            G_data[group] = Gs
            D_data[group] = Ds
            GxD_data[group] = Gs*Ds
            Valid_data[group] = valids
            Stage_data[group] = Stages
        else:   
            LG_data[group] = np.concatenate([LG_data[group], LGs])
            G_data[group] = np.concatenate([G_data[group], Gs])
            D_data[group] = np.concatenate([D_data[group], Ds])
            GxD_data[group] = np.concatenate([GxD_data[group],  Gs*Ds])
            Valid_data[group] = np.concatenate([Valid_data[group], valids])
            Stage_data[group] = np.concatenate([Stage_data[group], Stages])

        # save estimated LG values per self-similarity segment score
        for LG, G, D, SS, VV, St in zip(LGs, Gs, Ds, SSs, valids, Stages):
            for i in np.arange(0, 10, 2):
                ran = f'seg_{i/10}-{(i+2)/10}'
                if SS>=i and SS<i+0.2: break
            SS_data[ran+'_SS'].append(SS)
            SS_data[ran+'_LG'].append(LG)
            SS_data[ran+'_G'].append(G)
            SS_data[ran+'_D'].append(D)
            SS_data[ran+'_VV'].append(VV)
            SS_data[ran+'_St'].append(St)
            
    # sort dictionary
    sorted_dics = sort_dic_keys([LG_data, G_data, D_data, GxD_data, Valid_data, Stage_data])
    names = ['LG_data', 'G_data', 'D_data', 'GxD_data', 'Valid_data', 'Stage_data']
    # save per group estimations
    for group in LG_data.keys():
        out_path = f'{interm_folder}/{group}.csv'
        df = pd.DataFrame([], dtype=float)
        for dic, name in zip(sorted_dics, names):
            df[name] = dic[group]
        df.to_csv(f'{interm_folder}/{group}.csv', header=df.columns, index=None, mode='w+')
    # save per segment estimations
    for i in np.arange(0, 10, 2):
        df = pd.DataFrame([], dtype=float)
        ran = f'seg_{i/10}-{(i+2)/10}'
        for param in ['SS', 'LG', 'G', 'D', 'VV', 'St']:
            df[param] = SS_data[f'{ran}_{param}']
        df.to_csv(f'{interm_folder}/{ran}.csv', header=df.columns, index=None, mode='w+')
    
# analysis functions 
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


if __name__ == '__main__':
    version = 'SS_range'
    error_thresh = 1.8
    dataset = 'mgh' 
    Ut_smooth = 'non-smooth'  # smooth   non-smooth
    base_folder = f'./interm_Results/{Ut_smooth}/{dataset}_{version}/'

    ####################################
    #### LG Hyponogram Swimmer-PLOT ####
    ####################################
    if False:
        # setup figure
        fig = plt.figure(figsize=(14, 12))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        axes = [ax1, ax2]

        shift = 0
        max_time = 0
        factor = 20 * 10 * 200
        thresh = 0.7

        # run over all groups
        for i in range(1, 101):
            csv = base_folder + f'hypnograms/Study {i}.csv'
            data = np.squeeze(pd.read_csv(csv).values)
            # fill to 9 hours
            LG_array = np.zeros(int(9.5 * 60 * 60 * 10)) * np.nan
            LG_array[:len(data)] = data

            x = np.arange(len(LG_array)) / factor
            y = LG_array

            # default plot
            red = np.array(y)
            red[red<thresh] = np.nan
            ax1.plot(x, y+i+shift, 'k', alpha=0.8, lw=0.8)
            ax1.plot(x, red+i+shift, 'r', lw=1.2)

            # graph
            height = 2
            bottom = (i-1)*height
            ax2.fill_between(x, bottom, i*height, where=np.isnan(y), facecolor='lightgray')
            ax2.fill_between(x, bottom, i*height, where=y<0.3, facecolor='tab:gray')
            medium = np.logical_and(y>=0.3, y<thresh)
            ax2.fill_between(x, bottom, i*height, where=medium, facecolor='tab:orange')
            ax2.fill_between(x, bottom, i*height, where=y>=thresh, facecolor='tab:red')
            
            # correct shift
            maxi = np.nanmax(LG_array)
            shift += maxi
            # save max time
            if len(LG_array) > max_time:
                max_time = len(LG_array)

        # layout adjustments across axes
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Time\n(hr)', fontsize=12)
        
        # setup legend
        handles = []
        for label, color in zip(['Wake', f'LG<{thresh}', f'LG>={thresh}'], ['k', 'b', 'r']):
            handle = Line2D([0],[0], label=label, c='none')
            handles.append(handle)

        # ax2.legend(handles=handles, loc=0)
        ax1.set_yticks([])
        ax1.set_yticklabels([])
        ax2.set_yticks(range(0, 201, 50))
        ax2.set_yticklabels(range(0, 101, 25))


        plt.show()

    ##################################
    #### NREM vs REM  LG BOX-PLOT ####
    ##################################
    if False:
        # setup figure
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        # boxplot properties
        med_prop = {'color':'k', 'linestyle':'solid', 'linewidth':2}
        men_prop = {'color':'r', 'linestyle':'dashed', 'linewidth':2}
        boxdic = {'showmeans':True, 'meanline':True, 'showfliers':False, 'widths':0.2, 'medianprops':med_prop, 'meanprops':men_prop}

        # create boxplot for each stage NREM vs REM
        stages = ['nrem', 'rem']
        for pos, stage in enumerate(stages):
            xs, ys, zs = [], [], []

            # run over all groups
            groups, total_seg = [], 0
            for p, csvs in enumerate(np.sort(glob.glob(os.path.join(base_folder, 'seg*.csv')))):
                # extract intermediate results from sheet
                group = csvs.split('/seg_')[-1].split('.csv')[0]
                groups.append(group)
                SS_data = pd.read_csv(csvs)
                if len(SS_data)==0: continue
                total_seg += len(SS_data)
                SS = SS_data['SS'].values
                LG = SS_data['LG'].values
                G = SS_data['G'].values
                D = SS_data['D'].values
                valid_data = SS_data['VV'].values
                Stage_data = SS_data['St'].values
                # only include valid estimations
                no_lower_bound = np.logical_and(G==0.1, D==5)
                no_upper_bound = np.logical_and(G==0.1, D==50)
                no_edges = ~np.logical_or(no_lower_bound, no_upper_bound)
                valid = np.logical_and(no_edges, valid_data==True)
                inds = np.logical_and(np.isfinite(SS), valid)
                stage_ind = np.logical_and(inds, Stage_data==stage)
                # save
                xs = np.concatenate([xs, SS[stage_ind]])
                ys = np.concatenate([ys, LG[stage_ind]])
                zs = np.concatenate([zs, G[stage_ind]])

            # sort data, and remove outliers
            xs, ys, zs = xs[np.argsort(xs)], ys[np.argsort(xs)], zs[np.argsort(xs)]
            inds = ~np.logical_or(np.logical_and(xs>0, zs==0.1), np.logical_and(xs==0, zs!=0.1))
            x, y, z, n = xs[inds], ys[inds], zs[inds], len(inds)

            # boxplot 
            ax.boxplot(y, positions=[(pos+1)*0.3], **boxdic)

        # layout adjustments across axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # layout adjustments per axis
        ax.set_xlim([0.1, 0.8])
        ax.set_xticklabels(stages, fontsize=12)
        ax.set_ylabel('LG', fontsize=12)

        # overall layout
        main = f'Boxplots showing the estimated LG distribution NREM vs REM'
        # fig.suptitle(f'{main}', fontsize=14)
        ax.legend(loc=0, frameon=False)


    ##################################
    ###### per segment SS range ######
    ##################################
    if True:
        # setup figure
        fig, ax = plt.subplots(1, 1, figsize=(11, 10))
        xs, ys, zs = [], [], []

        # run over all groups
        groups, total_seg = [], 0
        for p, csvs in enumerate(np.sort(glob.glob(os.path.join(base_folder, 'seg*.csv')))):
            # extract intermediate results from sheet
            group = csvs.split('/seg_')[-1].split('.csv')[0]
            groups.append(group)
            SS_data = pd.read_csv(csvs)
            if len(SS_data)==0: continue
            total_seg += len(SS_data)
            SS = SS_data['SS'].values
            LG = SS_data['LG'].values
            G = SS_data['G'].values
            D = SS_data['D'].values
            valid_data = SS_data['VV'].values
            # only include valid estimations
            no_lower_bound = np.logical_and(G==0.1, D==5)
            no_upper_bound = np.logical_and(G==0.1, D==50)
            no_edges = ~np.logical_or(no_lower_bound, no_upper_bound)
            valid = np.logical_and(no_edges, valid_data==True)
            inds = np.logical_and(np.isfinite(SS), valid)
            # save
            xs = np.concatenate([xs, SS[inds]])
            ys = np.concatenate([ys, LG[inds]])
            zs = np.concatenate([zs, G[inds]])
        
        # sort data, and remove outliers
        xs, ys, zs = xs[np.argsort(xs)], ys[np.argsort(xs)], zs[np.argsort(xs)]
        inds = ~np.logical_or(np.logical_and(xs>0, zs==0.1), np.logical_and(xs==0, zs!=0.1))
        x, y, z, n = xs[inds], ys[inds], zs[inds], len(inds)
        xx = np.linspace(0, 1, 100)

        # scatter plot + polyfit
        scatter_dic = {'color':'tab:blue', 'alpha':0.2}
        poly_dic = {'color':'black', 'linestyle':'solid', 'label':'$2^{nd}$ order polynomial [95% CI]'}
        sns.regplot(x=x, y=y, scatter=True, order=2, ax=ax,
            scatter_kws=scatter_dic, line_kws=poly_dic)
        
        # reconstruct 2nd order polynomial
        popt, pcov = curve_fit(func, x, y)
        
        # compute r^2
        a, b, c = popt
        r2 = round(np.sqrt(1.0 - (sum((y-func(x, a, b, c))**2) / ((n-1.0) * np.var(y, ddof=1)))), 2)

        # calculate parameter confidence interval
        a, b, c= unc.correlated_values(popt, pcov)

        # calculate regression confidence interval
        py = a*xx*xx + b*xx + c
        nom = unp.nominal_values(py)
        std = unp.std_devs(py)
        lpb, upb = predband(xx, x, y, popt, func, conf=0.95)

        # prediction band (95% confidence)
        # ax.plot(xx, lpb, 'k--', label='$5^{th}-95^{th}$ percentile prediction range')
        # ax.plot(xx, upb, 'k--')

        # manually compute 5-95 prediction range
        xi, q5, q95, win = [], [], [], 0.3
        for i in np.arange(0.325, 1+win*2, win):
            lo, up = i-win/2, i+win/2
            y_vals = y[np.logical_and(x>=lo, x<up)]
            if len(y_vals)==0: continue
            xi.append(i)
            q5.append(np.quantile(y_vals, 0.05))
            q95.append(np.quantile(y_vals, 0.95))
        # plot range
        ax.plot(xi, q5, 'k--', label='$5^{th}-95^{th}$ percentile prediction range')
        ax.plot(xi, q95, 'k--')

        # layout adjustments per axis
        fz = 16
        ax.set_xlabel('\nSS', fontsize=fz)
        ax.set_ylabel('LG\n', fontsize=fz)
        xmin, xmax = 0, 1
        ymin, ymax = 0, 2.5
        margin = 0.025
        ax.set_xlim([xmin-margin*xmax, xmax+margin*xmax])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.set_ylim([ymin-margin*ymax, ymax+margin*ymax])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # overall layout
        # main = f'Scatterplot - Relationship SS and Gain (N={len(inds)} segments)'
        # fig.suptitle(f'{main}', fontsize=12, fontweight='bold')

        # add custom handles
        handles, _ = ax.get_legend_handles_labels()
        r2_dic = {'color':'none', 'linestyle':'', 'label':f'$r^{2}$: {r2}'}
        r2_handle = Line2D([0],[0], **r2_dic) 
        handles += [r2_handle]
        ax.legend(handles=handles, loc=0, frameon=False, fontsize=fz-1, title_fontsize=fz)  

        # plt.show()
        # import pdb; pdb.set_trace()

        #fit the model
        # x_data = np.arange(0, 1.1, 0.1)
        # x, y  = xs, ys
        # b, a = np.polyfit(x, np.log(y), 1)
        # curve = np.exp(a) * np.exp(b*x_data)
        # ax.plot(x_data, curve, 'b', lw=2)
        
        plt.show()

