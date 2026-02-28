import glob, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# import local functions
from Event_array_modifiers import find_events
from Convert_SS_seg_scores import convert_ss_seg_scores_into_arrays
from EM_output_to_Figures import (add_arousals, match_EM_with_SS_output,
                                  post_process_EM_output)
from Recreate_LG_array import create_total_LG_array
from SS_output_to_EM_input import load_sim_output


# extraction functions
def extract_EM_output(input_files, interm_folder, hf5_folder, version, dataset):
    # set empty dictionaries
    LG_data, G_data, D_data, GxD_data, SS_data, valid_data = [], [], [], [], [], []
    SS_dic = {}
    for i in np.arange(0, 10, 2):
        for param in ['SS', 'LG', 'G', 'D', 'VV']:
            SS_dic[f'seg_{i/10}-{(i+2)/10}_{param}'] = []

    # run over all fies
    for i, input_file in enumerate(input_files):
        num = input_file.split('/Study')[-1].split('.csv')[0]
        print(f'Extracting Study {num} ({i+1}/{len(input_files)}) ..   ', end='\r')
        
        # extract data
        data = pd.read_csv(input_file)
        
        # convert SS scores into columns
        data = convert_ss_seg_scores_into_arrays(data)

        # post-process EM output
        data = post_process_EM_output(data)

        # set/extract header fields
        hdr = {'Study_num': f'Study {num}'}
        for col in ['patient_tag', 'Fs', 'original_Fs']:
            hdr[col] = data.loc[0, col]

        # add arousals
        if 'MGH' in version:
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
                # valid_seg.append(len(find_events(data.d_i_ABD_smooth<1))>=4)
                valid_seg.append(len(find_events(SS_df.loc[start:end, 'flow_reductions']>0))>=5)
                SSs_seg.append(data.loc[start, 'SS_score'])
            
        # save estimated LG values per patient self-similarity group
        inds = np.array(Errors) < error_thresh
        LGs = np.array(LGs)[inds]
        Gs = np.array(Gs)[inds]
        Ds = np.array(Ds)[inds]
        SSs = np.array(SSs_seg)[inds]
        valids = np.array(valid_seg)[inds]

        LG_data = np.concatenate([LG_data, LGs])
        G_data = np.concatenate([G_data, Gs])
        D_data = np.concatenate([D_data, Ds])
        GxD_data = np.concatenate([GxD_data, Gs*Ds])
        SS_data = np.concatenate([SS_data, SSs])
        valid_data = np.concatenate([valid_data, valids])

        # save estimated LG values per self-similarity segment score
        for LG, G, D, SS, VV in zip(LGs, Gs, Ds, SSs, valids):
            for i in np.arange(0, 10, 2):
                ran = f'seg_{i/10}-{(i+2)/10}'
                if SS>=i and SS<i+0.2: break
            SS_dic[ran+'_SS'].append(SS)
            SS_dic[ran+'_LG'].append(LG)
            SS_dic[ran+'_G'].append(G)
            SS_dic[ran+'_D'].append(D)
            SS_dic[ran+'_VV'].append(VV)

    sorted_data = [LG_data, G_data, D_data, GxD_data, SS_data, valid_data]
    names = ['LG_data', 'G_data', 'D_data', 'GxD_data', 'SS_data', 'valid_data']
    # save per group estimations
    out_path = f'{interm_folder}/all_segments.csv'
    df = pd.DataFrame([], dtype=float)
    for dat, name in zip(sorted_data, names):
        df[name] = dat
    os.makedirs(interm_folder, exist_ok=True)
    df.to_csv(out_path, header=df.columns, index=None, mode='w+')
    # save per segment estimations
    for i in np.arange(0, 10, 2):
        df = pd.DataFrame([], dtype=float)
        ran = f'seg_{i/10}-{(i+2)/10}'
        for param in ['SS', 'LG', 'G', 'D', 'VV']:
            df[param] = SS_dic[f'{ran}_{param}']
        df.to_csv(f'{interm_folder}{ran}.csv', header=df.columns, index=None, mode='w+')
        
# plotting functions
def add_statistical_significance(data, ref_data, pos, ax, i):
    # compute significance
    U, p = stats.mannwhitneyu(ref_data, data, alternative='two-sided')
    # U, p = stats.ttest_ind(ref_data, data)
    if p < 0.001:
        sig_symbol = '***'
    elif p < 0.01:
        sig_symbol = '**'
    elif p < 0.05:
        sig_symbol = '*'
    else:
        return

    # plot *stars
    if i == 0:
        level = -0.05
        offset = - 0.1*(pos-1)
        hook = 0.0275
    else:
        level = 0.05
        offset = - 0.06*(pos-1)
        hook = 0.015
    ax.plot([1, pos], [level+offset]*2, color='k', lw=1)   # <duration>
    ax.plot([1, 1], [level+offset, level+offset+hook], color='k', lw=1)     # left va
    ax.plot([pos, pos], [level+offset, level+offset+hook], color='k', lw=1) # right va
    ax.text((1+pos)/2, level+offset-0.005, sig_symbol, va='top', ha='center')

# helper functions
def select_highest_LG_block(data, block):
    # assess blocks with a stride of 30min
    step = int(0.5 * 60 * 60 * 10)
    means = []
    for i in range(0, 100, step):
        if i*step+block > len(data):
            break
        means.append(np.nanmean(data[i*step:i*step+block]))

    # select block with hightest mean LG
    loc = np.argmax(means)
    LG_array = data[loc*step:loc*step+block]

    return LG_array


if __name__ == '__main__':
    versions = ['REM_OSA', 'NREM_OSA', 'high_CAI', 'SS_OSA', 'Heart_Failure']
    version_tags = [f'REDEKER_{v}_V2' if 'Heart' in v else f'MGH_{v}_V2' for v in versions]
    error_thresh = 1.8
    smooth = 'non-smooth'  # smooth   non-smooth

    # SS output folder
    date = '11_09_2023'
    exp = 'Expansion' if os.path.exists(f'/media/cdac/Expansion/CAISR data1/Rule_based') else 'Expansion1'
    

    # extract data for all versions
    lens = []
    ref_data = []
    group_shift = 0
    for v, version in enumerate(version_tags):
        # set hf5 folder
        dataset = 'redeker' if 'REDEKER' in version else 'mgh'
        hf5_folder = f'/media/cdac/{exp}/LG project/hf5data/{dataset}_{date}/'

        # set input folder TODO: set path to this dropbox folder
        input_folder = f'Dropbox/Final Code Thijs/{smooth}/{version}/'
        input_files = glob.glob(input_folder + '*.csv')
        lens.append(len(input_files))

        # check if intermediate results already available
        interm_folder = f'interm_Results/{smooth}/group_analysis/{version}/'
    
        # if intermediate results do not exist, extract EM output
        if not os.path.exists(interm_folder):
            import pdb; pdb.set_trace()
            print(f'\n>> {version} <<')
            extract_EM_output(input_files, interm_folder, hf5_folder, version, dataset)
            
        ##########################
        #### Cohort BOX-PLOTS ####
        ##########################
        if True:
            # setup figure
            if v==0:
                fig = plt.figure(figsize=(8,10))
                ax1 = fig.add_subplot(311)
                ax2 = fig.add_subplot(312)
                ax3 = fig.add_subplot(313)
                axes = [ax1, ax2, ax3]
            # boxplot properties
            med_prop = {'color':'r', 'linestyle':'dashed', 'linewidth':1.5}
            # men_prop = {'color':'r', 'linestyle':'dashed', 'linewidth':1.5}
            boxdic = {'showmeans':True, 'meanline':False, 'showfliers':False,
                        'widths':0.8, 'medianprops':med_prop}

            # extract intermediate results from sheet
            interm_result = pd.read_csv(f'{interm_folder}all_segments.csv')
            LG_data = interm_result['LG_data']
            G_data = interm_result['G_data']
            D_data = interm_result['D_data']
            GxD_data = interm_result['GxD_data']
            SS_data = interm_result['SS_data']
            valid_data = interm_result['valid_data']
        
            # plot boxplots (+custom means)
            for i, (ax, vals) in enumerate(zip(axes, [LG_data, G_data, D_data])):
                # only include valid estimations
                # inds = np.logical_and(np.isfinite(SS_data), valid_data==True)
                no_lower_bound = np.logical_and(G_data==0.1, D_data==5)
                no_upper_bound = np.logical_and(G_data==0.1, D_data==50)
                no_edges = ~np.logical_or(no_lower_bound, no_upper_bound)
                inds = np.logical_and(no_edges, valid_data==True)
                # create boxplots
                ax.boxplot(vals[inds], positions=[v+1], **boxdic)
                print(f'{version}: {np.median(vals[inds])}')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xlim([0.2, len(versions)+1])    
                ax.tick_params(left=False)   
                ax.tick_params(bottom=False)   
                ax.set_yticklabels([])

                # add statitical significance
                if i != 2:
                    if v==0:
                        ref_data.append(vals[inds])
                    else:
                        add_statistical_significance(vals[inds], ref_data[i], v+1, ax, i) 
            
            # set layout
            if v==len(versions)-1:
                # custom LG
                ax1.plot([0.4, 0.4], [-0.225, 1.725], 'k')
                yticks = [round(v, 1) for v in np.arange(0, 1.7, 0.3)]
                for y in yticks:
                    ax1.plot([0.35, 0.40], [y, y], 'k')
                    ax1.text(0.30, y, str(y), ha='right', va='center', fontsize=11)
                ax1.set_ylabel('LG\n', fontsize=12)
                ax1.set_ylim([-0.5, 1.85])
                # add legend
                ax1.plot([0.6, 0.8], [1.6]*2, 'r--', lw=2) 
                ax1.text(0.9, 1.6, 'mean', ha='left', va='center', fontsize=11)
                ax1.plot([0.6, 0.8], [1.4]*2, 'k', lw=2) 
                ax1.text(0.9, 1.4, 'median', ha='left', va='center', fontsize=11)
                ax1.text(0.715, 1.18, '***', ha='center', va='center', fontsize=11)
                ax1.text(0.9, 1.2, 'p < 0.001', ha='left', va='center', fontsize=11)

                # custom GAIN
                ax2.plot([0.4, 0.4], [-0.05, 1.25], 'k')
                yticks = [round(v, 1) for v in np.arange(0.1, 1.2, 0.2)]
                for y in yticks:
                    ax2.plot([0.35, 0.40], [y, y], 'k')
                    ax2.text(0.30, y, str(y), ha='right', va='center', fontsize=11)
                ax2.set_ylabel('$\\gamma$\n', fontsize=12)
                ax2.set_ylim([-0.2, 1.22])

                # custom TAU
                ax3.plot([0.4, 0.4], [5, 55], 'k')
                yticks = [round(v, 1) for v in np.arange(10, 55, 10)]
                for y in yticks:
                    ax3.plot([0.35, 0.40], [y, y], 'k')
                    ax3.text(0.30, y, str(y), ha='right', va='center', fontsize=11)
                ax3.set_ylabel('$\\tau$\n', fontsize=12)
                ax3.set_ylim([2.5, 55])  

                # x axis
                ax1.set_xticks([]) 
                ax1.set_xticklabels([])
                ax2.set_xticks([]) 
                ax2.set_xticklabels([])
                xticks = ['\n' + v.replace('_',' ') + f'\n(N={n})' for v, n in zip(versions, lens)]
                ax3.set_xticklabels(xticks, fontsize=11)

                # set title
                # main = 'Boxplots of the estimated "$\\gamma$" and "$\\tau$" distribution across cohorts\n'
                # fig.suptitle(f'{main}', fontsize=12, fontweight='bold')
    
        if False:
            # setup figure
            if v==0:
                fig = plt.figure(figsize=(14, 12))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                axes = [ax1, ax2]

            height = 2
            shift = 0
            factor = 20 * 10 * 200
            thresh1, thresh2 = 0.7, 1.0
            select = 20
            hr = 8.25

            # run over all groups
            cnt = 0
            len_input_files = 65 if 'REDEKER' in version else 100
            for i in range(len_input_files-select+1, len_input_files+1, 1):
                cnt+=1 
                csv = interm_folder + f'hypnograms/Study  {i}.csv'
                data = np.squeeze(pd.read_csv(csv).values)
                # fill to <hr> hours
                block = int(hr * factor)
                LG_array = np.zeros(block) * np.nan
                if len(data)<=block:
                    LG_array[:len(data)] = data
                else:
                    LG_array = select_highest_LG_block(data, block)

                x = np.arange(len(LG_array)) / factor
                y = LG_array

                # default plot
                shade = np.array(y)
                shade[np.isnan(shade)] = 0
                red = np.array(y)
                red[red<thresh2] = np.nan
                ax1.plot(x, shade+cnt+shift+group_shift + (select*v) + 7.5*v, color='lightgray', lw=1)
                ax1.plot(x, y+cnt+shift+group_shift     + (select*v) + 7.5*v, color='tab:gray', lw=1.2)
                ax1.plot(x, red+cnt+shift+group_shift   + (select*v) + 7.5*v, color='tab:red', lw=1.5)

                # bar graph
                bottom = (cnt-0.85)*height  + select*v*height + v*(height+1)
                top = (cnt-0.15)*height     + select*v*height + v*(height+1)
                ax2.fill_between(x, bottom, top, where=np.isnan(y), facecolor='lightgray')
                ax2.fill_between(x, bottom, top, where=y<thresh1, facecolor='tab:gray')
                medium = np.logical_and(y>=thresh1, y<thresh2)
                ax2.fill_between(x, bottom, top, where=medium, facecolor='tab:orange')
                ax2.fill_between(x, bottom, top, where=y>=thresh2, facecolor='tab:red')   
                             
                # correct shift
                maxi = np.nanmax(LG_array)
                shift += maxi

                # label
                if cnt == 1:
                    bottom_bar = bottom-1
                elif cnt == 20:
                    top_line = cnt + shift + group_shift + (select*v) + 7.5*v + 3.75
                    top_bar = top+1    
            
            # update group shift
            group_shift += shift

            # set group clamps
            clamp_dic = {'color':'black', 'linewidth':1, 'linestyle':'solid'}
            # ax1.plot([0, hr], [top_line]*2, **clamp_dic)
            ax2.plot([-0.2, -0.1], [bottom_bar]*2, **clamp_dic)
            ax2.plot([-0.2, -0.1], [top_bar]*2, **clamp_dic)
            ax2.plot([-0.2]*2, [bottom_bar, top_bar], **clamp_dic)
            tag = versions[v].replace('_',' ')
            y = (bottom_bar+top_bar) / 2
            ax2.text(-0.3, y, tag, ha='right', va='center', fontsize=11, rotation=90)

            if v==len(versions)-1:
                # layout adjustments across both axes
                for ax in axes:
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_xlabel('Time\n(hr)', fontsize=11)
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                    
                # set x/y axes
                ax1.set_ylim([-12, top_line-2.5])
                ax1.plot([0, hr], [-12]*2, 'k', lw=2.5)
                ax2.set_ylim([-8, top_bar+1])
                ax2.plot([0, hr], [-8]*2, 'k', lw=2.5)

                # legend
                start = np.array([0.5, 0.75])
                bottom, top = [-5]*2, [-3]*2
                tags = ['Wake', 'Low LG', 'Elevated LG', 'High LG']
                colors = ['lightgray', 'tab:gray', 'tab:orange', 'tab:red']
                text_dic = {'ha':'left', 'va':'center', 'fontsize':11}
                for i, (tag, c) in enumerate(zip(tags, colors)):
                    # ax1
                    if i in [0, 1]:
                        x = start + (i+0.5)*2
                        ax1.plot(x, [-6]*2, color=c)
                        ax1.text(x[1]+0.1, -6, tag, **text_dic)
                    elif i==3:
                        x = start + (i-0.5)*2
                        ax1.plot(x, [-6]*2, color=c)
                        ax1.text(x[1]+0.1, -6, tag, **text_dic)
                    x = start + i*2
                    # ax2
                    ax2.fill_between(x, bottom, top, facecolor=c)
                    ax2.text(x[1]+0.1, -4, tag, **text_dic)
                



    plt.tight_layout()
    plt.show()
