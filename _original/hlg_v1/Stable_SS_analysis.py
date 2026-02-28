import glob, h5py, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby

# load SS functions
from Compute_sleep_metrics import compute_sleep_metrics

# import utils functions
from Preprocessing import *
from Event_array_modifiers import *
# import local functions
from SS_output_to_EM_input import sort_input_files

# loading functions
def load_sim_output(path, cols=[]):
    # init DF
    hdr_cols = ['patient_tag', 'test_type', 'rec_type', 'cpap_start', 'Fs', 'SS_threshold']
    if len(cols) == 0:
        cols = ['abd', 'spo2', 'apnea', 'arousal', 'flow_reductions', 'sleep_stages', 'tagged', 'self similarity', 'ss_conv_score']
        cols += hdr_cols 
    # cols = hdr_cols  
    data = pd.DataFrame([], columns=cols)
    
    f = h5py.File(path, 'r')
    for key in cols:
        if key not in f.keys(): continue
        vals = f[key][:]
        # vals[np.isnan(vals)] = 0
        data[key] = vals
    f.close()

    # header:
    hdr = {}
    hdr['newFs'] = 10
    for hf in hdr_cols:
        if not hf in data.columns: continue
        val = data.loc[0, hf]
        try: val = val.decode('utf-8')
        except: pass
        hdr[hf] = val
        data.drop(columns=[hf], inplace=True)

    # sleep metrics
    if 'sleep_stages' in data.columns:
        data['patient_asleep'] = np.logical_and(data.sleep_stages < 5, data.sleep_stages > 0)
        RDI, AHI, CAI, sleep_time = compute_sleep_metrics(data.apnea, data.sleep_stages, exclude_wake=True)
        hdr['RDI'], hdr['AHI'], hdr['CAI'], hdr['sleep_time'] = RDI, AHI, CAI, sleep_time

    return data, hdr

# SS segmentation functions
def compute_osc_chains(data, hdr):
    # set sleep period and REM
    data['REM_breakpoints'] = np.nan
    notnan = np.where(data.sleep_stages>0)[0]
    data.loc[notnan, 'REM_breakpoints'] = 0
    rem = np.where(data.sleep_stages==4)[0]
    data.loc[rem, 'REM_breakpoints'] = 1

    # create breahting oscillation chains
    data['Osc_chain'] = 0
    data.loc[np.where(data['ss_conv_score']>=hdr['SS_threshold'])[0], 'Osc_chain'] = 1
    data['Osc_chain'] = data['Osc_chain'].rolling(int(3*60*hdr['newFs']), center=True).max()
    data.loc[np.where(np.logical_and(data['REM_breakpoints']==1, data['Osc_chain']==1))[0], 'Osc_chain'] = 2
    data.loc[np.where(data['REM_breakpoints'].isna())[0], 'Osc_chain'] = 0

    # remove short segments
    for i in range(1,3):
        for st, end in find_events(data['Osc_chain']==i):
            if end-st < 2*60*hdr['newFs']:
                data.loc[st:end, 'Osc_chain'] = 0

    return data, hdr

def compute_change_points_ruptures(data, hdr):
    # compute change points
    SS_trace = data['ss_conv_score'].values
    SS_trace[SS_trace<hdr['SS_threshold']] = np.nan
    win = int(3*60*hdr['newFs'])
    SS_trace = np.squeeze(pd.DataFrame(data=SS_trace).rolling(win, min_periods=1, center=True).median().fillna(0).values)
    scale = int(10*hdr['newFs'])
    detector = rpt.Pelt(model="rbf").fit(SS_trace[::scale])
    change_points = detector.predict(pen=4) #penalty
    change_points = np.array(change_points)*scale

    # create stabe SS array
    data['stable_SS'] = 0
    for i, loc0 in enumerate(change_points[:-1]):
        loc1 = change_points[i+1]
        if data.loc[loc0:loc1, 'SS_trace'].median() > 0:
            data.loc[loc0:loc1, 'stable_SS'] = 1
    # correct REM
    rem = np.where(np.logical_and(data.sleep_stages==4, data.stable_SS==1))[0]
    data.loc[rem, 'stable_SS'] = 2

    # remove short segments
    for i in range(1,3):
        for st, end in find_events(data['stable_SS']==i):
            if end-st < win:
                data.loc[st:end, 'stable_SS'] = 0

    return data

# plotting function
def plot_SS(data, hdr, out_path=''):
    # take middle 5hr segment --> // 10 rows == 30 min per row
    Fs = hdr['newFs']
    SS_thresh = hdr['SS_threshold']
    
    # set signal variables
    signal = data.abd.values
    sleep_stages = data.sleep_stages.values
    y_tech = data.apnea.values
    y_algo = data.flow_reductions.values
    osc_chain = data.Osc_chain.values
    stable_SS = data.stable_SS.values
    tagged_breaths = data.tagged.values
    ss_conv_score = data.ss_conv_score.values
    ss_trace = data.SS_trace.values
    
    # setup figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    row_height = 30

    # define the ids each row
    nrow = 10
    row_ids = np.array_split(np.arange(len(signal)), nrow)
    row_ids.reverse()

    # set sleep array
    sleep = np.array(signal)
    sleep[np.isnan(sleep_stages)] = np.nan
    sleep[sleep_stages==5] = np.nan
    # set wake array
    wake = np.zeros(signal.shape)
    wake[np.isnan(sleep_stages)] += signal[np.isnan(sleep_stages)]
    wake[sleep_stages==5] += signal[sleep_stages==5]
    wake[wake==0] = np.nan
    # set rem array
    rem = np.array(signal)
    rem[sleep_stages!=4] = np.nan
    # SS array
    SS = np.array(ss_trace)
    SS_none = np.array(ss_trace)
    SS_none[SS>=SS_thresh] = np.nan

    # PLOT SIGNALS
    for ri in range(nrow):
        a = 1 
        # plot signal
        ax.plot(sleep[row_ids[ri]]+ ri*row_height, c='k', lw=.3, alpha=a)
        ax.plot(wake[row_ids[ri]] + ri*row_height, c='r', lw=.3, alpha=a)
        ax.plot(rem[row_ids[ri]] + ri*row_height, c='b', lw=.3, alpha=a)

        # plot SS trace
        offset = -10
        factor = 5
        ax.plot(SS[row_ids[ri]]*factor + ri*row_height + offset, c='r', lw=1)
        ax.plot(SS_none[row_ids[ri]]*factor + ri*row_height + offset, c='k', lw=1)
        # SS reference lines
        ref = np.ones(len(row_ids[ri]))
        ax.plot(ref*0 + ri*row_height + offset, c='k', lw=.3, alpha=0.2)
        ax.plot(ref*0 + ri*row_height + offset+factor, c='k', lw=.3, alpha=0.2)
        ax.plot(ref*SS_thresh*factor + ri*row_height + offset, c='k', linestyle='dotted', lw=0.3)
        
        # plot split line for PTAF <--> CPAP
        if hdr['cpap_start'] in row_ids[ri]:
            loc = np.where(row_ids[ri]==hdr['cpap_start'])[0]
            min_ = -20 + ri*row_height
            max_ =  20 + ri*row_height
            ax.plot([loc, loc], [min_, max_], c='r', linestyle='dashed', zorder=10, lw=4)

    # PLOT LABELS
    for yi in range(6):
        if yi==0:
            labels = y_tech                 # tech label
            label_color = [None, 'k', 'b', 'b', 'k', 'k', None, 'b']
        elif yi==1:
            labels = y_algo                 # algo label
            label_color = [None, 'b', 'g', 'c', 'm', 'r', None, 'g']
        if yi == 2:
            labels = osc_chain               # black/blue marking for chains of breathing oscillations 
            label_color = [None, 'k', 'b']
        if yi == 3:
            labels = tagged_breaths            # '*' for HLG breathing oscillations
            label_color = [None, 'k']
        if yi == 4:
            labels = stable_SS               # black/blue marking for stable SS
            label_color = [None, 'k', 'b']

        # run over each plot row
        for ri in range(nrow):
            # group all labels and plot them
            loc = 0
            for i, j in groupby(labels[row_ids[ri]]):
                len_j = len(list(j))
               
                if np.isfinite(i) and label_color[int(i)] is not None:
                    if yi < 1:
                        # add scored events
                        sub = 0 if int(i) < 7 else 2
                        minus = 3 if not 'fc811d4b' in out_path else 4
                        ax.plot([loc, loc+len_j], [ri*row_height-minus*(2**yi)]*2, c=label_color[int(i)], lw=1)
                        if int(i) == 7:
                            ax.plot([loc, loc+len_j], [ri*row_height-minus*(2**1)]*2, c='m', lw=1)
                    if yi == 2:
                        # plot chains of breathing oscillations
                        ax.plot([loc, loc+len_j], [ri*row_height+8]*2, c=label_color[int(i)], lw=3, alpha=1)
                    if yi == 3:
                        c_score = np.round(ss_conv_score[row_ids[ri]][loc], 2)
                        if np.isfinite(c_score):
                            ax.text(loc, ri*row_height+10, '*', c='k', ha='center')
                            ax.text(loc, ri*row_height+15, str(c_score), ha='center', fontsize=3)
                    if yi == 4:
                        # plot stable SS regions
                        ax.plot([loc, loc+len_j], [ri*row_height+5]*2, c=label_color[int(i)], lw=3, alpha=1)
                        
                # update loc
                loc += len_j
                
    # plot layout setup
    ax.set_xlim([0, max([len(x) for x in row_ids])])
    ax.axis('off')

    # create title
    # AHI, CAI, T_SS = str(round(hdr['AHI'], 1)), str(round(hdr['CAI'], 1)), str(round(hdr['T_SS'], 1))
    # title = f'SS group: {group}\n'
    # subtitle = f'AHI: {AHI}; CAI: {CAI}; SS>0.8: {T_SS}%'
    # plt.title(title + subtitle) 
    plt.tight_layout()

    # save the figure
    if len(out_path) > 0:
        # plt.savefig(out_path)
        plt.savefig(out_path, dpi=900)
    plt.close()

def create_length_histogram(sim_df, result, version='Osc_chain'):
    # init dic with lengths
    len_dic = dict.fromkeys(np.sort(sim_df['SS group'].dropna().unique()))
    
    # run over all results
    for data, hdr in result:
        if data is None or hdr is None: continue

        # compute lens of Osc chains
        lens = np.array([(end-st)/hdr['newFs']/60 for st, end in find_events(data[version]==1)])

        # insert in group dictionary
        ID = hdr['patient_tag']
        group = hdr['group']
        len_dic[group] = lens if len_dic[group] is None else np.concatenate([len_dic[group], lens])

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    colors = ['blue', 'lightskyblue', 'khaki', 'darkorange', 'red']
    bar_width = 0.5 if version=='Osc_chain' else 1
    lw = 0.1
    xs = np.arange(0, 50+bar_width, bar_width)
    bottom = np.zeros(len(xs))
    
    # run over each group
    total_lens = []
    for i, group_key in enumerate(len_dic.keys()):
        # compute bar heights
        lens = len_dic[group_key]
        if lens is None: continue
        total_lens += lens.tolist()
        ys = np.array([sum(np.logical_and(lens>=x, lens<x+bar_width)) for x in xs])

        # create label        
        if i==0: lab = '< ' + group_key.split('-')[-1]
        elif i==len(len_dic)-1: lab = '> ' + group_key.split('-')[0][-3:]
        else: lab = group_key.replace('SS ', '').replace('-', ' - ')
        label = f'{lab}   [{round(np.median(lens), 1)} min]'

        # create bars
        ax.bar(xs, ys, color=colors[i], bottom=bottom, ec='k', width=bar_width, lw=lw, label=label)

        # update bottom
        bottom = ys if i==0 else bottom+ys  

    median = round(np.median(total_lens), 2)
    print(f'{version}:\nMedian length across all segments: {median}\n')

    # set layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-1, 51])
    ax.set_ylim([0, 1.05*max(bottom)])
    ax.set_xlabel('Window length\n(minutes)', fontsize=11)
    ax.set_ylabel('# of segments', fontsize=11)
    title = f'Patient bins based on ratio of expressed HLG [median segment length]'
    ax.legend(loc=0, facecolor='k', title=title, ncol=2, handletextpad=0.8,
                frameon=False, fontsize=10, title_fontsize=11)  
    if version=='Osc_chain':
        fig_title = 'Histogram of chains of breathingoscillation segment lengths during NREM.'
    elif version=='stable_SS':
        fig_title = 'Histogram of stable SS segment lengths during NREM.'
    subtitle = 'Apnea patients with AHI>10 were included (N=100).'
    # ax.set_title(f'{fig_title}\n{subtitle}')

##################################################################################
def run_stable_SS_detector(i, path, sim_df, output_folder, dataset):
    subjectID = path.split('/')[-1].split('.hf')[0]           
    out_path = f'{output_folder}{subjectID}.png'
    already = len(os.listdir(output_folder))
    ratio = f'{i+1}/{len(sim_df)}'
    print(f'Assessing {dataset.upper()} recording: {ratio} ({already})')

    # load recording
    try:
        data, hdr = load_sim_output(path)
        hdr['group'] = sim_df.loc[np.where(sim_df.SS_path==subjectID)[0][0], 'SS group']
        hdr['SS_threshold'] = 0.5
    except Exception as error: print('Loading error: ', error); return (None, None)

    # compute SS trace
    win = int(3*60*hdr['newFs'])
    data['SS_trace'] = data['ss_conv_score'].rolling(win, min_periods=1, center=True).median().fillna(0)

    # create chains of breathing oscillations 
    data, hdr = compute_osc_chains(data, hdr)

    # Apply stabe SS based on change point detection
    data = compute_change_points_ruptures(data, hdr)

    # skip already created figures
    if not os.path.exists(out_path):
        # create plot
        plot_SS(data, hdr, out_path=out_path)

    return (data, hdr)
##################################################################################

if __name__ == '__main__':
    dataset = 'mgh'
    date = '11_09_2023'
    multiprocess = False
    version = 'SS_cases'    # 'SS_cases'  'high_CAI'  'HLG_OSA'  'REM_OSA  'NREM_OSA'  'Heart_Failure'

    # set input folder
    exp = 'Expansion' if os.path.exists(f'/media/cdac/Expansion/CAISR data1/Rule_based') else 'Expansion1'
    input_folder = f'/media/cdac/{exp}/LG project/hf5data/{dataset}_{date}/'
    all_paths = glob.glob(input_folder + '*.hf5')

    # set output folder
    output_folder = f'/media/cdac/{exp}/LG project/HLG_figures_{dataset}_{version}/'
    os.makedirs(output_folder, exist_ok=True)

    # set all recording paths from dataset
    sim_df = pd.read_csv('csv_files/mgh_table1 100_SS_cases.csv' )
    all_paths = sort_input_files(all_paths, sim_df, version)
    
    # create futures
    result = []
    for i, p in enumerate(all_paths):
        r = run_stable_SS_detector(i, p, sim_df, output_folder, 'mgh') 
        result.append(r)

    # compute histograms
    create_length_histogram(sim_df, result, version='Osc_chain')
    plt.savefig('Osc_chain_paper_Figure.png', dpi=1200)
    create_length_histogram(sim_df, result, version='stable_SS')
    plt.savefig('Stable_SS_paper_Figure.png', dpi=1200)
    plt.show()
    import pdb; pdb.set_trace()



        


