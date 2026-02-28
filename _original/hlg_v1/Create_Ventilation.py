import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from scipy.signal import detrend, savgol_filter

# import algorithm functions
from Ventilation_envelope import compute_ventilation_envelopes
# import utils functions
from Event_array_modifiers import find_events, window_correction


def create_Ventilation_trace(data, Fs, plot=False):
    assert Fs == int(Fs), 'Proveded "Fs" is a double'
    Fs = int(Fs)
    # compute Ventilation trace for both NPT and ABD
    for col in ['breathing_trace', 'ABD']:
        if not col in data.columns: continue
        # compute mean envelopes
        df = compute_ventilation_envelopes(data.copy(), Fs, channels=[col])
        
        # compute normalized positive and negative envelope
        Q = df.Ventilation_pos_envelope.mean() - df.Ventilation_neg_envelope.mean()
        sig = detrend(data[col].fillna(0)) / Q
        df['pos_env'] = np.array(sig)
        df['neg_env'] = np.array(sig)
        df['pos_env'] = df['pos_env'].rolling(20, center=True, min_periods=1).quantile(0.95)
        df['neg_env'] = df['neg_env'].rolling(20, center=True, min_periods=1).quantile(0.05)
        # soft smooth
        df['pos_env'] = savgol_filter(df['pos_env'], 51, 1)
        df['neg_env'] = savgol_filter(df['neg_env'], 51, 1)
        
        # create Ventilation* trace
        Ventilation = df['pos_env'].values[::10] - df['neg_env'].values[::10]
        Ventilation[Ventilation<0] = 0
        df[f'Ventilation_{col}'] = np.repeat(Ventilation, 10)[:len(df)]

        # compute median Ventilation --> Eupnea
        df[f'average_Ventilation_{col}'] = df[f'Ventilation_{col}'].rolling(Fs*60*30, center=True, min_periods=1).median()
        # compute distance to Eupnea
        df[f'd_i_{col}'] = df[f'Ventilation_{col}'] / df[f'average_Ventilation_{col}']

        # insert to DataFrame
        data[f'Ventilation_{col}'] = df[f'Ventilation_{col}'].values
        data[f'Eupnea_{col}'] = df[f'average_Ventilation_{col}'].values
        data[f'd_i_{col}'] = df[f'd_i_{col}'].values
        data.loc[np.where(data[f'd_i_{col}']>1)[0], f'd_i_{col}'] = 1

        # find breathing below eupnea
        below_eupnea = df[f'Ventilation_{col}'] < df[f'average_Ventilation_{col}']
        df[f'below_Eupnea_{col}'] = below_eupnea.rolling(6*Fs, center=True).mean() == 1
        df[f'below_Eupnea_{col}'] = window_correction(df[f'below_Eupnea_{col}'], window_size=6*Fs)

        # find large ventilation spikes for possible arousals locations
        trailing_min = df[f'Ventilation_{col}'].rolling(3*Fs, min_periods=Fs).min()
        trailing_max = df[f'Ventilation_{col}'].rolling(3*Fs, min_periods=Fs).max()
        biggest_arousal = trailing_max - trailing_min
        biggest_arousal = biggest_arousal.rolling(10*Fs, min_periods=Fs).max()
        df['possible_large_locs'] = 0
        for loc, _ in find_events(biggest_arousal>0.5):
            # skip arousal locs on descending Ventilation (FPs), or <0
            if any(df.loc[loc-2*Fs:loc, f'Ventilation_{col}']>df.loc[loc, f'Ventilation_{col}']): continue
            if df.loc[loc, f'Ventilation_{col}'] <=0: continue
            df.loc[loc, 'possible_large_locs'] = 1

        # create smooth d_i below eupnea 
        # 1. only when meeting hyponpea threshold
        # 2. for more than 6 seconds
        data[f'd_i_{col}_smooth'] = 1
        data[f'd_i_{col}_smooth'] = data[f'd_i_{col}_smooth'].astype(float)
        data['arousal_locs'] = 0
        for st, end in find_events(df[f'below_Eupnea_{col}']>0):
            avg_decrease = np.mean(data.loc[st:end, f'd_i_{col}'])
            if avg_decrease <= 0.85:
                data.loc[st:end, f'd_i_{col}_smooth'] = avg_decrease
                # if no large loc is found, keep end of resp. event
                left, right = end-2*Fs, end+10*Fs
                plus = np.argmax(df.loc[left:right, f'Ventilation_{col}'])
                data.loc[left+plus, 'arousal_locs'] = 1
                df.loc[left:left+plus, 'possible_large_locs'] = 0
        
        # correct left over additional arousals
        for st, end in find_events(df['possible_large_locs']>0):
            # if no large loc is found, keep end of resp. event
            left, right = end-2*Fs, end+10*Fs
            plus = np.argmax(df.loc[left:right, f'Ventilation_{col}'])
            loc = left+plus
            close_double = np.where(data.loc[loc-3*Fs:loc+3*Fs, 'arousal_locs']==1)[0]
            if len(close_double)>0:
                if close_double[0]!=30:
                    data.loc[loc-3*Fs:loc+3*Fs, 'arousal_locs'] = 0
            df.loc[left:loc, 'possible_large_locs'] = 0
            df.loc[loc, 'possible_large_locs'] = 1

        # add large arousals unassociated with large detected events
        data['arousal_locs'] += df['possible_large_locs']

        # plot
        if plot:
            # plot breathing trace
            plt.plot(df[f'{col}'].mask(data.Stage<5)-1,'y', alpha=0.4)
            plt.plot(df[f'{col}'].mask(data.Stage==5)-1,'k', alpha=0.4)
            # plt.plot(trailing_min+ df[f'pos_env'].mean(),c='tab:orange', alpha=0.4)
            # plt.plot(trailing_max+ df[f'pos_env'].mean(),c='c', alpha=0.4)
            # plot Ventilation 
            vent_plot = df[f'Ventilation_{col}'] + df[f'pos_env'].mean()
            plt.plot(vent_plot,'b')
            plt.plot([df[f'pos_env'].mean()]*len(df), 'r')
            plt.plot(df[f'average_Ventilation_{col}']+df[f'pos_env'].mean(),'g', alpha=0.4)
            plt.plot(data[f'd_i_{col}']*df[f'average_Ventilation_{col}']+df[f'pos_env'].mean(),'g')
            plt.plot(data[f'd_i_{col}_smooth']*df[f'average_Ventilation_{col}']+df[f'pos_env'].mean(),'r')
            # add labels
            label_color = [None, 'b', 'g', 'c', 'm', 'r', None, 'g']
            for li, labels in enumerate([data.Apnea, data.Apnea_algo, df[f'below_Eupnea_{col}']]):
                loc = 0
                for i, j in groupby(labels):
                    len_j = len(list(j))
                    if not np.isnan(i) and label_color[int(i)] is not None:
                        if li==0:
                            plt.plot([loc, loc+len_j], [0]*2, c=label_color[int(i)], lw=2)
                            if int(i) == 7: plt.plot([loc, loc+len_j], [-0.1]*2, c='m', lw=1)
                        if li==1:
                            plt.plot([loc, loc+len_j], [-0.5]*2, c=label_color[int(i)], lw=2)
                        if li==2:
                            plt.plot([loc, loc+len_j], [-0.25]*2, c='k', lw=2)
                    loc += len_j

            # print possible arousal locations
            for loc in np.where(data['arousal_locs']==1)[0]+1:
                if loc >= len(df)-1: loc = len(df)-1 # correct for end of recording
                y = (data.loc[loc, f'Ventilation_{col}'] + df.loc[loc, f'pos_env'].mean()) + 0.2
                plt.text(loc, y, '*', c='k', ha='center')

            # print possible arousal locations
            for loc in np.where(df['possible_large_locs']==1)[0]+1:
                if loc >= len(df)-1: loc = len(df)-1 # correct for end of recording
                y = (data.loc[loc, f'Ventilation_{col}'] + df.loc[loc, f'pos_env'].mean()) + 0.2
                plt.text(loc, y, '*', c='r', ha='center')

            plt.show()
            import pdb; pdb.set_trace()

    # remove DF
    del df

    return data



