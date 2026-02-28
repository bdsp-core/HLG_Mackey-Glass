import numpy as np
import pandas as pd

# import utils functions
from Data_writers import write_to_hdf5_file
from Event_array_modifiers import find_events

# save
def save_output(data, hdr, out_file, channels):
    # put data in DataFrame
    df = pd.DataFrame([])
    for ch in channels + ['spo2', 'arousal']: 
        if ch in data.columns:
            df[ch] = data[ch].values
    if 'resp' in data.columns:
        df['apnea'] = data.resp.values
    df['flow_reductions'] = data.flow_reductions.values
    df['sleep_stages'] = data.stage.values
    df['self similarity'] = data.T_sim.values
    df['tagged'] = data.TAGGED.values
    df['ss_conv_score'] = data.ss_conv_score.values

    # add header info
    for key in hdr.keys():
        df[key] = hdr[key]

    write_to_hdf5_file(df, out_file, overwrite=True)

# Report
def create_report(output_data, hdr):
    # set sampling frequencies
    originalFs = hdr['Fs']
    newFs = hdr['newFs']
    finalFs = 1

    # Init DF
    original_cols = ['flow_reductions', 'T_sim', 'stage']
    data = pd.DataFrame([], columns=['start_idx', 'end_idx'] + original_cols)
    
    # Resample data to 1 Hz
    for sig in original_cols: 
        image = np.repeat(output_data[sig].values , finalFs)
        image = image[::newFs]    
        # 4. Insert in new DataFrame
        data[sig] = image
    
    # save columns of interest
    factor = originalFs // finalFs
    ind0 = np.arange(0, data.shape[0]) * factor
    ind1 = np.concatenate([ind0[1:], [ind0[-1]+factor]])
    data['second'] = range(len(data))
    data['start_idx'] = ind0
    data['end_idx'] = ind1

    # count number of central apneas/hypopneas
    lengths = {'apnea': 0, 'hypopnea': 0}
    for st, end in find_events(output_data['ss_conv_score']>=hdr['SS_threshold']):
        if any(data.loc[st:end, 'stage']==0): continue
        tag = 'hypopnea'
        if np.any(output_data.loc[st:st+30*newFs, 'flow_reductions']==1):
            tag = 'apnea'
        lengths[tag] += 1

    # compute number of tags --> set as hypopnea estimate*
    SS_num = len(find_events(output_data['ss_conv_score']>=hdr['SS_threshold']))

    # create summary report
    summary_report = pd.DataFrame([])
    duration = sum(data.stage!=0) / finalFs / 3600
    summary_report['signal duration (h)'] = [np.round(duration, 2)]
    summary_report['detected central apneas'] = [lengths[f'apnea']]
    summary_report['detected central hypopneas'] = [lengths[f'hypopnea']]
    summary_report['cai'] = [np.round(lengths[f'apnea'] / duration, 1)]
    summary_report['cahi'] = [np.round(sum(lengths.values()) / duration, 1)]
    summary_report['SS%'] = np.round((np.sum(data['T_sim']==1) / (len(data))) * 100, 1)  

    # remove original columns
    for col in original_cols: 
        if col in data.columns: 
            data = data.drop(columns=col)   
    
    # save data into .csv files
    full_report = pd.concat([data, summary_report], axis=1)
    
    return full_report, summary_report