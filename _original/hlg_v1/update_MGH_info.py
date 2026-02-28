import glob, h5py, os  
import numpy as np
import pandas as pd

# load SS functions
from Compute_sleep_metrics import compute_sleep_metrics

# import utils functions
from Event_array_modifiers import find_events



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



if __name__ == '__main__':
    dataset = 'mgh'
    date = '11_09_2023'

    # set input folder
    exp = 'Expansion' if os.path.exists(f'/media/cdac/Expansion/CAISR data1/Rule_based') else 'Expansion1'
    input_folder = f'/media/cdac/{exp}/LG project/hf5data/{dataset}_{date}/'
    input_files = glob.glob(input_folder + '*.hf5')
    stripped_paths = np.array([p.split('/')[-1].split('.hf5')[0] for p in input_files])
    
    # set all recording paths from dataset
    table_path = 'csv_files/caisr_mgh_v4_table1.csv'
    caisr_table = pd.read_csv(table_path) 
    caisr_table = caisr_table.drop(columns=['mgh_v4', 'path_prepared'])
    sim_df = pd.DataFrame([])

    # run over all files 
    not_in_T1, DOV_mismatch = 0, 0
    for i, path in enumerate(stripped_paths):
        print(f'extracting SS output {i}/{len(stripped_paths)} ..', end='\r')

        ID = path.split('_')[0]
        DOV = path.split('_')[1]
        if ID not in caisr_table.HashID.values: 
            not_in_T1 += 1
            continue
        loc = np.where(ID==caisr_table.HashID.values)[0]
        table_DOV = caisr_table.loc[loc[0], 'DOVshifted'].split(' ')[0].replace('/', '')
        if DOV != table_DOV:
            DOV_mismatch += 1
            continue

        # copy original table row
        sim_df = pd.concat([sim_df, caisr_table.loc[loc, :]], ignore_index=True)
        sim_df.loc[len(sim_df)-1, 'SS_path'] = path
        
        # load data
        try:
            cols = ['apnea', 'sleep_stages', 'self similarity', 'tagged']
            data, hdr = load_sim_output(input_files[i], cols=cols)
        except:
            continue

        # insert 3% metrics
        for metric in ['AHI', 'RDI', 'CAI']:
            sim_df.loc[len(sim_df)-1, f'{metric.lower()}_3%'] = hdr[metric]

        # compute individual event indices
        resp_map = {1:'Obs', 2:'Cen', 3:'Mix', 4:'Hyp', 5:'RERA'}
        for key in resp_map.keys():
            num = len(find_events(np.logical_and(data.patient_asleep, data.apnea==key))) / hdr['sleep_time']
            tag = resp_map[key] + '_i'
            sim_df.loc[len(sim_df)-1, tag] = num

        # insert SS metrics
        SS_time = np.sum(np.logical_and(data.patient_asleep, data['self similarity']))/(3600* hdr['newFs'])
        sim_df.loc[len(sim_df)-1, 'T_SS'] = round(SS_time / hdr['sleep_time'], 2)
        osc_num = len(find_events(np.logical_and(data.patient_asleep, data.tagged>0))) / hdr['sleep_time']
        sim_df.loc[len(sim_df)-1, 'T_osc'] = osc_num

        # set (N)REM regions
        REM_region = np.where(data.sleep_stages==4)[0]
        NREM_region = np.where(np.logical_and(data.sleep_stages>0, data.sleep_stages<4))[0]
        regions, tags = [REM_region, NREM_region], ['REM', 'NREM']       
        # compute AHI in (N)REM
        for region, tag in zip(regions, tags):
            if len(region)>0:
                apneas = data.loc[region, 'apnea'].values
                stages = data.loc[region, 'sleep_stages'].values
                RDI, AHI, CAI, sleep_time = compute_sleep_metrics(apneas, stages, exclude_wake=True)
            else:
                RDI, AHI, CAI, sleep_time = 0, 0, 0, 0
            # insert in DF
            sim_df.loc[len(sim_df)-1, f'RDI_{tag}'] = RDI
            sim_df.loc[len(sim_df)-1, f'AHI_{tag}'] = AHI
            sim_df.loc[len(sim_df)-1, f'CAI_{tag}'] = CAI
            sim_df.loc[len(sim_df)-1, f'{tag}_time'] = sleep_time

    print(f'{not_in_T1} not in Table 1')
    print(f'{DOV_mismatch} DOV mismatch')
    sim_path_updated = table_path.replace('.csv', '_updated.csv')
    import pdb; pdb.set_trace()
    sim_df.to_csv(sim_path_updated, header=sim_df.columns, index=None, mode='w+')




