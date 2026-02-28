import glob, os, h5py
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count


# set dbx pfx
from Event_array_modifiers import find_events

# import local functions
from EM_output_to_Figures import add_arousals, match_EM_with_SS_output, post_process_EM_output
from Convert_SS_seg_scores import convert_ss_seg_scores_into_arrays
from SS_output_to_EM_input import load_sim_output
from Recreate_LG_array import create_total_LG_array
from EM_output_histograms import compute_histogram




# extraction functions
def extract_EM_output(input_files, interm_folder, hf5_folder, version, dataset, csv_file, bar_folder):
    # set empty lists / dictionaries
    LG_data, G_data, D_data, GxD_data, SS_data, valid_data, ID_data, Stages_data = [], [], [], [], [], [], [], []
    SS_dic = {}
    for i in np.arange(0, 10, 2):
        for param in ['SS', 'LG', 'G', 'D', 'VV', 'St']:
            SS_dic[f'seg_{i/10}-{(i+2)/10}_{param}'] = []

    # Create a pool of workers
    num_workers = cpu_count()
    pool = Pool(num_workers)

    # Prepare arguments for parallel processing
    process_args = [(input_file, 
                    interm_folder,
                    hf5_folder,
                    version,
                    dataset,
                    csv_file,
                    bar_folder) for input_file in input_files]

    # Process files in parallel and collect results
    for i, (data_dic) in enumerate(pool.starmap(process_EM_output, process_args)):
        num = process_args[i][1]

        LG_data = np.concatenate([LG_data, data_dic['LGs']])
        G_data = np.concatenate([G_data, data_dic['Gs']]) 
        D_data = np.concatenate([D_data, data_dic['Ds']])
        GxD_data = np.concatenate([GxD_data, data_dic['Gs']*data_dic['Ds']])
        SS_data = np.concatenate([SS_data, data_dic['SSs']])
        valid_data = np.concatenate([valid_data, data_dic['valids']])
        ID_data = np.concatenate([ID_data, np.ones(len(data_dic['valids']))*int(num.strip())])
        Stages_data = np.concatenate([Stages_data, data_dic['Stages']])

        # save estimated LG values per self-similarity segment score
        for LG, G, D, SS, VV, ST in zip(data_dic['LGs'], data_dic['Gs'], data_dic['Ds'], data_dic['SSs'], data_dic['valids'], data_dic['Stages']):
            for j in np.arange(0, 10, 2):
                ran = f'seg_{j/10}-{(j+2)/10}'
                if SS>=j and SS<j+0.2: break
            SS_dic[ran+'_SS'].append(SS)
            SS_dic[ran+'_LG'].append(LG)
            SS_dic[ran+'_G'].append(G)
            SS_dic[ran+'_D'].append(D)
            SS_dic[ran+'_VV'].append(VV)
            SS_dic[ran+'_St'].append(ST)
    pool.close()
    pool.join()

    # save per group estimations
    sorted_data = [LG_data, G_data, D_data, GxD_data, SS_data, valid_data, ID_data, Stages_data]
    names = ['LG_data', 'G_data', 'D_data', 'GxD_data', 'SS_data', 'valid_data', 'ID_data', 'Stages_data']
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
        for param in ['SS', 'LG', 'G', 'D', 'VV', 'St']:
            df[param] = SS_dic[f'{ran}_{param}']
        df.to_csv(f'{interm_folder}{ran}.csv', header=df.columns, index=None, mode='w+')


def process_EM_output(input_file, interm_folder, hf5_folder, version, dataset, csv_file, bar_folder):
    # extract data
    data = pd.read_csv(input_file)
    
    # convert SS scores into columns
    data = convert_ss_seg_scores_into_arrays(data)

    # post-process EM output
    data = post_process_EM_output(data)

    # set/extract header fields
    num = input_file.split('/Study')[-1].split('\\Study')[-1].split('.csv')[0].strip()
    hdr = {'Study_num': f'Study {num}'}
    for col in ['patient_tag', 'Fs', 'original_Fs']:
        hdr[col] = data.loc[0, col]

    # add arousals
    if 'MGH' in version:
        _, hdr['SS group'] = add_arousals(data, version, 'mgh', hf5_folder, csv_file)
        
    # find matching SS output path, and extract SS data
    sim_path, _ = match_EM_with_SS_output(data, version, dataset, csv_file)
    path = hf5_folder + sim_path + '.hf5'
    failed = False
    try:
        SS_df, SS_hdr = load_sim_output(path, ['flow_reductions'])
    except:
        failed = True
    if failed or 'flow_reductions' not in SS_df.columns:
        SS_df, SS_hdr = load_sim_output(path, ['apnea', 'cpap_start'])
    assert len(SS_df)>0.99*len(data), 'matching SS output does not match EM data'

    # Cut only before CPAP start
    if 'CPAP' in version:   
        data = data.loc[:SS_hdr['cpap_start'], :].copy()
        SS_df = SS_df.loc[:SS_hdr['cpap_start'], :].copy()

    # recreate LG array for "LG hypnogram"
    total_LG = create_total_LG_array(data)
    total_LG = pd.DataFrame(total_LG, columns=['LG_hypno'])
    hypno_folder = os.path.join(interm_folder, 'hypnograms/')
    os.makedirs(hypno_folder, exist_ok=True)
    out_path = os.path.join(hypno_folder, f'Study {num}.csv')
    total_LG.to_csv(out_path, header=total_LG.columns, index=None, mode='w+')

    # set data before CPAP and compute histograms
    data['total_LG'] = total_LG
    compute_histogram(data, hdr, bar_folder)

    # run over all segments
    Errors, Vmaxs, LGs, Gs, Ds, Ls, SSs_seg, valid_seg, Stages = [], [], [], [], [], [], [], [], []
    for stage in ['nrem', 'rem']:
        # set run variables
        starts = data[f'{stage}_starts'].dropna().values.astype(int)
        ends = data[f'{stage}_ends'].dropna().values.astype(int)
        for start, end in zip(starts, ends):
            if 'CPAP' in version and end > SS_hdr['cpap_start']:
                continue
            # Extract parameters, and set out_path
            loc = np.where(data[f'{stage}_starts']==start)[0][0]
            Errors.append(round(data.loc[loc, 'rmse_Vo'], 2))
            Ls.append(data.loc[loc, f'L_{stage}'])
            Vmaxs.append(round(data.loc[loc, 'Vmax'], 2))
            LGs.append(data.loc[loc, f'LG_{stage}_corrected'])
            Gs.append(data.loc[loc, f'G_{stage}'])
            Ds.append(data.loc[loc, f'D_{stage}'])
            # valid_seg.append(len(find_events(data.d_i_ABD_smooth<1))>=4)
            valid_seg.append(len(find_events(SS_df.loc[start:end]>0))>=5)
            SSs_seg.append(data.loc[start, 'SS_score'])
            Stages.append(stage)

    # save estimated LG values per patient self-similarity group
    error_thresh = 1.8
    inds = np.array(Errors) < error_thresh
    LGs = np.array(LGs)[inds]
    Gs = np.array(Gs)[inds]
    Ds = np.array(Ds)[inds]
    SSs = np.array(SSs_seg)[inds]
    valids = np.array(valid_seg)[inds]
    Stages = np.array(Stages)[inds]
    data_dic = {
        'LGs': LGs,
        'Gs': Gs, 
        'Ds': Ds,
        'SSs': SSs,
        'valids': valids,
        'Stages': Stages,
    }

    return data_dic

        
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


if __name__ == '__main__':
    # dataset: bdsp, mgh, redeker, rt
    # versions: [CPAP_success, CPAP_failure, high_CAI, NREM_OSA, REM_OSA, SS_OSA, SS_range, Heart_Failure, Altitude]
    Ut_smooth = 'non-smooth'  # smooth   non-smooth

    dataset: str = 'mgh'
    versions: list[str] = ['SS_range']


    # extract data for all versions
    lens = []
    for i, version in enumerate(versions):
        # set hf5 folder
        hf5_folder = 'SS paper files/'
        bar_folder = f'./bars/{dataset}_{version}/'
        os.makedirs(bar_folder, exist_ok=True)

        # set input folder TODO: set path to this dropbox folder
        input_folder = f'Drobbox/Final Code Revision/{Ut_smooth}/{dataset.upper()}_{version}/'
        input_files = glob.glob(input_folder + '*.csv')
        lens.append(len(input_files))

        # set csv file
        csv_file = f'csv_files/{dataset}_table1 100_{version}_cases.csv'
        if 'CPAP' in version:
            csv_file = csv_file.replace('100_', '200_')

        # Set intermediate results folder
        base_folder = f'./interm_Results/{Ut_smooth}/group_analysis/'
        interm_folder = base_folder + f'{dataset}_{version}/'
    
        # if intermediate results do not exist, extract EM output
        # if not os.path.exists(interm_folder):
        print(f'\n>> {version} <<')
        extract_EM_output(input_files, interm_folder, hf5_folder, version, dataset, csv_file, bar_folder)
            
    