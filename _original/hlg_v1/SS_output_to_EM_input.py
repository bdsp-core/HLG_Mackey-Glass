import glob, h5py, random, os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# load SS functions
from Create_Ventilation import create_Ventilation_trace
from Compute_sleep_metrics import compute_sleep_metrics

# import utils functions
from Event_array_modifiers import find_events


# loading functions
def load_sim_output(path, cols=[]):
    # init DF
    hdr_cols = ['patient_tag', 'test_type',
                'rec_type', 'cpap_start', 'Fs', 'SS_threshold']
    if len(cols) == 0:
        cols = ['abd', 'chest', 'spo2', 'apnea', 'arousal', 'sleep_stages']
        cols += ['flow_reductions', 'tagged', 'self similarity', 'ss_conv_score']
        cols += hdr_cols
    # cols = hdr_cols
    data = pd.DataFrame([], columns=cols)

    f = h5py.File(path, 'r')
    for key in cols:
        if key not in f.keys():
            continue
        vals = f[key][:]
        # vals[np.isnan(vals)] = 0
        data[key] = vals
    f.close()

    # set chest to ABD if no ABD is available
    if 'abd' in data.columns and 'chest' in data.columns and all(data.abd.isna()) and not all(data.chest.isna()):
        print('"CHEST" replaced "ABD"!')
        data['abd'] = data.chest.values
        data = data.drop(columns=['chest'])

    # header:
    hdr = {}
    hdr['newFs'] = 10
    for hf in hdr_cols:
        if not hf in data.columns:
            continue
        val = data.loc[0, hf]
        try:
            val = val.decode('utf-8')
        except:
            pass
        hdr[hf] = val
        data = data.drop(columns=[hf])

    # sleep metrics
    if 'sleep_stages' in data.columns:
        data['patient_asleep'] = np.logical_and(
            data.sleep_stages < 5, data.sleep_stages > 0)
        RDI, AHI, CAI, sleep_time = compute_sleep_metrics(
            data.apnea, data.sleep_stages, exclude_wake=True)
        hdr['RDI'], hdr['AHI'], hdr['CAI'], hdr['sleep_time'] = RDI, AHI, CAI, sleep_time

    return data, hdr


def remove_bad_signal_recordings(df):
    # list of bad signals quality recordings
    bad_recs = [
        'b346da6',
        'dd20181',
        'b551d67',
        '0be8481',
        'fe5cdc8',
        '97a9256',
        'a58f9df',
        '76f60a8',
        '9b91a5f',
        'ad0ee71',
        'f3a9d3e',
        'f1487f4',
        'd300222',
        '2fbff9f',
        '8ed09d7',
        'e7de9ac',
        'e0e70ec',
        '55a330f',
        'd886538',
        '204b19a',
        '15a8620',
        '8c4d264',
    ]
    # filter good ids
    IDs = df.SS_path.values
    good_ids = [i for i, ID in enumerate(IDs) if ID[:len(bad_recs[0])] not in bad_recs]
    df = df.loc[good_ids].reset_index(drop=True)

    return df


def extract_latest_SS_outputs(sim_df, input_files):
    # run over all input files
    for p, path in enumerate(input_files):
        print(f'extracting SS output {p}/{len(input_files)} ..', end='\r')
        # set ID
        ID = path.split('/')[-1].split('.hf5')[0]
        if not ID in sim_df.path_name.values:
            continue

        # load sim data
        cols = ['sleep_stages', 'apnea', 'self similarity', 'tagged']
        data, hdr = load_sim_output(path, cols=cols)

        # insert info in sim_df
        loc = np.where(ID == sim_df.path_name)[0][0]
        sim_df.loc[loc, 'RDI_3%'] = hdr['RDI']
        sim_df.loc[loc, 'AHI_3%'] = hdr['AHI']
        sim_df.loc[loc, 'CAI_3%'] = hdr['CAI']
        sim_df.loc[loc, 'sleep_time'] = hdr['sleep_time']
        SS_time = np.sum(np.logical_and(data.patient_asleep,
                         data['self similarity']))/(3600 * hdr['newFs'])
        sim_df.loc[loc, 'T_SS_new'] = round(SS_time / hdr['sleep_time'], 2)
        osc_time = len(find_events(np.logical_and(
            data.patient_asleep, data.tagged > 0))) / hdr['sleep_time']
        sim_df.loc[loc, 'T_osc_new'] = round(osc_time, 2)

    return sim_df


def patient_selection(sim_df, version, sim_info_subset_path):
    # create selection based on <version>
    random.seed(0)
    if version == 'SS_cases':
        # create 5 groups with ranging SS
        ranges = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 1)]
        for i, maxi in ranges:
            ran = np.where(np.logical_and(sim_df['T_SS'] > i, sim_df['T_SS'] < maxi))[0]
            ran = random.sample(ran.tolist(), k=20)
            sim_df.loc[ran, 'SS group'] = f'SS {i}-{maxi}'
        # save subset
        selection_df = sim_df.dropna(subset=['SS group']).reset_index(drop=True)

    elif version == 'high_CAI':
        # find all patients with an AHI>10
        selection_df = sim_df.loc[sim_df.ahi > 10]

        # find patients where CAI > 5
        selection_df = selection_df.loc[selection_df['cai_3%'] > 5].sort_values(
            by=['cai'], ascending=False)

        # find all patients where >50% of apneas are central
        selection_df = selection_df.query('cai > oai').reset_index(drop=True)

        # set SS group to N/A
        selection_df.loc[:, 'SS group'] = 'N/A'

    elif version == 'HLG_OSA':
        # filter patients with a CAI<1
        selection_df = sim_df.loc[sim_df['cai_3%'] < 5]

        # filter patients with an AHI >30
        selection_df = selection_df.loc[selection_df['ahi_3%'] > 15].sort_values(
            by=['T_SS'], ascending=False)

        # set SS group to N/A
        selection_df.loc[:, 'SS group'] = 'N/A'

    elif version == 'REM_OSA':
        # find patients where REM sleep time > 10% of total sleep time
        selection_df = sim_df.loc[sim_df.REM_time > 0.1*sim_df.h_sleep]

        # find all patients with an AHI>15
        selection_df = selection_df.loc[selection_df.ahi > 15]

        # find all patients where 3x the amount of apneas are found in REM vs NREM
        selection_df = selection_df.query('AHI_REM > 3*AHI_NREM').reset_index(drop=True)

        # set SS group to N/A
        selection_df.loc[:, 'SS group'] = 'N/A'

    elif version == 'NREM_OSA':
        # find patients where REM sleep time > 10% of total sleep time
        selection_df = sim_df.loc[sim_df.REM_time > 0.1*sim_df.h_sleep]

        # find all patients with an AHI>15
        selection_df = selection_df.loc[selection_df.ahi > 15]

        # find all patients where 2x the amount of apneas are found in NREM vs REM
        selection_df = selection_df.query('AHI_NREM > 2*AHI_REM').reset_index(drop=True)

        # set SS group to N/A
        selection_df.loc[:, 'SS group'] = 'N/A'

    elif 'CPAP' in version:
        selection_dict = {}
        for tag in ['failure', 'success']:
            # filter successful CPAP patients
            cpap_csv_path = 'csv_files/sim_df_03_20_2023_all.csv'
            cpap_df = pd.read_csv(cpap_csv_path)
            if tag == 'success':
                cpap_df = cpap_df.loc[cpap_df['CPAP success 3%']
                                      == 'True'].reset_index(drop=True)
            else:
                cpap_df = cpap_df.loc[cpap_df['CPAP success 3%']
                                      == 'False'].reset_index(drop=True)

            # remove patients that are not in cpap_df
            subjectIDs = np.array([s.split('_')[0] for s in cpap_df['subjectID']])
            remove_locs = []
            metrics = ['T_SS1', 'AHI1_3%', 'CAI1_3%', 'subjectID']
            for i, ID in enumerate(sim_df['HashID'].values):
                loc = np.where(ID == subjectIDs)[0]
                if len(loc) == 0:
                    remove_locs.append(i)
                    continue

                # insert metrics
                for metric in metrics:
                    sim_df.loc[i, metric] = cpap_df.loc[loc[0], metric]

            # sort by ascending SS
            selection_dict[tag] = sim_df.drop(remove_locs)
            selection_dict[tag] = selection_dict[tag]
            cai_mask = selection_dict[tag]['CAI1_3%'] < 10
            if tag == 'failure':
                low_cai_df = selection_dict[tag][cai_mask].sort_values(
                    by=['CAI1_3%'], ascending=True)
                selection_dict[tag] = low_cai_df[:300].sort_values(by=['T_SS1'], ascending=False)[
                    :200].reset_index(drop=True)
            else:
                selection_dict[tag] = selection_dict[tag][cai_mask].sample(
                    n=200, random_state=1).reset_index(drop=True)

        # select based on version
        if 'success' in version:
            selection_df = selection_dict['success']
        else:
            selection_df = selection_dict['failure']

    selection_df.to_csv(sim_info_subset_path,
                        header=selection_df.columns, index=None, mode='w+')


def sort_input_files(all_paths, sim_df, version):
    stripped_paths = np.array(
        [p.split('/')[-1].split('\\')[-1].split('.hf5')[0] for p in all_paths])

    # sort paths ..
    sorted_paths = []
    if version == 'SS_cases':
        # based on SS groups
        for ID in sim_df.sort_values(by=['SS group']).SS_path:
            assert ID in stripped_paths, 'Somehow ID not found in "all_paths".'
            loc = np.where(ID == stripped_paths)[0][0]
            sorted_paths.append(all_paths[loc])
    elif version == 'high_CAI':
        # based on ascending CAI
        for ID in sim_df.sort_values(by=['T_SS']).SS_path:
            assert ID in stripped_paths, 'Somehow ID not found in "all_paths".'
            loc = np.where(ID == stripped_paths)[0][0]
            sorted_paths.append(all_paths[loc])
    elif version == 'HLG_OSA':
        # based on ascending CAI
        for ID in sim_df.sort_values(by=['T_SS']).SS_path:
            assert ID in stripped_paths, 'Somehow ID not found in "all_paths".'
            loc = np.where(ID == stripped_paths)[0][0]
            sorted_paths.append(all_paths[loc])
    elif version == 'REM_OSA':
        # based on ascending SS
        for ID in sim_df.sort_values(by=['T_SS']).SS_path:
            assert ID in stripped_paths, 'Somehow ID not found in "all_paths".'
            loc = np.where(ID == stripped_paths)[0][0]
            sorted_paths.append(all_paths[loc])
    elif version == 'NREM_OSA':
        # based on ascending SS
        for ID in sim_df.sort_values(by=['T_SS']).SS_path:
            assert ID in stripped_paths, 'Somehow ID not found in "all_paths".'
            loc = np.where(ID == stripped_paths)[0][0]
            sorted_paths.append(all_paths[loc])
    elif version == 'Heart_Failure':
        # based on decending EF
        stripped_paths = np.array([int(p[:4]) for p in stripped_paths])
        for ID in sim_df.sort_values(by=['EF'], ascending=False).ID:
            if ID not in stripped_paths:
                print(f'{ID} from redeker table not found among recordings')
                continue
            # assert ID in stripped_paths, 'Somehow ID not found in "all_paths".'
            loc = np.where(ID == stripped_paths)[0][0]
            sorted_paths.append(all_paths[loc])

    elif 'CPAP' in version:
        # based on ascending SS
        for ID in sim_df.sort_values(by=['T_SS1'])['subjectID']:
            assert ID in stripped_paths, 'Somehow ID not found in "all_paths".'
            loc = np.where(ID == stripped_paths)[0][0]
            sorted_paths.append(all_paths[loc])

    return sorted_paths


def sort_altitude_files(all_paths, date):
    sim_df = pd.DataFrame([], columns=['num', 'patient_num', 'altitude'])
    sorted_paths = []
    # run over all patient numbers
    for n, num in enumerate(range(1, 12)):
        patient_paths = [p for p in all_paths if f'P40-{num}' in p]
        # run over all altitudes
        for a, alt in enumerate(range(1, 5)):
            print(f'extracting SS output P40-{num}-{alt}..', end='\r')
            path = [p for p in patient_paths if f'P40-{num}-{alt}' in p]
            if len(path) == 0:
                continue
            sorted_paths += path

            # insert into DF
            loc = len(sim_df)
            sim_df.loc[loc, 'num'] = n*10 + a
            sim_df.loc[loc, 'patient_num'] = path[0].split(date)[-1].split('/')[1]
            sim_df.loc[loc, 'altitude'] = alt

            # add patient metrics
            data, hdr = load_sim_output(path[0])
            sim_df.loc[loc, 'RDI_3%'] = hdr['RDI']
            sim_df.loc[loc, 'AHI_3%'] = hdr['AHI']
            sim_df.loc[loc, 'CAI_3%'] = hdr['CAI']
            sim_df.loc[loc, 'sleep_time'] = hdr['sleep_time']
            SS_time = np.sum(np.logical_and(data.patient_asleep,
                             data['self similarity']))/(3600 * hdr['newFs'])
            sim_df.loc[loc, 'T_SS_new'] = round(SS_time / hdr['sleep_time'], 2)
            osc_time = len(find_events(np.logical_and(
                data.patient_asleep, data.tagged > 0))) / hdr['sleep_time']
            sim_df.loc[loc, 'T_osc_new'] = round(osc_time, 2)

    return sorted_paths, sim_df

# exporting functions


def segment_and_export_recordings(all_paths, version, dataset, save_folder):
    import pdb
    pdb.set_trace()
    # create out paths
    save_folder = os.path.join(save_folder, f'{dataset}_{version}_V7/')
    os.makedirs(save_folder, exist_ok=True)
    out_paths = [save_folder + f'Study {p+1}.csv' for p in range(len(all_paths))]

    # Create a pool of workers
    num_workers = cpu_count()
    pool = Pool(num_workers)

    # Prepare arguments for parallel processing
    process_args = [(path, out_path) for path, out_path in zip(all_paths, out_paths)]

    # Process files in parallel and collect results
    for i, _ in enumerate(pool.starmap(segment_and_export_recording, process_args)):
        num = process_args[i][1]


def segment_and_export_recording(path, out_path):
    # skip already processed
    if os.path.exists(out_path):
        return

    # load recording
    data, hdr = load_sim_output(path)
    data = data.rename(columns={"apnea": "Apnea"})
    data = data.rename(columns={"flow_reductions": "Apnea_algo"})
    data = data.rename(columns={"sleep_stages": "Stage"})
    data = data.rename(columns={"abd": "ABD"})
    data = data.rename(columns={"spo2": "SpO2"})

    # create Ventilation* signal
    data = create_Ventilation_trace(data, hdr['newFs'], plot=False)

    # crate index columns
    factor = hdr['Fs'] // hdr['newFs']
    ind0 = np.arange(0, data.shape[0]) * factor
    ind1 = np.concatenate([ind0[1:], [ind0[-1]+factor]])
    data['ind0'] = ind0
    data['ind1'] = ind1

    # add breathing tag
    tags = ['ptaf']*len(data)
    if hdr['cpap_start'] > 0:
        tags[hdr['cpap_start']:] = ['cflow']*(len(data)-hdr['cpap_start'])
    data['breath_tag'] = tags

    # save data into .csv files
    export_cols = ['ind0', 'ind1', 'ABD', 'Ventilation_ABD', 'Eupnea_ABD', 'd_i_ABD', 'd_i_ABD_smooth', 'arousal_locs',
                   'Apnea', 'Apnea_algo', 'Stage', 'SpO2', 'breath_tag']
    export_data = pd.DataFrame([], columns=export_cols)
    for c in export_cols:
        export_data[c] = data[c]

    # add all segment indices to DF
    seg_dic = segment_data_based_on_nrem(data, hdr['newFs'])
    for key in seg_dic.keys():
        export_data.loc[:len(seg_dic[key])-1, key] = seg_dic[key]

    # add SS scores per segment to DF
    SS_seg_scores = compute_SS_score_per_segement(data, seg_dic)
    for key in SS_seg_scores.keys():
        export_data.loc[:len(SS_seg_scores[key])-1,
                        f'{key}_SS_score'] = SS_seg_scores[key]

    # add patient tag
    export_data['patient_tag'] = hdr['patient_tag']
    export_data['Fs'] = hdr['newFs']
    export_data['original_Fs'] = hdr['Fs']

    # save in .csv file
    export_data.to_csv(out_path, header=export_data.columns, index=None, mode='w+')


def segment_data_based_on_nrem(data, Fs, block_size=8):
    # segment based on sleep stages
    stages = data.Stage.values
    nrem = np.logical_and(stages > 0, stages < 4)
    rem = stages == 4

    # find all blocks within REM
    seg_dic = {}
    block = int(block_size*60*Fs)
    for SS, tag in zip([nrem, rem], ['nrem', 'rem']):
        starts = np.array([])
        for st, end in find_events(SS):
            blocks = (end-st) // block
            if blocks == 0:
                continue
            shift = ((end-st) - blocks*block) / 2
            starts = np.concatenate(
                [starts, np.arange(st+shift, end-block, block/2).tolist()])
        seg_dic[f'{tag}_starts'] = starts.astype(int)
        seg_dic[f'{tag}_ends'] = starts.astype(int) + block

    return seg_dic


def compute_SS_score_per_segement(data, seg_dic):
    SS_seg_scores = {}
    for SS in ['rem', 'nrem']:
        SS_list = []
        # run over all segments
        for st, end in zip(seg_dic[f'{SS}_starts'], seg_dic[f'{SS}_ends']):
            SS_score = np.round(data.loc[st:end, 'ss_conv_score'].mean(), 2)
            # apped SS score
            if not np.isnan(SS_score):
                SS_list.append(SS_score)
            # if all NaN ..
            else:
                # .. and multiple events are found ..
                if len(find_events(data.loc[st:end, 'Apnea'] > 0)) >= 4:
                    # set SS score to 0
                    SS_list.append(0)
                else:
                    # else, keep NaN
                    SS_list.append(SS_score)
        SS_seg_scores[SS] = SS_list

    return SS_seg_scores


if __name__ == '__main__':
    dataset = 'bdsp'
    date = '11_30_2023' if dataset == 'altitude' else '11_09_2023'
    multiprocess = False
    # 'SS_cases'  'high_CAI'  'HLG_OSA'  'REM_OSA  'NREM_OSA'  'Heart_Failure'  'Altitude'
    version = 'CPAP_failure'
    if dataset == 'redeker':
        assert version == 'Heart_Failure'

    # set input folder
    input_folder = 'SS paper files/'
    all_paths = glob.glob(input_folder + '*.hf5')

    # set output folder
    save_folder = r"C:\Users\thijs\KoGES Scoring Dropbox\Thijs Nassi\ThijsNassi\ThijsTemp\EM_input_csv_files/"

    if dataset == 'bdsp':
        # set all recording paths from dataset
        sim_info_path = 'csv_files/mgh_table1.csv'
        sim_split_path = 'csv_files/SS_split-nights_2150_updated.csv'
        sim_info_subset_path = sim_info_path.replace(
            '.csv', f' 100_{version}_cases.csv')
        sim_info_subset_path = sim_info_subset_path.replace('mgh_table1', 'bdsp_table1')
        if not os.path.exists(sim_info_subset_path):
            # filter sim_df based on sim_split_df
            sim_df = pd.read_csv(sim_info_path)
            sim_split_df = pd.read_csv(sim_split_path)
            sim_df = sim_df.loc[sim_df['HashID'].isin(sim_split_df['HashID'])]

            # filter AHI>10, h_sleep>4, psg_type=split
            sim_df = sim_df.iloc[np.where(sim_df['ahi_3%'] > 10)[0]]
            sim_df = sim_df.iloc[np.where(sim_df['h_sleep'] > 4)[0]]
            sim_df = sim_df.iloc[np.where(sim_df['psg_type'] == 'split')[
                0]].reset_index(drop=True)

            # create selection based on <version>
            patient_selection(sim_df, version, sim_info_subset_path)

            # sort input files
        sim_df = pd.read_csv(sim_info_subset_path)
        all_paths = sort_input_files(all_paths, sim_df, version)

    elif dataset == 'mgh':
        # set all recording paths from dataset
        sim_info_path = 'csv_files/mgh_table1.csv'
        sim_info_subset_path = sim_info_path.replace(
            '.csv', f' 100_{version}_cases.csv')
        if not os.path.exists(sim_info_subset_path):
            sim_df = pd.read_csv(sim_info_path)
            sim_df = remove_bad_signal_recordings(sim_df)

            # filter AHI>15, h_sleep>4, psg_type=diagnostic
            sim_df = sim_df.iloc[np.where(sim_df['ahi_3%'] > 15)[0]]
            sim_df = sim_df.iloc[np.where(sim_df['h_sleep'] > 4)[0]]
            sim_df = sim_df.iloc[np.where(sim_df['psg_type'] == 'diagnostic')[
                0]].reset_index(drop=True)

            # create selection based on <version>
            patient_selection(sim_df, version)

        # sort input files
        sim_df = pd.read_csv(sim_info_subset_path)
        all_paths = sort_input_files(all_paths, sim_df, version)

    elif dataset == 'redeker':
        info_path = 'csv_files/redeker_table1.xls'
        info_subset_path = info_path.replace('.xls', f' 100_{version}_cases.csv')
        if not os.path.exists(info_subset_path):
            info_df = pd.read_excel(info_path)

            # filter AHI>10, h_sleep>4, psg_type=diagnostic
            info_df = info_df.iloc[np.where(info_df['AHI'] > 10)[0]]
            info_df = info_df.iloc[np.where(info_df['TSPP'] > 4*60)[0]]
            info_df = info_df.iloc[np.where(info_df['EF'] < 50)[
                0]].reset_index(drop=True)
            # set SS group to N/A
            info_df.loc[:, 'SS group'] = 'N/A'
            # rename some columns
            map_change = {'AGE': 'age', 'GEND': 'Sex', 'ETHNIC': 'ethnicity',
                          'AHI': 'ahi_3%', 'AHI4': 'ahi', 'CSAIND': 'cai'}
            info_df = info_df.rename(columns=map_change)

            # add paths
            for i, ID in enumerate(info_df.ID):
                path = [p for p in all_paths if f'/{ID}' in p]
                if len(path) != 1:
                    info_df = info_df.drop(i)
                    continue
                info_df.loc[i, 'SS_path'] = path[0].split('/')[-1].split('.hf5')[0]

            # save selection
            info_df.reset_index(drop=True)
            info_df[:100].to_csv(
                info_subset_path, header=info_df.columns, index=None, mode='w+')
            import pdb
            pdb.set_trace()

        # sort input files
        sim_df = pd.read_csv(info_subset_path)
        all_paths = sort_input_files(all_paths, sim_df, version)

    elif dataset == 'altitude':
        sim_info_path = 'csv_files/rt_table1 100_Altitude_cases.csv'
        # extract all input paths
        all_paths = glob.glob(input_folder + '*.hf5')

        # sort input files
        all_paths, sim_df = sort_altitude_files(all_paths, date)

        # save selection
        sim_df.to_csv(sim_info_path, header=sim_df.columns, index=None, mode='w+')

        import pdb
        pdb.set_trace()

    # segment and export recordings
    segment_and_export_recordings(all_paths, version, dataset, save_folder)
