import glob, h5py
import numpy as np
import matplotlib.pyplot as plt




# Compute histogram
def compute_histogram(data, hdr, bar_folder):
    # set patient sleep array
    data['patient_asleep'] = np.logical_and(data.Stage.values>0, data.Stage.values<5)

    # segment data into 5 minute windows
    epoch_size = int(round(8*60*hdr['Fs']))
    epoch_inds = np.arange(0, len(data)-epoch_size+1, epoch_size)
    seg_ids = list(map(lambda x:np.arange(x, x+epoch_size), epoch_inds))

    # run over all segments, and save mean SS score
    score_present = data.total_LG.dropna().index.values
    bin_means = np.zeros(len(seg_ids))
    for i, seg_id in enumerate(seg_ids):
        # skip if patient is asleep for < 25% in this segment
        if data.loc[seg_id, 'patient_asleep'].sum() < 0.20*epoch_size: 
            bin_means[i] = -1
            continue
        # if no ss score is found, keep zero
        if not any([True if s in seg_id else False for s in score_present]): continue
        # compute average SS score within window
        bin_means[i] = data.loc[seg_id, 'total_LG'].dropna().mean()
     
    ### create histogram ###
    bars = histogram_bins_to_bars(bin_means)

    # Save bars
    save_histogram_bars(np.array(bars), hdr['Study_num'], bar_folder)

    return bars


# Convert bins to bars
def histogram_bins_to_bars(bins):
    ### create histogram ###
    bins = bins[bins>=0]
    step = 0.1
    steps = np.arange(0, 1.1, step)
    bars = []    
    for block in steps[:-1]:
        # normalize bins
        percentage = sum(np.logical_and(np.array(bins)>=block, np.array(bins)<block+step+0.0001)) / len(bins) * 100
        bars.append(percentage)
    
    return bars


# Save bars
def save_histogram_bars(LG_bars, ID, bar_output_folder):
    # save data in .hf5 file
    out_file = bar_output_folder + ID + '.hf5'
    with h5py.File(out_file, 'w') as f:
        dtypef = 'float32'
        dXy = f.create_dataset('LG_bars', shape=LG_bars.shape, dtype=dtypef, compression="gzip")
        dXy[:] = LG_bars.astype(float)


# Load bars
def load_histogram_bars(bar_output_folder):
    # extract bars from success/failure groups
    bars_success = []
    bars_failure = []
    for version in ['CPAP_success', 'CPAP_failure']:
        bar_folder = f'{bar_output_folder}{version}/'
        bar_files = glob.glob(bar_folder + '*.hf5')
        bars = []
        for i in range(1, len(bar_files)+1):
            bar_file = [f for f in bar_files if f'Study {i}.hf5' in f][0]
            with h5py.File(bar_file, 'r') as f:
                bars.append(f['LG_bars'][:])
        if 'success' in version.lower():
            bars_success = bars
        else:
            bars_failure = bars

    return bars_success, bars_failure
    

# Predict CPAP success from bars
def predict_CPAP_SUCCESS_from_bars(df, bars, bars_success, bars_failure):
    # set comparison bars
    bars_s = np.mean(bars_success, axis=0)
    bars_f = np.mean(bars_failure, axis=0)

    # compute error scores
    for i in range(len(df)):
        print(f'Comparing histogram bars: #{i}/{len(df)}', end='\r')
        # total distance
        _, s_score = custom_error(bars[i, :], ref=bars_s)
        _, f_score = custom_error(bars[i, :], ref=bars_f)
        df.loc[df.index[i], 'LG Bar'] = s_score - f_score

    return df


# Compute error
def custom_error(bar, ref):
    # mean error
    error = []
    for i, (b, r) in enumerate(zip(bar, ref)):
        error.append(abs(b-r))
    mean_error = np.mean(error)
    total_error = sum(error)

    return mean_error, total_error



def total_histogram_plot(bars1, bars2):
    # mean histogram
    _, ax = plt.subplots(1, 1, figsize=(8,6))
    mean1, mean2 = np.mean(bars1, 0), np.mean(bars2, 0)
    ranges = np.arange(0.05,1,0.1)
    ax.bar([r-0.023 for r in ranges], mean1, color='g', edgecolor='k', width=0.0455, label=f'success (N={len(bars1)})')
    ax.bar([r+0.023 for r in ranges], mean2, color='r', edgecolor='k', width=0.0455, label=f'failure (N={len(bars2)})')

    # set layout
    ax.set_ylabel('%', fontsize=10, fontweight='bold')
    ax.set_xlabel('LG score', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 101)

    plt.title('Mean Histogram: CPAP success vs failure\nAvg. LG estimation within 8 min segments.', fontweight='bold')
    plt.legend()
