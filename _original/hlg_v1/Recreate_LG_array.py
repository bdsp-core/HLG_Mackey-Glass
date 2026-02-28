import numpy as np

def create_total_LG_array(data):
    total_array = np.zeros(len(data))

    for stage in ['nrem', 'rem']:
        # set run variables
        starts = data[f'{stage}_starts'].dropna().values.astype(int)
        ends = data[f'{stage}_ends'].dropna().values.astype(int)
        for i, (start, end) in enumerate(zip(starts, ends)):
            LG = data.loc[i, f'LG_{stage}_corrected']
            total_array[start: end] = LG

    # correct for LG==0?

    # mask locations where no sleep
    total_array[np.isnan(data.Stage)] = np.nan
    total_array[data.Stage==5] = np.nan

    return total_array
    