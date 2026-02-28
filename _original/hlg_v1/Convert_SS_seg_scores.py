def convert_ss_seg_scores_into_arrays(data):
    # run over all segments from (N)REM
    for stage in ['nrem', 'rem']:
        starts = data[f'{stage}_starts'].dropna().values.astype(int)
        ends = data[f'{stage}_ends'].dropna().values.astype(int)
        seg_scores = data[f'{stage}_SS_score'].values
        # convert scores into array
        for st, end, score in zip(starts, ends, seg_scores):
            data.loc[st-1:end-2, 'SS_score'] = score
        # remove singe scores
        data.drop(columns=[f'{stage}_SS_score'])

    return data
    