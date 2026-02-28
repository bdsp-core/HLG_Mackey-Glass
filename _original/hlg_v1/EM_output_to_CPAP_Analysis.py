import random, sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegressionCV
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, confusion_matrix,
                                precision_recall_curve, roc_curve)

from EM_output_histograms import load_histogram_bars, predict_CPAP_SUCCESS_from_bars



# Logistic regression
def compute_logistic_regression(x, y, tag, axs=[], CV_folds=5):
    # divide data into CV folds
    xs, ys, inds = set_cross_validation_folds(x, y, folds=CV_folds)

    # run over all CV folds
    for i in range(1, CV_folds+1):
        # do logistic regression on training data
        clf = LogisticRegressionCV(Cs=10, random_state=0, cv=3, class_weight='balanced', scoring='roc_auc').fit(xs[f'tr_fold_{i}'], ys[f'tr_fold_{i}'])

        # save x, y, predictions, and probabilities
        x_te, y_te = xs[f'te_fold_{i}'], ys[f'te_fold_{i}']
        x = x_te if i==1 else np.concatenate([x, x_te])
        y = y_te if i==1 else np.concatenate([y, y_te])
        pred = clf.predict(x_te) if i==1 else np.concatenate([pred, clf.predict(x_te)])
        prob = clf.predict_proba(x_te)[:,1] if i==1 else np.concatenate([prob, clf.predict_proba(x_te)[:,1]]) 

    # print performance
    print(f' {tag} CMT:\n{confusion_matrix(y, pred, labels=[0, 1])}')
    print(f' {tag} Acc: {np.round(sum(y==pred)/len(y), 2)}')

    # do bootstrapping for confidence intervals
    mean_roc, CI_roc = do_bootstrapping(y, prob, my_auc_roc)
    mean_pr, CI_pr = do_bootstrapping(y, prob, my_auc_pr)

    # set line color
    line_color = set_line_color(tag)

    # plot ROC curve
    fpr, tpr, _ = roc_curve(y, prob)
    area = '%.2f [%.2f-%.2f]'%(mean_roc, CI_roc[0], CI_roc[1])
    axs[0].plot(fpr, tpr, line_color, label=f'{tag}${area}')

    # plot PR curve
    precision, recall, _ = precision_recall_curve(y, prob)
    area = '%.2f [%.2f-%.2f]'%(mean_pr, CI_pr[0], CI_pr[1])
    axs[1].plot(recall, precision, line_color, label=f'{tag}${area}')

    return prob, y

def set_cross_validation_folds(x, y, folds):
    # shuffle array
    inds = np.array(sklearn.utils.shuffle(range(len(y)), random_state=0))

    # create CV folds
    xs, ys = {}, {}
    split = len(y)//folds
    for i in range(folds):
        if i < folds-1:
            test_inds = inds[split*i:split*(i+1)]
        else: 
            test_inds = inds[split*i:]
        xs[f'tr_fold_{i+1}'] = x[[j for j in range(len(y)) if j not in test_inds]]
        ys[f'tr_fold_{i+1}'] = y[[j for j in range(len(y)) if j not in test_inds]]
        xs[f'te_fold_{i+1}'] = x[test_inds]
        ys[f'te_fold_{i+1}'] = y[test_inds]

    return xs, ys, inds

def do_bootstrapping(y, proba, my_stat, n_bootstraps=100, percentage='95%'):
    # init empty arrays
    n_options = y.shape[0]
    index_original = np.arange(n_options).astype(int)
    metrics = []

    # run over all bootstraps
    for n in range(n_bootstraps):
        print('bootstrap #%s / %s'%(n+1, n_bootstraps), end='\r')

        # take random samples 
        index_bootstrap = random.choices(index_original, k=n_options) 

        # compute performance metrics
        true = y[index_bootstrap]
        prob = proba[index_bootstrap]

        # compute metric
        metric = my_stat(true, prob)
        metrics.append(metric)

    # compute mean +- CI intervals
    perc = int(percentage[:-1]) / 100
    metrics = np.array(metrics)
    mean = np.round(np.mean(metrics), 2)
    lower_bound = np.round(np.quantile(metrics, 1-perc, axis=0), 2)
    upper_bound = np.round(np.quantile(metrics, perc, axis=0), 2)
    
    return mean, [lower_bound, upper_bound]


# ROC PR curves 
def my_auc_roc(y, p):
    # compute ROC AUC
    fpr, tpr, _ = roc_curve(y, p)
    AUC = auc(fpr, tpr)

    return AUC

def my_auc_pr(y, p):
    # compute ROC AUC
    precision, recall, _ = precision_recall_curve(y, p)
    AUC = auc(recall, precision)

    return AUC

def set_line_color(tag):
    # single tags
    if tag == 'LG':
        line_color = 'b'
    elif tag == 'LG range':
        line_color = 'b'
    elif tag == 'LG bar':
        line_color = 'm'
    elif tag == 'CAI':
        line_color = 'k'
    elif tag == 'AHI':
        line_color = 'k--'
    elif tag == 'SS':
        line_color = 'y'
    elif tag == 'Combined':
        line_color = 'r'

    return line_color

def set_AUC_curve_layout(axs, subtitle):
    # set layout for both curves
    for n in range(2):
        ax = axs[n]
        # set layout
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        # setup legend
        lines, labels = ax.get_legend_handles_labels()
        tags = [l.split('$')[0] for l in labels] 
        vals = [l.split('$')[1] for l in labels]
        # features
        handles = []
        for line, tag in zip(lines, tags):
            handle = Line2D([0],[0], label=tag, c=line.get_c(), linestyle=line.get_linestyle())
            handles.append(handle)
        # auc values
        for val in vals:
            handle = Line2D([0],[0], label=val, c='none')
            handles.append(handle)

        title = ''.join([r"$\bf{" + x + "}$" + ' ' for x in subtitle.split(' ')]) + '[95% CI]'
        ax.legend(handles=handles, loc=4, ncol=2, fontsize=9, title=title, title_fontsize=10, alignment='left', 
                    facecolor='grey', framealpha=0.5, edgecolor='k', columnspacing=0)  
        
        # axes
        if n == 0:
            if 'Individual' in subtitle:
                ax.set_title(f'ROC', fontsize=11, weight='bold')
            ax.set_xlabel('1 - specificity', weight='bold', fontsize=10)
            ax.set_ylabel('sensitivity', weight='bold', fontsize=10)
            ax.plot([-0.1, 1.1], [-0.1, 1.1], 'grey', lw=1)
        if n == 1:
            # ax.set_yticks([])
            if 'Individual' in subtitle:
                ax.set_title(f'PR', fontsize=11, weight='bold')
            ax.set_xlabel('sensitivity', weight='bold', fontsize=10)
            ax.set_ylabel('precision', weight='bold', fontsize=10)
            ax.plot([-0.1, 1.1], [1.1, -0.1], 'grey', lw=1)


# Calibration curve
def compute_calibration_curve(prob, y, tag, ax):    
    # compute calibration curve
    yy, xx = calibration_curve(y, prob)

    # set line color
    line_color = set_line_color(tag)

    # plot calibration curve
    marker = 'o'

    ax.plot(xx, yy, line_color, marker=marker, label=tag, markersize=6)

def set_calibration_curve_layout(axs: list[plt.Axes]):
    for ax in axs:
        # add reference line
        ax.plot([-0.1, 1.1], [-0.1, 1.1], 'grey', lw=1)
        
        # set axes
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_ylabel('Ratio of patients that fail CPAP', weight='bold', fontsize=10)

        # add legend and title
        ax.set_title('Calibration', fontsize=11, weight='bold')
        title = ''.join([r"$\bf{" + x + "}$" + ' ' for x in 'Features'.split(' ')])

        ax.legend(loc=4, ncol=1, fontsize=9, title=title, title_fontsize=10, alignment='left', facecolor='grey', 
                        framealpha=0.5, edgecolor='k')  
        ax.set_xlabel(f'Predicted CPAP failure risk', weight='bold', fontsize=10)




if __name__ == '__main__':
    dataset = 'bdsp' # bdsp, mgh, redeker, rt
    versions = ['CPAP_success', 'CPAP_failure'] # CPAP_success, CPAP_failure, high_CAI, NREM_OSA, REM_OSA, SS_OSA, SS_range, Heart_Failure, Altitude
    Ut_smooth = 'non-smooth'  # smooth   non-smooth
    base_folder = f'./interm_Results/{Ut_smooth}/group_analysis/'
    

    # extract intermediate results from sheet
    success_results = pd.read_csv(f'{base_folder}BDSP_CPAP_success/all_segments.csv')
    failure_results = pd.read_csv(f'{base_folder}BDSP_CPAP_failure/all_segments.csv')
    LG_data = np.concatenate([success_results['LG_data'].values, failure_results['LG_data'].values])
    valid_data = np.concatenate([success_results['valid_data'].values, failure_results['valid_data'].values])
    ID_data = np.concatenate([success_results['ID_data'].values*-1, failure_results['ID_data'].values])

    # info file
    success_info_df = pd.read_csv(f'csv_files/bdsp_table1 200_CPAP_success_cases.csv')
    failure_info_df = pd.read_csv(f'csv_files/bdsp_table1 200_CPAP_failure_cases.csv')
    maxi = len(success_info_df)
    success_info_df['ID'] = range(-1, -maxi-1, -1)
    failure_info_df['ID'] = range(1, maxi+1)
    cols = ['ID', 'CAI1_3%', 'T_SS1', 'AHI1_3%']
    info_df = pd.concat([success_info_df[cols], failure_info_df[cols]])

    # compute mean across bars
    bar_folder = f'./bars/BDSP_'
    bars_success, bars_failure = load_histogram_bars(bar_folder)
    all_bars = np.array(bars_success + bars_failure)
    # total_histogram_plot(bars_success, bars_failure)
    info_df = predict_CPAP_SUCCESS_from_bars(info_df, all_bars, bars_success, bars_failure)

    # Set x and y
    x = np.expand_dims(LG_data[valid_data.astype(bool)], axis=1)
    y = np.concatenate([np.zeros(len(success_results)), np.ones(len(failure_results))]).astype(int)[valid_data.astype(bool)]       

    # Compute median across ID
    IDs = ID_data[valid_data.astype(bool)]
    unique_IDs = np.unique(IDs)
    LG_median = np.zeros(len(unique_IDs))
    LG_mean = np.zeros(len(unique_IDs))
    LG_25 = np.zeros(len(unique_IDs))
    LG_75 = np.zeros(len(unique_IDs))
    cai = np.zeros(len(unique_IDs))
    ahi = np.zeros(len(unique_IDs))
    ss = np.zeros(len(unique_IDs))
    LG_bar = np.zeros(len(unique_IDs))
    yy = np.zeros(len(unique_IDs))
    for i, ID in enumerate(unique_IDs):
        inds = IDs == ID
        LG_median[i] = np.median(LG_data[valid_data.astype(bool)][inds])
        LG_mean[i] = np.mean(LG_data[valid_data.astype(bool)][inds])
        LG_25[i] = np.percentile(LG_data[valid_data.astype(bool)][inds], 25)
        LG_75[i] = np.percentile(LG_data[valid_data.astype(bool)][inds], 75)
        yy[i] = np.median(y[inds])
        cai[i] = info_df.loc[info_df['ID'] == ID, 'CAI1_3%'].values[0]
        ss[i] = info_df.loc[info_df['ID'] == ID, 'T_SS1'].values[0]
        ahi[i] = info_df.loc[info_df['ID'] == ID, 'AHI1_3%'].values[0]
        LG_bar[i] = info_df.loc[info_df['ID'] == ID, 'LG Bar'].values[0]

    # Set x and y
    LG_median = np.expand_dims(LG_median, axis=1)
    LG_mean = np.expand_dims(LG_mean, axis=1)
    LG_25 = np.expand_dims(LG_25, axis=1)
    LG_75 = np.expand_dims(LG_75, axis=1)
    LG_range = np.concatenate([LG_25, LG_median, LG_75], axis=1)
    cai = np.expand_dims(cai, axis=1)
    ahi = np.expand_dims(ahi, axis=1)
    ss = np.expand_dims(ss, axis=1)
    LG_bar = np.expand_dims(LG_bar, axis=1)
    combined_x = np.concatenate([LG_range, ss], axis=1)


    # Compute logistic regression
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    prob_cai, y_cai = compute_logistic_regression(cai, yy, 'CAI', axs=axs, CV_folds=5)
    # prob_ahi, y_ahi = compute_logistic_regression(ahi, yy, 'AHI', axs=axs, CV_folds=5)
    prob_LG_range, y_LG_range = compute_logistic_regression(LG_range, yy, 'LG', axs=axs, CV_folds=5)
    # prob_ss, y_ss = compute_logistic_regression(ss, yy, 'SS', axs=axs, CV_folds=5)
    # prob_LG_bar, y_LG_bar= compute_logistic_regression(LG_bar, yy, 'LG bar', axs=axs, CV_folds=5)
    prob_combined, y_combined = compute_logistic_regression(combined_x, yy, 'Combined', axs=axs, CV_folds=5)
    set_AUC_curve_layout(axs, 'Features              AUC')
    
    # Compute calibration curve
    compute_calibration_curve(prob_cai, y_cai, 'CAI', axs[2])
    # compute_calibration_curve(prob_ahi, y_ahi, 'AHI', axs[2])
    compute_calibration_curve(prob_LG_range, y_LG_range, 'LG', axs[2])
    # compute_calibration_curve(prob_ss, y_ss, 'SS', axs[2])
    # compute_calibration_curve(prob_LG_bar, y_LG_bar, 'LG bar', axs[2])
    compute_calibration_curve(prob_combined, y_combined, 'Combined', axs[2])
    set_calibration_curve_layout([axs[2]])

    plt.tight_layout()
    plt.show()
    import pdb; pdb.set_trace()