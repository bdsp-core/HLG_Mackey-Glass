% validate_matlab_vs_python.m
% Run the EM algorithm on the first NREM segment of Study_example.csv
% via Octave and save results for comparison with Python.
%
% Usage (from python/tests/):
%   octave --no-gui validate_matlab_vs_python.m

addpath('../../_original/matlab');

% Load study data using Octave-compatible CSV reading
csv_path = '../data/em_example/Study_example.csv';

% Read header to get column names
fid = fopen(csv_path, 'r');
header_line = fgetl(fid);
fclose(fid);
col_names = strsplit(header_line, ',');

% Read numeric data (skip header)
data = dlmread(csv_path, ',', 1, 0);
fprintf('Loaded: %d rows x %d columns\n', size(data, 1), size(data, 2));

% Map column names to indices
col_idx = containers.Map();
for i = 1:length(col_names)
    col_idx(strtrim(col_names{i})) = i;
end

% Extract required columns
Fs = data(1, col_idx('Fs'));
w = 5 * Fs;
L = 0.05;
gamma_init = 0.5;
tau_init = 15 * Fs;

% Get NREM segment boundaries
nrem_starts_col = data(:, col_idx('nrem_starts'));
nrem_ends_col = data(:, col_idx('nrem_ends'));
nrem_starts = nrem_starts_col(~isnan(nrem_starts_col));
nrem_ends = nrem_ends_col(~isnan(nrem_ends_col));
fprintf('NREM segments: %d\n', length(nrem_starts));

% First NREM segment
starting = nrem_starts(1);
ending = nrem_ends(1) - 1;
if starting == 0
    starting = 1;
    ending = ending + 1;
end
ending = min(ending, size(data, 1));
fprintf('First segment: [%d, %d] = %d samples\n', starting, ending, ending - starting + 1);

% Build a struct that mimics a MATLAB table for fcn_em_algorithm_real_data
seg_data = data(starting:ending, :);

% Create a simple struct with the needed fields
T_seg = struct();
T_seg.Ventilation_ABD = seg_data(:, col_idx('Ventilation_ABD'));
T_seg.d_i_ABD = seg_data(:, col_idx('d_i_ABD'));
T_seg.d_i_ABD_smooth = seg_data(:, col_idx('d_i_ABD_smooth'));
T_seg.arousal_locs = seg_data(:, col_idx('arousal_locs'));

K = size(seg_data, 1);
fprintf('Segment length: %d\n', K);

% Run EM directly (bypass fcn_em_algorithm_real_data which expects a table)
rng('default');
V_max = 1;
s = 10^-8;
V_o = T_seg.Ventilation_ABD;
u = T_seg.d_i_ABD;  % non-smooth version
u_min = min(u);
dit = T_seg.arousal_locs;
Iter = 5;

tic;
[upAlpha, upgamma, uptau, h] = fcnEMAlgorithm_TN_v5(K, L, V_max, gamma_init, tau_init, s, V_o, u, dit, Iter, w);
elapsed = toc;

% Compute loop gain
G_est = upgamma(end);
D_est = uptau(end) / Fs;
LG_est = fcnGetLoopGain2(L, G_est, u_min);

% Print results
fprintf('\n=== MATLAB/Octave EM Results ===\n');
fprintf('gamma  = %.4f\n', G_est);
fprintf('tau    = %.1f sec\n', D_est);
fprintf('alpha  = %.4f\n', upAlpha(end));
fprintf('LG     = %.4f\n', LG_est);
fprintf('u_min  = %.4f\n', u_min);
fprintf('arousals = %d\n', length(h));
fprintf('time   = %.1f sec\n', elapsed);

fprintf('\nPer-iteration convergence:\n');
for i = 1:Iter
    fprintf('  iter %d: gamma=%.4f, tau=%.1fs, alpha=%.4f\n', i, upgamma(i), uptau(i)/Fs, upAlpha(i));
end

% Save results
fid = fopen('matlab_em_results.csv', 'w');
fprintf(fid, 'gamma,tau_sec,alpha,LG,u_min,n_arousals\n');
fprintf(fid, '%.6f,%.1f,%.4f,%.6f,%.6f,%d\n', G_est, D_est, upAlpha(end), LG_est, u_min, length(h));
fclose(fid);
fprintf('\nSaved: matlab_em_results.csv\n');
