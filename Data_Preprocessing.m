%% Motor Data Preprocessing Framework - Core Parameter Configuration and File Processing
clear; clc; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Core Parameter Configuration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Root folder path
root_path = 'G:\IEEE_data';
save_path = 'G:\IEEE_data\FD_data';
% Mapping relationship between folders and labels
folder_list = {'A0.01', 'A1', 'B0.01', 'B1', 'C0.01', 'C1', 'Health'};
label_list  = {'AHF', 'ALF', 'BHF', 'BLF', 'CHF', 'CLF', 'Health'};
% Filter condition configuration (XHF: 0.01Ω+45.63%; XLF:1Ω+12.30%; No filter for Health)
filter_rules = containers.Map(...
    {'AHF', 'BHF', 'CHF', 'ALF', 'BLF', 'CLF', 'Health'}, ...
    {{'short_res', 0.01, 'turn_percent', 45.63}, ...  % Filter conditions for AHF/BHF/CHF
    {'short_res', 0.01, 'turn_percent', 45.63}, ...
    {'short_res', 0.01, 'turn_percent', 45.63}, ...
    {'short_res', 1, 'turn_percent', 12.30}, ...     % Filter conditions for ALF/BLF/CLF
    {'short_res', 1, 'turn_percent', 12.30}, ...
    {'short_res', 1, 'turn_percent', 12.30}, ...
    {}});  % No filter conditions for Health

% Fixed preprocessing parameters (global)
fs_original = 1e6;                  % Original sampling frequency
samples_per_cycle_target = 512;     % Target number of sampling points per fundamental cycle
window_cycles = 2;                  % Window length (number of fundamental cycles) → 1024 points
slide_step_ratio = 1/8;             % Sliding step (ratio of fundamental cycle)
start_time = 1;                     % Sliding window start time (after 1 second)
cutoff_freq = 1e3;                  % Filter cutoff frequency
idx = 0;                            % Three-phase alignment mode, 0: original waveform, 1: Phase A zero-crossing, 2: Three-phase zero-crossing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Traverse Folders and Files, Execute Filtering, Parsing, Preprocessing and Saving
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for folder_idx = 1:length(folder_list)
    % Get current folder name and corresponding label
    current_folder = folder_list{folder_idx};
    current_label  = label_list{folder_idx};
    % Concatenate the full path of the current folder
    folder_fullpath = fullfile(root_path, current_folder);
    % Check if the folder exists
    if ~exist(folder_fullpath, 'dir')
        warning('Folder does not exist: %s, skipping this folder', folder_fullpath);
        continue;
    end

    % Get all .mat files in the current folder
    mat_files = dir(fullfile(folder_fullpath, '*.MAT'));
    if isempty(mat_files)
        disp(['No .mat files found in folder ', current_folder, ', skipping']);
        continue;
    end

    % Traverse each .mat file
    for file_idx = 1:length(mat_files)
        % Get the full file path and file name (without extension)
        file_fullpath = fullfile(folder_fullpath, mat_files(file_idx).name);
        [~, filename, ~] = fileparts(mat_files(file_idx).name);

        % Parse filename parameters: [Frequency]-[Torque]-[Short-circuit Resistance]-[Fault Phase Sequence]-[Short-circuit Turn Percentage]-[Control Mode]
        try
            % Split the filename by '-' to extract parameters
            params = split(filename, '-');
            freq       = str2double(params{1});    % Frequency (Hz)
            torque     = str2double(params{2});    % Torque (% of rated torque)
            short_res  = str2double(params{3});    % Short-circuit resistance (Ω)
            fault_phase= params{4};                % Fault phase sequence
            turn_percent= str2double(params{5});   % Short-circuit turn percentage (%)
            control    = params{6};                % Control mode

            % Skip files with failed parameter parsing
            if any(isnan([freq, torque, short_res, turn_percent]))
                warning('Failed to parse parameters of file %s, skipping', filename);
                continue;
            end
        catch
            warning('File %s format does not comply with rules, skipping', filename);
            continue;
        end

        % Filter files that meet the conditions according to labels
        filter_cond = filter_rules(current_label);
        is_pass = true;
        if ~isempty(filter_cond)
            % Check if short-circuit resistance and turn percentage meet filter conditions
            target_res = filter_cond{2};
            target_turn = filter_cond{4};
            if ~(abs(short_res - target_res) < 1e-6 && abs(turn_percent - target_turn) < 1e-6)
                is_pass = false;
            end
        end
        % Skip if filter conditions are not met
        if ~is_pass
            continue;
        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 3. Sliding Window Processing
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Read original data
        disp(['File is being preprocessed:', file_fullpath]);
        load_data = load(file_fullpath);

        % Call sliding window function, return all sample data + corresponding original indices
        [samples_data, samples_indices] = sliding_window_preprocess(...
            load_data, freq, fs_original, samples_per_cycle_target, ...
            start_time, window_cycles, slide_step_ratio, cutoff_freq);

        % Skip if no valid samples
        if isempty(samples_data)
            warning('No valid sliding window samples found in file %s, skipping', filename);
            continue;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 4. Save Each Sliding Window Sample According to Rules
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        save_folder_name = sprintf('%s_Hz_%.2f_Te_%.2f_%d', current_label, freq, torque, idx);
        save_fullpath = fullfile(save_path, save_folder_name);
        % Create save folder (create if not exists)
        if ~exist(save_fullpath, 'dir')
            mkdir(save_fullpath);
        end

        % Traverse each sample and save with "X_Y_Z.mat" naming convention
        for sample_idx = 1:length(samples_data)
            sample_data = samples_data{sample_idx};   % (1024,6) matrix
            start_idx = samples_indices{sample_idx}(1);  % Original data index
            end_idx   = samples_indices{sample_idx}(2);  % Original data index


            % Construct sample filename: X_Y_Z.mat
            sample_filename = sprintf('%d_%d_%d.mat', sample_idx, start_idx, end_idx);
            sample_save_path = fullfile(save_fullpath, sample_filename);

            % Save sample data (variable name is uniformly 'sample_data' for subsequent calls)
            save(sample_save_path, 'sample_data');
            disp(['Sliding window sample saved: ', sample_save_path]);
        end
    end
end

disp('Processing of all eligible files completed!');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sliding Window Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [samples_data, samples_indices] = sliding_window_preprocess(raw_data, f0, fs_original, samples_per_cycle_target, start_time, window_cycles, slide_step_ratio, cutoff_freq)
% Input:
%   raw_data: Original .mat data (structure)
%   f0: Fundamental frequency (Hz, parsed from filename)
%   Other parameters: Sampling frequency/Points per cycle/Start time/Window cycle count/Sliding step ratio/Filter cutoff frequency
% Output:
%   samples_data: Cell array, each element is a (1024,6) sample (columns: IA,IB,IC,UA,UB,UC)
%   samples_indices: Cell array, each element is [original start index, original end index]

% Initialize output
samples_data = {};
samples_indices = {};

% Channel mapping (adjust column order: first current IA/IB/IC, then voltage UA/UB/UC)
channels = {
    {'IA', 'CH02'},...  % Column 1: IA
    {'IB', 'CH04'},...  % Column 2: IB
    {'IC', 'CH06'},...  % Column 3: IC
    {'UA', 'CH01'},...  % Column 4: UA
    {'UB', 'CH03'},...  % Column 5: UB
    {'UC', 'CH05'}   % Column 6: UC
    };

% Step 1: Get the length of original data for any channel
ref_ch_var = channels{1}{2};
if ~isfield(raw_data, ref_ch_var)
    warning('Reference channel %s does not exist, exiting preprocessing', ref_ch_var);
    return;
end
ref_data = raw_data.(ref_ch_var);
total_samples = length(ref_data);

% Step 2: Calculate key parameters of sliding window (original data dimension)
T0 = 1/f0;                                  % Fundamental cycle (s)
samples_per_cycle_original = floor(fs_original * T0);  % Original points per cycle
window_samples_original = window_cycles * samples_per_cycle_original;  % Original window points
slide_step_samples_original = floor(slide_step_ratio * samples_per_cycle_original);  % Original sliding step points
start_idx_original = floor(start_time * fs_original) + 1;  % Original start index corresponding to 1 second

% Step 3: Sliding window loop (until end index exceeds data range)
current_start_idx = start_idx_original;
sample_count = 0;

while true
    % Calculate end index of current window
    current_end_idx = current_start_idx + window_samples_original - 1;

    % Termination condition: window end index exceeds original data length
    if current_end_idx > total_samples
        break;
    end

    %% Step 4: Preprocess 6 channels of the current window
    sample_matrix = [];
    valid_channel = true;

    for ch_idx = 1:length(channels)
        ch_name = channels{ch_idx}{1};
        ch_var = channels{ch_idx}{2};

        % Check if channel field exists
        if ~isfield(raw_data, ch_var)
            warning('Channel %s does not exist, skipping current window', ch_var);
            valid_channel = false;
            break;
        end

        % Extract original channel data of current window
        ch_data_original = raw_data.(ch_var);
        ch_window_data = ch_data_original(current_start_idx:current_end_idx);

        % Perform filtering + downsampling on current window data (call preprocessing subfunction)
        try
            [ch_processed, ~] = preprocess_signal_window(...
                ch_window_data, f0, fs_original, cutoff_freq, samples_per_cycle_target, window_cycles);
        catch
            warning('Failed to preprocess channel %s, skipping current window', ch_var);
            valid_channel = false;
            break;
        end

        % Concatenate to sample matrix (ensure it is a column vector)
        sample_matrix = [sample_matrix, ch_processed(:)];
    end

    % Verify sample matrix dimensions (must be 1024x6)
    if valid_channel && size(sample_matrix,1)==1024 && size(sample_matrix,2)==6

        % Amplitude normalization and DC component removal
        u_abc = [sample_matrix(:,4),...  % u_a
            sample_matrix(:,5),...  % u_b
            sample_matrix(:,6)];    % u_c
        i_abc = [sample_matrix(:,1),...  % i_a
            sample_matrix(:,2),...  % i_b
            sample_matrix(:,3)];    % i_c

        % Calculate global maximum value of three-phase voltage/current
        U_abc_max = max(abs(u_abc), [], 'all');  % Global maximum value of three-phase voltage (scalar)
        I_abc_max = max(abs(i_abc), [], 'all');  % Global maximum value of three-phase current (scalar)

        % Amplitude normalization
        u_abc_nor = u_abc / U_abc_max;  % Three-phase voltage normalization
        i_abc_nor = i_abc / I_abc_max;  % Three-phase current normalization

        % Remove DC component
        sample_matrix(:,1:3) = i_abc_nor - mean(i_abc_nor);
        sample_matrix(:,4:6) = u_abc_nor - mean(u_abc_nor);

        sample_count = sample_count + 1;
        samples_data{sample_count} = sample_matrix;
        samples_indices{sample_count} = [current_start_idx, current_end_idx];
    else
        warning('Preprocessing failed for sliding window %d-%d, its size does not meet 1024*6', current_start_idx, current_end_idx);
    end

    % Sliding window: increase start index by step size
    current_start_idx = current_start_idx + slide_step_samples_original;

    disp("The " + sample_count + "th sample has been processed");
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Core Function 2: Single Window Signal Preprocessing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_processed, t_processed] = preprocess_signal_window(ch_window_data, f0, fs_original, cutoff_freq, samples_per_cycle_target, window_cycles)
% Input: ch_window_data - Original window data of a single channel (column vector)
% Output: x_processed - Preprocessed window data (1024 points, corresponding to 2 fundamental cycles)

%% ==================== 1. Basic Parameter Calculation ====================
fs_target = f0 * samples_per_cycle_target;    % Target sampling rate after downsampling
T0 = 1/f0;                                    % Fundamental cycle
samples_per_cycle = floor(fs_original * T0);  % Original sampling points per cycle
N_cycle = 2;                                  % Extract data of two integer cycles

%% ==================== 2. Adaptive Anti-Aliasing Filter Design ====================
% Calculate target Nyquist frequency and normalized cutoff frequency of filter
nyquist_target = fs_target / 2;
Wn = cutoff_freq / (fs_original / 2);

% Automatically determine optimal filter order
transition_width = (nyquist_target - cutoff_freq) / (fs_original/2);
stopband_attenuation = 80;
Rp = 0.05;
min_order = ceil(log10((10^(stopband_attenuation/20)-1) / ...
    (10^(Rp/20)-1)) / log10(1/transition_width));
min_order = min_order * 2;  % Ensure even order

% Limit maximum order
max_order = 12;
filter_order = min(min_order, max_order);
filter_order = filter_order + mod(filter_order, 2);  % Ensure even order

% Design elliptic filter (second-order section form, better numerical stability)
[z, p, k] = ellip(filter_order/2, Rp, stopband_attenuation, Wn, 'low');
[sos, g] = zp2sos(z, p, k);

%% ==================== 3. Filtering Processing: Signal Extension + Controllable Edge Truncation + Filtering ====================
% Calculate effective transient length of filter
[imp_resp, ~] = impz(sos, [], fs_original);
cum_resp = cumsum(abs(imp_resp));
cum_resp_normalized = cum_resp / sum(abs(imp_resp));
transient_len = 1*find(cum_resp_normalized > 0.999, 1);
if isempty(transient_len)
    transient_len = filter_order * 2;
end

% Signal extension (mirror extension to avoid DC offset)
x_padded = [flip(ch_window_data(1:transient_len));
    ch_window_data;
    flip(ch_window_data(end-transient_len+1:end))];

% Zero-phase filtering (filtfilt avoids phase distortion)
x_filtered_padded = filtfilt(sos, g, double(x_padded));

% Truncate extended part, retain original length
x_filtered = x_filtered_padded(transient_len+1:end-transient_len);
x_filtered_clean = x_filtered;

%% ==================== 4. Downsampling to Target Sampling Rate ====================
% Calculate numerator and denominator of resampling (simplest fraction form)
g = gcd(fs_target, fs_original);
p_resamp = fs_target / g;
q_resamp = fs_original / g;
resample_filter_len = 10 * max(p_resamp, q_resamp);

% Calculate total transient length
resample_transient_base = floor(resample_filter_len / 2);
transient_len_scaled = floor(transient_len * fs_target / fs_original);
total_transient_len = transient_len_scaled + resample_transient_base;

% Dynamically supplement extension length (ensure sufficient valid signal)
target_samples = N_cycle * samples_per_cycle_target;
min_padded_len = target_samples + 2 * total_transient_len;
current_clean_len = length(x_filtered_clean);
if current_clean_len < min_padded_len - 2*total_transient_len
    extra_pad = min_padded_len - current_clean_len - 2*total_transient_len;
    total_transient_len = total_transient_len + extra_pad;
    fprintf('Dynamically supplementing extension length: %d points (to ensure sufficient valid signal)\n', extra_pad);
end

% Signal extension
x_resample_padded = [flip(x_filtered_clean(1:total_transient_len));
    x_filtered_clean;
    flip(x_filtered_clean(end-total_transient_len+1:end))];

% Rational resampling
x_resampled_padded = resample(double(x_resample_padded), p_resamp, q_resamp);

% Truncate extended part
resample_pad_len = floor(total_transient_len * p_resamp / q_resamp);
x_resampled = x_resampled_padded(resample_pad_len+1 : end-resample_pad_len);
t_resampled = (0:length(x_resampled)-1)' / fs_target;

%% ==================== 5. Secondary Integer Cycle Extraction ====================
% Calculate target extraction length
total_periods = N_cycle;
cut_length = total_periods * samples_per_cycle_target;

% Extract from the middle to avoid residual transients
if length(x_resampled) >= cut_length
    start = floor((length(x_resampled)-cut_length)/2);
    x_processed = x_resampled(start+1:start+cut_length);
    t_processed = t_resampled(start+1:start+cut_length);
else
    x_processed = x_resampled;
    t_processed = t_resampled;
    warning('Insufficient signal length after resampling, unable to extract specified number of cycles');
end
end