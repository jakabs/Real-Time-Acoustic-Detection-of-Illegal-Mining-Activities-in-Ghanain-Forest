%% MODULE 1 - Hybrid Feature Preprocessing (Log-Mel + MFCC)
% ============================================================
% Author: Okoampah Ernest
% Date: 24/11/2025
% Description: 
%   1. Extracts Log-Mel (Texture) and MFCC (Timbre) features ONLY.
%   2. Uses Low-Noise Augmentation (0.1% - 1%) for high-quality training.
%   3. Generates 3 key figures to support the study methodology.
% ============================================================

clearvars; clc; close all;
fprintf('=== MODULE 1: Hybrid Preprocessing (Log-Mel + MFCC) ===\n');

%% -------------------------
%% 0. Configuration
%% -------------------------
scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder), scriptFolder = pwd; end
project_root = fullfile(scriptFolder, '..');

data_raw     = fullfile(project_root, 'data_raw');
feat_dir     = fullfile(project_root, 'data_processed', 'features');
manifest_dir = fullfile(project_root, 'manifests');

% Create required folders
for p = {feat_dir, manifest_dir}
    if ~exist(p{1}, 'dir'), mkdir(p{1}); end
end

% --- Class mapping ---
source_folders = {'background', 'chainsaw', 'engine'};
class_map = containers.Map(...
    {'background','chainsaw','engine'}, ...
    {'background','chainsaw','engine'});
final_classes = {'background','chainsaw','engine'};

% --- Audio Parameters ---
params = struct();
params.fs_target       = 22050;
params.clip_dur        = 1.5;                     
params.clip_samples    = params.fs_target * params.clip_dur;
params.overlap_ratio   = 0.75;                    
params.step_samples    = round(params.clip_samples * (1 - params.overlap_ratio));

% --- Feature Parameters ---
params.win_len_sec     = 0.025;
params.hop_len_sec     = 0.010;
params.num_mel_bands   = 64;                      
params.fft_len         = 512;
params.freq_range      = [50, 8000];
params.target_time_frames = 128; 
params.mfcc_coeffs     = 13;                      

% --- NOISE SETTINGS (Maintained Low Noise) ---
params.noise_level_min = 0.001; % 0.1%
params.noise_level_max = 0.010; % 1.0%

% Derived parameters
params.win_len_samples = round(params.win_len_sec  * params.fs_target);
params.hop_len_samples = round(params.hop_len_sec  * params.fs_target);
if params.fft_len < params.win_len_samples
    params.fft_len = 2^nextpow2(params.win_len_samples);
end

fprintf('CONFIGURATION:\n');
fprintf('  Features: Log-Mel (Primary) + MFCC (Secondary)\n');
fprintf('  Noise: %.3f%% to %.3f%% (Low Intensity)\n', ...
    params.noise_level_min*100, params.noise_level_max*100);

%% -------------------------
%% 1. Cleanup
%% -------------------------
fprintf('\n--- 1. Cleaning previous results ---\n');
delete(fullfile(feat_dir, '*.mat'));
delete(fullfile(manifest_dir, '*.csv'));
delete(fullfile(manifest_dir, '*.json'));
delete(fullfile(manifest_dir, 'figure_*.png'));
fprintf('Cleanup done.\n');

%% -------------------------
%% 2. Hybrid Feature Extraction
%% -------------------------
fprintf('\n--- 2. Extracting Features ---\n');
feature_count = 0;
rng(42); 

% Track counts
actual_counts = containers.Map(final_classes, {0, 0, 0});

for ci = 1:numel(source_folders)
    folder = source_folders{ci};
    target = class_map(folder);
    src_dir = fullfile(data_raw, folder);
    
    if ~exist(src_dir,'dir'), warning('Missing: %s',src_dir); continue; end
    
    files = dir(fullfile(src_dir,'*.wav'));
    fprintf('Processing %s → %s (%d files)\n', folder, target, numel(files));
    
    for fi = 1:numel(files)
        filepath = fullfile(files(fi).folder, files(fi).name);
        
        % Robust File Reading
        try
            [x, fs_orig] = audioread(filepath);
        catch ME
            warning('Corrupt file skipped: %s (%s)', files(fi).name, ME.message);
            continue; 
        end
        
        [x, ~] = preprocess_audio(x, fs_orig, params.fs_target);
        
        if isempty(x), continue; end
        if length(x) < params.clip_samples
            x = pad_audio(x, params.clip_samples);
        end
        
        starts = 1 : params.step_samples : (length(x) - params.clip_samples + 1);
        
        for startIdx = starts
            clip = x(startIdx : startIdx + params.clip_samples - 1);
            
            % Augmentation (Low Noise)
            noise_lvl = params.noise_level_min + rand*(params.noise_level_max-params.noise_level_min);
            clip = add_noise(clip, noise_lvl);
            
            % Extract Features (Mel + MFCC Only)
            hybrid_features = extract_hybrid_features(clip, params);
            
            if ~isempty(hybrid_features)
                feature_count = feature_count + 1;
                fname = sprintf('%s_%06d.mat', target, feature_count);
                className = target;
                
                % Save
                save(fullfile(feat_dir, fname), 'hybrid_features', 'className', 'params');
                actual_counts(target) = actual_counts(target) + 1;
            end
        end
    end
end

fprintf('=== Extraction complete: %d samples ===\n', feature_count);
k = keys(actual_counts);
for i = 1:length(k)
    fprintf('  %s: %d samples\n', k{i}, actual_counts(k{i}));
end

%% -------------------------
%% 3. Save Configuration & Stratify
%% -------------------------
config_path = fullfile(manifest_dir, 'preprocessing_config.json');
fid = fopen(config_path,'w');
fprintf(fid, '%s', jsonencode(params, 'PrettyPrint', true));
fclose(fid);

fprintf('\n--- 3. Creating Manifests ---\n');
create_stratified_manifests(feat_dir, manifest_dir, final_classes);

%% -------------------------
%% 4. Study Visualizations
%% -------------------------
fprintf('\n--- 4. Generating Study Figures ---\n');

% Figure A: Hybrid Features Samples (Mel + MFCC)
generate_hybrid_samples_figure(feat_dir, manifest_dir, final_classes);

% Figure B: Noise Analysis (Clean vs Low Noise)
generate_noise_effect_figure(feat_dir, manifest_dir, params);

% Figure C: Dataset Distribution
generate_distribution_figure(actual_counts, manifest_dir, final_classes);

fprintf('\n=== MODULE 1 FINISHED ===\n');


%% =============================================================
%% HELPER FUNCTIONS
%% =============================================================

function [x,fs] = preprocess_audio(x,fs_orig,fs_target)
    if size(x,2)>1, x = mean(x,2); end 
    if fs_orig ~= fs_target
        x = resample(x, fs_target, fs_orig);
        fs = fs_target;
    else
        fs = fs_orig;
    end
    if max(abs(x)) > 0
        x = x / max(abs(x)) * 0.99;
    end
end

function clip = add_noise(clip, level)
    noise = randn(size(clip));
    rms_signal = rms(clip);
    rms_noise  = rms(noise);
    if rms_noise < 1e-12, rms_noise = 1e-12; end 
    clip = clip + noise*(rms_signal/rms_noise)*level;
    clip = max(min(clip,1),-1);
end

function x = pad_audio(x, target_samples)
    if length(x) < target_samples
        x(target_samples) = 0; 
    end
end

function feats = extract_hybrid_features(clip, p)
    try
        normalize_safe = @(m) (m - min(m(:))) / (max(m(:)) - min(m(:)) + 1e-8);
        pad_time = @(m) perform_time_padding(m, p.target_time_frames);
        
        % 1. Log-Mel (Primary)
        mel = melSpectrogram(clip, p.fs_target, ...
            'Window',           hann(p.win_len_samples,'periodic'), ...
            'OverlapLength',    p.win_len_samples - p.hop_len_samples, ...
            'FFTLength',        p.fft_len, ...
            'NumBands',         p.num_mel_bands, ...
            'FrequencyRange',   p.freq_range, ...
            'SpectrumType',     'power');
        mel = log10(mel + 1e-6); 
        mel = normalize_safe(mel);
        mel = pad_time(mel);
        
        % 2. MFCC (Secondary) - Transposed
        [coeffs, ~] = mfcc(clip, p.fs_target, ...
            'NumCoeffs', p.mfcc_coeffs, ...
            'FFTLength', p.fft_len, ...
            'Window', hann(p.win_len_samples));
        coeffs = coeffs'; % [Coeffs x Time]
        coeffs = normalize_safe(coeffs);
        coeffs = pad_time(coeffs);
        
        % Store (Linear REMOVED as requested)
        feats.primary_mel = mel;
        feats.secondary_mfcc = coeffs;
    catch 
        feats = [];
    end
end

function padded = perform_time_padding(matrix, target_cols)
    [rows, cols] = size(matrix);
    if cols < target_cols
        pad_width = target_cols - cols;
        padded = [matrix, zeros(rows, pad_width)];
    elseif cols > target_cols
        padded = matrix(:, 1:target_cols);
    else
        padded = matrix;
    end
end

function create_stratified_manifests(feat_dir, manifest_dir, classes)
    files = dir(fullfile(feat_dir,'*.mat'));
    paths = cell(length(files),1);
    labels = cell(length(files),1);
    for i = 1:length(files)
        paths{i}  = fullfile('data_processed','features',files(i).name);
        parts = strsplit(files(i).name, '_');
        labels{i} = parts{1};
    end
    T = table(paths, labels, 'VariableNames',{'filepath','label'});
    
    rng(42);
    trainIdx = []; valIdx = []; testIdx = [];
    for i = 1:length(classes)
        c = classes{i};
        idx = find(strcmp(T.label, c));
        n = numel(idx);
        perm = idx(randperm(n));
        nTrain = round(0.70*n); 
        nVal   = round(0.15*n);
        trainIdx = [trainIdx; perm(1:nTrain)];
        valIdx   = [valIdx;   perm(nTrain+1:nTrain+nVal)];
        testIdx  = [testIdx;  perm(nTrain+nVal+1:end)];
    end
    writetable(T(trainIdx,:), fullfile(manifest_dir,'train.csv'));
    writetable(T(valIdx,:),   fullfile(manifest_dir,'val.csv'));
    writetable(T(testIdx,:),  fullfile(manifest_dir,'test.csv'));
end

%% --- STUDY VISUALIZATIONS ---

function generate_hybrid_samples_figure(feat_dir, manifest_dir, classes)
    fig = figure('Position',[100 100 1200 800], 'Color','w', 'Visible', 'off');
    
    for i = 1:numel(classes)
        c = classes{i};
        f = dir(fullfile(feat_dir, [c '_*.mat']));
        if isempty(f), continue; end
        
        d = load(fullfile(f(1).folder,f(1).name));
        
        % Mel
        subplot(3, 2, (i-1)*2 + 1);
        imagesc(d.hybrid_features.primary_mel); axis xy; colormap('jet');
        title(sprintf('%s: Log-Mel (Texture)', c), 'FontWeight','bold');
        ylabel('Mel Bands'); xlabel('Time');
        
        % MFCC
        subplot(3, 2, (i-1)*2 + 2);
        imagesc(d.hybrid_features.secondary_mfcc); axis xy; colormap('jet');
        title(sprintf('%s: MFCC (Timbre)', c), 'FontWeight','bold');
        ylabel('Coefficients'); xlabel('Time');
    end
    
    out = fullfile(manifest_dir,'figure_hybrid_samples.png');
    exportgraphics(fig, out, 'Resolution', 300);
    fprintf('Saved Figure: Hybrid Samples -> %s\n', out);
    close(fig);
end

function generate_noise_effect_figure(feat_dir, manifest_dir, params)
    rng(999); 
    dummy_sig = sin(2*pi*440*(0:1/params.fs_target:0.5));
    
    % Clean
    mel_clean = melSpectrogram(dummy_sig, params.fs_target, ...
         'Window', hann(params.win_len_samples,'periodic'), ...
         'OverlapLength', params.win_len_samples - params.hop_len_samples, ...
         'FFTLength', params.fft_len, 'NumBands', params.num_mel_bands, ...
         'FrequencyRange', params.freq_range);
    mel_clean = log10(mel_clean + 1e-6);

    % Noisy (Max Low Noise)
    noisy_sig = add_noise(dummy_sig, params.noise_level_max);
    mel_noisy = melSpectrogram(noisy_sig, params.fs_target, ...
         'Window', hann(params.win_len_samples,'periodic'), ...
         'OverlapLength', params.win_len_samples - params.hop_len_samples, ...
         'FFTLength', params.fft_len, 'NumBands', params.num_mel_bands, ...
         'FrequencyRange', params.freq_range);
    mel_noisy = log10(mel_noisy + 1e-6);
     
    fig = figure('Position',[100 100 1000 400], 'Color','w', 'Visible', 'off');
    
    subplot(1,2,1); imagesc(mel_clean); axis xy; colormap('jet'); colorbar;
    title('Clean Signal', 'FontWeight', 'bold');
    
    subplot(1,2,2); imagesc(mel_noisy); axis xy; colormap('jet'); colorbar;
    title(sprintf('Augmented Signal (%.1f%% Noise)', params.noise_level_max*100), ...
        'FontWeight', 'bold');
    
    out = fullfile(manifest_dir,'figure_noise_effect.png');
    exportgraphics(fig, out, 'Resolution', 300);
    fprintf('Saved Figure: Noise Analysis -> %s\n', out);
    close(fig);
end

function generate_distribution_figure(counts_map, manifest_dir, classes)
    fig = figure('Position',[100 100 800 500], 'Color','w', 'Visible', 'off');
    
    counts = zeros(1, numel(classes));
    for i = 1:numel(classes)
        counts(i) = counts_map(classes{i});
    end
    
    b = bar(categorical(classes), counts);
    b.FaceColor = 'flat';
    b.CData = [0.2 0.6 0.8; 0.8 0.4 0.2; 0.2 0.8 0.4]; 
    
    title('Dataset Distribution', 'FontWeight', 'bold');
    ylabel('Number of Samples');
    grid on;
    
    text(1:length(counts), counts, num2str(counts'), ...
        'vert','bottom','horiz','center', 'FontWeight', 'bold');
    
    out = fullfile(manifest_dir,'figure_class_distribution.png');
    exportgraphics(fig, out, 'Resolution', 300);
    fprintf('Saved Figure: Distribution -> %s\n', out);
    close(fig);
end