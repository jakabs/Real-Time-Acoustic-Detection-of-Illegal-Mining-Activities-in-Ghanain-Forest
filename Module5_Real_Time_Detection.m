%% MODULE 5 - Hybrid Feature Inference (Real-Time Detection Demo)
% ============================================================
% Author: Okoampah Ernest
% Date: 24/11/2025
% Description: 
%   1. Loads Models.
%   2. Records/Loads Audio.
%   3. Applies Robust Feature Extraction.
%   4. **PHYSICS CHECK (BAND POWER)**: Uses Bass/Treble ratio to
%      hard-correct Engine vs Chainsaw confusion.
% ============================================================

clearvars; clc; close all;
fprintf('=== MODULE 5: Real-Time Detection System (Band Power Logic) ===\n');

%% -------------------------
%% 0. Configuration & Loading
%% -------------------------
scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder), scriptFolder = pwd; end
project_root = fullfile(scriptFolder, '..');

results_dir = fullfile(project_root, 'results');
manifest_dir = fullfile(project_root, 'manifests');
figures_dir = fullfile(project_root, 'figures');

% 1. Load Configuration
config_path = fullfile(manifest_dir, 'preprocessing_config.json');
if ~exist(config_path, 'file'), error('Config not found. Run Module 1.'); end
params = jsondecode(fileread(config_path));

% 2. Load Models
fprintf('Loading optimized models...\n');
if ~exist(fullfile(results_dir, 'primary_model.mat'), 'file'), error('Primary missing.'); end
if ~exist(fullfile(results_dir, 'secondary_model.mat'), 'file'), error('Secondary missing.'); end

tmp_prim = load(fullfile(results_dir, 'primary_model.mat'));
net_prim = tmp_prim.net_prim;
stats_prim.mu = tmp_prim.mu_prim;
stats_prim.sigma = tmp_prim.sigma_prim;

tmp_sec = load(fullfile(results_dir, 'secondary_model.mat'));
net_sec = tmp_sec.net_sec;
stats_sec.mu = tmp_sec.mu_sec;
stats_sec.sigma = tmp_sec.sigma_sec;

classes = {'background', 'chainsaw', 'engine'};

% 3. Set Weights & Thresholds
w_prim = 0.62; 
w_sec  = 0.38;
rms_threshold = 0.002;  % Lower RMS threshold for better mic sensitivity

fprintf('System Ready.\n');
fprintf(' - RMS Threshold: %.3f\n', rms_threshold);
fprintf(' - Physics Logic: Bass/Treble Ratio Check ENABLED.\n');

%% -------------------------
%% 1. Get Audio Input
%% -------------------------
choice = menu('Select Input Source', 'Live Microphone (10s)', 'Load WAV File');

if choice == 1
    fs_rec = params.fs_target;
    recObj = audiorecorder(fs_rec, 16, 1);
    fprintf('\n>>> RECORDING 10 SECONDS... (Make noise!) <<<\n');
    recordblocking(recObj, 10);
    x = getaudiodata(recObj);
    fprintf('>>> Recording Complete. <<<\n');
    fs_orig = fs_rec;
elseif choice == 2
    [fname, fpath] = uigetfile({'*.wav';'*.mp3'}, 'Select Audio File');
    if isequal(fname,0), disp('Cancelled.'); return; end
    full_audio_path = fullfile(fpath, fname);
    fprintf('Loading: %s\n', fname);
    [x, fs_orig] = audioread(full_audio_path);
else
    return;
end

if size(x, 2) > 1, x = mean(x, 2); end % Mono
if fs_orig ~= params.fs_target
    x = resample(x, params.fs_target, fs_orig);
end
x = x / (max(abs(x)) + eps); % Normalize microphone recording

%% -------------------------
%% 2. Process Audio Stream
%% -------------------------
fprintf('\n--- Analyzing Audio Content ---\n');

clip_len = params.clip_samples;
step_len = round(clip_len * 0.5); 
num_clips = floor((length(x) - clip_len) / step_len) + 1;

if num_clips < 1
    x = [x; zeros(clip_len - length(x), 1)]; 
    num_clips = 1;
end

votes = zeros(3, 1); 
conf_scores = zeros(3, num_clips);
is_silence = false(1, num_clips);

fprintf('Processing %d segments...\n', num_clips);

for i = 1:num_clips
    start_idx = (i-1)*step_len + 1;
    end_idx   = start_idx + clip_len - 1;
    clip = x(start_idx:end_idx);
    
    % --- 1. NOISE GATE ---
    current_rms = rms(clip);
    if current_rms < rms_threshold
        conf_scores(:, i) = [1.0; 0.0; 0.0]; % Background
        votes(1) = votes(1) + 1;
        is_silence(i) = true;
        continue; 
    end
    
    % --- 2. FEATURE EXTRACTION ---
    feats = extract_inference_features(clip, params);
    if isempty(feats), continue; end
    
    % --- 3. CNN INFERENCE ---
    in_prim = (feats.mel - stats_prim.mu) / (stats_prim.sigma + eps);
    in_prim = reshape(in_prim, [64, 128, 1, 1]); 
    p_prim = predict(net_prim, in_prim);
    
    in_sec = (feats.mfcc - stats_sec.mu) / (stats_sec.sigma + eps);
    in_sec = reshape(in_sec, [13, 128, 1, 1]); 
    p_sec = predict(net_sec, in_sec);
    
    p_final = (w_prim * p_prim) + (w_sec * p_sec);
    [~, raw_idx] = max(p_final);
    
    % --- 4. PHYSICS LOGIC CHECK (BAND POWER RATIO) ---
    % This function prints debug info so you can see the ratio
    corrected_idx = verify_prediction_physics(clip, params.fs_target, raw_idx, classes);
    
    % Update Scores based on logic correction
    if corrected_idx ~= raw_idx
        % Swap probabilities to reflect the logic override
        temp = p_final(raw_idx);
        p_final(raw_idx) = p_final(corrected_idx);
        p_final(corrected_idx) = temp;
        fprintf('  [LOGIC OVERRIDE] Seg %d: Forced %s -> %s\n', ...
            i, upper(classes{raw_idx}), upper(classes{corrected_idx}));
    end
    
    % Accumulate
    conf_scores(:, i) = p_final';
    [~, final_clip_idx] = max(p_final);
    votes(final_clip_idx) = votes(final_clip_idx) + 1;
    
    % Debug output
    fprintf('RMS: %.4f, Predicted: %s, Confidence: %.2f%%\n', current_rms, classes{final_clip_idx}, max(p_final)*100);
    
    if mod(i, 5) == 0
        fprintf('  Seg %d: %s (Conf: %.1f%%)\n', ...
            i, upper(classes{final_clip_idx}), max(p_final)*100);
    end
end

%% -------------------------
%% 3. Final Decision & Visualization
%% -------------------------
[~, final_idx] = max(sum(conf_scores, 2)); 
final_pred = classes{final_idx};
final_conf = mean(conf_scores(final_idx, :)) * 100;

fprintf('\n=== FINAL VERDICT ===\n');
fprintf('DETECTED: %s\n', upper(final_pred));

fig = figure('Position', [50, 50, 1200, 900], 'Color', 'w', 'Name', 'Detection Results');

% 1. Waveform
subplot(4,1,1);
t = (0:length(x)-1) / params.fs_target;
plot(t, x, 'k', 'Color', [0.4 0.4 0.4]); hold on;
title(['Verdict: ' upper(final_pred)], 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (s)'); ylabel('Amp'); axis tight; ylim([-1.1 1.1]);

% Paint Red Zones (Engine/Chainsaw)
threat_indices = find((conf_scores(2,:) > 0.5 | conf_scores(3,:) > 0.5) & ~is_silence);
if ~isempty(threat_indices)
    y_lim = ylim;
    for idx = threat_indices
        t_s = (idx-1)*step_len / params.fs_target;
        t_e = t_s + (clip_len / params.fs_target);
        fill([t_s t_e t_e t_s], [y_lim(1) y_lim(1) y_lim(2) y_lim(2)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
end

% 2. Confidence
subplot(4,1,2);
time_axis = ((0:num_clips-1)*step_len) / params.fs_target;
plot(time_axis, conf_scores(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis, conf_scores(2,:), 'r-', 'LineWidth', 1.5);
plot(time_axis, conf_scores(3,:), 'g-', 'LineWidth', 1.5);
title('Confidence Stream'); legend('Background', 'Chainsaw', 'Engine'); grid on;

% 3. Spectrogram
subplot(4,1,3);
spectrogram(x, hann(512), 256, 512, params.fs_target, 'yaxis');
title('Spectral Analysis (Look for Bass vs Treble)'); colormap('jet');

% 4. Votes
subplot(4,1,4);
b = bar(categorical(classes), sum(conf_scores, 2));
b.FaceColor = 'flat'; b.CData = [0.2 0.6 0.8; 0.8 0.4 0.2; 0.2 0.8 0.4];
title('Total Evidence');

saveas(fig, fullfile(figures_dir, 'final_demo_result.png'));
fprintf('Saved: figures/final_demo_result.png\n');

%% -------------------------
%% HELPER: Physics Logic Check (Band Power)
%% -------------------------
function corrected_idx = verify_prediction_physics(clip, fs, pred_idx, classes)
    corrected_idx = pred_idx;
    pred_class = classes{pred_idx};
    
    % Only check if prediction is Chainsaw or Engine
    if strcmp(pred_class, 'chainsaw') || strcmp(pred_class, 'engine')
        
        N = 1024;
        Y = fft(clip, N);
        P = abs(Y(1:N/2+1)).^2; % Power
        f = fs*(0:(N/2))/N;     % Frequency
        
        % Define Bands
        idx_bass   = f >= 50 & f <= 600;    % Engine Core
        idx_treble = f >= 1500 & f <= 8000; % Chainsaw Core
        
        p_bass   = sum(P(idx_bass));
        p_treble = sum(P(idx_treble));
        
        ratio = p_bass / (p_treble + eps);
        
        % LOGIC:
        % If AI says Chainsaw, but Ratio > 1.5 (Bass Heavy) -> Force Engine
        if strcmp(pred_class, 'chainsaw') && ratio > 1.5
            corrected_idx = 3; % Engine
        end
        
        % If AI says Engine, but Ratio < 0.5 (Treble Heavy) -> Force Chainsaw
        if strcmp(pred_class, 'engine') && ratio < 0.5
            corrected_idx = 2; % Chainsaw
        end
    end
end

%% -------------------------
%% HELPER: Feature Extraction
%% -------------------------
function feats = extract_inference_features(clip, p)
    try
        normalize_safe = @(m) (m - min(m(:))) / (max(m(:)) - min(m(:)) + 1e-8);
        
        mel = melSpectrogram(clip, p.fs_target, ...
            'Window', hann(p.win_len_samples,'periodic'), ...
            'OverlapLength', p.win_len_samples - p.hop_len_samples, ...
            'FFTLength', p.fft_len, 'NumBands', p.num_mel_bands, ...
            'FrequencyRange', p.freq_range, 'SpectrumType', 'power');
        mel = log10(mel + 1e-6);
        mel = normalize_safe(mel);
        
        [coeffs, ~] = mfcc(clip, p.fs_target, ...
            'NumCoeffs', p.mfcc_coeffs, 'FFTLength', p.fft_len, ...
            'Window', hann(p.win_len_samples));
        coeffs = coeffs'; 
        coeffs = normalize_safe(coeffs);
        
        feats.mel = enforce_dims(mel, [64, 128]);
        feats.mfcc = enforce_dims(coeffs, [13, 128]);
    catch
        feats = [];
    end
end

function fixed_mat = enforce_dims(matrix, target_size)
    t_r = target_size(1); t_c = target_size(2);
    [r, c] = size(matrix);
    if r > t_r, matrix = matrix(1:t_r, :); elseif r < t_r, matrix = [matrix; zeros(t_r - r, c)]; end
    [r, c] = size(matrix);
    if c > t_c, matrix = matrix(:, 1:t_c); elseif c < t_c, matrix = [matrix, zeros(r, t_c - c)]; end
    fixed_mat = matrix;
end
