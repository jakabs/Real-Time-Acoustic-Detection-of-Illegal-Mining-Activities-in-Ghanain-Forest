%% MODULE 2 - CNN Hybrid Feature Training & Analysis
% ============================================================
% Author: Okoampah Ernest
% Date: 24/11/2025
% Description: 
%   1. Trains Primary (Log-Mel) and Secondary (MFCC) CNNs.
%   2. Implements Robust Data Loading (Auto-Transpose).
%   3. Generates 3 Key Study Figures:
%      - Training Dynamics (Loss/Accuracy Curves)
%      - Confusion Matrices
%      - Per-Class Metrics (Precision/Recall/F1)
% ============================================================

clearvars; clc; close all;
fprintf('=== MODULE 2: Hybrid Feature Training & Analysis ===\n');

%% -------------------------
%% 0. Configuration
%% -------------------------
scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder), scriptFolder = pwd; end
project_root = fullfile(scriptFolder, '..');

manifest_dir = fullfile(project_root, 'manifests');
results_dir  = fullfile(project_root, 'results');
figures_dir  = fullfile(project_root, 'figures');

% Ensure output directories exist
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
if ~exist(figures_dir, 'dir'), mkdir(figures_dir); end

% Load Config and Class Names
config_path = fullfile(manifest_dir, 'preprocessing_config.json');
if ~exist(config_path, 'file'), error('Run Module 1 first!'); end
params = jsondecode(fileread(config_path));

classes = {'background','chainsaw','engine'};

%% -------------------------
%% 1. Load Data (Robust Loader)
%% -------------------------
fprintf('\n--- 1. Loading and Shaping Data ---\n');

trainTbl = readtable(fullfile(manifest_dir,'train.csv'));
valTbl   = readtable(fullfile(manifest_dir,'val.csv'));
testTbl  = readtable(fullfile(manifest_dir,'test.csv'));

% --- LOAD PRIMARY (Log-Mel) ---
% Target: [64 Freq x 128 Time]
fprintf('Loading Primary Features (Log-Mel)...\n');
[Xtrain_prim, Ytrain] = load_features_safe(trainTbl, project_root, 'primary_mel', [64, 128], classes);
[Xval_prim, Yval]     = load_features_safe(valTbl, project_root, 'primary_mel', [64, 128], classes);
[Xtest_prim, Ytest]   = load_features_safe(testTbl, project_root, 'primary_mel', [64, 128], classes);

% --- LOAD SECONDARY (MFCC) ---
% Target: [13 Coeffs x 128 Time]
% The helper function will automatically TRANSPOSE if it finds [128x13] data.
fprintf('Loading Secondary Features (MFCC)...\n');
[Xtrain_sec, ~] = load_features_safe(trainTbl, project_root, 'secondary_mfcc', [13, 128], classes);
[Xval_sec, ~]   = load_features_safe(valTbl, project_root, 'secondary_mfcc', [13, 128], classes);
[Xtest_sec, ~]  = load_features_safe(testTbl, project_root, 'secondary_mfcc', [13, 128], classes);

%% -------------------------
%% 2. Global Normalization (Z-Score)
%% -------------------------
fprintf('\n--- 2. Applying Global Normalization ---\n');

% Primary: Calculate stats on TRAIN only, apply to Val/Test
mu_prim = mean(Xtrain_prim(:)); 
sigma_prim = std(Xtrain_prim(:));
Xtrain_prim = (Xtrain_prim - mu_prim) / (sigma_prim + eps);
Xval_prim   = (Xval_prim   - mu_prim) / (sigma_prim + eps);
Xtest_prim  = (Xtest_prim  - mu_prim) / (sigma_prim + eps);
fprintf('Primary (Mel) Stats: mu=%.4f, sigma=%.4f\n', mu_prim, sigma_prim);

% Secondary: Calculate stats on TRAIN only
mu_sec = mean(Xtrain_sec(:)); 
sigma_sec = std(Xtrain_sec(:));
Xtrain_sec = (Xtrain_sec - mu_sec) / (sigma_sec + eps);
Xval_sec   = (Xval_sec   - mu_sec) / (sigma_sec + eps);
Xtest_sec  = (Xtest_sec  - mu_sec) / (sigma_sec + eps);
fprintf('Secondary (MFCC) Stats: mu=%.4f, sigma=%.4f\n', mu_sec, sigma_sec);

%% -------------------------
%% 3. Train Primary Model (Log-Mel)
%% -------------------------
fprintf('\n--- 3. Training Primary Model (Log-Mel) ---\n');

% CNN Architecture for 64x128 images
layers_prim = [
    imageInputLayer([64 128 1], 'Name', 'input', 'Normalization', 'none')
    
    convolution2dLayer(3, 32, 'Padding','same'); batchNormalizationLayer; reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding','same'); batchNormalizationLayer; reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 128, 'Padding','same'); batchNormalizationLayer; reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    globalAveragePooling2dLayer
    dropoutLayer(0.4) 
    fullyConnectedLayer(length(classes))
    softmaxLayer
    classificationLayer];

opts_prim = trainingOptions('adam', ...
    'MaxEpochs', 25, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-3, ...
    'ValidationData', {Xval_prim, Yval}, ...
    'ValidationPatience', 5, ...
    'Plots', 'training-progress', ... % ENABLE LIVE PLOTTING
    'Verbose', true);

fprintf('Starting Training (Primary)...\n');
[net_prim, info_prim] = trainNetwork(Xtrain_prim, Ytrain, layers_prim, opts_prim);

% Evaluate Primary
Ypred_prim = classify(net_prim, Xtest_prim);
acc_prim = mean(Ypred_prim == Ytest);
fprintf('Primary Model Accuracy: %.2f%%\n', acc_prim*100);

%% -------------------------
%% 4. Train Secondary Model (MFCC)
%% -------------------------
fprintf('\n--- 4. Training Secondary Model (MFCC) ---\n');

layers_sec = [
    imageInputLayer([13 128 1], 'Name', 'input', 'Normalization', 'none')
    
    % Block 1: Standard conv
    convolution2dLayer(3, 32, 'Padding','same'); batchNormalizationLayer; reluLayer
    maxPooling2dLayer(2, 'Stride', 2) 
    
    % Block 2: Standard conv
    convolution2dLayer(3, 64, 'Padding','same'); batchNormalizationLayer; reluLayer
    maxPooling2dLayer(2, 'Stride', 2) 
    
    globalAveragePooling2dLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(length(classes))
    softmaxLayer
    classificationLayer];

opts_sec = trainingOptions('adam', ...
    'MaxEpochs', 25, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-3, ...
    'ValidationData', {Xval_sec, Yval}, ...
    'ValidationPatience', 5, ...
    'Plots', 'training-progress', ... % ENABLE LIVE PLOTTING
    'Verbose', true);

fprintf('Starting Training (Secondary)...\n');
[net_sec, info_sec] = trainNetwork(Xtrain_sec, Ytrain, layers_sec, opts_sec);

% Evaluate Secondary
Ypred_sec = classify(net_sec, Xtest_sec);
acc_sec = mean(Ypred_sec == Ytest);
fprintf('Secondary Model Accuracy: %.2f%%\n', acc_sec*100);

%% -------------------------
%% 5. Study Visualizations 
%% -------------------------
fprintf('\n--- 5. Generating Study Figures ---\n');

% --- FIGURE A: Training Dynamics (Loss & Accuracy) ---
% Critical for showing convergence and lack of overfitting
fig1 = figure('Position', [100 100 1200 600], 'Color', 'w', 'Visible', 'off');

% Primary Model - Loss
subplot(2,2,1);
plot_training_curve(info_prim, 'Loss');
title('Primary (Mel): Loss', 'FontWeight', 'bold');

% Primary Model - Accuracy
subplot(2,2,2);
plot_training_curve(info_prim, 'Accuracy');
title('Primary (Mel): Accuracy', 'FontWeight', 'bold');

% Secondary Model - Loss
subplot(2,2,3);
plot_training_curve(info_sec, 'Loss');
title('Secondary (MFCC): Loss', 'FontWeight', 'bold');

% Secondary Model - Accuracy
subplot(2,2,4);
plot_training_curve(info_sec, 'Accuracy');
title('Secondary (MFCC): Accuracy', 'FontWeight', 'bold');

out_fig1 = fullfile(figures_dir, 'figure_training_dynamics.png');
exportgraphics(fig1, out_fig1, 'Resolution', 300);
fprintf('Saved: %s\n', out_fig1);
close(fig1);


% --- FIGURE B: Confusion Matrix Comparison ---
fig2 = figure('Position', [100 100 1200 500], 'Color', 'w', 'Visible', 'off');

subplot(1,2,1);
confusionchart(Ytest, Ypred_prim);
title({['Primary (Mel): ' num2str(acc_prim*100,'%.1f') '%'], 'Texture Features'});

subplot(1,2,2);
confusionchart(Ytest, Ypred_sec);
title({['Secondary (MFCC): ' num2str(acc_sec*100,'%.1f') '%'], 'Timbre Features'});

out_fig2 = fullfile(figures_dir, 'figure_confusion_matrices.png');
exportgraphics(fig2, out_fig2, 'Resolution', 300);
fprintf('Saved: %s\n', out_fig2);
close(fig2);


% --- FIGURE C: Per-Class Performance Metrics (Precision/Recall/F1) ---
% Useful for showing if one class is harder than others
fig3 = figure('Position', [100 100 800 500], 'Color', 'w', 'Visible', 'off');
plot_per_class_metrics(Ytest, Ypred_prim, Ypred_sec, classes);

out_fig3 = fullfile(figures_dir, 'figure_per_class_metrics.png');
exportgraphics(fig3, out_fig3, 'Resolution', 300);
fprintf('Saved: %s\n', out_fig3);
close(fig3);


%% -------------------------
%% 6. Save Models & Data for Ensemble
%% -------------------------
fprintf('\n--- 6. Saving Data for Module 3 ---\n');

save(fullfile(results_dir, 'primary_model.mat'), 'net_prim', 'mu_prim', 'sigma_prim', 'acc_prim', 'info_prim');
save(fullfile(results_dir, 'secondary_model.mat'), 'net_sec', 'mu_sec', 'sigma_sec', 'acc_sec', 'info_sec');
save(fullfile(results_dir, 'data_splits.mat'), 'Yval', 'Ytest', 'valTbl', 'testTbl', 'classes');

fprintf('Models and Splits saved to: %s\n', results_dir);
fprintf('=== MODULE 2 COMPLETE ===\n');

%% -------------------------
%% HELPER FUNCTIONS
%% -------------------------
function [X_4d, Y_cat] = load_features_safe(tbl, proj_root, field_name, target_size, class_names)
    num_samples = height(tbl);
    target_H = target_size(1); 
    target_W = target_size(2);
    X_4d = zeros(target_H, target_W, 1, num_samples, 'single');
    for i = 1:num_samples
        fname = fullfile(proj_root, tbl.filepath{i});
        if ~exist(fname, 'file'), warning('Missing: %s', fname); continue; end
        tmp = load(fname, 'hybrid_features');
        feat = tmp.hybrid_features.(field_name);
        [r, c] = size(feat);
        if (r == target_W) && (c == target_H), feat = feat'; [r, c] = size(feat);
        elseif (r == target_W) && (target_H < target_W) && (c < r), feat = feat'; [r, c] = size(feat); end
        if r > target_H, feat = feat(1:target_H, :); elseif r < target_H, feat = [feat; zeros(target_H-r, c)]; end
        if c > target_W, feat = feat(:, 1:target_W); elseif c < target_W, feat = [feat zeros(target_H, target_W-c)]; end
        X_4d(:, :, 1, i) = single(feat);
    end
    if nargout > 1, Y_cat = categorical(tbl.label, class_names); end
end

function plot_training_curve(info, type)
    if strcmp(type, 'Loss')
        trainData = info.TrainingLoss;
        valData = info.ValidationLoss;
        yLab = 'Cross-Entropy Loss';
    else
        trainData = info.TrainingAccuracy;
        valData = info.ValidationAccuracy;
        yLab = 'Accuracy (%)';
    end
    
    % Interpolate validation data to match training iterations
    iters = 1:length(trainData);
    valIters = 1:floor(length(trainData)/length(valData)):length(trainData);
    valIters = valIters(1:length(valData)); % Ensure dimensions match
    
    plot(iters, trainData, 'b-', 'LineWidth', 1.5); hold on;
    % Plot validation as dots/markers
    plot(valIters, valData, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'r');
    
    xlabel('Iteration'); ylabel(yLab);
    legend('Training', 'Validation', 'Location', 'best');
    grid on; box on;
end

function plot_per_class_metrics(Ytest, Ypred1, Ypred2, classes)
    % Calculate F1 Score per class for both models
    f1_model1 = zeros(1, numel(classes));
    f1_model2 = zeros(1, numel(classes));
    
    for i = 1:numel(classes)
        c = classes{i};
        % Model 1
        tp = sum(Ytest == c & Ypred1 == c);
        fp = sum(Ytest ~= c & Ypred1 == c);
        fn = sum(Ytest == c & Ypred1 ~= c);
        f1_model1(i) = 2*tp / (2*tp + fp + fn + eps);
        
        % Model 2
        tp = sum(Ytest == c & Ypred2 == c);
        fp = sum(Ytest ~= c & Ypred2 == c);
        fn = sum(Ytest == c & Ypred2 ~= c);
        f1_model2(i) = 2*tp / (2*tp + fp + fn + eps);
    end
    
    data = [f1_model1; f1_model2]';
    b = bar(data);
    b(1).FaceColor = [0.2 0.6 0.8]; % Blue
    b(2).FaceColor = [0.8 0.4 0.2]; % Orange
    
    set(gca, 'XTickLabel', classes);
    ylabel('F1-Score');
    legend('Primary (Mel)', 'Secondary (MFCC)', 'Location', 'SouthEast');
    title('Per-Class F1-Score Comparison', 'FontWeight', 'bold');
    ylim([0.8 1.05]); grid on;
end