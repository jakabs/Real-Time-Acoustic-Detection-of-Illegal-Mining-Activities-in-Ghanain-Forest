%% MODULE 3 - Ensemble Generation & Final Evaluation
% ============================================================
% Author: Okoampah Ernest
% Date: 24/11/2025
% Description: 
%   1. Loads Trained Models (Primary & Secondary).
%   2. Optimizes Ensemble Weights using VALIDATION set (Soft Voting).
%   3. Evaluates Final Performance on TEST set.
%   4. Generates "Final Report" figures.
% ============================================================
clearvars; clc; close all;
fprintf('=== MODULE 3: Ensemble Generation & Evaluation ===\n');

%% -------------------------
%% 0. Configuration
%% -------------------------
scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder), scriptFolder = pwd; end
project_root = fullfile(scriptFolder, '..');
manifest_dir = fullfile(project_root, 'manifests');
results_dir  = fullfile(project_root, 'results');
figures_dir  = fullfile(project_root, 'figures');

% Check if models exist
if ~exist(fullfile(results_dir, 'primary_model.mat'), 'file') || ...
   ~exist(fullfile(results_dir, 'secondary_model.mat'), 'file')
    error('Models not found! Run Module 2 first.');
end

% Load Config
params = jsondecode(fileread(fullfile(manifest_dir, 'preprocessing_config.json')));
classes = {'background','chainsaw','engine'};

%% -------------------------
%% 1. Load Trained Models & Splits
%% -------------------------
fprintf('\n--- 1. Loading Pre-trained Models ---\n');

% Load Primary
tmp = load(fullfile(results_dir, 'primary_model.mat'));
net_prim = tmp.net_prim;
stats_prim.mu = tmp.mu_prim;
stats_prim.sigma = tmp.sigma_prim;
fprintf('Loaded Primary Model (Acc: %.2f%%)\n', tmp.acc_prim*100);

% Load Secondary
tmp = load(fullfile(results_dir, 'secondary_model.mat'));
net_sec = tmp.net_sec;
stats_sec.mu = tmp.mu_sec;
stats_sec.sigma = tmp.sigma_sec;
fprintf('Loaded Secondary Model (Acc: %.2f%%)\n', tmp.acc_sec*100);

% Load Splits (Crucial for consistent evaluation)
load(fullfile(results_dir, 'data_splits.mat'), 'valTbl', 'testTbl', 'Yval', 'Ytest');

%% -------------------------
%% 2. Load Evaluation Features
%% -------------------------
fprintf('\n--- 2. Loading Features for Ensemble ---\n');

% Load Primary Features (Mel)
fprintf('Loading Primary Features...\n');
[Xval_prim, ~]     = load_features_safe(valTbl, project_root, 'primary_mel', [64, 128], classes);
[Xtest_prim, ~]    = load_features_safe(testTbl, project_root, 'primary_mel', [64, 128], classes);

% Load Secondary Features (MFCC)
fprintf('Loading Secondary Features...\n');
[Xval_sec, ~]      = load_features_safe(valTbl, project_root, 'secondary_mfcc', [13, 128], classes);
[Xtest_sec, ~]     = load_features_safe(testTbl, project_root, 'secondary_mfcc', [13, 128], classes);

%% -------------------------
%% 3. Normalize & Generate Probabilities
%% -------------------------
fprintf('\n--- 3. Generating Soft Probabilities ---\n');

% Normalize Primary (using SAVED stats)
Xval_prim  = (Xval_prim  - stats_prim.mu) / (stats_prim.sigma + eps);
Xtest_prim = (Xtest_prim - stats_prim.mu) / (stats_prim.sigma + eps);

% Normalize Secondary (using SAVED stats)
Xval_sec   = (Xval_sec   - stats_sec.mu) / (stats_sec.sigma + eps);
Xtest_sec  = (Xtest_sec  - stats_sec.mu) / (stats_sec.sigma + eps);

% Predict Probabilities
probs_prim_val  = predict(net_prim, Xval_prim);
probs_sec_val   = predict(net_sec, Xval_sec);

probs_prim_test = predict(net_prim, Xtest_prim);
probs_sec_test  = predict(net_sec, Xtest_sec);

%% -------------------------
%% 4. Optimize Ensemble Weights (Grid Search)
%% -------------------------
fprintf('\n--- 4. Optimizing Weights (Validation Set) ---\n');

best_acc = 0;
best_w = 0.5; 
weights = 0:0.01:1; % Fine-grained search

for w = weights
    % Soft Voting
    probs_ensemble = (w * probs_prim_val) + ((1-w) * probs_sec_val);
    
    % Get Class Indices
    [~, idx] = max(probs_ensemble, [], 2);
    Y_pred_val = categorical(classes(idx), classes);
    
    % Check Accuracy (reshape to ensure column vector alignment)
    acc = mean(Y_pred_val(:) == Yval(:));
    
    if acc > best_acc
        best_acc = acc;
        best_w = w;
    end
end

fprintf('Optimal Weight Found: w_prim = %.2f | w_sec = %.2f\n', best_w, 1-best_w);
fprintf('Best Validation Accuracy: %.2f%%\n', best_acc*100);

%% -------------------------
%% 5. Final Test Evaluation (FIXED)
%% -------------------------
fprintf('\n--- 5. Final Test Set Evaluation ---\n');

% 1. Ensemble Prediction (Apply Best Weights)
final_probs_test = (best_w * probs_prim_test) + ((1-best_w) * probs_sec_test);
[~, idx_ens] = max(final_probs_test, [], 2);
Y_final_pred = categorical(classes(idx_ens), classes);

% 2. Individual Model Predictions (Robust Indexing)
[~, idx_prim] = max(probs_prim_test, [], 2);
Y_prim_pred = categorical(classes(idx_prim), classes);

[~, idx_sec] = max(probs_sec_test, [], 2);
Y_sec_pred = categorical(classes(idx_sec), classes);

% 3. Calculate Final Accuracies
acc_prim_test = mean(Y_prim_pred(:) == Ytest(:));
acc_sec_test  = mean(Y_sec_pred(:)  == Ytest(:));
acc_ens_test  = mean(Y_final_pred(:) == Ytest(:));

% 4. Report
fprintf('=== FINAL TEST RESULTS ===\n');
fprintf('Primary Model (Mel):    %.2f%%\n', acc_prim_test*100);
fprintf('Secondary Model (MFCC): %.2f%%\n', acc_sec_test*100);
fprintf('Ensemble Model:         %.2f%% (Improvement: +%.2f%%)\n', ...
    acc_ens_test*100, (acc_ens_test - max(acc_prim_test, acc_sec_test))*100);

%% -------------------------
%% 6. Visualization
%% -------------------------
fprintf('\n--- 6. Generating Final Report Figures ---\n');

fig = figure('Position',[100 100 1200 600], 'Color', 'w', 'Visible', 'off');

% Subplot 1: Bar Chart Comparison
subplot(1, 2, 1);
b = bar([acc_prim_test, acc_sec_test, acc_ens_test]*100);
b.FaceColor = 'flat';
b.CData(1,:) = [0.2 0.6 0.8]; % Blue
b.CData(2,:) = [0.8 0.4 0.2]; % Orange
b.CData(3,:) = [0.2 0.8 0.4]; % Green
xticklabels({'Primary', 'Secondary', 'Ensemble'});
ylabel('Accuracy (%)');
ylim([0 105]); 
title('Model Performance Comparison', 'FontWeight', 'bold');
grid on;
% Add text labels
text(1:3, [acc_prim_test, acc_sec_test, acc_ens_test]*100 + 2, ...
    string(round([acc_prim_test, acc_sec_test, acc_ens_test]*100, 2)) + "%", ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% Subplot 2: Ensemble Confusion Matrix (FIXED)
subplot(1, 2, 2);
cm = confusionchart(Ytest, Y_final_pred);
% FIX: Use the Title property directly instead of title() function
cm.Title = sprintf('Ensemble Confusion Matrix (Acc: %.2f%%)', acc_ens_test*100);
cm.RowSummary = 'row-normalized'; % Adds Precision/Recall bars automatically

% Save
exportgraphics(fig, fullfile(figures_dir, 'figure_ensemble_results.png'), 'Resolution', 300);
fprintf('Final Report saved to: %s\n', figures_dir);

% Save Predictions CSV
results_tbl = table(testTbl.filepath, testTbl.label, Y_final_pred(:), ...
    'VariableNames', {'File', 'TrueLabel', 'PredictedLabel'});
writetable(results_tbl, fullfile(results_dir, 'final_test_predictions.csv'));
fprintf('Predictions CSV saved.\n');

fprintf('=== MODULE 3 FINISHED ===\n');

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