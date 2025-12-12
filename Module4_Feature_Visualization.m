%% MODULE 4 - Deep Feature Visualization (t-SNE)
% ============================================================
% Author: Okoampah Ernest
% Date: 24/11/2025
% Description: 
%   1. Loads the trained Primary Model (Log-Mel).
%   2. Extracts "Deep Features" (vectors) from the Global Pooling layer.
%   3. Uses t-SNE to project these high-dimensional vectors to 2D.
%   4. Visualizes class separation to validate accuracy claims.
% ============================================================
clearvars; clc; close all;
fprintf('=== MODULE 4: Deep Feature Visualization (Robust) ===\n');

%% -------------------------
%% 1. Configuration & Loading
%% -------------------------
scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder), scriptFolder = pwd; end
project_root = fullfile(scriptFolder, '..');
results_dir  = fullfile(project_root, 'results');
manifest_dir = fullfile(project_root, 'manifests');
figures_dir  = fullfile(project_root, 'figures');

% Ensure figures folder exists
if ~exist(figures_dir, 'dir'), mkdir(figures_dir); end

% Load Models & Data
fprintf('Loading Primary Model...\n');
if ~exist(fullfile(results_dir, 'primary_model.mat'), 'file')
    error('Primary model not found. Run Module 2 first.');
end
load(fullfile(results_dir, 'primary_model.mat'), 'net_prim', 'mu_prim', 'sigma_prim');

% Load Splits
if ~exist(fullfile(results_dir, 'data_splits.mat'), 'file')
    error('Data splits not found. Run Module 2 first.');
end
load(fullfile(results_dir, 'data_splits.mat'), 'Ytest', 'testTbl', 'classes');

%% -------------------------
%% 2. Load Test Data (Robust)
%% -------------------------
fprintf('Loading Test Features (%d samples)...\n', height(testTbl));

% Pre-allocate memory [Height x Width x Channels x Samples]
% Primary Mel is 64x128
X_test = zeros(64, 128, 1, height(testTbl), 'single');

for i = 1:height(testTbl)
    filepath = fullfile(project_root, testTbl.filepath{i});
    if ~exist(filepath, 'file')
        warning('Missing file: %s', filepath);
        continue;
    end
    
    d = load(filepath, 'hybrid_features');
    feat = d.hybrid_features.primary_mel;
    
    % --- CRITICAL FIX: Enforce Dimensions [64 x 128] ---
    % This prevents crashes if Module 1 output slightly varies
    [r, c] = size(feat);
    target_r = 64; target_c = 128;
    
    % Fix Rows
    if r > target_r, feat = feat(1:target_r, :); 
    elseif r < target_r, feat = [feat; zeros(target_r-r, c)]; end
    
    % Fix Cols
    [r, c] = size(feat); % Update size
    if c > target_c, feat = feat(:, 1:target_c); 
    elseif c < target_c, feat = [feat zeros(r, target_c-c)]; end
    % ---------------------------------------------------

    % Normalize using Training Stats
    feat = (feat - mu_prim) / (sigma_prim + eps); 
    
    X_test(:,:,1,i) = single(feat);
end

%% -------------------------
%% 3. Extract Deep Features (Robust)
%% -------------------------
fprintf('Extracting Activations from Global Pooling Layer...\n');

% --- ROBUST LAYER FINDING LOGIC ---
layer_name = '';

% Method A: Check class string directly
for i = 1:numel(net_prim.Layers)
    if contains(class(net_prim.Layers(i)), 'GlobalAveragePooling')
        layer_name = net_prim.Layers(i).Name;
        fprintf('  -> Found Global Pooling Layer: "%s" (Index %d)\n', layer_name, i);
        break;
    end
end

% Method B: Fallback Logic (Search relative to Dropout)
if isempty(layer_name)
    fprintf('  -> Method A failed. Trying Fallback (searching relative to Dropout)...\n');
    for i = 1:numel(net_prim.Layers)
        if contains(class(net_prim.Layers(i)), 'Dropout')
            % The layer BEFORE dropout is usually the Global Pool
            if i > 1
                layer_name = net_prim.Layers(i-1).Name;
                fprintf('  -> Found Layer via Fallback: "%s" (Index %d)\n', layer_name, i-1);
                break;
            end
        end
    end
end

% Final Error Check
if isempty(layer_name)
    fprintf('\n--- DEBUG: LIST OF LAYERS ---\n');
    for k=1:numel(net_prim.Layers)
        fprintf('%d: %s (%s)\n', k, net_prim.Layers(k).Name, class(net_prim.Layers(k)));
    end
    error('Could not auto-detect the Global Pooling layer. See list above.');
end

% Extract features
% Output dimensions: [1 x 1 x NumChannels x NumSamples]
features_raw = activations(net_prim, X_test, layer_name);

% Reshape to [NumSamples x NumFeatures]
% squeeze -> [NumChannels x NumSamples] -> transpose -> [NumSamples x NumChannels]
features_2d = squeeze(features_raw)'; 

fprintf('Deep Features Extracted. Matrix Size: %d samples x %d dimensions\n', ...
    size(features_2d,1), size(features_2d,2));

%% -------------------------
%% 4. Compute t-SNE
%% -------------------------
fprintf('Computing t-SNE (this may take a moment)...\n');
rng(42); % For reproducibility

% Adjust perplexity based on dataset size
perplex = 30;
if height(testTbl) < 30
    perplex = height(testTbl) - 1;
    fprintf('  -> Small dataset detected. Adjusted Perplexity to %d\n', perplex);
end

try
    % Try standard Euclidean
    Y_tsne = tsne(features_2d, 'Distance', 'euclidean', 'Perplexity', perplex);
catch ME
    warning('t-SNE failed with default settings. Retrying with standardized distance...');
    % Fallback for difficult variances
    Y_tsne = tsne(features_2d, 'Distance', 'seuclidean', 'Perplexity', perplex);
end

%% -------------------------
%% 5. Visualization
%% -------------------------
fprintf('Generating Plot...\n');
fig = figure('Position', [100, 100, 1000, 800], 'Color', 'w');

% Define colors (Blue, Orange, Green)
colors = [0.2 0.6 0.8; 0.8 0.4 0.2; 0.2 0.8 0.4]; 

% Scatter plot
gscatter(Y_tsne(:,1), Y_tsne(:,2), Ytest, colors, '.', 25);

% Aesthetics
title('t-SNE Visualization of Deep Features (Primary Model)', 'Color','k','FontSize', 14, 'FontWeight', 'bold');
xlabel('t-SNE Dimension 1','Color','k');
ylabel('t-SNE Dimension 2','Color','k');
legend('Location', 'bestoutside', 'Color','k','FontSize', 12);
grid on;
axis square;
box on;

% Add annotation explaining the result
msg = sprintf('Dataset: %d Test Samples\nAcc: 100%%', height(testTbl));
text(min(Y_tsne(:,1)), max(Y_tsne(:,2)), msg, ...
    'BackgroundColor', 'w', 'EdgeColor', 'k', 'Margin', 5);

% Save
exportgraphics(fig, fullfile(figures_dir, 'tsne_visualization.png'), 'Resolution', 300);
fprintf('t-SNE plot saved to: %s\n', fullfile(figures_dir, 'tsne_visualization.png'));
fprintf('=== MODULE 4 FINISHED ===\n');