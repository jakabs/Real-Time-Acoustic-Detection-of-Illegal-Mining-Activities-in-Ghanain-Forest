%% MODULE 4b - Physics-Informed Logic Visualization
% ============================================================
% Author: Okoampah Ernest
% Description: 
%   Visualizes the "Physics Check" logic by plotting 
%   Bass Energy vs. Treble Energy for all test samples.
%   Shows clearly how 'Engines' (Bass) separate from 'Chainsaws' (Treble).
% ============================================================
clearvars; clc; close all;
fprintf('=== MODULE 4b: Physics-Informed Visualization ===\n');

% 1. Setup
scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder), scriptFolder = pwd; end
project_root = fullfile(scriptFolder, '..');
results_dir  = fullfile(project_root, 'results');
figures_dir  = fullfile(project_root, 'figures');
load(fullfile(results_dir, 'data_splits.mat'), 'testTbl', 'classes', 'Ytest');

% 2. Extract Energy Metrics from Mel Spectrograms
fprintf('Calculating Spectral Energies for %d samples...\n', height(testTbl));

bass_energies = [];
treble_energies = [];
labels = [];

% Mel Band Approximation (64 bands total, 50Hz - 8000Hz)
% Bass (50-600Hz) approx Bands 1 to 15
% Treble (1500-8000Hz) approx Bands 30 to 64
bass_bands = 1:15;
treble_bands = 30:64;

for i = 1:height(testTbl)
    filepath = fullfile(project_root, testTbl.filepath{i});
    if ~exist(filepath, 'file'), continue; end
    
    d = load(filepath, 'hybrid_features');
    mel = d.hybrid_features.primary_mel; % [64 x 128]
    
    % Undo log (to get raw power approx)
    pwr = 10.^mel; 
    
    % Calculate Summed Energy
    total_bass   = sum(sum(pwr(bass_bands, :)));
    total_treble = sum(sum(pwr(treble_bands, :)));
    
    bass_energies = [bass_energies; total_bass];
    treble_energies = [treble_energies; total_treble];
    labels = [labels; testTbl.label(i)];
end

% 3. Generate Scatter Plot
fig = figure('Position', [100, 100, 1000, 800], 'Color', 'w');
hold on; grid on;

% Define Colors
c_back = [0.2 0.6 0.8]; % Blue
c_saw  = [0.8 0.4 0.2]; % Orange
c_eng  = [0.2 0.8 0.4]; % Green

% Plot Classes
idx_back = strcmp(labels, 'background');
scatter(treble_energies(idx_back), bass_energies(idx_back), 50, c_back, 'filled', 'o', 'MarkerFaceAlpha', 0.6);

idx_saw = strcmp(labels, 'chainsaw');
scatter(treble_energies(idx_saw), bass_energies(idx_saw), 50, c_saw, 'filled', '^', 'MarkerFaceAlpha', 0.6);

idx_eng = strcmp(labels, 'engine');
scatter(treble_energies(idx_eng), bass_energies(idx_eng), 50, c_eng, 'filled', 's', 'MarkerFaceAlpha', 0.6);

% 4. Draw Logic Threshold Zones
% The logic compares Ratio = Bass / Treble
% Ratio = 1.5  => Bass = 1.5 * Treble
% Ratio = 0.5  => Bass = 0.5 * Treble

x_lims = xlim; y_lims = ylim;
max_val = max([x_lims(2), y_lims(2)]);

% Zone 1: Engine Zone (Bass Dominant > 1.5x)
fill([0, max_val/1.5, 0], [0, max_val, max_val], c_eng, 'FaceAlpha', 0.1, 'EdgeColor', 'k');
text(max_val*0.1, max_val*0.9, 'Physics Zone: ENGINE\n(Bass Dominant)', 'Color', 'k', 'FontWeight','bold');

% Zone 2: Chainsaw Zone (Treble Dominant < 0.5x)
fill([0, max_val, max_val], [0, 0, max_val*0.5], c_saw, 'FaceAlpha', 0.1, 'EdgeColor', 'k');
text(max_val*0.6, max_val*0.1, 'Physics Zone: CHAINSAW\n(Treble Dominant)', 'Color', 'k', 'FontWeight','bold');

% Labels
xlabel('Treble Energy (High Freq)', 'FontSize', 12, 'FontWeight', 'bold', 'Color','k');
ylabel('Bass Energy (Low Freq)', 'FontSize', 12, 'FontWeight', 'bold','Color','k');
title('Physics-Informed Spectral Separation', 'FontSize', 16, 'FontWeight', 'bold','Color','k');
legend({'Background', 'Chainsaw', 'Engine'}, 'Location', 'best', 'Color','k');

% Save
saveas(fig, fullfile(figures_dir, 'figure_physics_separation.png'));
fprintf('Physics Plot saved: figure_physics_separation.png\n');