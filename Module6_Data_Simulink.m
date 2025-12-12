%%% MODULE 6 - Prepare Models for Simulink
% ============================================================
% Author: Okoampah Ernest
% Date: 24/11/2025

% This script packs the Primary Model, Secondary Model, and
% Config into a single file 'simulink_data_package.mat'.
% This makes the Simulink MATLAB Function block much simpler.
% ============================================================
clearvars; clc;

% 1. Setup Paths
project_root = pwd; % Assuming running from 'scripts' folder
results_dir = fullfile(project_root, '..', 'results');
manifest_dir = fullfile(project_root, '..', 'manifests');

% 2. Load Components
fprintf('Loading components...\n');

% Load Config
conf_data = jsondecode(fileread(fullfile(manifest_dir, 'preprocessing_config.json')));

% Load Primary (Mel)
d1 = load(fullfile(results_dir, 'primary_model.mat'));
net_prim = d1.net_prim;
stats_prim.mu = d1.mu_prim;
stats_prim.sigma = d1.sigma_prim;

% Load Secondary (MFCC)
d2 = load(fullfile(results_dir, 'secondary_model.mat'));
net_sec = d2.net_sec;
stats_sec.mu = d2.mu_sec;
stats_sec.sigma = d2.sigma_sec;

% 3. Define Weights & Thresholds (Hardcoded for Simulink)
ensemble_weights = [0.62, 0.38]; % Primary, Secondary
class_names = {'background', 'chainsaw', 'engine'};

% 4. Pack into a structure
simulinkModel = struct();
simulinkModel.net_prim = net_prim;
simulinkModel.stats_prim = stats_prim;
simulinkModel.net_sec = net_sec;
simulinkModel.stats_sec = stats_sec;
simulinkModel.config = conf_data;
simulinkModel.weights = ensemble_weights;
simulinkModel.classes = class_names;

% 5. Save
outfile = fullfile(project_root, '..', 'results', 'simulink_data_package.mat');
save(outfile, 'simulinkModel');

fprintf('SUCCESS! Data packed to: %s\n', outfile);
fprintf('You can now point your Simulink function to this file.\n');