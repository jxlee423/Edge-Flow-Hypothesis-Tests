%% =================================================================
%  STEP 1: BATCH CONFIGURATION (The only place you need to edit)
% =================================================================
clear
clc

% --- 1. Choose graph model ---
% Options: 'Small World', 'Stochastic Block', or 'Scale Free'
% parameters.graph_mode = 'Small World'; 
% parameters.graph_mode = 'Stochastic Block';
parameters.graph_mode = 'Scale Free';

% --- 2. SHARED Fixed Parameters (Same for all models) ---
% distribution
parameters.distribution = 'Gaussian';

% cov
parameters.cov.rhos = 0.5;
parameters.cov.correlation_ds = 5;
parameters.cov.averaging_weights = 0.5; 
parameters.cov.averaging_signs = 1;

% sample
parameters.n_real.graph = 1;
parameters.n_real.flow = 50;

% Manual Graph ID (e.g., 'G1', 'G2')
manual_graph_id = 1;


% --- 3. MODEL-SPECIFIC Configuration (Grid & Fixed Params) ---
fprintf('Configuring batch for: %s\n', parameters.graph_mode);

if strcmp(parameters.graph_mode, 'Small World')

    % --- Define the grid of parameters to iterate over ---
    grid_parameters = {
        'Ns', [1000, 2000, 4000, 8000, 16000];
        'cov.rhos', [0, 0.1, 0.2, 0.3, 0.4, 0.5];
    };
    
    % --- Set Fixed Parameters for Small World ---
    parameters.default_ranges = 0;
    parameters.Ks = 13; % mean node degree/2
    parameters.betas = 0.7; % rewiring probability
    
elseif strcmp(parameters.graph_mode, 'Stochastic Block') 
    % --- Define the grid of parameters to iterate over ---
    grid_parameters = {
        'Ns', [64000, 128000];
        'cov.averaging_weights', [0,0.1,0.2,0.3,0.4];
    };

    % --- Set Fixed Parameters for Stochastic Block ---
    parameters.default_ranges = 0;
    parameters.a = 15;
    parameters.b = 1;
    parameters.n_communities = 3;
elseif strcmp(parameters.graph_mode, 'Scale Free')
    % --- Define the grid of parameters to iterate over ---
    grid_parameters = {
        'Ns', [250, 500, 1000, 2000, 4000, 8000, 16000];
        'd_mins', [2, 6, 8, 10];
    };
    
    % --- Set Fixed Parameters for Scale Free ---
    parameters.default_ranges = 0;
    parameters.gammas = 2.3;
    parameters.debug_scalefree = false;
    parameters.python_path = '"/Users/jiaxin/Documents/Summer 2024/EFHT/Edge-Flow-Hypothesis-Tests/.conda/bin/python3"'; % please change it to your own path
    
else
    error('Invalid GRAPH_MODEL_TO_RUN. Check STEP 1. Must be ''Small World'', ''Stochastic Block'', or ''Scale Free''.');
end

%% =================================================================
%  STEP 2: AUTOMATIC EXECUTION (No modification needed below)
% =================================================================

% --- Add function library path once ---
current_folder = fileparts(mfilename('fullpath'));
utilities_folder_path = fullfile(current_folder, '..', 'utilities', 'data_generation');
addpath(utilities_folder_path);
fprintf('Successfully added function library to path: \n\t%s\n', utilities_folder_path);

% --- Generate all possible combinations from the parameter grid ---
param_names = grid_parameters(:, 1);
param_values = grid_parameters(:, 2);
[grid_outputs{1:length(param_values)}] = ndgrid(param_values{:});
num_runs = numel(grid_outputs{1});

fprintf('\nStarting batch generation for %d total parameter combinations...\n', num_runs);
fprintf('==========================================================================\n');

% --- A single, generic loop to execute all jobs ---
for i = 1:num_runs
    
    % A. Start with the fixed parameters
    params_for_this_run = parameters;
    
    fprintf('\n--- RUN %d / %d ---\n', i, num_runs);
    
    % B. Dynamically set the variable parameters for this specific run
    for j = 1:length(param_names)
        current_value = grid_outputs{j}(i);
        eval(sprintf('params_for_this_run.%s = %s;', param_names{j}, num2str(current_value)));
        fprintf('  Setting %s = %s\n', param_names{j}, num2str(current_value));
    end
    
    % C. Add the manual graph ID
    params_for_this_run.manual_graph_id = manual_graph_id;
    
    % D. Call the "executor" function to do the actual work
    single_run_synthetic_data_generator(params_for_this_run);
    
end

fprintf('\n==========================================================================\n');
fprintf('🎉 Batch processing complete! All %d combinations were successfully processed.\n', num_runs);