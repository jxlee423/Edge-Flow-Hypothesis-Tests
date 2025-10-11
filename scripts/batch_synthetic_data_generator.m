%% =================================================================
%  STEP 1: BATCH CONFIGURATION (The only place you need to edit)
% =================================================================
clear
clc

% --- Define the grid of parameters to iterate over ---
% Format: {'field.name.in.parameters.struct', [array of values]}

grid_parameters = {
    'Ns', [1000, 2000, 4000, 8000, 16000];
    'cov.averaging_weights', [0, 0.1, 0.2, 0.3, 0.4]
};

% !!! Remember to commment parameters you already put in the grid parameters.

% --- Set Fixed Parameters that are the same for all runs ---
% distribution
parameters.distribution = 'Gaussian';

% covariance models
parameters.cov.rhos = 0.5;
parameters.cov.correlation_ds = 5;
% parameters.cov.averaging_weights = 0.4;
parameters.cov.averaging_signs = 1;

%% Graph Settings

% Small World graph parameters
parameters.graph_mode = 'Small World';
parameters.default_ranges = 0;
% parameters.Ns = 1000; % size
parameters.Ks = 13; % mean node degree/2, or E = K*N
parameters.betas = 0.7; % rewiring probability


parameters.n_real.graph = 1;
parameters.n_real.flow = 50;

% Please set the number for the 'G' part of the filename yourself.
% For example, if you already have a G1 and want to create G2, set this to 2.
manual_graph_id = 1;

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