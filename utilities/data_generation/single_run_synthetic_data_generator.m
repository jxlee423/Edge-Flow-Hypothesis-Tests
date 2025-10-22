% This is the "executor" function, which generates one set of data
% based on the input parameters structure.
function single_run_synthetic_data_generator(parameters)
    
    % =================================================================
    %  STEP 1: Automatic Naming and Path Setup
    % =================================================================
    
    % --- Generate baseName from the incoming parameters structure ---
    graph_mode_str = strrep(parameters.graph_mode, ' ', '');
    
    switch parameters.graph_mode
        case 'Small World'
            graph_params_str = sprintf('Beta%.1f_K%d', parameters.betas, parameters.Ks);
        case 'Stochastic Block'
            graph_params_str = sprintf('Nc%d_A%.d_B%.d', parameters.n_communities, parameters.a, parameters.b);
        otherwise
            graph_params_str = 'UnknownParams';
    end
    
    cov_params_str = sprintf('Rho%.1f_D%d_W%.1f', ...
        parameters.cov.rhos, ...
        parameters.cov.correlation_ds, ...
        parameters.cov.averaging_weights);
    if parameters.cov.averaging_signs == 1, sign_str = 'Pos'; else, sign_str = 'Neg'; end
    
    size_str = sprintf('%gkNodes', parameters.Ns/1000);
    % Use the manual_graph_id passed from the controller
    graph_id_str = sprintf('G%d', parameters.manual_graph_id);
    
    baseName = strjoin({graph_mode_str, graph_params_str, [cov_params_str, '_', sign_str], graph_id_str, size_str}, '-');
    
    fprintf('Generated baseName: %s\n', baseName);
    
    % --- Setup Paths ---
    current_folder = fileparts(mfilename('fullpath'));
    output_folder = fullfile(current_folder, '..', '..', 'data', 'synthetic', graph_mode_str, baseName);
    
    if ~exist(output_folder, 'dir')
       mkdir(output_folder);
       fprintf('Created output folder: %s\n', output_folder);
    else
       fprintf('Output folder already exists: %s\n', output_folder);
    end

    % =================================================================
    %  STEP 2: Execution Area
    % =================================================================
    
    % --- 1. Generate data ---
    fprintf('Calling Random_Flow_Sampler...\n');

    [Examples] = Random_Flow_Sampler(parameters);
    
    % --- 2. Extract data ---
    try
        edges = Examples.Graphs.edge_to_endpoints{1, 1};
        flows = Examples.Flows{1, 1};
    catch ME
        disp('ERROR: Failed to extract data. The structure of "Examples" is not as expected.');
        rethrow(ME);
    end
    
    % --- 3. Write CSV files ---
    num_flows = size(flows, 2);
    fprintf('Writing %d sample files...\n', num_flows);
    headers = {'node1', 'node2', 'flow'};
    
    for i = 1:num_flows
        output_filename = fullfile(output_folder, sprintf('%s_Sample%d.csv', baseName, i));
        current_data = [edges, flows(:, i)];
        output_table = array2table(current_data, 'VariableNames', headers);
        writetable(output_table, output_filename);
    end
    fprintf('Successfully saved all %d files for this run.\n', num_flows);
end