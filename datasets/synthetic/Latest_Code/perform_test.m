function pass_rate = perform_test(flow_data, subgraph_edges, test_mode)
    switch test_mode
        case 'test1_seperated'
            pass_rate = perform_test1_seperated(flow_data, subgraph_edges);
        case 'test2_seperated'
            pass_rate = perform_test2_seperated(flow_data, subgraph_edges);
        case 'test3_seperated'
            pass_rate = perform_test3_seperated(flow_data, subgraph_edges);
        case 'test_united'
            pass_rate = perform_test_united(flow_data, subgraph_edges);
        case 'simple_test_debug'
            pass_rate = perform_test_simple_debug(flow_data, subgraph_edges);
        otherwise
            error('Unknown test mode: %s', test_mode);
    end
end

% Perform tests on flow data and return the pass rate
% Input:
%   flow_data - Flow data matrix
%   subgraph_edges - Subgraph edge information
% Output:
%   pass_rate - Test pass rate


%%test1_seperated
function pass_rate = perform_test1_seperated(flow_data, subgraph_edges)

num_tests = size(flow_data, 2); % Number of columns in flow data
num_passes = 0; % Count of tests passed

for col = 1:num_tests
    % Extract subgraph flows for the current column
    subgraph_flows = extract_subgraph_flows(flow_data(:, col), subgraph_edges);

    if ~isempty(subgraph_flows)
        % Compute covariance matrix
        cov_matrix = cov(subgraph_flows);

        % Test 1: Bootstrap the entire covariance matrix and compute the 95% confidence interval for diagonal elements
        ci_matrix = bootstrap_cov_matrix(cov_matrix);
        if overlap_diagonal
            num_passes = num_passes + 1; 
        end
    end
end
% Calculate pass rate
pass_rate = num_passes / num_tests;
end

%%test2_seperated
function pass_rate = perform_test2_seperated(flow_data, subgraph_edges)
        % Test 2: Compute the 95% confidence interval for the element in the first row and second column
num_tests = size(flow_data, 2); % Number of columns in flow data
num_passes = 0; % Count of tests passed

for col = 1:num_tests
    % Extract subgraph flows for the current column
    subgraph_flows = extract_subgraph_flows(flow_data(:, col), subgraph_edges);

    if ~isempty(subgraph_flows)
        % Compute covariance matrix
        cov_matrix = cov(subgraph_flows);
        ci_matrix = bootstrap_cov_matrix(cov_matrix);
        ci_off_diag_1_2 = ci_matrix(1, 2, :);
        if in_interval(0, ci_off_diag_1_2(:))
            num_passes = num_passes + 1; 
        end
    end
end
% Calculate pass rate
pass_rate = num_passes / num_tests;
end

%%test3_seperated
function pass_rate = perform_test3_seperated(flow_data, subgraph_edges)
    %for test3_separated
end


%%test_united
function pass_rate = perform_test_united(flow_data, subgraph_edges)
num_tests = size(flow_data, 2); % Number of columns in flow data
num_passes = 0; % Count of tests passed

for col = 1:num_tests
    % Extract subgraph flows for the current column
    subgraph_flows = extract_subgraph_flows(flow_data(:, col), subgraph_edges);

    if ~isempty(subgraph_flows)
        % Compute covariance matrix
        cov_matrix = cov(subgraph_flows);

        % Test 1: Bootstrap the entire covariance matrix and compute the 95% confidence interval for diagonal elements
        ci_matrix = bootstrap_cov_matrix(cov_matrix);
        % Check if the confidence intervals of the diagonal elements overlap
        overlap_diagonal = check_overlap(diagonal_ci(ci_matrix));
        if ~overlap_diagonal
            continue;
        end

        % Test 2: Compute the 95% confidence interval for the element in the first row and second column
        ci_off_diag_1_2 = ci_matrix(1, 2, :);
        if ~in_interval(0, ci_off_diag_1_2(:))
            continue;
        end

        % Test 3: Compute 95% confidence intervals for other selected elements
        selected_elements = [ci_matrix(1, 3, :); ci_matrix(1, 4, :); ci_matrix(2, 3, :); ci_matrix(2, 4, :); ci_matrix(3, 4, :)];
        ci_selected_normalized = normalize_by_overlap(selected_elements, diagonal_ci(ci_matrix));

        % Check if the calculated intervals overlap and intersect with [0, 0.5)
        overlap_selected = check_overlap(ci_selected_normalized);
        if overlap_selected && in_interval([0, 0.5], ci_selected_normalized)
            num_passes = num_passes + 1;
        end
    end
end

% Calculate pass rate
pass_rate = num_passes / num_tests;
end


%%simple_test_debug
function pass_rate = perform_test_simple_debug(flow_data, subgraph_edges)
num_tests = size(flow_data, 2); % Number of columns in flow data
num_passes = 0; % Count of tests passed

for col = 1:num_tests
    % Extract subgraph flows for the current column
    subgraph_flows = extract_subgraph_flows(flow_data(:, col), subgraph_edges);

    if ~isempty(subgraph_flows)
    num_passes = num_passes + 1;    
    end
end
% Calculate pass rate
pass_rate = num_passes / num_tests;
end


%%some functions to support test

function subgraph_flows = extract_subgraph_flows(flow_column, subgraph_edges)
    % Extract flow data of the subgraph
    % Input:
    %   flow_column - A column of flow data
    %   subgraph_edges - Subgraph edge information
    % Output:
    %   subgraph_flows - Flow data of the subgraph

    subgraph_flows = [];

    % Get the size of the flow_column
    flow_size = size(flow_column, 1);  % Define flow_size as the number of rows in flow_column

    % For each subgraph, match and extract its flow data
    for i = 1:size(subgraph_edges, 1)
        edge_indices = subgraph_edges(i, :); % Extract edge indices of the subgraph

        % Ensure all indices are positive integers and within the valid range
        if all(edge_indices > 0) && all(edge_indices <= flow_size)
            subgraph_flow = flow_column(edge_indices);  % Extract corresponding flow data
            subgraph_flows = [subgraph_flows; subgraph_flow'];
        end
    end

end

function ci_matrix = bootstrap_cov_matrix(cov_matrix)
    % Bootstrap each element of the covariance matrix to compute the 95% confidence interval
    % Input:
    %   cov_matrix - Covariance matrix
    % Output:
    %   ci_matrix - 95% confidence interval matrix for each element

    num_bootstrap = 1000;
    [rows, cols] = size(cov_matrix);
    ci_matrix = zeros(rows, cols, 2); % Preallocate the confidence interval matrix

    for i = 1:rows
        for j = 1:cols
            %bootstrao all elements in cov_matrix
            bootstrap_samples = bootstrp(num_bootstrap, @(x) x(i,j), cov_matrix);
            ci_matrix(i, j, :) = prctile(bootstrap_samples, [2.5, 97.5]);
        end
    end
end


function diagonal_ci = diagonal_ci(ci_matrix)
    % Extract the confidence intervals of the diagonal elements
    % Input:
    %   ci_matrix - 95% confidence interval matrix for each element
    % Output:
    %   diagonal_ci - Confidence interval matrix for diagonal elements

    diagonal_ci = zeros(size(ci_matrix, 1), 2);
    for i = 1:size(ci_matrix, 1)
        diagonal_ci(i, :) = ci_matrix(i, i, :);
    end
end

function overlap = check_overlap(ci_matrix)
    % Check if confidence intervals overlap
    % Input:
    %   ci_matrix - Confidence interval matrix, each row an interval
    % Output:
    %   overlap - Returns true if there is overlap; otherwise returns false

    overlap = true;
    for i = 1:size(ci_matrix, 1) - 1
        for j = i+1:size(ci_matrix, 1)
            if ci_matrix(i, 2) < ci_matrix(j, 1) || ci_matrix(i, 1) > ci_matrix(j, 2)
                overlap = false;
                return;
            end
        end
    end
end

function normalized_ci = normalize_by_overlap(ci_matrix, overlap_ci)
    % Normalize confidence intervals by the overlap part
    % Input:
    %   ci_matrix - Original confidence intervals
    %   overlap_ci - Overlap part of the confidence interval from test one
    % Output:
    %   normalized_ci - Normalized confidence intervals

    overlap_length = overlap_ci(2) - overlap_ci(1);
    normalized_ci = (ci_matrix - overlap_ci(1)) / overlap_length;
end

function in_interval_flag = in_interval(value, ci)
    % Check if a value or interval is within another interval
    % Input:
    %   value - Value or interval to check
    %   ci - Confidence interval
    % Output:
    %   in_interval_flag - Returns true if within the interval; otherwise returns false

    if numel(value) == 2
        in_interval_flag = (value(1) < ci(2)) && (value(2) > ci(1));
    else
        in_interval_flag = (value > ci(1)) && (value < ci(2));
    end
end
