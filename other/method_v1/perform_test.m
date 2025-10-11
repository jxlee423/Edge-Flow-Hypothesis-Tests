function pass_rate = perform_test(flow_data, subgraph_edges, test_modes)
% If test_modes is a single string, convert it into a cell array
if ischar(test_modes)
    test_modes = {test_modes};
end

need_covariance_bootstrap = any(ismember(test_modes, {'test1_seperated', 'test2_seperated', 'test3_with_test1', 'test_united'}));
need_alpha_subgraph = any(ismember(test_modes, {'test1', 'test2', 'test3', 'testAll'}));

num_tests = size(flow_data, 2); % Number of columns in the flow data
num_modes = length(test_modes);
num_passes = zeros(1, num_modes); % Initialize pass count

% for test that needs alpha_subgraph
if need_alpha_subgraph
    alpha = 0.05;
    subgraph = [1 0 1 1;
        0 1 1 1;
        1 1 1 1;
        1 1 1 1];
end

for col = 1:num_tests
    % Extract subgraph flows
    subgraph_flows = extract_subgraph_flows(flow_data(:, col), subgraph_edges);

    if ~isempty(subgraph_flows)
        % for test that needs covariance_bootstrap
        if need_covariance_bootstrap
            % Compute covariance matrix and bootstrap confidence interval matrix only once
            ci_matrix = bootstrap_cov_matrix(subgraph_flows);
            % fprintf('Pause for No.%d of %d tests\n', col,num_tests);
        end

        % Perform corresponding test for each test mode
        for idx = 1:num_modes
            test_mode = test_modes{idx};
            switch test_mode
                case 'test1_seperated'
                    pass = perform_test1_seperated(ci_matrix);
                case 'test2_seperated'
                    pass = perform_test2_seperated(ci_matrix);
                case 'test3_with_test1'
                    pass = perform_test3_with_test1(ci_matrix);
                case 'test_united'
                    pass = perform_test_united(ci_matrix);
                case 'test1'
                    pass = test1(subgraph_flows, alpha);
                case 'test2'
                    pass = test2(subgraph_flows, subgraph, alpha);
                case 'test3'
                    pass = test3(subgraph_flows, subgraph, alpha);
                case 'testAll'
                    pass = testAll(subgraph_flows, subgraph, alpha);
                otherwise
                    error('unknown_test_mode: %s', test_mode);
            end
            if pass
                num_passes(idx) = num_passes(idx) + 1;
            end
        end
    end
end
% Calculate the pass rate for each test mode
pass_rate = num_passes / num_tests;
% disp('pass_rate:');
% disp(pass_rate);
end

% Perform tests on flow data and return the pass rate
% Input:
%   flow_data - Flow data matrix
%   subgraph_edges - Subgraph edge information
% Output:
%   pass_rate - Test pass rate

%%test1_seperated
function pass = perform_test1_seperated(ci_matrix)
    diag_ci = diagonal_ci(ci_matrix);
    overlap_diagonal = check_overlap(diag_ci);
    pass = overlap_diagonal;
end

%%test2_seperated
function pass = perform_test2_seperated(ci_matrix)
% Extract the confidence interval of the first row and second column
    ci_off_diag_1_2 = ci_matrix(1, 2, :);
        % Check if 0 is within this confidence interval
    pass = in_interval(0, ci_off_diag_1_2(:));
end

%%test3_with_test1
function pass = perform_test3_with_test1(ci_matrix)
    % First perform test 1
    diag_ci = diagonal_ci(ci_matrix);  % Get the confidence interval of the diagonal elements
    [overlap_diagonal, overlap_ci] = check_overlap(diag_ci);  % Check if there is overlap and get the overlapping interval
    
    if ~overlap_diagonal
        pass = false;
        return;
    end

    % Then perform test 3
    % Select the confidence intervals of relevant covariance matrix elements
    selected_elements = [ci_matrix(1, 3, :); ci_matrix(1, 4, :); ci_matrix(2, 3, :); ci_matrix(2, 4, :); ci_matrix(3, 4, :)];

    % Normalize these elements based on the overlap interval obtained from test 1
    ci_selected_normalized = normalize_by_overlap(selected_elements, overlap_ci);

    [overlap_selected, overlap_selected_ci] = check_overlap(ci_selected_normalized);
    % Check if the normalized intervals overlap, and whether the overlap is within [0, 0.5]
    if overlap_selected
        pass = in_interval([0, 0.5], overlap_selected_ci);
    else
        pass = false;
    end
end

%%test_united
function pass = perform_test_united(ci_matrix)
    diag_ci = diagonal_ci(ci_matrix);
    overlap_diagonal = check_overlap(diag_ci);
    if ~overlap_diagonal
        pass = false;
        return;
    end

    % Extract the confidence interval of the first row and second column
    ci_off_diag_1_2 = ci_matrix(1, 2, :);
        % Check if 0 is within this confidence interval
    in_interval_flag = in_interval(0, ci_off_diag_1_2(:));
    if ~in_interval_flag
        pass = false;
        return;
    end

    % Select the confidence intervals of relevant covariance matrix elements
    selected_elements = [ci_matrix(1, 3, :); ci_matrix(1, 4, :); ci_matrix(2, 3, :); ci_matrix(2, 4, :); ci_matrix(3, 4, :)];

    % Normalize these elements based on the overlap interval obtained from test 1
    ci_selected_normalized = normalize_by_overlap(selected_elements, overlap_ci);

    [overlap_selected, overlap_selected_ci] = check_overlap(ci_selected_normalized);
    % Check if the normalized intervals overlap, and whether the overlap is within [0, 0.5]
    if overlap_selected
        pass = in_interval([0, 0.5], overlap_selected_ci);
    else
        pass = false;
    end

end

function pass = test1(subgraph_flows, alpha)
    % subgraph_flows is a matrix, where each column corresponds to an edge class.
    n = size(subgraph_flows, 2);
    for i = 1:n-1
        for j = i+1:n
            xs = subgraph_flows(:, i);
            ys = subgraph_flows(:, j);
            %u = xs + ys;
            %disp(u(:))
            ci = slope_ci(xs + ys, xs - ys, alpha / (n * (n + 1) / 2));
            if (ci(1) > 0) || (ci(2) < 0)
                pass = false;
                return
            end
        end
    end
    pass = true;
end

function pass = test2(subgraph_flows, subgraph, alpha)
    % subgraph_flows is a matrix of flows, where each column corresponds to an edge class.
    % subgraph is a matrix of 1s and 0s which describes the subgraph, where a 1 corresponds to edge i and j being connected, and 0 means they are disjoint.
    n = size(subgraph_flows, 2);
    n_tests = (n * n - sum(subgraph, "all")) / 2;
    for i = 1:n-1
        for j = i+1:n
            if ~subgraph(i, j)
                ci = slope_ci(subgraph_flows(:, i), subgraph_flows(:, j), alpha / n_tests);
                if ci(1) > 0 || ci(2) < 0
                    pass = false;
                    return
                end
            end
        end
    end
    pass = true;
end

function pass = test3(subgraph_flows, subgraph, alpha)
    minUpperBound = intmax;
    maxLowerBound = intmin;
    n = size(subgraph_flows, 2);
    n_tests = (sum(subgraph, "all")) / 2;
    for i = 1:n-1
        for j = i+1:n
            if subgraph(i, j)
                ci = slope_ci(subgraph_flows(:, i), subgraph_flows(:, j), alpha / n_tests);
                if ci(2) < minUpperBound
                    minUpperBound = ci(2);
                end
                if ci(1) > maxLowerBound
                    maxLowerBound = ci(1);
                end
            end
        end
    end
    pass = minUpperBound > maxLowerBound;
end

function pass = testAll(subgraph_flows, subgraph, alpha)
    pass = test1(subgraph_flows, alpha / 3) && test2(subgraph_flows, subgraph, alpha / 3) && test3(subgraph_flows, subgraph, alpha / 3);
end





% %%simple_test_debug
% %%This test is for debugging if the whole program can run properly instead of performing extensive calculations.
% %%To use this, enable the quick_debug test option in Flow_Example.m (don't forget to disable other options).
% function pass_rate = perform_test_simple_debug(flow_data, subgraph_edges)
% num_tests = size(flow_data, 2); % Number of columns in flow data
% num_passes = 0; % Count of tests passed
% 
% for col = 1:num_tests
%     % Extract subgraph flows for the current column
%     subgraph_flows = extract_subgraph_flows(flow_data(:, col), subgraph_edges);
% 
%     if ~isempty(subgraph_flows)
%     num_passes = num_passes + 1;    
%     end
% end
% % Calculate pass rate
% pass_rate = num_passes / num_tests;
% end

%%Supporting functions for tests

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
    % disp('Current subgraph_flows:');
    % disp(subgraph_flows);
end

function ci_matrix = bootstrap_cov_matrix(subgraph_flows)
    % Input:
    %   subgraph_flows - Original subgraph flow data
    % Output:
    %   ci_matrix - 95% confidence interval for each covariance value
    num_bootstrap = 500;
    [n, p] = size(subgraph_flows); % n is the number of samples, p is the number of variables
    bootstrap_cov_values = zeros(p, p, num_bootstrap); % Store each covariance matrix

    % Perform bootstrap
    for i = 1:num_bootstrap
        % Random sampling with replacement
        resample_indices = randsample(n, n, true); 
        resample_data = subgraph_flows(resample_indices, :);
        
        % Compute covariance matrix and store
        bootstrap_cov_values(:, :, i) = cov(resample_data);
    end

    % Initialize confidence interval matrix to store upper and lower bounds for each covariance value
    ci_matrix = zeros(p, p, 2);

    % Calculate 95% confidence interval for each covariance value
    for j = 1:p
        for k = 1:p
            % Extract the (j, k) element from all bootstrap samples
            cov_samples = squeeze(bootstrap_cov_values(j, k, :));
            
            % Calculate the 2.5% and 97.5% percentiles as the 95% confidence interval
            ci_matrix(j, k, :) = prctile(cov_samples, [2.5, 97.5]);
        end
    end
    % % Display the confidence intervals
    % disp('95% Confidence Intervals for Covariance Matrix:');
    % for j = 1:p
    %     for k = 1:p
    %         fprintf('Covariance between edge %d and edge %d: [%.5f, %.5f]\n', j, k, ci_matrix(j, k, 1), ci_matrix(j, k, 2));
    %     end
    % end
end

function diagonal_ci = diagonal_ci(ci_matrix)
% Extract the confidence intervals of the diagonal elements
    diagonal_ci = zeros(size(ci_matrix, 1), 2);
    for i = 1:size(ci_matrix, 1)
        diagonal_ci(i, :) = ci_matrix(i, i, :);
    end
end


function [overlap, overlap_ci] = check_overlap(ci_matrix)
    % Check if confidence intervals overlap and return the overlapping part
    % Input:
    %   ci_matrix - Confidence interval matrix, each row an interval
    % Output:
    %   overlap - Returns true if there is overlap; otherwise returns false
    %   overlap_ci - The overlap part of the intervals, or an empty array if no overlap

    overlap = true;
    overlap_start = -inf;  % Initial left boundary of the overlap interval
    overlap_end = inf;     % Initial right boundary of the overlap interval

    for i = 1:size(ci_matrix, 1)
        overlap_start = max(overlap_start, ci_matrix(i, 1));  % Update the left boundary
        overlap_end = min(overlap_end, ci_matrix(i, 2));      % Update the right boundary
        if overlap_start > overlap_end
            overlap = false;  % If the left boundary exceeds the right, there is no overlap
            overlap_ci = [];  % Return an empty overlap interval
            return;
        end
    end
    
    overlap_ci = [overlap_start, overlap_end];  % Return the overlap interval
end

function normalized_ci = normalize_by_overlap(ci_matrix, overlap_ci)
    % Normalize confidence intervals by the overlap part
    % Input:
    %   ci_matrix - Original confidence intervals
    %   overlap_ci - Overlap part of the confidence interval from test one
    % Output:
    %   normalized_ci - Normalized confidence intervals

    overlap_length = overlap_ci(2) - overlap_ci(1);  % Length of the overlap interval
    normalized_ci = (ci_matrix - overlap_ci(1)) / overlap_length;  % Normalize
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

function conf_interval_b1 = slope_ci(xs, ys, alpha)
    n = length(xs);

    % Create a column of ones and concatenate with xs
    ones_column = ones(size(xs, 1), 1);
    xs_column = xs(:); % Reshape xs to a column vector

    X = [ones_column, xs_column]; % Concatenate

    % Compute the matrix XtX and its inverse
    XtX = X' * X;
    c_matrix = inv(XtX);

    % Initialize H matrix
    H = zeros(n, 1);

    % Compute H values
    for i = 1:n
        H(i) = X(i, :) * (c_matrix * X(i, :)');
    end

    % Compute average of the diagonal elements of H
    h_bar = mean(H);

    % Compute e_ii and d_ii for each i
    e = H / h_bar;
    d = min(4, e);

    % Compute the Ordinary Least Squares (OLS) estimate

    beta_ols = inv(X' * X) * (X' * ys(:));

    y_hat = X * beta_ols;
    residuals = ys(:) - y_hat;

    % Construct diagonal matrix A
    A_diag = residuals.^2 .* (1 - H).^-d;
    A = diag(A_diag);
    
    % Calculate matrix S
    S = c_matrix * (X' * (A * (X * c_matrix)));

    % Extract standard errors for intercept and slope
    S_0 = sqrt(S(1, 1));  % Standard error for intercept
    S_1 = sqrt(S(2, 2));  % Standard error for slope

    % Calculate confidence interval for the slope (b1)
    t_value = tinv(1 - alpha / 2, n - 2);  % t critical value
    b1 = beta_ols(2);
    conf_interval_b1 = [b1 - t_value * S_1, b1 + t_value * S_1];

end


