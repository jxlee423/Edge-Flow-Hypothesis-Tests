function [edge_to_endpoints,A] = degree_distribution_sampler(N,degree_dist,debug,python_path)
%% Samples a connected Erdos-Renyi random graph with N nodes and connection probability p

%% Inputs
% 1. N: the number of nodes
% 2. degree_dist: the degree distribution, passed in as a vector of
% probabilities for each degree

%% outputs
% 1. edge_to_endpoints: a matrix whith a row for each edge, and each rows
% stores the endpoints of the edge

%% Method
% 1. Sample an initial degree for every node from the degree distribution.
% 2. Assign to each node as many stubs as its sampled degree.
% 3. Randomly match the stubs in pairs to obtain a (multi)graph, which may
%    contain self-loops and multi-edges.
% 4. Call a degree-preserving rewiring procedure (implemented in Python)
%    to remove self-loops and multi-edges, yielding a simple graph.
% 5. Construct the adjacency matrix A from the resulting edge list and
%    compute the actual degrees.
% 6. Check that all degrees lie in the support of the prescribed
%    degree distribution and that the graph is connected.
% 7. If these conditions are not met, restart from step 1.
    if nargin < 3
        debug = false;
    end

    if nargin < 4
        python_path = 'python'; 
    end

    %% 1. Preprocessing: CDF, d_min
    cdf = cumsum(degree_dist);
    max_support_deg = numel(degree_dist);
    d_min = find(degree_dist > 0, 1, 'first');

    %% 2. Outer loop: repeat until a connected graph is obtained
    stop = 0;
    outer_iter = 0;

    while stop == 0
        outer_iter = outer_iter + 1;
        if debug
            fprintf('\n[ITER %d] Entering outer loop\n', outer_iter);
        end

        %% 2.1 Sample degrees according to degree_dist and build stub list
        t_sampling = tic;
        stub_assignment = [];
        degrees_sampled = NaN(N, 1);

        for j = 1:N
            r = rand();
            degrees_sampled(j) = find(cdf >= r, 1, 'first');
            stub_assignment = [stub_assignment, j * ones(1, degrees_sampled(j))];
        end

        % Ensure that the total number of stubs is even
        if mod(numel(stub_assignment), 2) == 1
            idx_del = randi(numel(stub_assignment));
            stub_assignment(idx_del) = [];
        end

        E_target = numel(stub_assignment) / 2;
        t_sampling = toc(t_sampling);
        if debug
            fprintf('[ITER %d] Sampling degrees and constructing stubs took %.3f seconds, target number of edges E_target = %d\n', ...
                outer_iter, t_sampling, E_target);
        end

        %%---------------------------------------------------
        %% 2.2 Randomly match all stubs (allowing self-loops and multi-edges)
        %%---------------------------------------------------
        t_pair = tic;
        stub_order = randperm(numel(stub_assignment));
        stub_assignment = stub_assignment(stub_order);
        edge_to_endpoints = zeros(E_target, 2);
        for e = 1:E_target
            edge_to_endpoints(e, 1) = stub_assignment(2*e - 1);
            edge_to_endpoints(e, 2) = stub_assignment(2*e);
        end
        E = E_target;

        t_pair = toc(t_pair);
        if debug
            fprintf('[ITER %d] Initial random stub pairing (with self-loops and multi-edges allowed) took %.3f seconds, initial number of edges E = %d\n', ...
                outer_iter, t_pair, E);
        end

        %%---------------------------------------------------
        %% 2.3 Degree-preserving rewiring: remove self-loops and multi-edges
        %%      (implemented in Python)
        %%---------------------------------------------------
        t_rewire = tic;
        
        % 1. Save data for Python
        try
            save('temp_data.mat', 'N', 'E_target', 'edge_to_endpoints', 'debug', 'outer_iter');
        catch ME
            fprintf('[ITER %d] MATLAB error: failed to save temp_data.mat: %s\n', outer_iter, ME.message);
            continue; % Restart outer loop
        end
        
        % 2. Invoke Python script
        if debug
             fprintf('[ITER %d] Calling Python script (rewire.py)...\n', outer_iter);
        end
        
        % python_path = '"/Users/jiaxin/Documents/Summer 2024/EFHT/Edge-Flow-Hypothesis-Tests/.conda/bin/python3"';

        % 2. Build the system command
        current_func_dir = fileparts(mfilename('fullpath'));
        rewire_script_full_path = fullfile(current_func_dir, 'rewire.py');
        % command_str = "/path/to/python" "/path/to/rewire.py"
        command_str = sprintf('%s "%s"', python_path, rewire_script_full_path);
        
        if debug
             fprintf('[ITER %d] Calling Python via ...\n', outer_iter);
             fprintf('[ITER %d] CMD: %s\n', outer_iter, command_str);
        end
        
        [status, cmdout] = system(command_str);
        
        if debug
            fprintf('--- Python stdout begins ---\n');
            disp(cmdout);
            fprintf('--- Python stdout ends ---\n');
        end
        
        if status ~= 0
            fprintf('[ITER %d] Python script execution failed (status = %d).\n', ...
                outer_iter, status);
            continue;
        end
        
        % 3. Load the result from Python
        try
            load('result_data.mat', 'cleaned_edges', 'success');
        catch ME
            fprintf('[ITER %d] MATLAB error: failed to load result_data.mat: %s\n', outer_iter, ME.message);
            continue;
        end
        
        % 4. Check whether Python rewiring succeeded
        if ~success
             fprintf('[ITER %d] Python rewiring exceeded iteration limit. Restarting outer loop.\n', outer_iter);
             continue;
        end
        
        % Python succeeded: update edge_to_endpoints
        edge_to_endpoints = double(cleaned_edges);
        
        t_rewire = toc(t_rewire);
        if debug
            fprintf('[ITER %d] Python rewiring total time = %.3f seconds\n', outer_iter, t_rewire);
        end

        %%---------------------------------------------------
        %% 2.4 Construct adjacency matrix A, compute degrees,
        %%     and check that degrees lie within the support
        %%---------------------------------------------------
        t_adj = tic;

        A = sparse(edge_to_endpoints(:,1), edge_to_endpoints(:,2), 1, N, N);
        A = A + A.';

        degrees = full(sum(A, 2));
        deg_min_obs = min(degrees);
        deg_max_obs = max(degrees);

        t_adj = toc(t_adj);
        if debug
            fprintf('[ITER %d] Constructing adjacency matrix and computing degrees took %.3f seconds, min(deg) = %d, max(deg) = %d\n', ...
                outer_iter, t_adj, deg_min_obs, deg_max_obs);
        end

        if any(degrees < d_min) || any(degrees > max_support_deg)
            if debug
                fprintf('[ITER %d] At least one node has degree outside [%d, %d]. Restarting outer loop.\n', ...
                    outer_iter, d_min, max_support_deg);
            end
            continue;
        end

        %%---------------------------------------------------
        %% 2.5 Check connectivity
        %%---------------------------------------------------
        t_conn = tic;

        G = graph(A);
        [~, component_sizes] = conncomp(G);

        t_conn = toc(t_conn);
        if debug
            fprintf('[ITER %d] conncomp took %.3f seconds, number of connected components = %d\n', ...
                outer_iter, t_conn, numel(component_sizes));
        end

        if numel(component_sizes) == 1
            stop = 1;
            if debug
                fprintf('[ITER %d] Graph is connected. Sampling succeeded.\n', outer_iter);
            end
        else
            if debug
                fprintf('[ITER %d] Graph is not connected. Restarting outer loop.\n', outer_iter);
            end
            stop = 0;
        end

    end

end