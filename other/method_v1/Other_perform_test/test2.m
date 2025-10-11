function pass = test2(subgraph_flows, subgraph, alpha)
    % subgraph_flows is a matrix of flows, where each column corresponds to an edge class.
    % subgraph is a matrix of 1s and 0s which describes the subgraph, where a 1 corresponds to edge i and j being connected, and 0 means they are disjoint.
    n = size(subgraph_flows, 2);
    num_tests = (n * n - sum(subgraph, "all")) / 2;
    for i = 1:n-1
        for j = i+1:n
            if ~subgraph(i, j)
                ci = slope_ci(subgraph_flows(:, i), subgraph_flows(:, j), alpha / num_tests);
                if ci(1) > 0 || ci(2) < 0
                    pass = false;
                    return
                end
            end
        end
    end
    pass = true;
end
