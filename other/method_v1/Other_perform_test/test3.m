function pass = test3(subgraph_flows, subgraph, alpha)
    minUpperBound = intmax;
    maxLowerBound = intmin;
    n = size(subgraph_flows, 2);
    num_tests = (sum(subgraph, "all")) / 2;
    for i = 1:n-1
        for j = i+1:n
            if subgraph(i, j)
                ci = slope_ci(subgraph_flows(:, i), subgraph_flows(:, j), alpha / num_tests);
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
