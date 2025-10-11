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
