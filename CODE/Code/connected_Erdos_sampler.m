function [edge_to_endpoints] = connected_Erdos_sampler(N,p)
%% Samples a connected Erdos-Renyi random graph with N nodes and connection probability p

%% Inputs
% 1. N: the number of nodes
% 2. p: the connection probability

%% outputs
% 1. edge_to_endpoints: a matrix whith a row for each edge, and each rows
% stores the endpoints of the edge

%% preallocate
all_possible_edges = NaN([N*(N-1)/2,2]);
k = 0;
for i = 1:N
    for j = i+1:N
        k = k+1;
        all_possible_edges(k,:) = [i,j];
    end
end

%% loop until sample is connected
stop = 0;
while stop == 0
    %% sample adjacency
    edge_or_not = (rand([N*(N-1)/2,1]) <= p);
    edge_to_endpoints = all_possible_edges(edge_or_not,:);
    A = sparse(edge_to_endpoints(:,1),edge_to_endpoints(:,2),1,N,N) + ...
        sparse(edge_to_endpoints(:,2),edge_to_endpoints(:,1),1,N,N);

    %% check if connected
    [~,component_sizes] = conncomp(graph(A)); 
    if length(component_sizes) == 1
        stop = 1;
    end

end



end