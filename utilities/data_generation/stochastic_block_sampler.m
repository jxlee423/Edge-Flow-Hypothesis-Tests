function [edge_list, classes] = stochastic_block_sampler(community_size_list, P)
% STOCHASTIC_BLOCK_SAMPLER - Generates a graph based on Stochastic Block Model parameters.
%
% Inputs:
%   community_size_list: A row vector where each element represents the number of nodes in a community.
%                        e.g., [50, 60, 70] means there are 3 communities with sizes 50, 60, and 70.
%   P:                   A K x K probability matrix (where K is the number of communities),
%                        where P(i, j) is the probability of an edge existing between a node in community i and a node in community j.
%
% Outputs:
%   edge_list:           An M x 2 matrix (where M is the total number of edges in the graph). Each row [u, v] represents an edge between node u and node v.
%   classes:             An N x 1 column vector (where N is the total number of nodes). classes(i) indicates the community ID for node i.

%% 1. Initialize parameters and community assignment vector
n_communities = length(community_size_list); % Number of communities
N = sum(community_size_list); % Total number of nodes in the graph

% Create the classes vector to store the community assignment for each node.
% For example, if community_size_list = [3, 2], classes will be [1; 1; 1; 2; 2].
classes = repelem(1:n_communities, community_size_list)';

%% 2. Generate the adjacency matrix A
A = sparse(N, N);

% Iterate over all unique pairs of nodes (i, j) where i < j to avoid self-loops and duplicate checks.
for i = 1:N
    for j = i + 1:N
        % Get the communities to which nodes i and j belong.
        community_i = classes(i);
        community_j = classes(j);
        
        % Get the connection probability from the probability matrix P.
        p_connection = P(community_i, community_j);
        
        % Decide whether to connect the two nodes based on the probability.
        if rand() < p_connection
            A(i, j) = 1;
            A(j, i) = 1;
        end
    end
end

%% 3. Convert the adjacency matrix to an edge list format
% The find function locates the indices of all non-zero elements in a matrix.
% triu(A) takes the upper triangular part of the adjacency matrix to ensure each edge is listed only once.
[row, col] = find(triu(A));
edge_list = [row, col];

end