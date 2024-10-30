function [edge_to_endpoints,A] = fixed_degree_distribution_sampler(N,degree_dist);
%% Samples a connected Erdos-Renyi random graph with N nodes and connection probability p

%% Inputs
% 1. N: the number of nodes
% 2. degree_dist: the degree distribution, passed in as a vector of
% probabilities for each degree

%% outputs
% 1. edge_to_endpoints: a matrix whith a row for each edge, and each rows
% stores the endpoints of the edge

%% Method
% sample an initial degree for every node from the degree distribution,
% then assign each node as many stubs as its assigned degree
% randomly match the stub lists
% search for any self-stubs, if there are self-stubs start a rewiring
% process
% to rewire use ~ MCMC
% propose an initial graph by deleting one end of each stub that forms a
% self loop and assigning uniform randomly
% this defines an initial seed graph for the MCMC
% then run MCMC using uniform half edge reassignment on a small fraction of
% edges, and with likelihood of each graph given by the degree distribution
% check connectivity, reject if not connected and start over

%% create cdf
cdf = cumsum(degree_dist); 

%% loop until connected
stop = 0;
while stop == 0
    %% initialize
    edge_to_endpoints = [];

    %% sample degrees
    stub_assignment = [];
    degrees = NaN([N,1]);
    for j = 1:N
        r = rand([1,1]);
        degrees(j) = find(cdf >= r, 1, 'first');
        stub_assignment = [stub_assignment,j*ones([1,degrees(j)])];
    end
    if mod(sum(degrees),2) == 1
        stub_assignment(randperm(length(stub_assignment),1)) = [];
    end
    E = length(stub_assignment)/2;
    

    %% assign stubs
    % assign each stub to a node, keeping the degrees of nodes preserved
    stub_order = randperm(2*E,2*E);
    stub_assignment = stub_assignment(stub_order);
    
    % match stubs at random
    stub_list_left = randperm(E,E);
    stub_list_right = randperm(E,E);

    % make edge to endpoints
    edge_to_endpoints(:,1) = stub_assignment(stub_list_left);
    edge_to_endpoints(:,2) = stub_assignment(E + stub_list_right);

    %% find self loops
    self_loops = find(edge_to_endpoints(:,1) - edge_to_endpoints(:,2) == 0);

    %% rewire self loops for initial seed
    while isempty(self_loops) ~= 1
        for j = 1:length(self_loops)
            k = self_loops(j);
            node = edge_to_endpoints(k,1);
            attachment_p = degrees; % attach preferentially (for scale free)
            attachment_p(node) = 0; % don't self reattach
            attachment_p = attachment_p/sum(attachment_p);
            attachment_cdf = cumsum(attachment_p);

            r = rand([1,1]);
            edge_to_endpoints(k,2) = find(attachment_cdf >= r, 1, 'first');
        end
        self_loops = find(edge_to_endpoints(:,1) - edge_to_endpoints(:,2) == 0);
    end

    %% run MCMC
    stop_MCMC = 1; % only try this if we need it
    while stop_MCMC == 0
        %% evaluate likelihood


        %% propose update to network

        %% accept or reject

        %% check stopping
    end
       

    %% compute adjacency
    A = sparse(edge_to_endpoints(:,1),edge_to_endpoints(:,2),1,N,N) + ...
        sparse(edge_to_endpoints(:,2),edge_to_endpoints(:,1),1,N,N);

    %% check if connected
    [~,component_sizes] = conncomp(graph(A)); 
    if length(component_sizes) == 1
        stop = 1;
    end

end