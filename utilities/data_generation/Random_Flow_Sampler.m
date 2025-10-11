function [Examples] = Random_Flow_Sampler(parameters)
%% Random graph and edge flow generator
% samples a sequence of random graphs
% builds empirical spectrum (Laplacian and normalized Laplacian)
% computes empirical complexity

%% Inputs: parameters struct
% this is a structure with fields for graph parameters, sample sizes, and
% covariance parameters

%% Graph parameters
% 1. parameters.graph_mode = 'Erdos', 'Small World', 'Scale Free', 'Block
% Stochastic','Near Complete Bipartite','One-D Spatial', or 'Two-D Spatial'
% sets the family of random graphs
% 2. parameters.default_ranges = Boolean, if equals 1, then the parameter
% ranges are chosen by default

% 3. parameters.Ns = set of graph sizes (number of nodes) to generate

% (for Erdos) 
% 4. parameters.degs = set average degrees


% (for small world model/Watts Strogatz)
% 4. parameters.Ks = mean node degree/2 
% 5. parameters.betas = rewiring probability (0 is perfect ring lattice, 1 is Erdos Renyi) controls degree of symmetry

% (for scale free)
% 4. parameters.d_mins = minimum degrees
% 5. parameters.gammas = decay rate in degree distribution

% (for block stochastic)
% 4. parameters.n_communities =  number of communities
% 5. parameters.expected_community_sizes = expected community sizes
% 6. parameters.p_intras = within community connection probabilities
% 7. parameters.p_inters = across community connection probabilities

% (for near complete bipartite)
% 3. parameters.Ns = size of community 1
% 4. parameters.Ms = size of community 2
% 5. parameters.p_inters = probability of intercommunity connection (should
% be near to 1
% 6. parameters.p_intras = probability of intracommunity connection (should
% be near to zero)

% (for one-D and two-D spatial models)
% 4. parameters.sigmas = bandwidth for kernel used to build connections
% (should be positive valued, values >> 1 produce near to complete graphs,
% values << 1 produce spatially organized graphs with a large diameter

%% Covariance parameters
% will generate covariances from the power series family
% will use the iterated averaging model with initial flows sampled from the
% null model (trait-performance/first order power series)
% will average out to a distance d, with averaging weight chosen by the
% user

% 1. parameters.cov.rhos = correlation coefficients before iterated
% averaging (null)
% 2. parameters.cov.correlation_ds = max correlation distance in the
% edge graph
% 3. parameters.cov.averaging_weights = parameter controlling the
% decay rate in correlation with distance, equivalently, the degree of
% averaging in the iterated averaging model
% 4. parameters.cov.averaging_signs = parameter controlling whether we
% average towards or away from transitive flows (use + or - A_edge), should
% be 1 or -1



%% Realization Count Parameters
% 1. parameters.n_real.graph = number of graph topologies to sample per
% graph family
% 2. parameters.n_real.flow = number of flows to sample per covariance and
% graph


if parameters.default_ranges == 1
    n_real.graph = 4; % small since graph generation is expensive, and results concentrate for large graphs
    n_real.flow = 100; % large since cheap to do at scale
else
    n_real.graph = parameters.n_real.graph;
    n_real.flow = parameters.n_real.flow;
end


%% draw the graphs
if strcmp(parameters.graph_mode,'Erdos')
    % parameters
    if parameters.default_ranges == 1
        parameters.Ns = unique(round(10.^linspace(2,4,5)));
        parameters.degs = 10.^linspace(log(2)/log(10),log(20)/log(10),6);
    end

    [parameter_grid.Ns,parameter_grid.degs] = meshgrid(parameters.Ns,parameters.degs);
    parameter_grid.size = [length(parameters.Ns),length(parameters.degs)];

    % sample
    for i = 1:prod(parameter_grid.size)
        %% print progress
        fprintf('\n Generating graphs for parameter set %d of %d',i,prod(parameter_grid.size))

        %% get parameters
        N = parameter_grid.Ns(i);
        p = min(parameter_grid.degs(i)/(N - 1),1);
        fprintf('\n    %d Nodes, %d Edges',N,p*N*(N-1)/2);

        if p >= log(N)/N
            for k = 1:n_real.graph
                Graphs.valid(i,k) = 1;
                Graphs.edge_to_endpoints{i,k} = connected_Erdos_sampler(N,p);
            end
        else % avoid sampling cases when the graph will almost never be connected
            for k = 1:n_real.graph
                Graphs.valid(i,k) = 0;
                Graphs.edge_to_endpoints{i,k} = nan;
            end
        end
    end

elseif strcmp(parameters.graph_mode,'Small World') % (Wattz Strogatz) standard random graph family for social networks, forms clusters, has short average path length
    % parameters
    if parameters.default_ranges == 1
        parameters.Ns = round(10.^linspace(2,3,4)); % size
        parameters.Ks = 10; % mean node degree/2, or E = K*N
        parameters.betas = [0,10.^(linspace(-1.25,-0.5,4))]; % rewiring probability (0 is perfect ring lattice, 1 is Erdos Renyi) controls degree of symmetry
    end

    [parameter_grid.Ns,parameter_grid.Ks,parameter_grid.betas] = meshgrid(parameters.Ns,parameters.Ks,parameters.betas);
    parameter_grid.size = [length(parameters.Ns),length(parameters.Ks),length(parameters.betas)];

    % sample
    for i = 1:prod(parameter_grid.size)
        %% print progress
        fprintf('\n Generating graphs for parameter set %d of %d',i,prod(parameter_grid.size))

        %% get parameters
        N = parameter_grid.Ns(i);
        K = parameter_grid.Ks(i);
        beta = parameter_grid.betas(i);

        if K <= N-1 && K/(N-1) >= log(N)/N
            for k = 1:n_real.graph
                Graphs.valid(i,k) = 1;
                graph_struct = WattsStrogatz(N,K,beta);
                Graphs.edge_to_endpoints{i,k} = table2array(graph_struct.Edges);
            end
        else % avoid sampling cases when the graph will almost never be connected
            for k = 1:n_real.graph
                Graphs.valid(i,k) = 0;
                Graphs.edge_to_endpoints{i,k} = nan;
            end
        end
    end

elseif strcmp(parameters.graph_mode,'Scale Free') % standard random graph family for social networks, power law degree distribution, use the BA model to get a power law with exponent 3
    % parameters
    if parameters.default_ranges == 1
        parameters.Ns = round(10.^linspace(log(80)/log(10),log(600)/log(10),10));
        parameters.d_mins = [5]; % minimum degree
        parameters.gammas = linspace(2,3,10); % decay rate in degree distribution
    end

    [parameter_grid.Ns,parameter_grid.d_mins,parameter_grid.gammas] = meshgrid(parameters.Ns,parameters.d_mins,parameters.gammas);
    parameter_grid.size = [length(parameters.Ns),length(parameters.d_mins),length(parameters.gammas)];

    % sample
    for i = 1:prod(parameter_grid.size)
        %% print progress
        fprintf('\n Generating graphs for parameter set %d of %d',i,prod(parameter_grid.size))

        %% get parameters
        N = parameter_grid.Ns(i);
        d_min = parameter_grid.d_mins(i);
        gamma = parameter_grid.gammas(i);

        degree_dist = (1:N-1).^(-gamma); % ~ Pareto
        degree_dist(1:d_min-1) = 0;
        degree_dist = degree_dist/sum(degree_dist);

        if d_min <= N-1
            for k = 1:n_real.graph
                Graphs.valid(i,k) = 1;
                [Graphs.edge_to_endpoints{i,k},~] = fixed_degree_distribution_sampler(N,degree_dist);
            end
        else
            for k = 1:n_real.graph
                Graphs.valid(i,k) = 0;
                Graphs.edge_to_endpoints{i,k} = nan;
            end
        end
    end

elseif strcmp(parameters.graph_mode,'Block Stochastic') % community structure, clustered spectrum?
    % parameters
    if parameters.default_ranges == 1
        parameters.n_communities = 5;% number of communities
        parameters.expected_community_sizes = 80; % expected community sizes
        parameters.p_intras = 0.975; % within community connection p
        parameters.p_inters = 0.025; % across community connection p
    end

    [parameter_grid.p_intras,parameter_grid.p_inters,parameter_grid.n_communities,parameter_grid.expected_community_sizes] =...
        ndgrid(parameters.p_intras,parameters.p_inters,parameters.n_communities,parameters.expected_community_sizes);
    parameter_grid.size = [length(parameters.p_inters),length(parameters.p_intras),...
        length(parameters.n_communities),length(parameters.expected_community_sizes)];

    % sample
    for i = 1:prod(parameter_grid.size)
        %% print progress
        fprintf('\n Generating graphs for parameter set %d of %d',i,prod(parameter_grid.size))

        %% get parameters
        n_communities = parameter_grid.n_communities(i);
        community_size_list = round(2*parameter_grid.expected_community_sizes(i)*rand(1,parameters.n_communities));
        p_inter = parameter_grid.p_inters(i);
        p_intra = parameter_grid.p_intras(i);

        P = (p_intra - p_inter)*eye(n_communities,n_communities) + p_inter*ones(n_communities,n_communities); % block connection probabilities

        for k = 1:n_real.graph
            Graphs.valid(i,k) = 1;
            [Graphs.edge_to_endpoints{i,k},classes{i}] = stochastic_block_sampler(community_size_list,P);
        end
    end

elseif strcmp(parameters.graph_mode,'Near Complete Bipartite') % cross community structure, near symmetry, clustered spectrum?
    % pick size of subgraphs, or size of graph and probability of falling into either
    % pick cross subgraph and within subgraph connection probability (1,0)
    % for complete bipartite
    
    % parameters
    if parameters.default_ranges == 1
        parameters.Ns = round(10.^linspace(1,log(150)/log(10),2)); % size
        parameters.Ms = round(10.^linspace(1,log(150)/log(10),2)); % size
        parameters.p_inters = 0.95 + linspace(0,0.05,5); % almost complete across
        parameters.p_intras = 0 + linspace(0,0.05,3); % not many connections within
    end


    [parameter_grid.p_inters,parameter_grid.p_intras,parameter_grid.Ns,parameter_grid.Ms] =...
        ndgrid(parameters.p_inters,parameters.p_intras,parameters.Ns,parameters.Ms);
    parameter_grid.size = [length(parameters.p_inters),length(parameters.p_intras),...
        length(parameters.Ns),length(parameters.Ms)];

    % sample
    for i = 1:prod(parameter_grid.size)
        %% print progress
        fprintf('\n Generating graphs for parameter set %d of %d',i,prod(parameter_grid.size))

        %% get parameters
        N = parameter_grid.Ns(i);
        M = parameter_grid.Ms(i);
        p_inter = parameter_grid.p_inters(i);
        p_intra = parameter_grid.p_intras(i);

        for k = 1:n_real.graph
            Graphs.valid(i,k) = 1;
            Graphs.edge_to_endpoints{i,k} = near_complete_bipartite_sampler(N,M,p_inter,p_intra);
        end
    end

elseif strcmp(parameters.graph_mode,'One-D Spatial') % sample along line with gaussian kernel connection
    % parameters
    if parameters.default_ranges == 1
        parameters.Ns = round(10.^linspace(1,log(200)/log(10),18));
        parameters.sigmas = round(10.^linspace(log(5)/log(10),log(100)/log(10),23));
    end

    [parameter_grid.Ns,parameter_grid.sigmas] = meshgrid(parameters.Ns,parameters.sigmas);
    parameter_grid.size = [length(parameters.Ns),length(parameters.sigmas)];

    % sample
    for i = 1:prod(parameter_grid.size)
        %% print progress
        fprintf('\n Generating graphs for parameter set %d of %d',i,prod(parameter_grid.size))

        %% get parameters
        N = parameter_grid.Ns(i);
        sigma = parameter_grid.sigmas(i);

        for k = 1:n_real.graph
            Graphs.valid(i,k) = 1;
            Graphs.edge_to_endpoints{i,k} = Line_Gaussian_Kernel_sampler(N,sigma);
        end
    end

elseif strcmp(parameters.graph_mode,'Two-D Spatial') % sample on lattice with gaussian kernel connection
    % parameters
    if parameters.default_ranges == 1
        parameters.Ns = round(10.^linspace(1,log(200)/log(10),10));
        parameters.Ns = round(sqrt(parameters.Ns)).^2;
        parameters.sigmas = round(10.^linspace(log(3)/log(10),log(10)/log(10),10));
    end

    [parameter_grid.Ns,parameter_grid.sigmas] = meshgrid(parameters.Ns,parameters.sigmas);
    parameter_grid.size = [length(parameters.Ns),length(parameters.sigmas)];

    % sample
    for i = 1:prod(parameter_grid.size)
        %% print progress
        fprintf('\n Generating graphs for parameter set %d of %d',i,prod(parameter_grid.size))

        %% get parameters
        N = parameter_grid.Ns(i);
        sigma = parameter_grid.sigmas(i);

        for k = 1:n_real.graph
            Graphs.valid(i,k) = 1;
            Graphs.edge_to_endpoints{i,k} = Grid_Gaussian_Kernel_sampler(N,sigma);
        end
    end
end

%% store parameter grids
parameters.graph_param_grid = parameter_grid;

%% print
fprintf('\n\n Graphs Complete! \n')


%% sample flows
[Flows,Cov_settings] = Flow_Sampler(Graphs,parameters);



%% structure output struct
Examples.input_parameters = parameters; % store inputs
Examples.Graph_settings = parameter_grid; % stores the settings for the random graph
%                                           organized as a look up for the
%                                           graph parameters given the
%                                           first index of the flows array
%                                           (stores parameters in grid,
%                                           call with a single index for
%                                           vectorized version)

Examples.Cov_settings = Cov_settings; % store the parameter settings corresponding to each graph
%                                       organized as a look up to find covariance parameters given the
%                                       third and fourth index of the flow array
%                                       Note: the null is true look-up
%                                       table only needs the last array
%                                       index of the flows cell

Examples.Graphs = Graphs; 

Examples.Flows = Flows;   % store the example flows. Cell array. 
%                           First index is graph parameter settings. 
%                           Second index is graph realization
%                           Third index is correlation for null model
%                           Fourth index is parameters for averaging model
%                           used to break null model

%% print
fprintf('\n\n Flows Complete! \n')

end




function [Flows,Cov_settings] = Flow_Sampler(Graphs,parameters)

%% set defaults
if parameters.default_ranges == 1
    parameters.cov.rhos = [0,0.1,0.2,0.3,0.4,0.5];
    parameters.cov.correlation_ds = [1,2,3,5,10];
    parameters.cov.averaging_weights = linspace(0,0.5,5);
end

%% reformat parameters
[n_graph_params,n_graph_real] = size(Graphs.edge_to_endpoints);


n_rhos = length(parameters.cov.rhos);
n_averaging_params = length(parameters.cov.correlation_ds)*length(parameters.cov.averaging_weights)*length(parameters.cov.averaging_signs);
[cov_ds_grid,cov_averaging_grid,cov_signs_grid] = meshgrid(parameters.cov.correlation_ds,parameters.cov.averaging_weights,parameters.cov.averaging_signs);

%% preallocate space for output
Flows = cell(n_graph_params,n_graph_real,n_rhos,n_averaging_params);

Cov_settings.rhos = nan(n_rhos,n_averaging_params);
Cov_settings.distances = nan(n_rhos,n_averaging_params);
Cov_settings.weights = nan(n_rhos,n_averaging_params);
Cov_settings.null_is_true = nan([n_averaging_params,1]);

%% draw the flows
for i = 1:n_graph_params
     %% print progress
     fprintf('\n Generating flows for graph set %d of %d',i,n_graph_params)

    for k = 1:n_graph_real
        fprintf('\n    Generating flows for graph %d of %d',k,n_graph_real)

        %% check if valid
        if Graphs.valid(i,k) == 1
            %% extract graph
            edge_to_endpoints = Graphs.edge_to_endpoints{i,k};
            V = parameters.graph_param_grid.Ns(i);
            E = length(edge_to_endpoints(:,1));

            fprintf('\n    %d Nodes, %d Edges',V,E);

            %% build gradient
            G = sparse((1:E),edge_to_endpoints(:,1),1,E,V) - sparse((1:E),edge_to_endpoints(:,2),1,E,V);

            %% build Laplacian
            Laplacian = G*G';

            %% build edge adjacency
            A_edge = Laplacian - 2*speye(E,E);

            %% loop over rhos
            for j = 1:n_rhos
                fprintf('\n       Generating flows for correlation %d of %d',j,n_rhos)
                rho = parameters.cov.rhos(j);

                %% build initial covariance
                Cov_0 = speye(E,E) + rho*A_edge;

                %% loop over averaging parameter grid
                for l = 1:n_averaging_params
                    % fprintf('\n          Generating flows for perturbation %d of %d',l,n_averaging_params)
                    cov_dist = cov_ds_grid(l);
                    cov_averaging_weight = cov_averaging_grid(l);
                    av_sign = cov_signs_grid(l);

                    %% compute edge degrees
                    edge_degrees = sum(abs(A_edge));

                    %% compute covariance after the iterated averaging procedure
                    if E < 500
                        Averaging_M = ((1 - cov_averaging_weight)*speye(E,E) + cov_averaging_weight*diag(1./edge_degrees)*av_sign*A_edge)^(cov_dist - 1);
                        Cov = Averaging_M*Cov_0*Averaging_M';

                        %% decompose
                        R = cholcov(Cov); % returns R such that R*R' = Cov
                        R = R';
                        [~, rank_of_Cov] = size(R);

                        %% sample
                        if strcmp(parameters.distribution,'Gaussian')
                            Flows{i,k,j,l} = R*randn(rank_of_Cov,parameters.n_real.flow);
                        elseif strcmp(parameters.distribution,'Laplace')
                            Flows{i,k,j,l} = R*randl(rank_of_Cov,parameters.n_real.flow);
                        elseif strcmp(parameters.distribution,'Student T')
                            Flows{i,k,j,l} = R*trnd(3,rank_of_Cov,parameters.n_real.flow)/sqrt(3);
                        elseif strcmp(parameters.distribution,'Uniform')
                            Flows{i,k,j,l} = sqrt(3)*R*(2*rand(rank_of_Cov,parameters.n_real.flow) - 1);
                        end
                    else
                        % sample potentials
                        if strcmp(parameters.distribution,'Gaussian')
                            U = randn(V,parameters.n_real.flow);
                        elseif strcmp(parameters.distribution,'Laplace')
                            U = randl(V,parameters.n_real.flow);
                        elseif strcmp(parameters.distribution,'Student T')
                            U = trnd(3,V,parameters.n_real.flow)/sqrt(3);
                        elseif strcmp(parameters.distribution,'Uniform')
                            U = sqrt(3)*(2*rand(V,parameters.n_real.flow) - 1);
                        end
                        
                        % convert to flow
                        flow = G*U; % has covariance G*G' = A_edge + 2 I
                        
                        % perturb
                        Flows{i,k,j,l} = sqrt(rho)*flow + sqrt(1 - 2*rho)*randn(E,parameters.n_real.flow); % has covariance (1 - 2 rho) I + rho G*G'= I + rho A_edge
                        % average
                        if cov_dist - 1 > 1
                            for products = 1:cov_dist - 1
                                Flows{i,k,j,l} = (1 - cov_averaging_weight)*Flows{i,k,j,l} + cov_averaging_weight*av_sign*sparse((1:E),(1:E),1./edge_degrees,E,E)*(A_edge*Flows{i,k,j,l});
                            end
                        end
                    end

                    %% store settings
                    if cov_dist <= 1 || cov_averaging_weight == 0
                        Cov_settings.null_is_true(l) = 1;
                    else
                        Cov_settings.null_is_true(l) = 0;
                    end
                    Cov_settings.rhos(j,l) = rho;
                    Cov_settings.distances(j,l) = cov_dist;
                    Cov_settings.weights(j,l) = cov_averaging_weight;
                end
            end
        end
    end
end
end

