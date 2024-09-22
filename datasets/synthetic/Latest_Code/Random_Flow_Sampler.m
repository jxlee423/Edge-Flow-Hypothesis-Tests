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
                % Generate subgraph edges

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
                % Generate subgraph edges
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
                [Graphs.edge_to_endpoints{i,k},~] = ADJUSTED_fixed_degree_distribution_sampler(N,degree_dist);
                % Generate subgraph edges
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
            % Generate subgraph edges
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
            % Generate subgraph edges

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
            % Generate subgraph edges

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
            % Generate subgraph edges

        end
    end
end

%% store parameter grids
parameters.graph_param_grid = parameter_grid;

%% print
fprintf('\n\n Graphs Complete! \n')

base_path_newtxt = 'G:\2024 Summer\EFHT\Synthetic Data Sets\TESTING_NEWTXT';
base_path_subtxt = 'G:\2024 Summer\EFHT\Synthetic Data Sets\TESTING_SUBTXT';

for i = 1:prod(parameter_grid.size)
    for k = 1:n_real.graph
        if Graphs.valid(i,k) == 0
            Graphs.edge_to_endpoints{i,k} = nan;
            continue; 
        end
        
        input_file = fullfile(base_path_newtxt, sprintf('graph_%d_%d.txt', i, k));
        output_file_base = fullfile(base_path_subtxt, sprintf('output_%d_%d', i, k));
        
        generate_txt_from_edges(Graphs.edge_to_endpoints{i,k}, input_file);
        gtrieScanner_mex(input_file, output_file_base);
        
        output_file = sprintf('%s_sub.txt', output_file_base);
        Graphs.subgraph_edges{i,k} = subgraph_selecting(output_file, Graphs.edge_to_endpoints{i,k}, parameters.subgraph_mode);  % Store subgraph edges
    end
end


%% sample flows
[Flows, Effect_sizes, Cov_settings, Test_Results] = Flow_Sampler(Graphs,parameters);

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


Examples.Test_Results = Test_Results; %store test results

Examples.Effect_sizes = Effect_sizes; % store the maximum violation of the null family covariances
                                      % organized to match the indexing for
                                      % the flow array

%% print
fprintf('\n\n Flows Complete! \n')

end



function [Flows, Effect_sizes, Cov_settings, Test_Results] = Flow_Sampler(Graphs, parameters)

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

Test_Results.test1 = cell(size(Flows));  % Test 1
Test_Results.test2 = cell(size(Flows));  % Test 2
Test_Results.test3 = cell(size(Flows));  % Test 3

Effect_sizes = nan(n_rhos,n_averaging_params,3); % entries of each cell are max violation on the diagonal, off the diagonal disjoint, off the diagonal connected

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
            %% get subgraph_edges
            subgraph_edges = Graphs.subgraph_edges{i,k};

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
                    edge_max = 500; % ~ large enough to concentrate, small enough to be tractable
                    if E < edge_max
                        Averaging_M = ((1 - cov_averaging_weight)*speye(E,E) + cov_averaging_weight*diag(1./edge_degrees)*av_sign*A_edge)^(cov_dist - 1);
                        Cov = Averaging_M*Cov_0*Averaging_M';

                        %% compute effect size
                        Effect_sizes(j,l,:) = Find_effect_size(Cov,A_edge,Cov_0);

                        %% decompose
                        R = cholcov(Cov); % returns R such that R*R' = Cov
                        R = R';

                        %% sample
                        if strcmp(parameters.distribution,'Gaussian')
                            Flows{i,k,j,l} = R'*randn(E,parameters.n_real.flow);
                        elseif strcmp(parameters.distribution,'Laplace')
                            Flows{i,k,j,l} = R'*randl(E,parameters.n_real.flow);
                        elseif strcmp(parameters.distribution,'Student T')
                            Flows{i,k,j,l} = R'*trnd(3,E,parameters.n_real.flow)/sqrt(3);
                        elseif strcmp(parameters.distribution,'Uniform')
                            Flows{i,k,j,l} = sqrt(3)*R'*(2*rand(E,parameters.n_real.flow) - 1);
                        end
                    else
                        %% Compute covariance on subset of 1000 edges
                        Averaging_M_subset = ((1 - cov_averaging_weight)*speye(edge_max,edge_max) + cov_averaging_weight*diag(1./edge_degrees(1:edge_max))*av_sign*A_edge(1:edge_max,1:edge_max))^(cov_dist - 1);
                        Cov = Averaging_M_subset*Cov_0(1:edge_max,1:edge_max)*Averaging_M_subset';
                        Effect_sizes(j,l,:) = Find_effect_size(Cov,A_edge(1:edge_max,1:edge_max),Cov_0(1:edge_max,1:edge_max));

                        % %% decompose Cov 0
                        % R = cholcov(Cov_0); % returns R such that R*R' = Cov
                        % R = R';
                        % [~,rank] = size(R);
                        %
                        % %% sample
                        % Flows{i,k,j,l} = R*randn(rank,parameters.n_real.flow);

                        %% sample potentials
                        if strcmp(parameters.distribution,'Gaussian')
                            U = randn(V,parameters.n_real.flow);
                        elseif strcmp(parameters.distribution,'Laplace')
                            U = randl(V,parameters.n_real.flow);
                        elseif strcmp(parameters.distribution,'Student T')
                            U = trnd(3,V,parameters.n_real.flow)/sqrt(3);
                        elseif strcmp(parameters.distribution,'Uniform')
                            U = sqrt(3)*(2*rand(V,parameters.n_real.flow) - 1);
                        end

                        %% convert to flow
                        flow = G*U; % has covariance G*G' = A_edge + 2 I

                        %% perturb
                        Flows{i,k,j,l} = rho*flow + (1 - 2*rho)*randn(E,parameters.n_real.flow); % has covariance (1 - 2 rho) I + rho G*G'= I + rho A_edge

                        %% average
                        if cov_dist - 1 > 1
                            for products = 1:cov_dist - 1
                                Flows{i,k,j,l} = (1 - cov_averaging_weight)*Flows{i,k,j,l} + cov_averaging_weight*av_sign*sparse((1:E),(1:E),1./edge_degrees,E,E)*(A_edge*Flows{i,k,j,l});
                            end
                        end

                    end
                    %% calculate test-pass rate
                    % Test 1
                    pass_rate_test1 = perform_test(Flows{i, k, j, l}, subgraph_edges, parameters.test{1});
                    Test_Results.test1{i, k, j, l} = pass_rate_test1;
    
                    % Test 2
                    pass_rate_test2 = perform_test(Flows{i, k, j, l}, subgraph_edges, parameters.test{2});
                    Test_Results.test2{i, k, j, l} = pass_rate_test2;
    
                    % Test 3
                    pass_rate_test3 = perform_test(Flows{i, k, j, l}, subgraph_edges, parameters.test{3});
                    Test_Results.test3{i, k, j, l} = pass_rate_test3;


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


function effect_size = Find_effect_size(C,A_edge,Cov_0)

%% find number of edges
[E,~] = size(A_edge);

%% define family of matrices
D = @(sigma,rho) (sigma.^2).*(speye(E,E) + rho*A_edge);

%% intialize covariance fit
sigma_0 = sqrt(mean(diag(Cov_0)));
rho_0 = mean(mean(A_edge.*Cov_0))/sigma_0^2;
best_fit_param = [sigma_0,rho_0];

%% find maximum and minimum entries on and off diagonal
diag_range = [min(diag(C)),max(diag(C))];
covariances = A_edge(A_edge ~= 0).*C(A_edge ~= 0);
off_diag_range = [min(covariances),max(covariances)];

% %% loop over ps
% for p = 2:2:6
%     %% define objective
%     elementwise_p_norm = @(Cov_true,Cov_fit) full(sum(sum( abs(Cov_true - Cov_fit).^p ))).^(1/p);
%     objective = @(param_vec) elementwise_p_norm(C,D(param_vec(1),param_vec(2)));
% 
%     %% constraints (constraint assumes A*param <= b)
%     % A_constraint = [0,-1;0,1];
%     % b_constraint = [0;0.5];
% 
%     options = optimoptions(@fmincon,'Display', 'off');
% 
%     %% optimize
%     best_fit_param = fmincon(objective,best_fit_param,[],[],[],[],[0,0],[Inf,0.5],[],options); %%%%% WARNING: THIS EATS THE VAST MAJORITY OF THE RUNTIME (97 - 98%)
% end

%% optimize on desired loss directly
%objective = @(param_vec) max(max(abs(C - D(param_vec(1),param_vec(2)))));
objective = @(param_vec) max(abs([param_vec(1).^2 - diag_range(1),diag_range(2) - param_vec(1).^2,...
    param_vec(1).^2.*param_vec(2) - off_diag_range(1),off_diag_range(2) - param_vec(1).^2.*param_vec(2)]));

options = optimoptions(@fmincon,'Display', 'off');
best_fit_param = fmincon(objective,[sigma_0,rho_0],[],[],[],[],[0,0],[Inf,0.5],[],options);

%% return effect size
D_best = D(best_fit_param(1),best_fit_param(2));
discrepancy = abs(full(C - D_best)); 
off_diag_disc = diag(diag(D_best).^(-1/2))*(discrepancy - diag(diag(discrepancy)))*diag(diag(D_best).^(-1/2));

effect_size(1) = max(diag(discrepancy)./diag(D_best));
effect_size(2) = max(max(off_diag_disc(A_edge == 0)));
effect_size(3) = max(max(off_diag_disc(A_edge ~= 0)));

end


