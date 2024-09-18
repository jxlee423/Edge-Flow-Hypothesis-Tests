%% generate examples for edge flow hypotheses
clear

%% shared settings
% distribution
parameters.distribution = 'Gaussian';

% test
parameters.test = 'test1_seperated';
% parameters.test = 'test2_seperated';
% parameters.test = 'test3_seperated';
% parameters.test = 'test_united';

%subgraph_mode
parameters.subgraph_mode = '0001001101011110'; % Subgraph code

% default parameter ranges for covariance models
parameters.cov.rhos = [0,0.1,0.2,0.3,0.4,0.49];
parameters.cov.correlation_ds = [1,2,3,4,5];
parameters.cov.averaging_weights = linspace(0,0.5,5);
parameters.cov.averaging_signs = [-1,1];

% default sample sizes
parameters.n_real.graph = 4;
parameters.n_real.flow = 20; 



% Erdos Renyi
parameters.graph_mode = 'Erdos';

% other ranges to default?
parameters.default_ranges = 0;

% default parameter ranges for graph sampling
parameters.Ns = unique(round(10.^linspace(2,3,4))); % default: 10^2 to 10^3
parameters.degs = 10.^linspace(log(5)/log(10),log(20)/log(10),5); % default: parameterize using degree so that graphs stay sparse

% parameters for medium to medium large graphs
% parameters.Ns = unique(round(10.^linspace(3.333,4,3))); % large graphs range: 10^3 to 10^4
% parameters.degs = 14;

% sample
[Examples.Erdos_Renyi] = Random_Flow_Sampler(parameters);


save('Example_Flows_Erdos','Examples','-v7.3')
 
% % Small World
% parameters.graph_mode = 'Small World';
% 
% % other ranges to default?
% parameters.default_ranges = 0;
% 
% % default parameter ranges for graph sampling
% parameters.Ns = round(10.^linspace(2,3,4)); % size
% parameters.Ks = 10; % mean node degree/2, or E = K*N
% parameters.betas = [0,10.^(linspace(-1.25,-0.5,4))]; % rewiring probability (0 is perfect ring lattice, 1 is Erdos Renyi) controls degree of symmetry
% 
% % % % large graph sampling
% % parameters.Ns = round(10.^linspace(3.333,4,3)); % size
% % parameters.Ks = 10; % mean node degree/2, or E = K*N
% % parameters.betas = [0.2];
% 
% 
% % sample
% [Examples.Small_World] = Random_Flow_Sampler(parameters);
% 
% 
% save('Example_Flows_Small_World','Examples','-v7.3')



% % Scale Free
% parameters.graph_mode = 'Scale Free';
% 
% % other ranges to default?
% parameters.default_ranges = 0;
% 
% % default parameter ranges for graph sampling
% parameters.Ns = round(10.^linspace(2,3,4));
% parameters.d_mins = [5]; % minimum degree
% parameters.gammas = linspace(2,3,5); % decay rate in degree distribution
% 
% 
% % parameters.Ns = round(10.^linspace(3.333,4,3));
% % parameters.d_mins = [5]; % minimum degree
% % parameters.gammas = 3; % decay rate in degree distribution
% 
% 
% % sample
% [Examples.Scale_Free] = Random_Flow_Sampler(parameters);
% 
% 
% save('Example_Flows_Scale_Free','Examples','-v7.3')
