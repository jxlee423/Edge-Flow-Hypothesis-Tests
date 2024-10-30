function pass = testAll(subgraph_flows, subgraph, alpha)
    pass = test1(subgraph_flows, alpha / 3) && test2(subgraph_flows, subgraph, alpha / 3) && test3(subgraph_flows, subgraph, alpha / 3);
end
