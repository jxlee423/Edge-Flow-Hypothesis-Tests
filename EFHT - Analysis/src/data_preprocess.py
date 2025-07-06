import pandas as pd
import networkx as nx
import numpy as np
import copy
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict

# Preliminary classification
def Classifying(df):
    """
    Classifies edges based on two metrics: "edge betweenness" and "sum of degrees".
    - Class 1: Edges where both edge betweenness and sum of degrees are in the top 25%.
    - Class 2: Edges where both edge betweenness and sum of degrees are in the bottom 25%.
    """
    df_class = df.copy()
    
    # 1. Preprocessing: ensure node1 < node2
    for index, row in df_class.iterrows():
        u, v = row["node1"], row["node2"]
        if u > v:
            df_class.at[index, "node1"] = v
            df_class.at[index, "node2"] = u
            
    # 2. Create graph and calculate the two core metrics
    G = nx.Graph()
    G = nx.from_pandas_edgelist(df_class, 'node1', 'node2')

    # --- Metric 1: Edge Betweenness ---
    edge_betweenness = nx.edge_betweenness_centrality(G)
    def _get_betweenness(row):
        u, v = row['node1'], row['node2']
        return edge_betweenness.get((u, v), edge_betweenness.get((v, u), 0.0))
    df_class['betweenness'] = df_class.apply(_get_betweenness, axis=1)

    # --- Metric 2: Sum of Degrees ---
    node_degrees = dict(G.degree())
    def _get_edge_degree(row):
        return node_degrees.get(row['node1'], 0) + node_degrees.get(row['node2'], 0)
    df_class['degree'] = df_class.apply(_get_edge_degree, axis=1)
    
    # 3. Determine the Top 75% and Bottom 25% thresholds
    betweenness_top = df_class['betweenness'].quantile(0.75)
    betweenness_bottom = df_class['betweenness'].quantile(0.25)
    
    degree_top = df_class['degree'].quantile(0.75)
    degree_bottom = df_class['degree'].quantile(0.25)
    
    # 4. Define classification masks based on the dual conditions
    # Condition for Class 1: both betweenness and degree must be at or above the 75th percentile
    mask_class1 = (df_class['betweenness'] >= betweenness_top) & \
                  (df_class['degree'] >= degree_top)
                  
    # Condition for Class 2: both betweenness and degree must be at or below the 25th percentile
    mask_class2 = (df_class['betweenness'] <= betweenness_bottom) & \
                  (df_class['degree'] <= degree_bottom)
                  
    # 5. Filter data and assign classes
    # First, only keep rows that satisfy either class condition
    df_filtered = df_class[mask_class1 | mask_class2].copy()
    
    # Then, assign class labels on the filtered data
    # If an edge's betweenness is above the high threshold, it must be Class 0 (because we have already filtered)
    df_filtered['class'] = np.where(
        df_filtered['betweenness'] >= betweenness_top,
        0,  # Assign as Class 1
        1   # Otherwise, assign as Class 2
    )
    
    return df_filtered

# Select edges for KS test
def KS_Data_Preprocessing(df):
    np.random.seed(423) 
    
    class_data = defaultdict(lambda: {'edges': [], 'flows': []})
    for _, row in df.iterrows():
        cls = row['class']
        edge = (row['node1'], row['node2'])
        class_data[cls]['edges'].append(edge)
        class_data[cls]['flows'].append(row['flow'])
    
    result = {0: [], 1: []} 
    used_nodes = set()
    
    while True:
        available = {
            0: [i for i, e in enumerate(class_data[0]['edges']) 
                if e[0] not in used_nodes and e[1] not in used_nodes],
            1: [i for i, e in enumerate(class_data[1]['edges']) 
                if e[0] not in used_nodes and e[1] not in used_nodes]
        }
        
        remain_0 = len(available[0])
        remain_1 = len(available[1])
        
        if remain_0 + remain_1 == 0:
            break
            
        if remain_0 <= remain_1 and remain_0 > 0:
            target_cls = 0
        elif remain_1 < remain_0 and remain_1 > 0:
            target_cls = 1
        else:
            target_cls = 0 if remain_0 > 0 else 1
        
        success = False
        np.random.shuffle(available[target_cls])  
        
        for idx in available[target_cls]:
            u, v = class_data[target_cls]['edges'][idx]
            if u not in used_nodes and v not in used_nodes:
                result[target_cls].append(class_data[target_cls]['flows'][idx])
                used_nodes.update([u, v])
                success = True
                break
                
        if not success:
            alt_cls = 1 if target_cls == 0 else 0
            for idx in available[alt_cls]:
                u, v = class_data[alt_cls]['edges'][idx]
                if u not in used_nodes and v not in used_nodes:
                    result[alt_cls].append(class_data[alt_cls]['flows'][idx])
                    used_nodes.update([u, v])
                    success = True
                    break
            if not success:
                break  

    df_class0_KS = pd.DataFrame({'flow': [abs(f) for f in result[0]]})
    df_class1_KS = pd.DataFrame({'flow': [abs(f) for f in result[1]]})
    df_class0_ANOVA = pd.DataFrame({'flow': [f for f in result[0]]})
    df_class1_ANOVA = pd.DataFrame({'flow': [f for f in result[1]]})
    return df_class0_KS, df_class1_KS, df_class0_ANOVA, df_class1_ANOVA

# Select edge pairs for Bivariate Test
def BEDT_Data_Preprocessing(df,dfall):
    np.random.seed(423) 
    edge_info = {}
    class_edges = defaultdict(list)
    node_map = defaultdict(set)
    
    for idx, row in df.iterrows():
        u, v = sorted([row['node1'], row['node2']])
        edge_info[idx] = (u, v)
        cls = row['class']
        class_edges[cls].append(idx)
        node_map[u].add(idx)
        node_map[v].add(idx)
    
    # Generate candidate pairs for each class and sort them internally
    class0_candidates = []
    class1_candidates = []
    
    for cls in class_edges:
        edges = class_edges[cls]
        cls_pairs = []
        for i in range(len(edges)):
            e1 = edges[i]
            u1, v1 = edge_info[e1]
            connected = node_map[u1].union(node_map[v1])
            for e2 in connected:
                if e2 <= e1 or df.loc[e2, 'class'] != cls:
                    continue
                u2, v2 = edge_info[e2]
                if {u1, v1}.isdisjoint({u2, v2}):
                    continue
                pair = tuple(sorted((e1, e2)))
                cls_pairs.append((pair, cls))
        
        # Sort by the number of shared nodes in descending order within the class
        cls_pairs.sort(key=lambda x: -len(set(edge_info[x[0][0]] + edge_info[x[0][1]])))
        
        if cls == 0:
            class0_candidates = cls_pairs
        else:
            class1_candidates = cls_pairs

    # Process candidate pairs, prioritizing the class with fewer selected pairs
    i0 = 0
    i1 = 0
    selected_0 = 0
    selected_1 = 0
    used_nodes = set()
    used_edges = set()
    selected = []
    
    while i0 < len(class0_candidates) or i1 < len(class1_candidates):
        # Determine which class to process next
        current_cls = None
        if selected_0 <= selected_1:
            if i0 < len(class0_candidates):
                current_cls = 0
            else:
                current_cls = 1 if i1 < len(class1_candidates) else None
        else:
            if i1 < len(class1_candidates):
                current_cls = 1
            else:
                current_cls = 0 if i0 < len(class0_candidates) else None
        
        if current_cls is None:
            break
        
        if current_cls == 0:
            current_pair, cls = class0_candidates[i0]
            i0 += 1
        else:
            current_pair, cls = class1_candidates[i1]
            i1 += 1
        
        e1, e2 = current_pair
        nodes = set(edge_info[e1] + edge_info[e2])
        edges = {e1, e2}
        
        if nodes.isdisjoint(used_nodes) and edges.isdisjoint(used_edges):
            u1, v1 = edge_info[e1]
            u2, v2 = edge_info[e2]
            shared = (set([u1, v1]) & set([u2, v2])).pop()
            
            flow1 = df.loc[e1, 'flow']
            flow2 = df.loc[e2, 'flow']
            if shared == u1:
                flow1 *= -1
            if shared == v2:
                flow2 *= -1
            
            selected.append({
                'flow1': flow1,
                'flow2': flow2,
                'edge1': f"{u1},{v1}",
                'edge2': f"{u2},{v2}",
                'class': cls
            })
            
            used_nodes.update(nodes)
            used_edges.update(edges)
            
            if cls == 0:
                selected_0 += 1
            else:
                selected_1 += 1
    
    # Split into class-specific DataFrames
    class0 = [item for item in selected if item['class'] == 0]
    class1 = [item for item in selected if item['class'] == 1]

    edge_info_all = {}
    node_map_all = defaultdict(set)
    
    for idx, row in dfall.iterrows():
        u, v = sorted([row['node1'], row['node2']])
        edge_info_all[idx] = (u, v)
        node_map_all[u].add(idx)
        node_map_all[v].add(idx)
    
    # 2. Generate all_candidates from dfall
    all_candidates = []
    for i in range(len(dfall.index)):
        u1, v1 = edge_info_all[i]
        connected = node_map_all[u1].union(node_map_all[v1])
        for e2 in connected:
            if e2 <= i: 
                continue
            u2, v2 = edge_info_all[e2]
            if {u1, v1}.isdisjoint({u2, v2}): 
                continue
            all_candidates.append(tuple(sorted((i, e2))))
    
    # 3. Sort and process global candidates with dfall
    all_candidates.sort(key=lambda x: -len(set(edge_info_all[x[0]] + edge_info_all[x[1]])))
    
    selectedAll = []
    used_nodes_all = set()
    used_edges_all = set()
    
    for pair in all_candidates:
        e1, e2 = pair
        nodes = set(edge_info_all[e1] + edge_info_all[e2])
        edges = {e1, e2}
        
        if nodes.isdisjoint(used_nodes_all) and edges.isdisjoint(used_edges_all):
            u1, v1 = edge_info_all[e1]
            u2, v2 = edge_info_all[e2]
            shared = (set([u1, v1]) & set([u2, v2])).pop()
            
            flow1 = dfall.loc[e1, 'flow']
            flow2 = dfall.loc[e2, 'flow']
            
            if shared == u1: flow1 *= -1
            if shared == v2: flow2 *= -1
            
            selectedAll.append({
                'flow1': flow1,
                'flow2': flow2,
                'edge1': f"{u1},{v1}",
                'edge2': f"{u2},{v2}",
            })
            
            used_nodes_all.update(nodes)
            used_edges_all.update(edges)
    
    df0 = pd.DataFrame(class0, columns=['flow1', 'flow2', 'edge1', 'edge2']) if class0 else pd.DataFrame()
    df1 = pd.DataFrame(class1, columns=['flow1', 'flow2', 'edge1', 'edge2']) if class1 else pd.DataFrame()

    dfAll = pd.DataFrame(selectedAll, columns=['flow1', 'flow2', 'edge1', 'edge2']) if selectedAll else pd.DataFrame()
    
    return df0, df1, dfAll


def Coloring(df):
    """
    Groups edges based on the graph's edge coloring algorithm and creates pairs of the closest edges within each color group.
    This is to prepare the data required for the independence test.
    """
    df_colored = df.copy()
    # Ensure node1 < node2, and adjust flow direction accordingly
    for index, row in df_colored.iterrows():
        u, v = row["node1"], row["node2"]
        if u > v:
            df_colored.at[index, "node1"] = v
            df_colored.at[index, "node2"] = u
            df_colored.at[index, "flow"] *= -1

    def _edge_coloring(edges):
        """Internal function: executes Vizing's theorem based edge coloring algorithm."""
        vertices = set()
        for u, v in edges:
            vertices.update({u, v})
        
        if not vertices:
            return {}, 0

        degree = {v: 0 for v in vertices}
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        max_degree = max(degree.values())
        
        color_map = {}
        used_colors = {v: set() for v in vertices}
        for edge in edges:
            u, v = sorted(edge)
            if (u, v) in color_map:
                continue
                
            forbidden = used_colors[u].union(used_colors[v])
            for color in range(max_degree + 1):
                if color not in forbidden:
                    color_map[(u, v)] = color
                    used_colors[u].add(color)
                    used_colors[v].add(color)
                    break
        return color_map

    edges = list(zip(df_colored["node1"], df_colored["node2"]))
    color_map = _edge_coloring(edges)

    if not color_map:
        return pd.DataFrame()
        
    df_colored["color"] = df_colored.apply(
        lambda r: color_map.get(tuple(sorted((r["node1"], r["node2"]))), -1), axis=1)

    original_G = nx.from_pandas_edgelist(
        df_colored, "node1", "node2", ["flow", "color"], create_using=nx.Graph()
    )

    def process_color(color):
        """Internal function: pairing within each color group."""
        G = copy.deepcopy(original_G)
        all_edges_in_color = [(u, v) for u, v, attr in G.edges(data=True) if attr.get("color") == color]
        target_edges = sorted([tuple(sorted(edge)) for edge in all_edges_in_color])
        
        total_edges = len(target_edges)  
        if total_edges < 2:
            return []

        all_pairs_sp = dict(nx.all_pairs_shortest_path_length(G))

        edge_distances = []
        for i in range(len(target_edges)):
            for j in range(i + 1, len(target_edges)):
                e1, e2 = target_edges[i], target_edges[j]
                pairs = [
                    (e1[0], e2[0]), (e1[0], e2[1]),
                    (e1[1], e2[0]), (e1[1], e2[1])
                ]
                distance = min(all_pairs_sp.get(a, {}).get(b, float('inf')) for a, b in pairs)
                edge_distances.append((distance, e1, e2))
        edge_distances.sort(key=lambda x: (x[0], x[1], x[2]))
        used_edges = set()
        color_pairs = []
        target_count = int(0.9 * total_edges)

        for dist, e1, e2 in edge_distances:
            if len(used_edges) >= target_count:
                break
            if e1 not in used_edges and e2 not in used_edges:
                color_pairs.append({
                    "flowX": abs(G.edges[e1]["flow"]),
                    "flowY": abs(G.edges[e2]["flow"]),
                    "edgeX": f"{e1[0]},{e1[1]}",
                    "edgeY": f"{e2[0]},{e2[1]}",
                    "color": color,
                    "distance": dist
                })
                used_edges.update({e1, e2})
        return color_pairs

    all_colors = df_colored["color"].unique()
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(process_color)(color)
        for color in tqdm(sorted(all_colors), desc="Processing Colors")
    )
    
    all_flow_pairs = pd.DataFrame([item for sublist in results for item in sublist])
    reflected = all_flow_pairs.copy()
    reflected['flowX'], reflected['flowY'] = all_flow_pairs['flowX'], all_flow_pairs['flowY']
    ind_data = pd.concat([all_flow_pairs, reflected], ignore_index = True)
    
    return ind_data