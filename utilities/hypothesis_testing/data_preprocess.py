import pandas as pd
import networkx as nx
import numpy as np
import copy
import graph_tool.all as gt
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

def standardize_graph_data(df):
    """
    1. Ensure node1 < node2.
    2. If node1 > node2, swap the nodes and reverse the flow's sign.
    """
    df = df.copy()
    mask = df['node1'] > df['node2']
    
    # swap node1 and node2 where node1 > node2
    df.loc[mask, ['node1', 'node2']] = df.loc[mask, ['node2', 'node1']].values
    
    # reverse the flow sign where node1 > node2
    df.loc[mask, 'flow'] *= -1
    
    return df

# Preliminary classification for ScaleFree
def Compute_Metrics_SF(df):
    """
    Classifies edges based on "Augmented Forman-Ricci Curvature (AFRC)".
    Formula: AFRC = 4 - deg(u) - deg(v) + 3 * Triangles(u,v)
    """
    df_class = df.copy()
    
    # 1. Preprocessing: ensure node1 < node2
    mask_swap = df_class['node1'] > df_class['node2']
    df_class.loc[mask_swap, ['node1', 'node2']] = df_class.loc[mask_swap, ['node2', 'node1']].values
            
    # 2. Calculate Degrees using graph-tool
    g = gt.Graph(directed=False)
    g.vp.ids = g.add_edge_list(df_class[['node1', 'node2']].values, 
                                hashed=True, 
                                hash_type='int64_t')

    deg = g.degree_property_map("total")
    node_degrees_dict = {g.vp.ids[v]: deg[v] for v in g.vertices()}
    
    d_u = df_class['node1'].map(node_degrees_dict).fillna(0).astype(int)
    d_v = df_class['node2'].map(node_degrees_dict).fillna(0).astype(int)
    
    # 3. Calculate Triangles using Sparse Matrix Multiplication
    unique_nodes = pd.unique(df_class[['node1', 'node2']].values.ravel('K'))
    node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
    
    row_idx = df_class['node1'].map(node_to_idx).values
    col_idx = df_class['node2'].map(node_to_idx).values
    data = np.ones(len(df_class), dtype=int)
    N = len(unique_nodes)
    
    # Create Symmetric Adjacency Matrix A
    A = csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
    A = A + A.T 
    
    # A * A [u, v] gives the number of common neighbors (triangles)
    A2 = A.dot(A)
    triangles = A2[row_idx, col_idx].A1
    
    # 4. Calculate AFRC
    # Formula: 4 - du - dv + 3 * triangles
    df_class['AFRC'] = 4 - d_u - d_v + 3 * triangles

    print("Metrics calculated (AFRC).")
    return df_class

def Apply_Thresholds_SF(df_metrics, data_type, fraction_top=0.05, fraction_bottom=0.70):
    """Determine thresholds"""
    if data_type == 'real':
        print("Auto-tuning balancing.")
        q_top = 1.0 - fraction_top
        q_bottom = fraction_bottom
    else:
        print("Fixed threshold.")
        q_top = 0.95
        q_bottom = 0.70

    afrc_top = df_metrics['AFRC'].quantile(q_top)
    afrc_bottom = df_metrics['AFRC'].quantile(q_bottom)
    
    mask_class0 = (df_metrics['AFRC'] >= afrc_top)
    mask_class1 = (df_metrics['AFRC'] <= afrc_bottom)
                
    df_filtered = df_metrics[mask_class0 | mask_class1].copy()
    df_filtered['class'] = np.where(df_filtered['AFRC'] >= afrc_top, 0, 1)
    return df_filtered

# def Classifying_SF(df, data_type, fraction_top=0.05, fraction_bottom=0.7):
#     """
#     Classifies edges based on "Augmented Forman-Ricci Curvature (AFRC)".
#     Formula: AFRC = 4 - deg(u) - deg(v) + 3 * Triangles(u,v)
#     - Class 0: AFRC in Top 5%.
#     - Class 1: AFRC in Bottom 70%.
#     Uses scipy.sparse
#     """
#     df_class = df.copy()
    
#     # 1. Preprocessing: ensure node1 < node2
#     mask_swap = df_class['node1'] > df_class['node2']
#     df_class.loc[mask_swap, ['node1', 'node2']] = df_class.loc[mask_swap, ['node2', 'node1']].values
            
#     # 2. Calculate Degrees using graph-tool
#     g = gt.Graph(directed=False)
#     g.vp.ids = g.add_edge_list(df_class[['node1', 'node2']].values, 
#                                 hashed=True, 
#                                 hash_type='int64_t')

#     deg = g.degree_property_map("total")
#     node_degrees_dict = {g.vp.ids[v]: deg[v] for v in g.vertices()}
    
#     d_u = df_class['node1'].map(node_degrees_dict).fillna(0).astype(int)
#     d_v = df_class['node2'].map(node_degrees_dict).fillna(0).astype(int)
    
#     # 3. Calculate Triangles using Sparse Matrix Multiplication
#     unique_nodes = pd.unique(df_class[['node1', 'node2']].values.ravel('K'))
#     node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
    
#     row_idx = df_class['node1'].map(node_to_idx).values
#     col_idx = df_class['node2'].map(node_to_idx).values
#     data = np.ones(len(df_class), dtype=int)
#     N = len(unique_nodes)
    
#     # Create Symmetric Adjacency Matrix A
#     A = csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
#     A = A + A.T 
    
#     # A * A [u, v] gives the number of common neighbors (triangles)
#     A2 = A.dot(A)
#     triangles = A2[row_idx, col_idx].A1
    
#     # 4. Calculate AFRC
#     # Formula: 4 - du - dv + 3 * triangles
#     df_class['AFRC'] = 4 - d_u - d_v + 3 * triangles

#     print("Metrics calculated (AFRC).")

#     if data_type == 'real':
#         print("Auto-tuning balancing.")
#         q_top = 1.0 - fraction_top
#         q_bottom = fraction_bottom

#     else:
#         print("Fixed threshold.")
#         q_top = 0.95
#         q_bottom = 0.70

#     afrc_top = df_class['AFRC'].quantile(q_top)
#     afrc_bottom = df_class['AFRC'].quantile(q_bottom)
    
#     # 6. Define classification masks
#     mask_class0 = (df_class['AFRC'] >= afrc_top)
#     mask_class1 = (df_class['AFRC'] <= afrc_bottom)
                
#     # 7. Filter data and assign classes
#     df_filtered = df_class[mask_class0 | mask_class1].copy()
#     df_filtered['class'] = np.where(df_filtered['AFRC'] >= afrc_top, 0, 1)
    
#     return df_filtered

# Preliminary classification for StocastcBlock
def Compute_Metrics_SB(df):
    """
    Classifies edges based on "sum of degrees".
    """
    df_class = df.copy()
    
    # 1. Preprocessing: ensure node1 < node2
    mask_swap = df_class['node1'] > df_class['node2']
    df_class.loc[mask_swap, ['node1', 'node2']] = df_class.loc[mask_swap, ['node2', 'node1']].values
            
    # 2. Create graph and calculate the two core metrics

    g = gt.Graph(directed=False)
    g.vp.ids = g.add_edge_list(df_class[['node1', 'node2']].values, 
                                hashed=True, 
                                hash_type='int64_t')

    # --- Metric 1: Sum of Degrees ---
    deg = g.degree_property_map("total")
    
    node_degrees_dict = {g.vp.ids[v]: deg[v] for v in g.vertices()}
    
    degree1 = df_class['node1'].map(node_degrees_dict)
    degree2 = df_class['node2'].map(node_degrees_dict)
    df_class['degree'] = degree1 + degree2
    print("   [Metrics computed: Sum of Degrees]")
    return df_class

def Apply_Thresholds_SB(df_metrics, data_type, fraction_top=0.20, fraction_bottom=0.20):
    """Determine thresholds"""
    if data_type == 'real':
        print("Auto-tuning balancing.")
        q_top = 1.0 - fraction_top
        q_bottom = fraction_bottom
    else:
        print("Fixed threshold.")
        q_top = 0.80
        q_bottom = 0.20

    degree_top = df_metrics['degree'].quantile(q_top)
    degree_bottom = df_metrics['degree'].quantile(q_bottom)
    
    # Define classification masks
    mask_class0 = (df_metrics['degree'] >= degree_top)
    mask_class1 = (df_metrics['degree'] <= degree_bottom)
                    
    # Filter data and assign classes
    df_filtered = df_metrics[mask_class0 | mask_class1].copy()
    df_filtered['class'] = np.where(df_filtered['degree'] >= degree_top, 0, 1)
 
    return df_filtered

# def Classifying_SB(df, data_type, fraction_top=0.20, fraction_bottom=0.20):
#     """
#     Classifies edges based on "sum of degrees".
#     - Class 0: Edges where sum of degrees is in the top 20%.
#     - Class 1: Edges where sum of degrees is in the bottom 20%.
#     Uses graph-tool
#     """
#     df_class = df.copy()
    
#     # 1. Preprocessing: ensure node1 < node2
#     mask_swap = df_class['node1'] > df_class['node2']
#     df_class.loc[mask_swap, ['node1', 'node2']] = df_class.loc[mask_swap, ['node2', 'node1']].values
            
#     # 2. Create graph and calculate the two core metrics

#     g = gt.Graph(directed=False)
#     g.vp.ids = g.add_edge_list(df_class[['node1', 'node2']].values, 
#                                 hashed=True, 
#                                 hash_type='int64_t')

#     # --- Metric 1: Sum of Degrees ---
#     deg = g.degree_property_map("total")
    
#     node_degrees_dict = {g.vp.ids[v]: deg[v] for v in g.vertices()}
    
#     degree1 = df_class['node1'].map(node_degrees_dict)
#     degree2 = df_class['node2'].map(node_degrees_dict)
#     df_class['degree'] = degree1 + degree2

#     print("Metrics calculated.")
    
#     # 3. Determine thresholds
#     if data_type == 'real':
#         print("Auto-tuning balancing.")
#         q_top = 1.0 - fraction_top
#         q_bottom = fraction_bottom
#     else:
#         print("Fixed threshold.")
#         q_top = 0.80
#         q_bottom = 0.20

#     degree_top = df_class['degree'].quantile(q_top)
#     degree_bottom = df_class['degree'].quantile(q_bottom)
    
#     # Define classification masks
#     mask_class0 = (df_class['degree'] >= degree_top)
#     mask_class1 = (df_class['degree'] <= degree_bottom)
                    
#     # Filter data and assign classes
#     df_filtered = df_class[mask_class0 | mask_class1].copy()
#     df_filtered['class'] = np.where(df_filtered['degree'] >= degree_top, 0, 1)
 
#     return df_filtered

# Preliminary classification for SmallWorld
def Compute_Metrics_SW(df):
    """
    Classifies edges based on two metrics: "edge betweenness" and "sum of degrees".
    """
    df_class = df.copy()
    
    # 1. Preprocessing: ensure node1 < node2
    mask_swap = df_class['node1'] > df_class['node2']
    df_class.loc[mask_swap, ['node1', 'node2']] = df_class.loc[mask_swap, ['node2', 'node1']].values
            
    # 2. Create graph and calculate the two core metrics

    g = gt.Graph(directed=False)
    g.vp.ids = g.add_edge_list(df_class[['node1', 'node2']].values, 
                                hashed=True, 
                                hash_type='int64_t')

    # --- Metric 1: Edge Betweenness ---
    _, e_bet = gt.betweenness(g, norm=True)
    df_class['betweenness'] = e_bet.fa

    # --- Metric 2: Sum of Degrees ---
    deg = g.degree_property_map("total")
    
    node_degrees_dict = {g.vp.ids[v]: deg[v] for v in g.vertices()}
    
    degree0 = df_class['node1'].map(node_degrees_dict)
    degree1 = df_class['node2'].map(node_degrees_dict)
    df_class['degree'] = degree0 + degree1

    print("Metrics calculated.")
    return df_class

def Apply_Thresholds_SW(df_metrics, data_type, fraction_top=0.25, fraction_bottom=0.25):
    if data_type == 'real':
        print("Auto-tuning balancing.")
        q_top = 1.0 - fraction_top
        q_bottom = fraction_bottom
    else:
        print("Fixed threshold.")
        q_top = 0.75
        q_bottom = 0.25

    betweenness_top = df_metrics['betweenness'].quantile(q_top)
    betweenness_bottom = df_metrics['betweenness'].quantile(q_bottom)
    degree_top = df_metrics['degree'].quantile(q_top)
    degree_bottom = df_metrics['degree'].quantile(q_bottom)

    # 4. Define classification masks
    mask_class0 = (df_metrics['betweenness'] >= betweenness_top) & \
                  (df_metrics['degree'] >= degree_top)
                  
    mask_class1 = (df_metrics['betweenness'] <= betweenness_bottom) & \
                  (df_metrics['degree'] <= degree_bottom)
                  
    # 5. Filter data and assign classes
    df_filtered = df_metrics[mask_class0 | mask_class1].copy()
    df_filtered['class'] = np.where(df_filtered['betweenness'] >= betweenness_top, 0, 1)
    
    return df_filtered

# def Classifying_SW(df, data_type, fraction_top=0.25, fraction_bottom=0.25):
#     """
#     Classifies edges based on two metrics: "edge betweenness" and "sum of degrees".
#     - Class 0: Edges where both edge betweenness and sum of degrees are in the top 25%.
#     - Class 1: Edges where both edge betweenness and sum of degrees are in the bottom 25%.
#     Uses graph-tool
#     """
#     df_class = df.copy()
    
#     # 1. Preprocessing: ensure node1 < node2
#     mask_swap = df_class['node1'] > df_class['node2']
#     df_class.loc[mask_swap, ['node1', 'node2']] = df_class.loc[mask_swap, ['node2', 'node1']].values
            
#     # 2. Create graph and calculate the two core metrics

#     g = gt.Graph(directed=False)
#     g.vp.ids = g.add_edge_list(df_class[['node1', 'node2']].values, 
#                                 hashed=True, 
#                                 hash_type='int64_t')

#     # --- Metric 1: Edge Betweenness ---
#     _, e_bet = gt.betweenness(g, norm=True)
#     df_class['betweenness'] = e_bet.fa

#     # --- Metric 2: Sum of Degrees ---
#     deg = g.degree_property_map("total")
    
#     node_degrees_dict = {g.vp.ids[v]: deg[v] for v in g.vertices()}
    
#     degree0 = df_class['node1'].map(node_degrees_dict)
#     degree1 = df_class['node2'].map(node_degrees_dict)
#     df_class['degree'] = degree0 + degree1

#     print("Metrics calculated.")
    
#     # 3. Determine thresholds
#     if data_type == 'real':
#         print("Auto-tuning balancing.")
#         q_top = 1.0 - fraction_top
#         q_bottom = fraction_bottom
#     else:
#         print("Fixed threshold.")
#         q_top = 0.75
#         q_bottom = 0.25

#     betweenness_top = df_class['betweenness'].quantile(q_top)
#     betweenness_bottom = df_class['betweenness'].quantile(q_bottom)
#     degree_top = df_class['degree'].quantile(q_top)
#     degree_bottom = df_class['degree'].quantile(q_bottom)
    
#     # 4. Define classification masks
#     mask_class0 = (df_class['betweenness'] >= betweenness_top) & \
#                   (df_class['degree'] >= degree_top)
                  
#     mask_class1 = (df_class['betweenness'] <= betweenness_bottom) & \
#                   (df_class['degree'] <= degree_bottom)
                  
#     # 5. Filter data and assign classes
#     df_filtered = df_class[mask_class0 | mask_class1].copy()
#     df_filtered['class'] = np.where(df_filtered['betweenness'] >= betweenness_top, 0, 1)
    
#     return df_filtered

def run_auto_tuning_loop(df_metrics, df_input, graph_type, data_type, target_test, auto_tune, init_top, init_bottom):
    """Hyperparameter tuning loop method"""
    fraction_top, fraction_bottom = init_top, init_bottom
    max_iterations = 5 if (auto_tune and data_type == 'real') else 1
    tolerance = 0.40
    loop_iter = 0
    
    best_class_data = None
    best_extracted_data = None

    print(f"\n   === Threshold Tuning for [{target_test}] ===")
    
    while loop_iter < max_iterations:
        loop_iter += 1
        if data_type == 'real':
            mode_str = f"Auto Iteration {loop_iter}/{max_iterations}" if auto_tune else "Manual/Fixed"
            print(f"   [{mode_str}] | fraction_top: {fraction_top:.2f}, fraction_bottom: {fraction_bottom:.2f}")

        if graph_type == 'SmallWorld':
            class_data = Apply_Thresholds_SW(df_metrics, data_type, fraction_top, fraction_bottom)
        elif graph_type == 'StochasticBlock':
            class_data = Apply_Thresholds_SB(df_metrics, data_type, fraction_top, fraction_bottom)
        elif graph_type == 'ScaleFree':
            class_data = Apply_Thresholds_SF(df_metrics, data_type, fraction_top, fraction_bottom)

        best_class_data = class_data

        if target_test == 'KS':
            df_class0, df_class1 = KS_Data_Preprocessing(class_data)
            len0, len1 = len(df_class0), len(df_class1)
            best_extracted_data = (df_class0, df_class1)
        elif target_test == 'BEDT':
            df0, df1, dfAll = BEDT_Data_Preprocessing(class_data, df_input)
            len0, len1 = len(df0), len(df1)
            best_extracted_data = (df0, df1, dfAll)

        if not (auto_tune and data_type == 'real'):
            print(f"   [Final Counts] Class 0: {len0}, Class 1: {len1}")
            break

        ratio = max(len0, len1) / min(len0, len1)

        if ratio <= 1.0 + tolerance:
            print(f"   Balance Reached (diff < {tolerance*100:.0f}%). Counts: {len0} vs {len1}")
            break
   
        step_size = 0.1
        if len0 < len1:
            fraction_top = min(0.90, fraction_top + step_size)
            fraction_bottom = max(0.05, fraction_bottom - 0.5*step_size)
        else:
            fraction_bottom = min(0.90, fraction_bottom + step_size)
            fraction_top = max(0.05, fraction_top - 0.5*step_size)

    return best_class_data, best_extracted_data, fraction_top, fraction_bottom

# Select edges for KS test
def KS_Data_Preprocessing(df):
    np.random.seed(423) 

    grouped = df.groupby('class')
    
    edges_by_class = {
        0: grouped.get_group(0)[['node1', 'node2']].to_numpy(),
        1: grouped.get_group(1)[['node1', 'node2']].to_numpy()
    }
    flows_by_class = {
        0: grouped.get_group(0)['flow'].to_numpy(),
        1: grouped.get_group(1)['flow'].to_numpy()
    }

    node_to_edge_map = defaultdict(list)
    for cls in [0, 1]:
        for i, (u, v) in enumerate(edges_by_class[cls]):
            node_to_edge_map[u].append((cls, i))
            node_to_edge_map[v].append((cls, i))

    is_edge_available = {
        0: np.ones(len(edges_by_class[0]), dtype=bool),
        1: np.ones(len(edges_by_class[1]), dtype=bool)
    }
    
    remain_counts = {
        0: len(edges_by_class[0]),
        1: len(edges_by_class[1])
    }
    
    used_nodes = set()
    result = {0: [], 1: []}

    while True:
        remain_0 = remain_counts[0]
        remain_1 = remain_counts[1]
        
        if remain_0 + remain_1 == 0:
            break
            
        if remain_0 <= remain_1 and remain_0 > 0:
            target_cls = 0
        elif remain_1 < remain_0 and remain_1 > 0:
            target_cls = 1
        else:
            target_cls = 0 if remain_0 > 0 else 1
            
        alt_cls = 1 - target_cls
        
        def find_and_process_edge(cls_to_process):
            available_indices = [i for i, is_avail in enumerate(is_edge_available[cls_to_process]) if is_avail]
            
            if not available_indices:
                return False
            
            np.random.shuffle(available_indices)
            
            for edge_idx in available_indices:
                u, v = edges_by_class[cls_to_process][edge_idx]
                
                if u not in used_nodes and v not in used_nodes:
                    result[cls_to_process].append(flows_by_class[cls_to_process][edge_idx])
                    used_nodes.update([u, v])
                    
                    for node in [u, v]:
                        if node in node_to_edge_map:
                            for conn_cls, conn_idx in node_to_edge_map[node]:
                                if is_edge_available[conn_cls][conn_idx]:
                                    is_edge_available[conn_cls][conn_idx] = False
                                    remain_counts[conn_cls] -= 1
                    return True
            
            return False

        success = find_and_process_edge(target_cls)
        if not success:
            success = find_and_process_edge(alt_cls)
            
        if not success:
            break
            
    df_class0_KS = pd.DataFrame({'flow': [abs(f) for f in result[0]]})
    df_class1_KS = pd.DataFrame({'flow': [abs(f) for f in result[1]]})
    return df_class0_KS, df_class1_KS

# Select edge pairs for Bivariate Test
# def BEDT_Data_Preprocessing(df,dfall):
#     np.random.seed(423) 

#     df_data = df.to_dict('index')

#     edge_info = {}
#     class_edges = defaultdict(list)
#     node_map = defaultdict(set)
    
#     for idx, row_data in df_data.items():
#         u, v = sorted([row_data['node1'], row_data['node2']])
#         edge_info[idx] = (u, v)
#         cls = row_data['class']
#         class_edges[cls].append(idx)
#         node_map[u].add(idx)
#         node_map[v].add(idx)
    
#     # Generate candidate pairs for each class and sort them internally
#     class0_candidates = []
#     class1_candidates = []
    
#     for cls in class_edges:
#         edges = class_edges[cls]
#         cls_pairs = []
#         for i in range(len(edges)):
#             e1 = edges[i]
#             u1, v1 = edge_info[e1]
#             connected = node_map[u1].union(node_map[v1])
#             for e2 in connected:
#                 if e2 <= e1 or df_data[e2]['class'] != cls:
#                     continue
#                 u2, v2 = edge_info[e2]
#                 if {u1, v1}.isdisjoint({u2, v2}):
#                     continue
#                 pair = tuple(sorted((e1, e2)))
#                 cls_pairs.append((pair, cls))
        
#         # Sort by the number of shared nodes in descending order within the class
#         cls_pairs.sort(key=lambda x: -len(set(edge_info[x[0][0]] + edge_info[x[0][1]])))
        
#         if cls == 0:
#             class0_candidates = cls_pairs
#         else:
#             class1_candidates = cls_pairs

#     # Process candidate pairs, prioritizing the class with fewer selected pairs
#     i0 = 0
#     i1 = 0
#     selected_0 = 0
#     selected_1 = 0
#     used_nodes = set()
#     used_edges = set()
#     selected = []
    
#     while i0 < len(class0_candidates) or i1 < len(class1_candidates):
#         # Determine which class to process next
#         current_cls = None
#         if selected_0 <= selected_1:
#             if i0 < len(class0_candidates):
#                 current_cls = 0
#             else:
#                 current_cls = 1 if i1 < len(class1_candidates) else None
#         else:
#             if i1 < len(class1_candidates):
#                 current_cls = 1
#             else:
#                 current_cls = 0 if i0 < len(class0_candidates) else None
        
#         if current_cls is None:
#             break
        
#         if current_cls == 0:
#             current_pair, cls = class0_candidates[i0]
#             i0 += 1
#         else:
#             current_pair, cls = class1_candidates[i1]
#             i1 += 1
        
#         e1, e2 = current_pair
#         nodes = set(edge_info[e1] + edge_info[e2])
#         edges = {e1, e2}
        
#         if nodes.isdisjoint(used_nodes) and edges.isdisjoint(used_edges):
#             u1, v1 = edge_info[e1]
#             u2, v2 = edge_info[e2]
#             shared = (set([u1, v1]) & set([u2, v2])).pop()
            
#             flow1 = df_data[e1]['flow']
#             flow2 = df_data[e2]['flow']
#             if shared == u1:
#                 flow1 *= -1
#             if shared == v2:
#                 flow2 *= -1
            
#             selected.append({
#                 'flow1': flow1,
#                 'flow2': flow2,
#                 'edge1': f"{u1},{v1}",
#                 'edge2': f"{u2},{v2}",
#                 'class': cls
#             })
            
#             used_nodes.update(nodes)
#             used_edges.update(edges)
            
#             if cls == 0:
#                 selected_0 += 1
#             else:
#                 selected_1 += 1
    
#     # Split into class-specific DataFrames
#     class0 = [item for item in selected if item['class'] == 0]
#     class1 = [item for item in selected if item['class'] == 1]

#     dfall_data = dfall.to_dict('index')
#     edge_info_all = {}
#     node_map_all = defaultdict(set)

#     for idx, row_data in dfall_data.items():
#         u, v = sorted([row_data['node1'], row_data['node2']])
#         edge_info_all[idx] = (u, v)
#         node_map_all[u].add(idx)
#         node_map_all[v].add(idx)
    
#     # 2. Generate all_candidates from dfall
#     all_candidates = []

#     dfall_indices = list(dfall_data.keys())

#     for i_idx in range(len(dfall_indices)):
#             i = dfall_indices[i_idx]
#             u1, v1 = edge_info_all[i]
#             connected = node_map_all[u1].union(node_map_all[v1])
#             for e2 in connected:
#                 if e2 <= i: 
#                     continue
#                 u2, v2 = edge_info_all[e2]
#                 if {u1, v1}.isdisjoint({u2, v2}): 
#                     continue
#                 all_candidates.append(tuple(sorted((i, e2))))
    
#     # 3. Sort and process global candidates with dfall
#     all_candidates.sort(key=lambda x: -len(set(edge_info_all[x[0]] + edge_info_all[x[1]])))
    
#     selectedAll = []
#     used_nodes_all = set()
#     used_edges_all = set()
    
#     for pair in all_candidates:
#         e1, e2 = pair
#         nodes = set(edge_info_all[e1] + edge_info_all[e2])
#         edges = {e1, e2}
        
#         if nodes.isdisjoint(used_nodes_all) and edges.isdisjoint(used_edges_all):
#             u1, v1 = edge_info_all[e1]
#             u2, v2 = edge_info_all[e2]
#             shared = (set([u1, v1]) & set([u2, v2])).pop()
            
#             flow1 = dfall_data[e1]['flow']
#             flow2 = dfall_data[e2]['flow']

#             if shared == u1: flow1 *= -1
#             if shared == v2: flow2 *= -1
            
#             selectedAll.append({
#                 'flow1': flow1,
#                 'flow2': flow2,
#                 'edge1': f"{u1},{v1}",
#                 'edge2': f"{u2},{v2}",
#             })
            
#             used_nodes_all.update(nodes)
#             used_edges_all.update(edges)
    
#     df0 = pd.DataFrame(class0, columns=['flow1', 'flow2', 'edge1', 'edge2']) if class0 else pd.DataFrame()
#     df1 = pd.DataFrame(class1, columns=['flow1', 'flow2', 'edge1', 'edge2']) if class1 else pd.DataFrame()

#     dfAll = pd.DataFrame(selectedAll, columns=['flow1', 'flow2', 'edge1', 'edge2']) if selectedAll else pd.DataFrame()
    
#     return df0, df1, dfAll

# optimized version (starting from 2026-Feb-2; Scale Free)
def BEDT_Data_Preprocessing(df, dfall):
    np.random.seed(423) 

    df_data = df.to_dict('index')
    edge_info = {}
    class_edges = defaultdict(list)
    node_map = defaultdict(set)
    
    for idx, row_data in df_data.items():
        u, v = sorted([row_data['node1'], row_data['node2']])
        edge_info[idx] = (u, v)
        cls = row_data['class']
        class_edges[cls].append(idx)
        node_map[u].add(idx)
        node_map[v].add(idx)


    def candidate_generator(target_cls, used_nodes_set, used_edges_set):
        edges = class_edges[target_cls]
        for i in range(len(edges)):
            e1 = edges[i]

            #  check if e1 is usable
            if e1 in used_edges_set: continue
            u1, v1 = edge_info[e1]
            if u1 in used_nodes_set or v1 in used_nodes_set: continue

            connected = node_map[u1].union(node_map[v1])
            
            for e2 in connected:
                if e2 <= e1 or df_data[e2]['class'] != target_cls:
                    continue
                
                # check if e2 is usable
                if e2 in used_edges_set: continue
                u2, v2 = edge_info[e2]
                if u2 in used_nodes_set or v2 in used_nodes_set: continue
                
                if {u1, v1}.isdisjoint({u2, v2}):
                    continue
                
                # yield usable pair
                yield (e1, e2)

    used_nodes = set()
    used_edges = set()
    selected = []
    
    selected_0 = 0
    selected_1 = 0

    # check candidate generators for both classes
    gen0 = candidate_generator(0, used_nodes, used_edges)
    gen1 = candidate_generator(1, used_nodes, used_edges)
    
    # check if generators are done
    gen0_exhausted = False
    gen1_exhausted = False
    
    while not (gen0_exhausted and gen1_exhausted):
        current_cls = None
        
        # Determine which class to process next
        if selected_0 <= selected_1:
            if not gen0_exhausted:
                current_cls = 0
            elif not gen1_exhausted:
                current_cls = 1
        else:
            if not gen1_exhausted:
                current_cls = 1
            elif not gen0_exhausted:
                current_cls = 0
        
        if current_cls is None:
            break
            
        try:
            if current_cls == 0:
                e1, e2 = next(gen0)
                selected_0 += 1
            else:
                e1, e2 = next(gen1)
                selected_1 += 1
                
            u1, v1 = edge_info[e1]
            u2, v2 = edge_info[e2]
            
            # update used nodes and edges
            used_nodes.update([u1, v1, u2, v2])
            used_edges.update([e1, e2])
            
            shared = (set([u1, v1]) & set([u2, v2])).pop()
            
            flow1 = df_data[e1]['flow']
            flow2 = df_data[e2]['flow']
            
            if shared == u1:
                flow1 *= -1
            if shared == v2:
                flow2 *= -1
            
            selected.append({
                'flow1': flow1,
                'flow2': flow2,
                'edge1': f"{u1},{v1}",
                'edge2': f"{u2},{v2}",
                'class': current_cls
            })
            
        except StopIteration:
            if current_cls == 0:
                gen0_exhausted = True
            else:
                gen1_exhausted = True

    # Split into class-specific DataFrames
    class0 = [item for item in selected if item['class'] == 0]
    class1 = [item for item in selected if item['class'] == 1]
    
    df0 = pd.DataFrame(class0, columns=['flow1', 'flow2', 'edge1', 'edge2']) if class0 else pd.DataFrame()
    df1 = pd.DataFrame(class1, columns=['flow1', 'flow2', 'edge1', 'edge2']) if class1 else pd.DataFrame()

    # Prepare for dfAll processing
    
    dfall_data = dfall.to_dict('index')
    edge_info_all = {}
    node_map_all = defaultdict(set)

    for idx, row_data in dfall_data.items():
        u, v = sorted([row_data['node1'], row_data['node2']])
        edge_info_all[idx] = (u, v)
        node_map_all[u].add(idx)
        node_map_all[v].add(idx)
        
    dfall_indices = list(dfall_data.keys())
    
    def all_candidate_generator(used_nodes_set, used_edges_set):
        for i_idx in range(len(dfall_indices)):
            i = dfall_indices[i_idx]
            
            if i in used_edges_set: continue
            u1, v1 = edge_info_all[i]
            if u1 in used_nodes_set or v1 in used_nodes_set: continue

            connected = node_map_all[u1].union(node_map_all[v1])
            
            for e2 in connected:
                if e2 <= i: 
                    continue
                
                if e2 in used_edges_set: continue
                u2, v2 = edge_info_all[e2]
                if u2 in used_nodes_set or v2 in used_nodes_set: continue

                if {u1, v1}.isdisjoint({u2, v2}): 
                    continue
                
                yield (i, e2)

    selectedAll = []
    used_nodes_all = set()
    used_edges_all = set()
    
    gen_all = all_candidate_generator(used_nodes_all, used_edges_all)
    
    for pair in gen_all:
        e1, e2 = pair
        u1, v1 = edge_info_all[e1]
        u2, v2 = edge_info_all[e2]
        
        used_nodes_all.update([u1, v1, u2, v2])
        used_edges_all.update([e1, e2])

        shared = (set([u1, v1]) & set([u2, v2])).pop()
        
        flow1 = dfall_data[e1]['flow']
        flow2 = dfall_data[e2]['flow']

        if shared == u1: flow1 *= -1
        if shared == v2: flow2 *= -1
        
        selectedAll.append({
            'flow1': flow1,
            'flow2': flow2,
            'edge1': f"{u1},{v1}",
            'edge2': f"{u2},{v2}",
        })
    
    dfAll = pd.DataFrame(selectedAll, columns=['flow1', 'flow2', 'edge1', 'edge2']) if selectedAll else pd.DataFrame()
    
    return df0, df1, dfAll


# def Coloring(df, jobs=-1):
#     """
#     Groups edges based on the graph's edge coloring algorithm and creates pairs of the closest edges within each color group.
#     This is to prepare the data required for the independence test.
#     """
#     df_colored = df.copy()
#     # Ensure node1 < node2, and adjust flow direction accordingly
#     swap_mask = df_colored["node1"] > df_colored["node2"]
#     df_colored.loc[swap_mask, ['node1', 'node2']] = \
#         df_colored.loc[swap_mask, ['node2', 'node1']].values
#     df_colored.loc[swap_mask, "flow"] *= -1

#     def _edge_coloring(edges):
#         """Internal function: executes Vizing's theorem based edge coloring algorithm."""
#         vertices = set()
#         for u, v in edges:
#             vertices.update({u, v})
        
#         if not vertices:
#             return {}

#         degree = {v: 0 for v in vertices}
#         for u, v in edges:
#             degree[u] += 1
#             degree[v] += 1
#         max_degree = max(degree.values())
        
#         color_map = {}
#         used_colors = {v: set() for v in vertices}
#         for edge in edges:
#             u, v = sorted(edge)
#             if (u, v) in color_map:
#                 continue
                
#             forbidden = used_colors[u].union(used_colors[v])
#             for color in range(max_degree*2 + 1):
#                 if color not in forbidden:
#                     color_map[(u, v)] = color
#                     used_colors[u].add(color)
#                     used_colors[v].add(color)
#                     break
#         return color_map

#     edges = list(zip(df_colored["node1"], df_colored["node2"]))
#     color_map = _edge_coloring(edges)

#     if not color_map:
#         return pd.DataFrame()
        
#     color_series = pd.Series(color_map)
#     color_series.index = pd.MultiIndex.from_tuples(color_series.index)
#     df_index = pd.MultiIndex.from_frame(df_colored[['node1', 'node2']])
    
#     df_colored['color'] = df_index.map(color_series).fillna(-1).astype(int)

#     df_colored = df_colored[df_colored['color'] != -1].copy()

#     original_G = nx.from_pandas_edgelist(
#         df_colored, "node1", "node2", ["flow", "color"], create_using=nx.Graph()
#     )
    
#     def process_color(color, G):
#         """Internal function: pairing within each color group."""
#         all_edges_in_color = [(u, v) for u, v, attr in G.edges(data=True) if attr.get("color") == color]
#         target_edges = sorted([tuple(sorted(edge)) for edge in all_edges_in_color])
        
#         total_edges = len(target_edges)  
#         if total_edges < 2:
#             return []

#         color_nodes = set()
#         for u, v in target_edges:
#             color_nodes.update([u, v])
            
#         local_sp = {}
#         for node in color_nodes:
#             dists = nx.single_source_shortest_path_length(G, node)
#             local_sp[node] = {target: dists[target] for target in color_nodes if target in dists}

#         edge_distances = []
#         for i in range(len(target_edges)):
#             for j in range(i + 1, len(target_edges)):
#                 e1, e2 = target_edges[i], target_edges[j]
#                 pairs = [
#                     (e1[0], e2[0]), (e1[0], e2[1]),
#                     (e1[1], e2[0]), (e1[1], e2[1])
#                 ]
#                 distance = min(local_sp.get(a, {}).get(b, float('inf')) for a, b in pairs)
#                 edge_distances.append((distance, e1, e2))
                
#         edge_distances.sort(key=lambda x: (x[0], x[1], x[2]))
#         used_edges = set()
#         color_pairs = []
#         target_count = int(0.9 * total_edges)

#         for dist, e1, e2 in edge_distances:
#             if len(used_edges) >= target_count:
#                 break
#             if e1 not in used_edges and e2 not in used_edges:
#                 color_pairs.append({
#                     "flowX": abs(G.edges[e1]["flow"]),
#                     "flowY": abs(G.edges[e2]["flow"]),
#                     "edgeX": f"{e1[0]},{e1[1]}",
#                     "edgeY": f"{e2[0]},{e2[1]}",
#                     "color": color,
#                     "distance": dist
#                 })
#                 used_edges.update({e1, e2})
#         return color_pairs

#     color_counts = df_colored["color"].value_counts()

#     top_3_colors = color_counts.nlargest(3).index.tolist()
    
#     print(f"Identified top 3 colors: {top_3_colors}")
#     print(f"Edge counts for top 3: {color_counts.nlargest(3).to_dict()}")

#     results = Parallel(n_jobs=jobs, prefer="processes")(
#         delayed(process_color)(color, original_G)
#         for color in tqdm(sorted(top_3_colors), desc="Processing Top 3 Colors")
#     )
    
#     all_flow_pairs = pd.DataFrame([item for sublist in results for item in sublist])
#     reflected = all_flow_pairs.copy()
#     reflected['flowX'], reflected['flowY'] = all_flow_pairs['flowY'], all_flow_pairs['flowX']
#     ind_data = pd.concat([all_flow_pairs, reflected], ignore_index = True)
    
#     return ind_data


def Coloring(df, jobs=-1):
    """
    Greedy Coloring based on RCM topological dimensionality reduction.
    """
    df_colored = df.copy()
    swap_mask = df_colored["node1"] > df_colored["node2"]
    df_colored.loc[swap_mask, ['node1', 'node2']] = \
        df_colored.loc[swap_mask, ['node2', 'node1']].values
    df_colored.loc[swap_mask, "flow"] *= -1

    # 1. Greedy Coloring
    def _edge_coloring(edges):
        """Internal function: executes Vizing's theorem based edge coloring algorithm."""
        vertices = set()
        for u, v in edges:
            vertices.update({u, v})
        if not vertices:
            return {}
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
            for color in range(max_degree*2 + 1):
                if color not in forbidden:
                    color_map[(u, v)] = color
                    used_colors[u].add(color)
                    used_colors[v].add(color)
                    break
        return color_map

    edges = list(zip(df_colored["node1"], df_colored["node2"]))
    color_map_dict = _edge_coloring(edges)

    if not color_map_dict:
        return pd.DataFrame()
        
    color_series = pd.Series(color_map_dict)
    color_series.index = pd.MultiIndex.from_tuples(color_series.index)
    df_index = pd.MultiIndex.from_frame(df_colored[['node1', 'node2']])
    
    df_colored['color'] = df_index.map(color_series).fillna(-1).astype(int)
    df_colored = df_colored[df_colored['color'] != -1].copy()

    # 2. Construct graph-tool object
    unique_nodes = np.unique(df_colored[['node1', 'node2']].values)
    node_map = {node: i for i, node in enumerate(unique_nodes)}
    rev_node_map = {i: node for node, i in node_map.items()}
    N = len(unique_nodes)
    
    g = gt.Graph(directed=False)
    edge_list = df_colored[['node1', 'node2']].values
    mapped_edges = [(node_map[u], node_map[v]) for u, v in edge_list]
    g.add_edge_list(mapped_edges)
    
    ep_flow = g.new_edge_property("double")
    ep_flow.a = df_colored["flow"].values
    g.ep["flow"] = ep_flow
    
    ep_color = g.new_edge_property("int")
    ep_color.a = df_colored["color"].values
    g.ep["color"] = ep_color

    color_counts = df_colored["color"].value_counts()
    top_3_colors = color_counts.nlargest(3).index.tolist()
    
    print(f"Identified top 3 colors: {top_3_colors}")
    print(f"Edge counts for top 3: {color_counts.nlargest(3).to_dict()}")

    # 3. Compute topological sequence via RCM
    print("   - Computing topological sequence via RCM...")
    row = np.array([node_map[u] for u in df_colored['node1']])
    col = np.array([node_map[v] for v in df_colored['node2']])
    data = np.ones(len(row), dtype=int)
    
    # Build sparse adjacency matrix and ensure symmetry
    adj = csr_matrix((data, (row, col)), shape=(N, N))
    adj = adj + adj.T 
    
    # Get RCM ordering, nodes connected in the graph will be assigned nearby coordinates
    rcm_perm = reverse_cuthill_mckee(adj)
    node_to_pos = np.empty(N, dtype=float)
    node_to_pos[rcm_perm] = np.arange(N)

    # 4. Pair edges based on topological coordinates and compute precise distances only for selected pairs
    def process_color_gt(target_color, g_full):
        selected_edges = [e for e in g_full.edges() if g_full.ep.color[e] == target_color]
        total_edges = len(selected_edges)
        if total_edges < 2:
            return []

        # Score each edge: average of endpoint coordinates
        edge_scores = []
        for e in selected_edges:
            u, v = int(e.source()), int(e.target())
            score = (node_to_pos[u] + node_to_pos[v]) / 2.0
            edge_scores.append((score, e))

        # Sort: edges with similar scores are very close in the topological structure!
        edge_scores.sort(key=lambda x: x[0])

        color_pairs = []
        target_count = int(0.9 * total_edges) # Extract 50% of the edges

        # Directly pair adjacent siblings (step of 2, pairwise combination)
        for i in range(0, len(edge_scores) - 1, 2):
            if len(color_pairs) * 2 >= target_count:
                break
                
            e1 = edge_scores[i][1]
            e2 = edge_scores[i+1][1]
            
            u1, v1 = int(e1.source()), int(e1.target())
            u2, v2 = int(e2.source()), int(e2.target())
            
            # only query the true shortest distance for the already paired edges
            try:
                d1 = gt.shortest_distance(g_full, source=u1, target=u2)
                d2 = gt.shortest_distance(g_full, source=u1, target=v2)
                d3 = gt.shortest_distance(g_full, source=v1, target=u2)
                d4 = gt.shortest_distance(g_full, source=v1, target=v2)
                dist = min(d1, d2, d3, d4)
            except Exception:
                continue

            if dist >= 2147483647:
                continue
                
            color_pairs.append({
                "flowX": abs(g_full.ep.flow[e1]),
                "flowY": abs(g_full.ep.flow[e2]),
                "edgeX": f"{rev_node_map[u1]},{rev_node_map[v1]}",
                "edgeY": f"{rev_node_map[u2]},{rev_node_map[v2]}",
                "color": target_color,
                "distance": int(dist)
            })
            
        return color_pairs

    results = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(process_color_gt)(c, g)
        for c in tqdm(top_3_colors, desc="Processing Colors")
    )
    
    flattened_results = [item for sublist in results for item in sublist]
    if not flattened_results:
        print("   No edge pairs could be formed.")
        return pd.DataFrame()
        
    all_flow_pairs = pd.DataFrame(flattened_results)
    reflected = all_flow_pairs.copy()
    reflected['flowX'], reflected['flowY'] = all_flow_pairs['flowY'], all_flow_pairs['flowX']
    ind_data = pd.concat([all_flow_pairs, reflected], ignore_index = True)
    
    return ind_data