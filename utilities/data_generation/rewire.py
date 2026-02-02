import scipy.io
import numpy as np
import random
import sys
import time

def is_bad_edge(edge, A_count):
    """
    Helper function: determine whether an edge is invalid (self-loop or multi-edge).
    """
    u, v = edge
    # 1. Check for self-loop
    if u == v:
        return True
    
    # 2. Check for multi-edge
    if u > v:
        u, v = v, u
    
    return A_count.get((u, v), 0) > 1

def update_bad_edge_list(e_idx, edge_to_endpoints, A_count, bad_idx, bad_pos, num_bad):
    """
    Helper function: update the list of invalid edges (self-loops or multi-edges)
    """
    is_bad = is_bad_edge(edge_to_endpoints[e_idx], A_count)
    is_in_list = (e_idx in bad_pos)

    if is_bad and not is_in_list:
        # The edge was previously valid and now becomes invalid: append to bad_idx
        bad_idx.append(e_idx)
        bad_pos[e_idx] = num_bad
        num_bad += 1
        
    elif not is_bad and is_in_list:
        # The edge was previously invalid and now becomes valid: remove from bad_idx
        k = bad_pos[e_idx]
        num_bad -= 1
        
        if k < num_bad:
            # Fill the gap with the last element
            last_edge_idx = bad_idx.pop()
            bad_idx[k] = last_edge_idx
            bad_pos[last_edge_idx] = k 
        else:
            bad_idx.pop()
            
        del bad_pos[e_idx]

    return bad_idx, bad_pos, num_bad

def run_rewiring(N, E, edge_to_endpoints, debug=False, outer_iter=0):
    """
    MCMC Rewiring
    """
    
    t_start_rewire = time.time()
    
    # -----------------------------------------------------------------
    # Step 2.3.a: build A_count as a dictionary of edge multiplicities
    # -----------------------------------------------------------------
    if debug:
        print(f'[PY ITER {outer_iter}] Constructing A_count...', flush=True)
    
    A_count = {}
    for edge in edge_to_endpoints:
        u, v = edge
        if u > v: 
            u, v = v, u
        A_count[(u, v)] = A_count.get((u, v), 0) + 1
        
    if debug:
        print(f'[PY ITER {outer_iter}] A_count (dict) constructed.', flush=True)

    # -----------------------------------------------------------------
    # Step 2.3.b: construct the initial list of invalid edges
    # -----------------------------------------------------------------
    bad_idx = []
    bad_pos = {}
    
    for e_idx in range(E):
        if is_bad_edge(edge_to_endpoints[e_idx], A_count):
            bad_idx.append(e_idx)
            bad_pos[e_idx] = len(bad_idx) - 1
            
    num_bad = len(bad_idx)
    
    if debug:
        print(f'[PY ITER {outer_iter}] Initial number of invalid edges = {num_bad} (self-loops + multi-edges)', flush=True)
        
    if num_bad == 0:
        return edge_to_endpoints, True # success = True

    # -----------------------------------------------------------------
    # Step 2.3.c: main MCMC rewiring loop)
    # -----------------------------------------------------------------
    max_rewire_iter = 15 * num_bad # 15 * num_bad
    
    for iter in range(max_rewire_iter):
        
        if num_bad == 0:
            if debug:
               print(f'[PY ITER {outer_iter}] Rewiring completed in {iter} iterations.', flush=True)
            return edge_to_endpoints, True # success = True
            
        if debug and (iter % max(1, max_rewire_iter // 5) == 0):
            print(f'  [PY ITER {outer_iter}] Rewiring progress {iter} / {max_rewire_iter}, current number of invalid edges = {num_bad}', flush=True)
            
        # 1. Randomly select one invalid edge e1
        k = random.randint(0, num_bad - 1)
        e1_idx = bad_idx[k]
        a, b = edge_to_endpoints[e1_idx]
        
        # 2. Randomly select another edge e2 
        e2_idx = random.randint(0, E - 1)
        if e2_idx == e1_idx:
            e2_idx = (e2_idx + 1) % E
        c, d = edge_to_endpoints[e2_idx]
        
        # 3. Try two possible edge-switching scheme
        success_this_iter = False
        for choice in range(2):
            if choice == 0:
                u1, v1 = a, c
                u2, v2 = b, d
            else:
                u1, v1 = a, d
                u2, v2 = b, c
                
            # 3a. Reject self-loops
            if u1 == v1 or u2 == v2:
                continue
                
            # 3b. Prepare canonical keys (unordered pairs)
            uu1, vv1 = (u1, v1) if u1 < v1 else (v1, u1)
            uu2, vv2 = (u2, v2) if u2 < v2 else (v2, u2)
            aa, bb = (a, b) if a < b else (b, a)
            cc, dd = (c, d) if c < d else (d, c)
            
            # 3c. Check whether the proposed edges would create multi-edges
            cnt_aa = A_count.get((aa, bb), 0)
            cnt_cc = A_count.get((cc, dd), 0)
            cnt_11 = A_count.get((uu1, vv1), 0)
            cnt_22 = A_count.get((uu2, vv2), 0)
            
            cnt_aa_after = cnt_aa - 1
            cnt_cc_after = cnt_cc - 1
            
            removed_11 = (1 if (uu1 == aa and vv1 == bb) else 0) + (1 if (uu1 == cc and vv1 == dd) else 0)
            cnt_11_after = cnt_11 - removed_11 + 1
            
            removed_22 = (1 if (uu2 == aa and vv2 == bb) else 0) + (1 if (uu2 == cc and vv2 == dd) else 0)
            cnt_22_after = cnt_22 - removed_22 + 1
            
            if cnt_11_after > 1 or cnt_22_after > 1:
                continue
                
            # 4. Accept the proposal
            A_count[(aa, bb)] = cnt_aa_after
            A_count[(cc, dd)] = cnt_cc_after
            A_count[(uu1, vv1)] = cnt_11_after
            A_count[(uu2, vv2)] = cnt_22_after
            
            # 5. Update the edge list
            edge_to_endpoints[e1_idx] = [u1, v1]
            edge_to_endpoints[e2_idx] = [u2, v2]
            
            # 6. Update the list of invalid edges
            bad_idx, bad_pos, num_bad = update_bad_edge_list(
                e1_idx, edge_to_endpoints, A_count, bad_idx, bad_pos, num_bad)
            bad_idx, bad_pos, num_bad = update_bad_edge_list(
                e2_idx, edge_to_endpoints, A_count, bad_idx, bad_pos, num_bad)
                
            success_this_iter = True
            break
            
        # if not success_this_iter: continue
            
    t_end_rewire = time.time()
    if debug:
        print(f'[PY ITER {outer_iter}] Rewiring finished after reaching the iteration limit, total time = {t_end_rewire - t_start_rewire:.3f} seconds', flush=True)

    # If there are still invalid edges, report failure
    if num_bad > 0:
        if debug:
            print(f'[PY ITER {outer_iter}] Rewiring reached the iteration limit with {num_bad} invalid edges remaining. Aborting this outer iteration.', flush=True)
        return edge_to_endpoints, False # success = False
    
    return edge_to_endpoints, (num_bad == 0)

if __name__ == "__main__":
    
    # 1. Load data passed from MATLAB
    try:
        data = scipy.io.loadmat('temp_data.mat')
        N = int(data['N'][0, 0])
        E_target = int(data['E_target'][0, 0])
        edge_to_endpoints = data['edge_to_endpoints']
        debug = bool(data['debug'][0, 0])
        outer_iter = int(data['outer_iter'][0, 0])

        if N <= 0 or E_target <= 0:
            raise ValueError("N or E_target is non-positive.")
        if edge_to_endpoints.shape[0] != E_target or edge_to_endpoints.shape[1] != 2:
            raise ValueError(f"edge_to_endpoints has shape {edge_to_endpoints.shape}, which is inconsistent with E_target = {E_target}")
            
    except Exception as e:
        print(f"Python error: failed to load temp_data.mat: {e}", file=sys.stderr)
        scipy.io.savemat('result_data.mat', {'success': False})
        sys.exit(1)

    # 2. Run the rewiring algorithm
    try:
        cleaned_edges, success = run_rewiring(N, E_target, edge_to_endpoints, debug, outer_iter)
        
    except Exception as e:
        print(f"Python error: run_rewiring failed: {e}", file=sys.stderr)
        success = False
        cleaned_edges = edge_to_endpoints

    # 3. Save results to MATLAB
    try:
        scipy.io.savemat('result_data.mat', {'cleaned_edges': cleaned_edges, 'success': success})
    except Exception as e:
        print(f"Python error: failed to save result_data.mat: {e}", file=sys.stderr)
        sys.exit(1)
        
    if debug:
        print(f"Python script finished, success = {success}", flush=True)