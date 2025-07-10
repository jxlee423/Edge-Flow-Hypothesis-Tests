import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from tqdm import tqdm

def run_bivariate_test(D1_df, D2_df, D_all_df, config, results_manager):
    print(f"\n ============ Running Bivariate Test ============")
    # --- 1. Extract parameters from config and data ---
    D1 = D1_df[['flow1', 'flow2']].values
    D2 = D2_df[['flow1', 'flow2']].values
    D3 = D_all_df[['flow1', 'flow2']].values
    n_BED1 = len(D1)
    n_BED2 = len(D2)
    n_BED3 = len(D3)
    print(f"# of Edge Pairs in Class1 : {n_BED1}")
    print(f"# of Edge Pairs in Class2 : {n_BED2}")
    
    if n_BED1 < 50 or n_BED2 < 50:
        print("ERROR: Insufficient bivariate equivalence of distributions test data")
        return None
    
    if n_BED3 < (n_BED1 + n_BED2):
        print("ERROR: n_BED3 < (n_BED1 + n_BED2)")
        return None

    # Getting parameters from the config module
    GRID_SIZE = config.BIVARIATE_GRID_SIZE
    EXTEND_RATIO = config.BIVARIATE_EXTEND_RATIO
    N_PERMUTATIONS = config.BIVARIATE_N_PERMUTATIONS
    EPSILON = config.EPSILON
    alpha = config.ALPHA
    RANDOM_STATE = config.RANDOM_STATE
    
    # --- 2. Internal helper function definitions ---
    def generate_k_candidates_dynamic(data_size, num_candidates=20):
        # 1. Calculate the central K value
        k_min = 8
        k_max = int(np.sqrt(data_size))

        # 2. Determine the starting point of the candidate range, ensuring it's not less than 8
        if k_max <= k_min:
            return [k_min]

        log_candidates = np.logspace(
            np.log10(k_min),
            np.log10(k_max),
            num=num_candidates
        )

        # 3. Process the float candidates into a unique, sorted list of integers
        candidates = np.unique(np.round(log_candidates)).astype(int)
        return candidates
    # ===================== KNN Density Estimation Function =====================    
    def knn_density(query_points, data_points, k):
        knn = NearestNeighbors(n_neighbors=k).fit(data_points)
        distances, _ = knn.kneighbors(query_points)
        r_k = distances[:, -1]
        volume = np.pi * r_k**2
        return k / (len(data_points) * (volume + EPSILON))
    # ===================== Optimal_K Function =====================
    def find_optimal_k_cv(data, k_values, n_splits=5):
        cv_scores = {}
        for k in k_values:
            min_train_size = len(data) * (n_splits - 1) / n_splits
            if k >= min_train_size:
                print(f"Skipping K={k} because it's too large for the training fold size.")
                continue
            
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            fold_log_likelihoods = []
            
            for train_index, val_index in kf.split(data):
                train_data, val_data = data[train_index], data[val_index]
                
                if k >= len(train_data): 
                    continue
                # Calculate densities on the validation set
                densities = knn_density(val_data, train_data, k)
                # Calculate log-likelihood 
                log_likelihood = np.sum(np.log(np.maximum(densities, EPSILON)))
                fold_log_likelihoods.append(log_likelihood)
            # Calculate the average log-likelihood for this K value
            if fold_log_likelihoods:
                cv_scores[k] = np.mean(fold_log_likelihoods)
        # Find the K with the highest score
        best_k = max(cv_scores, key=cv_scores.get)
        
        return best_k
    # ===================== SKL Computation Function =====================
    def compute_skl(d1_sub, d2_sub, k):
        all_points = np.concatenate([d1_sub, d2_sub])
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        
        delta_x = (x_max - x_min) * EXTEND_RATIO
        delta_y = (y_max - y_min) * EXTEND_RATIO
        x_grid = np.linspace(x_min - delta_x, x_max + delta_x, GRID_SIZE)
        y_grid = np.linspace(y_min - delta_y, y_max + delta_y, GRID_SIZE)
        grid_points = np.column_stack([arr.ravel() for arr in np.meshgrid(x_grid, y_grid)])

        P = knn_density(grid_points, d1_sub, k)
        Q = knn_density(grid_points, d2_sub, k)
        
        P_safe = np.maximum(P, EPSILON)
        Q_safe = np.maximum(Q, EPSILON)
        
        kl_pq = np.sum(P_safe * (np.log(P_safe) - np.log(Q_safe)))
        kl_qp = np.sum(Q_safe * (np.log(Q_safe) - np.log(P_safe)))
        
        dx_step = x_grid[1] - x_grid[0]
        dy_step = y_grid[1] - y_grid[0]
        
        return (kl_pq + kl_qp) * dx_step * dy_step
    # ===================== Permutation Function =====================
    def permutation_skl(seed, d_all, n1, n2, k):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(d_all))
        d1_perm = d_all[idx[:n1]]
        d2_perm = d_all[idx[n1:n1+n2]]
        return compute_skl(d1_perm, d2_perm, k)
    
    def plot_skl_distribution(null_distribution, obs_kl, p_value, results_manager):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(null_distribution, bins=30, alpha=0.7, label='Null Distribution')
        ax.axvline(obs_kl, color='r', linestyle='--', label=f'Observed KL = {obs_kl:.4f}')
        ax.set_xlabel('Symmetric KL Divergence')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Bivariate Equivalence Test (p-value = {p_value:.4f})')
        ax.legend()
        fig.tight_layout()
        results_manager.save_plot(fig, "skl_distribution.png", test_name="bivariate_test")

    def plot_scatter(d1, d2, results_manager):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(d1[:, 0], d1[:, 1], color='red', label='Edge Pair Class 1', alpha=0.6)
        ax.scatter(d2[:, 0], d2[:, 1], color='purple', label='Edge Pair Class 2', alpha=0.6)
        ax.set_xlabel('Flow 1')
        ax.set_ylabel('Flow 2')
        ax.set_title('Bivariate Data Scatter Plot')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        results_manager.save_plot(fig, "scatter_plot.png", test_name="bivariate_test")
    
    def plot_2d_density_comparison(d1, d2, k, config, results_manager):
        GRID_SIZE = config.BIVARIATE_GRID_SIZE
        EXTEND_RATIO = config.BIVARIATE_EXTEND_RATIO
        EPSILON = config.EPSILON

        all_points = np.concatenate([d1, d2])
        global_min = all_points.min()
        global_max = all_points.max()
        margin = (global_max - global_min) * EXTEND_RATIO

        axis_min = global_min - margin
        axis_max = global_max + margin
        
        x_grid = np.linspace(axis_min, axis_max, GRID_SIZE)
        y_grid = np.linspace(axis_min, axis_max, GRID_SIZE)
        XX, YY = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[XX.ravel(), YY.ravel()]

        P = knn_density(grid_points, d1, k).reshape(XX.shape)
        Q = knn_density(grid_points, d2, k).reshape(XX.shape)

        vmax_density = max(np.max(P), np.max(Q))

        fig, axes = plt.subplots(1, 3, figsize=(24, 7))

        for ax in axes:
            ax.set_xlabel('Flow 1')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            ax.grid(True)
        axes[0].set_ylabel('Flow 2')

        cp1 = axes[0].contourf(XX, YY, P, levels=20, cmap='viridis', vmin=0, vmax=vmax_density)
        fig.colorbar(cp1, ax=axes[0])
        axes[0].set_title('Density of Class 1 (P)')

        cp2 = axes[1].contourf(XX, YY, Q, levels=20, cmap='viridis', vmin=0, vmax=vmax_density)
        fig.colorbar(cp2, ax=axes[1])
        axes[1].set_title('Density of Class 2 (Q)')

        log_ratio = np.log(P + EPSILON) - np.log(Q + EPSILON)
        max_abs = np.max(np.abs(log_ratio))
        cp3 = axes[2].contourf(XX, YY, log_ratio, levels=20, cmap='coolwarm', vmin=-max_abs, vmax=max_abs)
        fig.colorbar(cp3, ax=axes[2])
        axes[2].set_title('Log Ratio: log(P / Q)')

        fig.tight_layout()
        
        results_manager.save_plot(fig, "density_comparison.png", test_name="bivariate_test")
    
    
    # --- 3. Finding the optimal K value through cross-validation---
    print("Finding optimal K using cross-validation on combined data...")
    D_all_cv = np.concatenate((D1, D2))
    k_candidates = generate_k_candidates_dynamic(len(D_all_cv))
    print(f"Dynamically generated K candidates: {k_candidates}")
    
    K_optimal = find_optimal_k_cv(D_all_cv, k_candidates)
    print(f"Optimal K selected via Cross-Validation: {K_optimal}")
    
    # --- 4. Oberserved KL computation ---
    obs_kl = compute_skl(D1, D2, K_optimal)
    print(f"Observed KL = {obs_kl:.4f}")

    # --- 5. KL under null computation---
    seeds = [RANDOM_STATE + i for i in range(N_PERMUTATIONS)]
    null_distribution = Parallel(n_jobs=-1)(
        delayed(permutation_skl)(s, D3, n_BED1, n_BED2, K_optimal) 
        for s in tqdm(seeds, desc="KL Under Null (Parallel)")
    )
    
    # --- 6. P-value computation ---
    p_value = (np.sum(np.array(null_distribution) >= obs_kl) + 1) / (N_PERMUTATIONS + 1)
    
    plot_2d_density_comparison(D1, D2, K_optimal, config, results_manager)

    # --- 7. Generate and save all visualizations ---  
    plot_skl_distribution(null_distribution, obs_kl, p_value, results_manager)
    plot_scatter(D1, D2, results_manager)
    plot_2d_density_comparison(D1, D2, K_optimal, config, results_manager)
        
    #save result
    test_results = {
        'observed_skl': float(obs_kl),
        'p_value': float(p_value),
        'optimal_k': int(K_optimal),
        'k_candidates': [int(k) for k in k_candidates],
        '#class1': int(n_BED1),
        '#class2': int(n_BED2)
    }
    results_manager.save_json(test_results, "result.json", test_name="bivariate_test")
    results_manager.add_to_report("Bivariate Equivalence Test", p_value, alpha/3, params=f"K={K_optimal}, #class1={n_BED1}, #class2={n_BED2}")
    
    print(f"\nBivariate Test Result:")
    print(f"Observed Symmetric KL Divergence: {obs_kl:.4f}")
    print(f"P value: {p_value:.4f}")
    if p_value < alpha/3:
        print(f"Conclusion: Reject the null hypothesis (p < {alpha/3}). The distributions are significantly different.")
    else:
        print(f"Conclusion: Fail to reject the null hypothesis (p >= {alpha/3}). There is no significant evidence that the distributions are different.")
        
    return test_results
