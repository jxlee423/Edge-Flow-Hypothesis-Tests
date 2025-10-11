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
    # ===================== KNN Density Estimation Function =====================    
    def knn_density(query_points, data_points, k):
        knn = NearestNeighbors(n_neighbors=k).fit(data_points)
        distances, _ = knn.kneighbors(query_points)
        r_k = distances[:, -1]
        volume = np.pi * r_k**2
        return k / (len(data_points) * (volume + EPSILON))
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
        global_min = all_points.min(axis=0)
        global_max = all_points.max(axis=0)
        margin = (global_max - global_min) * EXTEND_RATIO

        axis_min = global_min - margin
        axis_max = global_max + margin
        
        x_grid = np.linspace(axis_min[0], axis_max[0], GRID_SIZE)
        y_grid = np.linspace(axis_min[1], axis_max[1], GRID_SIZE)
        XX, YY = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[XX.ravel(), YY.ravel()]

        # Estimate densities for P (from d1) and Q (from d2)
        P = knn_density(grid_points, d1, k).reshape(XX.shape)
        Q = knn_density(grid_points, d2, k).reshape(XX.shape)

        vmax_density = max(np.max(P), np.max(Q))

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f'Bivariate Fit Diagnostic and Comparison (k={k})', fontsize=20)


        for i, ax in enumerate(axes):
            ax.set_xlabel('Flow 1')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(axis_min[0], axis_max[0])
            ax.set_ylim(axis_min[1], axis_max[1])
            ax.grid(True)
        axes[0].set_ylabel('Flow 2')

        # --- Plot 1: Density of Class 1 (P) with data points ---
        cp1 = axes[0].contourf(XX, YY, P, levels=20, cmap='viridis', vmin=0, vmax=vmax_density)
        fig.colorbar(cp1, ax=axes[0])
        axes[0].contour(XX, YY, P, levels=cp1.levels, colors='white', linewidths=0.5, alpha=0.8)
        axes[0].scatter(d1[:, 0], d1[:, 1], c='red', s=8, alpha=0.4, label='Class 1 Data Points')
        axes[0].set_title('Density Fit for Class 1 (P)', fontsize=16)
        axes[0].legend()


        # --- Plot 2: Density of Class 2 (Q) with data points ---
        cp2 = axes[1].contourf(XX, YY, Q, levels=20, cmap='viridis', vmin=0, vmax=vmax_density)
        fig.colorbar(cp2, ax=axes[1])
        axes[1].contour(XX, YY, Q, levels=cp2.levels, colors='white', linewidths=0.5, alpha=0.8)
        axes[1].scatter(d2[:, 0], d2[:, 1], c='red', s=8, alpha=0.4, label='Class 2 Data Points')
        axes[1].set_title('Density Fit for Class 2 (Q)', fontsize=16)
        axes[1].legend()

        # --- Plot 3: Log Ratio of Densities ---
        log_ratio = np.log(P + EPSILON) - np.log(Q + EPSILON)
        max_abs = np.max(np.abs(log_ratio))
        cp3 = axes[2].contourf(XX, YY, log_ratio, levels=20, cmap='coolwarm', vmin=-max_abs, vmax=max_abs)
        fig.colorbar(cp3, ax=axes[2])
        axes[2].set_title('Log Ratio: log(P / Q)', fontsize=16)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
        
        results_manager.save_plot(fig, "density_fit_comparison.png", test_name="bivariate_test")

    # --- 3. Finding the optimal K value through cross-validation---
    n_min = min(n_BED1, n_BED2)
    K_optimal = int(np.sqrt(n_min))
    K_optimal = max(2, K_optimal)
    print(f"Optimal K: {K_optimal}")

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
    
    # --- 7. Generate and save all visualizations ---  
    plot_skl_distribution(null_distribution, obs_kl, p_value, results_manager)
    plot_scatter(D1, D2, results_manager)
    plot_2d_density_comparison(D1, D2, K_optimal, config, results_manager)
        
    #save result
    test_results = {
        'observed_skl': float(obs_kl),
        'p_value': float(p_value),
        'optimal_k': int(K_optimal),
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
