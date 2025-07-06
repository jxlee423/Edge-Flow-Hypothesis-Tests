import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from tqdm import tqdm


def run_independence_test(df_reflected, config, results_manager):
    
    # --- 1. Get parameters from config ---
    N_SIMULATIONS = config.INDEPENDENCE_N_SIMULATIONS
    TOP_N_COLORS = config.INDEPENDENCE_TOP_N_COLORS
    EPSILON = config.EPSILON
    GRID_SIZE = config.INDEPENDENCE_GRID_SIZE
    RANDOM_STATE = config.RANDOM_STATE
    alpha = config.ALPHA
    
    # --- 2. Define internal classes and helper functions (encapsulate logic) ---
    def unified_knn_density(query_points, knn_model, k, dim):
        if query_points.shape[0] == 0:
            return np.array([])
            
        distances, _ = knn_model.kneighbors(query_points)
        r_k = distances[:, -1]
        
        if dim == 1:
            volume = 2 * r_k
        elif dim == 2:
            volume = np.pi * r_k**2
        else:
            raise ValueError("Only 1D or 2D data is supported")
            
        n_samples = knn_model.n_samples_fit_
        return k / (n_samples * (volume + EPSILON))
    
    class MarginalEstimator:
        def __init__(self, k):
            self.k = k
            self.X = None
            self.knn = None
            self.cdf_func = None
            
        def fit(self, data):
            self.X = np.sort(data.flatten())
            self.knn = NearestNeighbors(n_neighbors=self.k).fit(self.X.reshape(-1, 1))
            
        def pdf(self, x):
            x_reshaped = np.array(x).reshape(-1, 1)
            return unified_knn_density(x_reshaped, self.knn, self.k, dim=1)
            
        def compute_cdf(self, GRID_SIZE):
            """Numerically compute CDF and create interpolation function"""
            grid = np.linspace(self.X.min(), self.X.max(), GRID_SIZE)
            pdf_values = self.pdf(grid)
            cdf_values = np.cumsum(pdf_values) * (grid[1] - grid[0])
            cdf_values /= cdf_values[-1]
            self.cdf_func = interp1d(grid, cdf_values, kind='linear', fill_value=(0, 1), bounds_error=False)
            
        def sample(self, n_samples, rng):
            """Inverse transform sampling"""
            u = rng.random(n_samples)
            return np.interp(u, self.cdf_func.y, self.cdf_func.x) 
    
    def sample_product_distribution(estimator, n_samples, rng):
        """Sample from product distribution"""
        x_samples = estimator.sample(n_samples, rng)
        y_samples = estimator.sample(n_samples, rng)
        return np.column_stack((x_samples, y_samples))

    def compute_kl_divergence(samples, marginal_estimator, knn_joint, k_joint_val):
        """Calculate KL(product || joint)"""
            # Product density
        p_product_x = marginal_estimator.pdf(samples[:, 0])
        p_product_y = marginal_estimator.pdf(samples[:, 1])
        p_product = p_product_x * p_product_y
        # Joint density
        p_joint = unified_knn_density(samples, knn_joint, k_joint_val, dim=2)
        
        log_p_product = np.log(p_product)
        log_p_joint = np.log(p_joint)
        
        log_ratio = log_p_product - log_p_joint
        return np.mean(log_ratio)

    def generate_k_candidates_dynamic(data_size, num_candidates=20):
        k_min = 8
        k_max = int(np.sqrt(data_size))

        if k_max <= k_min:
            return [k_min]

        log_candidates = np.logspace(
            np.log10(k_min),
            np.log10(k_max),
            num=num_candidates
        )

        candidates = np.unique(np.round(log_candidates)).astype(int)
        return candidates

    def find_optimal_k_cv(data, k_candidates, dim, n_splits=5):
        cv_scores = {}
        for k in k_candidates:
            if k >= len(data) * (n_splits - 1) / n_splits:
                continue
            
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            fold_log_likelihoods = []
            
            for train_idx, val_idx in kf.split(data):
                train_data, val_data = data[train_idx], data[val_idx]
                if k >= len(train_data): continue
                
                if dim == 1:
                    train_data = train_data.reshape(-1, 1)
                    val_data = val_data.reshape(-1, 1)

                if val_data.shape[0] == 0: continue

                knn = NearestNeighbors(n_neighbors=k).fit(train_data)
                densities = unified_knn_density(val_data, knn, k, dim)
                fold_log_likelihoods.append(np.sum(np.log(np.maximum(densities, EPSILON))))
            
            if fold_log_likelihoods:
                cv_scores[k] = np.mean(fold_log_likelihoods)
        
        return max(cv_scores, key=cv_scores.get) if cv_scores else k_candidates[0]

    # --- 3. Filter color groups to be tested ---
    color_counts = df_reflected['color'].value_counts()
    top_colors = color_counts.head(TOP_N_COLORS).index.tolist()
    print(f"Performing independence test on the following Top {len(top_colors)} color groups: {top_colors}")

    all_results = []

    # --- 4. Iterate through each color group to perform the test ---
    for color in top_colors:
        print(f"\n--- Analyzing color group: {color} ---")
        
        current_edges = df_reflected[df_reflected['color'] == color]
        non_current_edges = df_reflected[df_reflected['color'] != color]
        
        # a. Dynamically select the optimal K value
        print("  - Selecting optimal K value via cross-validation...")
        marginal_train_data = np.concatenate([non_current_edges['flowX'], non_current_edges['flowY']])
        joint_train_data = current_edges[['flowX', 'flowY']].values
        
        k_candidates_marginal = generate_k_candidates_dynamic(len(marginal_train_data))
        k_candidates_joint = generate_k_candidates_dynamic(len(joint_train_data))

        print(f"   - Dynamically generated marginal K candidates: {k_candidates_marginal}")
        print(f"   - Dynamically generated joint K candidates: {k_candidates_joint}")
        
        # Ensure there is enough data for CV
        if len(marginal_train_data) < 10 * max(k_candidates_marginal) or len(joint_train_data) < 10 * max(k_candidates_joint):
             print(f"Warning: Not enough data for cross-validation in color group {color}, skipping.")
             continue
         
        optimal_k_marginal = find_optimal_k_cv(marginal_train_data, k_candidates_marginal, dim=1)
        optimal_k_joint = find_optimal_k_cv(joint_train_data, k_candidates_joint, dim=2)
        print(f"  - Optimal K_MARGINAL = {optimal_k_marginal}, Optimal K_JOINT = {optimal_k_joint}")

        rng = np.random.default_rng(RANDOM_STATE)
        # b. Estimate the marginal distribution
        marginal_estimator = MarginalEstimator(k=optimal_k_marginal)
        marginal_estimator.fit(marginal_train_data)
        marginal_estimator.compute_cdf(GRID_SIZE)
        
        # c. Estimate the joint distribution
        knn_joint = NearestNeighbors(n_neighbors=optimal_k_joint).fit(joint_train_data)

        # d. Calculate the observed KL divergence
        product_samples = sample_product_distribution(marginal_estimator, len(current_edges), rng)
        obs_kl = compute_kl_divergence(product_samples, marginal_estimator, knn_joint, optimal_k_joint)
        print(f"  - Observed KL divergence = {obs_kl:.4f}")

        # e. Simulate the null distribution
        def simulate_kl_divergence_job(seed):
            job_rng = np.random.default_rng(seed)
            sim_samples = sample_product_distribution(marginal_estimator, len(current_edges), job_rng)
            knn_sim = NearestNeighbors(n_neighbors=optimal_k_joint).fit(sim_samples)
            return compute_kl_divergence(sim_samples, marginal_estimator, knn_sim, optimal_k_joint)
        
        # Generate unique, reproducible seeds for each parallel task
        seeds = [RANDOM_STATE + i for i in range(N_SIMULATIONS)]
        
        null_kls = Parallel(n_jobs=-1)(
            delayed(simulate_kl_divergence_job)(s)
            for s in tqdm(seeds, desc=f"Simulating null distribution (Color {color})")
        )

        # e. Calculate the p-value
        p_value = (np.sum(np.array(null_kls) >= obs_kl) + 1) / (N_SIMULATIONS + 1)

        # f. Save results and plots for this color group
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))
        
        # Scatter Plot
        ax_scatter.scatter(
            non_current_edges['flowX'], 
            non_current_edges['flowY'], 
            label='Other Colors (Baseline)', 
            color='purple', 
            alpha=0.7 
        )

        ax_scatter.scatter(
            current_edges['flowX'], 
            current_edges['flowY'], 
            label=f'Color {color} Data', 
            color='red',
            alpha=0.7
        )

        ax_scatter.set_xlabel('flowX')
        ax_scatter.set_ylabel('flowY')
        ax_scatter.set_title(f'Data Scatter Plot: Color {color} vs. Others')
        ax_scatter.legend()
        ax_scatter.grid(True)
        ax_scatter.set_aspect('equal', adjustable='box')
        fig_scatter.tight_layout()

        # Null Distribution Plot
        fig_kl, ax_kl = plt.subplots(figsize=(10, 6))
        ax_kl.hist(null_kls, bins=30, alpha=0.7, label='Null Distribution')
        ax_kl.axvline(obs_kl, color='r', linestyle='--', label=f'Observed KL = {obs_kl:.4f}')
        ax_kl.set_title(f'Independence Test for Color {color} (p-value = {p_value:.4f})')
        ax_kl.set_xlabel('KL Divergence')
        ax_kl.set_ylabel('Frequency')
        ax_kl.legend()
        fig_kl.tight_layout()
        
        color_test_name = f"independence_test/color_{color}"    
        results_manager.save_plot(fig_kl, "kl_distribution.png", test_name=color_test_name)
        results_manager.save_plot(fig_scatter, "scatter_plot.png", test_name=color_test_name)
        
        
        color_result = {
        'color': int(color),
        'n_samples': int(len(current_edges)),
        'observed_kl': float(obs_kl),
        'p_value': float(p_value),
        'optimal_k_marginal': int(optimal_k_marginal),
        'optimal_k_joint': int(optimal_k_joint), 
        'k_candidates_marginal': [int(k) for k in k_candidates_marginal],
        'k_candidates_joint': [int(k) for k in k_candidates_joint]
    }
        results_manager.save_json(color_result, "result.json", test_name=color_test_name)
        
        # Add the result for this color group to the main report
        results_manager.add_to_report(f"Independence Test (Color: {color})", p_value, params=f"K_marg={optimal_k_marginal}, K_joint={optimal_k_joint}, n_samples={len(current_edges)}")
        
        print(f"\n--- Independence Test Result for Color: {color} ---")
        print(f"P value: {p_value:.4f}")
        
        if p_value < alpha:
            print(f"Conclusion: Reject the null hypothesis (p < {alpha}).")
            print("           The variables 'flowX' and 'flowY' are likely DEPENDENT for this color group.")
        else:
            print(f"Conclusion: Fail to reject the null hypothesis (p >= {alpha}).")
            print("           There is no significant evidence that 'flowX' and 'flowY' are dependent.")
            
        all_results.append(color_result)

    print("\nIndependence test completed.")
    return all_results