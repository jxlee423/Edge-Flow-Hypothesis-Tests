import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from tqdm import tqdm

def run_independence_test(df_reflected, config, results_manager):
    
    # --- 1. Get parameters from config ---
    N_SIMULATIONS = config.INDEPENDENCE_N_SIMULATIONS
    TOP_N_COLORS = config.INDEPENDENCE_TOP_N_COLORS
    EPSILON = config.EPSILON
    GRID_SIZE = config.INDEPENDENCE_GRID_SIZE
    EXTEND_RATIO = config.INDEPENDENCE_EXTEND_RATIO 
    RANDOM_STATE = config.RANDOM_STATE
    alpha = config.ALPHA

    # --- 2. Define plotting and help functions ---
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
            
        def compute_cdf(self, GRID_SIZE, extend_ratio):
            """Numerically compute CDF and create interpolation function"""
            data_min = self.X.min()
            data_max = self.X.max()
            data_range = data_max - data_min
            grid_min = data_min - extend_ratio * data_range
            grid_max = data_max + extend_ratio * data_range
            grid = np.linspace(grid_min, grid_max, GRID_SIZE)
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
        p_product_x = marginal_estimator.pdf(samples[:, 0])
        p_product_y = marginal_estimator.pdf(samples[:, 1])
        p_product = p_product_x * p_product_y
        p_joint = unified_knn_density(samples, knn_joint, k_joint_val, dim=2)
        log_p_product = np.log(p_product + EPSILON)
        log_p_joint = np.log(p_joint + EPSILON)
        log_ratio = log_p_product - log_p_joint
        return np.mean(log_ratio), log_ratio
    
    def simulate_kl_divergence_job(seed):
        job_rng = np.random.default_rng(seed)
        n_total_samples = len(current_edges) * 2
        all_sim_samples = sample_product_distribution(marginal_estimator, n_total_samples, job_rng)
        train_samples = all_sim_samples[:len(current_edges)]
        eval_samples = all_sim_samples[len(current_edges):]
        knn_sim = NearestNeighbors(n_neighbors=optimal_k_joint).fit(train_samples)
        mean_kl, log_ratios = compute_kl_divergence(eval_samples, marginal_estimator, knn_sim, optimal_k_joint)
        return mean_kl, log_ratios

    def find_k(data):
        n_samples = len(data)
        k = int(np.sqrt(n_samples))
        k = max(2, k)
        return k

    def visualize_1d_fit(marginal_estimator, data, k_value, title, color, results_manager):
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(data, bins=100, density=True, color='gray', alpha=0.5, label='Data Histogram (Ground Truth)')
        x_grid = np.linspace(data.min(), data.max(), 500)
        pdf_fit = marginal_estimator.pdf(x_grid)
        ax.plot(x_grid, pdf_fit, 'b-', lw=2, label=f'1D-KNN PDF Fit (k={k_value})')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Flow Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)
        test_name = f"independence_test/color_{color}"
        results_manager.save_plot(fig, "1d_knn_pdf_fit.png", test_name=test_name)

    def visualize_2d_fit(joint_data, knn_model, k_value, title, color, results_manager):
        fig, ax = plt.subplots(figsize=(10, 8))
        x_min, x_max = joint_data[:, 0].min(), joint_data[:, 0].max()
        y_min, y_max = joint_data[:, 1].min(), joint_data[:, 1].max()
        x_range, y_range = x_max - x_min, y_max - y_min
        grid_x = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 100)
        grid_y = np.linspace(y_min - 0.1 * y_range, y_max + 0.1 * y_range, 100)
        XX, YY = np.meshgrid(grid_x, grid_y)
        grid_points = np.c_[XX.ravel(), YY.ravel()]
        Z_joint = unified_knn_density(grid_points, knn_model, k_value, dim=2).reshape(XX.shape)
        contour = ax.contourf(XX, YY, Z_joint, levels=20, cmap='viridis', alpha=0.7)
        fig.colorbar(contour, ax=ax, label='Estimated Density')
        ax.contour(XX, YY, Z_joint, levels=contour.levels, colors='white', linewidths=0.5)
        ax.scatter(joint_data[:, 0], joint_data[:, 1], c='red', s=10, alpha=0.5, label='Original Data Points')
        ax.set_xlabel('flowX')
        ax.set_ylabel('flowY')
        ax.set_title(title, fontsize=16)
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        test_name = f"independence_test/color_{color}"
        results_manager.save_plot(fig, "2d_knn_pdf_fit.png", test_name=test_name)

    def plot_scatter(current_edges, non_current_edges, color, results_manager):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(
            non_current_edges['flowX'], 
            non_current_edges['flowY'], 
            label='Other Colors (Baseline)', 
            color='purple', 
            alpha=0.6 
        )
        ax.scatter(
            current_edges['flowX'], 
            current_edges['flowY'], 
            label=f'Color {color} Data', 
            color='red',
            alpha=0.6
        )
        ax.set_xlabel('flowX')
        ax.set_ylabel('flowY')
        ax.set_title(f'Data Scatter Plot: Color {color} vs. Others')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        color_test_name = f"independence_test/color_{color}"
        results_manager.save_plot(fig, "scatter_plot.png", test_name=color_test_name)
        
    def plot_kl_distribution(null_kls, obs_kl, p_value, color, results_manager):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(null_kls, bins=30, alpha=0.7, label='Null Distribution')
        ax.axvline(obs_kl, color='r', linestyle='--', label=f'Observed KL = {obs_kl:.4f}')
        ax.set_title(f'Independence Test for Color {color} (p-value = {p_value:.4f})')
        ax.set_xlabel('KL Divergence')
        ax.set_ylabel('Frequency')
        ax.legend()
        fig.tight_layout()
        color_test_name = f"independence_test/color_{color}"
        results_manager.save_plot(fig, "kl_distribution.png", test_name=color_test_name)
    
    def plot_2d_density_comparison(knn_joint, k_joint, marginal_estimator, current_edges, color, results_manager):
        # 1. Create a 2D grid covering the data range
        x = current_edges['flowX']
        y = current_edges['flowY']
        
        all_values = np.concatenate([x, y])
        axis_min, axis_max = all_values.min(), all_values.max()
        
        x_grid = np.linspace(axis_min, axis_max, 100)
        y_grid = np.linspace(axis_min, axis_max, 100)
        
        XX, YY = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[XX.ravel(), YY.ravel()]

        # 2. P_joint density on the grid
        Z_joint = unified_knn_density(grid_points, knn_joint, k_joint, dim=2).reshape(XX.shape)

        # 3. P_product density on the grid
        p_prod_x = marginal_estimator.pdf(grid_points[:, 0])
        p_prod_y = marginal_estimator.pdf(grid_points[:, 1])
        Z_product = (p_prod_x * p_prod_y).reshape(XX.shape)
        
        # 4. Plot
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
        for ax in axes:
            ax.set_xlabel('flowX')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            ax.grid(True)

        axes[0].set_ylabel('flowY')

        # Figure 1: P_joint (real joint distribution)
        cp1 = axes[0].contourf(XX, YY, Z_joint, levels=20, cmap='viridis')
        fig.colorbar(cp1, ax=axes[0])
        axes[0].set_title(f'P_joint for Color {color}\n(Observed Joint Density)')

        # Figure 2: P_product (Distribution under the assumption of independence)
        cp2 = axes[1].contourf(XX, YY, Z_product, levels=20, cmap='viridis')
        fig.colorbar(cp2, ax=axes[1])
        axes[1].set_title(f'P_product (from Others)\n(Independent Model Density)')

        # Figure 3: Variance map (log(P_joint / P_product))
        log_ratio = np.log(Z_joint + EPSILON) - np.log(Z_product + EPSILON)
        max_abs = np.max(np.abs(log_ratio))
        cp3 = axes[2].contourf(XX, YY, log_ratio, levels=20, cmap='coolwarm', vmin=-max_abs, vmax=max_abs)
        fig.colorbar(cp3, ax=axes[2])
        axes[2].set_title(f'Log Ratio: log(P_joint / P_product)')

        fig.tight_layout()
        
        color_test_name = f"independence_test/color_{color}"
        results_manager.save_plot(fig, "2d_density_comparison.png", test_name=color_test_name)

    def plot_log_ratio_distribution(obs_log_ratios, null_log_ratios_sample, color, results_manager):
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(obs_log_ratios, bins=50, density=True, color='red', alpha=0.6, label=f'Observed Log Ratios (Mean={np.mean(obs_log_ratios):.2f})')
        ax.hist(null_log_ratios_sample, bins=50, density=True, color='blue', alpha=0.6, label=f'Null Log Ratios (A Sample from Null Distribution, Mean={np.mean(null_log_ratios_sample):.2f})')

        ax.set_title(f'Distribution of Per-Point Log Ratios for Color {color}', fontsize=16)
        ax.set_xlabel('Log Ratio = log(P_product) - log(P_joint)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)
        ax.axvline(0, color='black', linestyle='--', lw=2, label='Zero Line')
        
        fig.tight_layout()
        color_test_name = f"independence_test/color_{color}"
        results_manager.save_plot(fig, "log_ratio_distribution.png", test_name=color_test_name)

    # --- 3. Filter color groups to be tested ---
    color_counts = df_reflected['color'].value_counts()
    top_colors = color_counts.head(TOP_N_COLORS).index.tolist()
    print(f"Performing independence test on the following Top {len(top_colors)} color groups: {top_colors}")

    all_results = []

    # --- 4. Main analysis loop for each color ---
    for color in top_colors:
        print(f"\n--- Analyzing color group: {color} ---")
        rng = np.random.default_rng(RANDOM_STATE)
        current_edges = df_reflected[df_reflected['color'] == color]
        non_current_edges = df_reflected[df_reflected['color'] != color]
        
        marginal_train_data = np.concatenate([non_current_edges['flowX'], non_current_edges['flowY']])
        joint_train_data = current_edges[['flowX', 'flowY']].values
                 
        # a. Select K using the robust heuristic
        optimal_k_marginal = find_k(marginal_train_data)
        optimal_k_joint = find_k(joint_train_data)
        print(f"  - K_MARGINAL = {optimal_k_marginal}, K_JOINT = {optimal_k_joint}")

        # b. Estimate the marginal distribution
        marginal_estimator = MarginalEstimator(k=optimal_k_marginal)
        marginal_estimator.fit(marginal_train_data)
        marginal_estimator.compute_cdf(GRID_SIZE, EXTEND_RATIO)

        # c. Estimate the joint distribution
        knn_joint = NearestNeighbors(n_neighbors=optimal_k_joint).fit(joint_train_data)
        
        print("  - Generating diagnostic plots for 1D and 2D fits...")
        visualize_1d_fit(marginal_estimator, marginal_train_data, optimal_k_marginal,
                         f"1D Marginal PDF Fit (k={optimal_k_marginal})", color, results_manager)
        visualize_2d_fit(joint_train_data, knn_joint, optimal_k_joint,
                         f"2D Joint PDF Fit for Color {color} (k={optimal_k_joint})", color, results_manager)

        # d. Calculate the observed KL divergence
        
        product_samples = sample_product_distribution(marginal_estimator, len(current_edges), rng)
        obs_kl, obs_log_ratios = compute_kl_divergence(product_samples, marginal_estimator, knn_joint, optimal_k_joint)
        print(f"  - Observed KL divergence = {obs_kl:.4f}")

        # e. Simulate the null distribution
        seeds = [RANDOM_STATE + i for i in range(N_SIMULATIONS)]
        simulate_results_list = Parallel(n_jobs=-1)(
            delayed(simulate_kl_divergence_job)(s)
            for s in tqdm(seeds, desc=f"Simulating null distribution (Color {color})")
        )


        null_kls = [item[0] for item in simulate_results_list]
        null_log_ratios_list = [item[1] for item in simulate_results_list]

        # f. Calculate the p-value
        p_value = (np.sum(np.array(null_kls) >= obs_kl) + 1) / (N_SIMULATIONS + 1)
        
        # g. Generating ALL visualizations
        plot_scatter(current_edges, non_current_edges, color, results_manager)
        plot_kl_distribution(null_kls, obs_kl, p_value, color, results_manager)
        plot_2d_density_comparison(knn_joint, optimal_k_joint, marginal_estimator, current_edges, color, results_manager)
        plot_log_ratio_distribution(obs_log_ratios, null_log_ratios_list[0], color, results_manager)
        
        
        color_result = {
        'color': int(color),
        'n_samples': int(len(current_edges)),
        'observed_kl': float(obs_kl),
        'p_value': float(p_value),
        'optimal_k_marginal': int(optimal_k_marginal),
        'optimal_k_joint': int(optimal_k_joint)
    }
        results_manager.save_json(color_result, "result.json", test_name=f"independence_test/color_{color}")
        results_manager.add_to_report(f"Independence Test (Color: {color})", p_value, alpha/(3*TOP_N_COLORS), params=f"K_marg={optimal_k_marginal}, K_joint={optimal_k_joint}, #target_color_samples={len(current_edges)}")
        
        print(f"\n--- Independence Test Result for Color: {color} ---")
        print(f"P value: {p_value:.4f}")
        
        if p_value < alpha/(3*TOP_N_COLORS):
            print(f"Conclusion: Reject the null hypothesis (p < {alpha/(3*TOP_N_COLORS)}).")
            print("           The variables 'flowX' and 'flowY' are likely DEPENDENT for this color group.")
        else:
            print(f"Conclusion: Fail to reject the null hypothesis (p >= {alpha/(3*TOP_N_COLORS)}).")
            print("           There is no significant evidence that 'flowX' and 'flowY' are dependent.")
            
        all_results.append(color_result)

    print("\nIndependence test completed.")
    return all_results