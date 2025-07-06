import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import ks_2samp

def ecdf(data):
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    return sorted_data, y

def run_ks_test(df_class1_KS, df_class2_KS, config, results_manager):
    print(f"\n ============ Running KS Test ============")
    sample1 = df_class1_KS['flow'].dropna().values
    sample2 = df_class2_KS['flow'].dropna().values
    n_ks1 = len(sample1)
    n_ks2 = len(sample2)
    print(f"# of Edges in Class1 : {n_ks1}")
    print(f"# of Edges in Class2 : {n_ks2}")
    
    if n_ks1 < 2 or n_ks2 < 2:
        print("ERROR: Insufficient KS test data")
        return None
    
    alpha = config.ALPHA
    
    KS_statistic, pvalue = ks_2samp(sample1, sample2)
    
    # Plot
    x1, y1 = ecdf(sample1)
    x2, y2 = ecdf(sample2)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, alpha=0.3)
    ax.step(x1, y1, where='post', label='Class 0', lw=2, color='#1f77b4')
    ax.step(x2, y2, where='post', label='Class 1', lw=2, color='#ff7f0e')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f'KS Test Distribution Comparison\n(p-value = {pvalue:.3e})', fontsize=14)
    ax.set_xlabel('Flow Value', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.legend(loc='lower right', fontsize=12)
    fig.tight_layout()
    
    #save result
    test_results = {'KS_statistic': KS_statistic, 'p_value': pvalue}
    results_manager.save_json(test_results, "result.json", test_name="ks_test")
    results_manager.add_to_report("KS Test", pvalue)
    results_manager.save_plot(fig, "distribution_comparison.png", test_name="ks_test")
    
    print("\nKS Test Result:")
    print(f"KS_statistic D = {KS_statistic:.4f}")
    print(f"P value = {pvalue:.4e}")
    if pvalue < alpha:
        print(f"Conclusion: Reject the null hypothesis (p < {alpha}), distributions are different.")
    else:
        print(f"Conclusion: Fail to reject the null hypothesis (p >= {alpha}), distributions may be the same.")
    return test_results