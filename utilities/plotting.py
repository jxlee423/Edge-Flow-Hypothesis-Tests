import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import beta

# plotting function for sythetic results
def load_all_summary_data(root_dir):
    """
    Loads all summary_report.csv files into a single DataFrame at once.
    """
    all_dfs = []
    print(f"--- Starting to load all summary reports from '{root_dir}' ---")
    for dirpath, _, filenames in os.walk(root_dir):
        if 'summary_report.csv' in filenames:
            file_path = os.path.join(dirpath, 'summary_report.csv')
            try:
                setting_name = os.path.basename(dirpath)
                df = pd.read_csv(file_path, sep=',')
                df['setting'] = setting_name
                df['setting_group'] = df['setting'].str.replace(r'-G\d+-', '-', regex=True)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error processing file '{file_path}': {e}")

    if not all_dfs:
        print("No 'summary_report.csv' files were found.")
        return pd.DataFrame()

    master_df = pd.concat(all_dfs, ignore_index=True)
    print("All summary reports loaded successfully.")

    master_df['Test Name Clean'] = master_df['Test Name'].str.extract(r'([A-Za-z\s]+)')[0].fillna('').astype(str).str.strip()
    
    return master_df

def calculate_power(df, rejection_condition_func, test_name_filter=None):
    """
    A flexible function to calculate statistical power based on specified filtering criteria and rejection logic.
    """
    if test_name_filter:
        df_filtered = df[df['Test Name Clean'] == test_name_filter].copy()
    else:
        df_filtered = df.copy()

    results_data = []
    for setting_group, group_df in df_filtered.groupby('setting_group'):
        all_samples = group_df['Dataset ID'].unique()
        total_samples_count = len(all_samples)
        if total_samples_count == 0:
            continue
        
        rejected_samples_count = sum(1 for sample_id in all_samples if rejection_condition_func(group_df[group_df['Dataset ID'] == sample_id]))
        power = rejected_samples_count / total_samples_count if total_samples_count > 0 else 0
        
        results_data.append({
            'setting': setting_group,
            'power': power,
            'n_rejections': rejected_samples_count,
            'n_trials': total_samples_count
        })
    return pd.DataFrame(results_data)

def calculate_clopper_pearson_interval(k, n, alpha=0.05):
    """Calculates the Clopper-Pearson confidence interval."""
    if k < 0 or k > n:
        raise ValueError("k must be between 0 and n")
    if n == 0: return (0.0, 1.0)
    lower = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    upper = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lower, upper)

def add_plot_features(df):
    """
    Adds all possible columns required for plotting to the power_df (NodeCount, CI, Rho, Distance, etc.).
    """
    if df.empty:
        return df
    
    # Attempt to extract Averaging K
    if df['setting'].str.contains('_K').any():
        df['K'] = df['setting'].str.extract(r'_K(\d+)-').astype(float)
    
    # Attempt to extract Beta
    if df['setting'].str.contains('-Beta').any():
        df['beta'] = df['setting'].str.extract(r'-Beta([\d.]+)_').astype(float)

    # Attempt to extract Rho
    if df['setting'].str.contains('-Rho').any():
        df['rho'] = df['setting'].str.extract(r'-Rho([\d.]+)_').astype(float)

    # Attempt to extract Averaging Distance
    if df['setting'].str.contains('_D').any():
        df['averaging distance'] = df['setting'].str.extract(r'_D(\d+)_').astype(float)

    # Attempt to extract Averaging Weight
    if df['setting'].str.contains('_W').any():
        df['W'] = df['setting'].str.extract(r'_W([\d.]+)_').astype(float)

    # Extract NodeCount
    def convert_k_to_numeric(node_str):
        if isinstance(node_str, str):
            return float(node_str.replace('k', '')) * 1000
        return None
    
    nodes_str = df['setting'].str.extract(r'-(\d+(?:\.\d+)?k)Nodes')
    if not nodes_str.empty:
        df['NodeCount'] = nodes_str.iloc[:, 0].apply(convert_k_to_numeric)

    # Calculate confidence intervals
    ci_bounds = df.apply(lambda row: calculate_clopper_pearson_interval(row['n_rejections'], row['n_trials']), axis=1)
    df[['lower_bound', 'upper_bound']] = pd.DataFrame(ci_bounds.tolist(), index=df.index)
    df['y_err_lower'] = df['power'] - df['lower_bound']
    df['y_err_upper'] = df['upper_bound'] - df['power']
    
    return df

# =============================================================================
# REGION 2: GENERIC PLOTTING FUNCTION
# =============================================================================

def plot_power_chart(ax, power_df, title, group_by_col, legend_title, custom_palette):
    """
    A generic function to plot the power chart on a specified Axes object.
    """
    ax.clear()
    if power_df.empty or group_by_col not in power_df.columns or power_df[group_by_col].isnull().all():
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center')
        ax.set_title(title, fontsize=16)
        return

    for group_value in sorted(power_df[group_by_col].unique()):
        group_df = power_df[power_df[group_by_col] == group_value].sort_values('NodeCount')
        ax.errorbar(
            x=group_df['NodeCount'],
            y=group_df['power'],
            yerr=[group_df['y_err_lower'], group_df['y_err_upper']],
            label=group_value,
            color=custom_palette.get(group_value, 'gray'),
            fmt='-o', capsize=5, markersize=8
        )

    ax.set_xscale('log')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Number of Nodes (Log Scale)', fontsize=12)
    ax.set_ylabel('Power (#Rejection / #Samples)', fontsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.tick_params(axis='x', which='major', labelsize=10)

    unique_nodes = sorted(power_df['NodeCount'].dropna().unique())
    if unique_nodes:
        ax.set_xticks(unique_nodes)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:g}k'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, label='α = 0.05')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title=legend_title, fontsize=10, title_fontsize=11, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, which="both", ls="--")

# =============================================================================
# REGION 3: MAIN ANALYSIS AND PLOTTING WORKFLOW
# =============================================================================

def run_analysis_and_generate_plots(config):
    """
    A complete analysis workflow to load data, calculate power, and generate plots based on the provided configuration.
    """
    print(f"\n{'='*80}\nRunning analysis for: {config['main_title']}\n{'='*80}")
    
    # --- 1. Data Preparation ---
    master_df = load_all_summary_data(config['root_dir'])
    if master_df.empty:
        print(f"Skipping analysis for '{config['root_dir']}' due to no data.")
        return

    # --- 2. Define Rejection Logic ---
    reject_by_conclusion = lambda df: (df['Conclusion'] == 'Reject H0').any()
    reject_ks_bivar_uncorrected = lambda df: (df['P-Value'] < 0.05).any()
    reject_indep_uncorrected = lambda df: (df['P-Value'] < (0.05 / 3)).any()
    def reject_ks_bivar_combined(df):
        tests_to_consider = df[df['Test Name Clean'].isin(['KS Test', 'Bivariate Equivalence Test'])]
        return (tests_to_consider['P-Value'] < (0.05 / 2)).any()

    # --- 3. Calculate Data for All Plots ---
    print("Calculating power for all 8 scenarios...")
    power_dfs = {
        "Overall (Corrected)": add_plot_features(calculate_power(master_df, reject_by_conclusion)),
        "Combined KS & Bivariate (p < 0.05/2)": add_plot_features(calculate_power(master_df, reject_ks_bivar_combined)),
        "KS Test (Corrected)": add_plot_features(calculate_power(master_df, reject_by_conclusion, 'KS Test')),
        "KS Test (Uncorrected, p < 0.05)": add_plot_features(calculate_power(master_df, reject_ks_bivar_uncorrected, 'KS Test')),
        "Independence Test (Corrected)": add_plot_features(calculate_power(master_df, reject_by_conclusion, 'Independence Test')),
        "Independence Test (Uncorrected, p < 0.05/3)": add_plot_features(calculate_power(master_df, reject_indep_uncorrected, 'Independence Test')),
        "Bivariate Equivalence (Corrected)": add_plot_features(calculate_power(master_df, reject_by_conclusion, 'Bivariate Equivalence Test')),
        "Bivariate Equivalence (Uncorrected, p < 0.05)": add_plot_features(calculate_power(master_df, reject_ks_bivar_uncorrected, 'Bivariate Equivalence Test'))
    }
    print("All calculations complete.")

    # --- 4. Plotting ---
    print("Generating plots...")
    fig, axes = plt.subplots(4, 2, figsize=(25, 33))
    axes = axes.flatten()
    
    plot_titles = list(power_dfs.keys())
    for i, title in enumerate(plot_titles):
        plot_power_chart(
            ax=axes[i],
            power_df=power_dfs[title],
            title=title,
            group_by_col=config['group_by_col'],
            legend_title=config['legend_title'],
            custom_palette=config['palette']
        )
    
    fig.suptitle(config['main_title'], fontsize=24, weight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.figtext(0.5, 0.01, config['caption'], wrap=True, horizontalalignment='center', fontsize=12, fontstyle='italic')
    
    # --- 5. Save and Display ---
    plt.savefig(f"{config['filename_base']}.pdf", format='pdf', bbox_inches='tight')
    # plt.savefig(f"{config['filename_base']}.png", format='png', bbox_inches='tight')
    plt.show()
    print(f"Analysis complete. Figures saved as '{config['filename_base']}.pdf/.png'.")