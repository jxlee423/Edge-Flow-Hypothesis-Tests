import numpy as np
from scipy.stats import levene, f_oneway, ttest_ind

def run_variance_and_mean_tests(df_class1, df_class2, config, results_manager):
    """
    Performs a two-step statistical test procedure:
    1. Use Levene's Test to check if the variances of two samples are equal.
    2. Based on the result of Levene's Test, automatically select the appropriate test 
       (standard ANOVA or Welch's ANOVA) to compare the means of the two groups.
    """
    print(f"\n============ Running Levene's and Mean Comparison Tests ============")
    
    # 1. Prepare data
    sample1 = df_class1['flow'].dropna().values
    sample2 = df_class2['flow'].dropna().values
    
    n1 = len(sample1)
    n2 = len(sample2)
    
    print(f"# of Samples in Class 1: {n1}")
    print(f"# of Samples in Class 2: {n2}")

    if n1 < 2 or n2 < 2:
        print("ERROR: Insufficient data for statistical tests.")
        return None

    alpha = config.ALPHA
    
    # =========================================================================
    # Step 1: Levene's Test (for Equality of Variances)
    # =========================================================================
    print("\n--- Step 1: Levene's Test for Equality of Variances ---")
    
    levene_stat, levene_pvalue = levene(sample1, sample2, center='median')
    
    print(f"Levene's Statistic = {levene_stat:.4f}")
    print(f"P-value = {levene_pvalue:.4e}")

    if levene_pvalue < alpha:
        print(f"Conclusion: Reject the null hypothesis (p < {alpha}). Variances are considered UNEQUAL.")
    else:
        print(f"Conclusion: Fail to reject the null hypothesis (p >= {alpha}). Variances are considered EQUAL.")

    # Save the results of Levene's Test
    levene_results = {'levene_statistic': levene_stat, 'p_value': levene_pvalue}
    results_manager.save_json(levene_results, "levene_test_result.json", test_name="levene_test")
    results_manager.add_to_report("Levene's Test (Variance)", levene_pvalue)
    
    # =========================================================================
    # Step 2: Based on the variance test result, select the appropriate test for means
    # =========================================================================
    print("\n--- Step 2: Test for Equality of Means ---")

    mean_test_name = ""
    mean_stat = None
    mean_pvalue = None

    if levene_pvalue < alpha:
        # Unequal variances -> Use Welch's t-test (the two-sample form of Welch's ANOVA)
        print("Levene's test indicates unequal variances. Running Welch's t-test.")
        mean_test_name = "Welch's t-test"
        mean_stat, mean_pvalue = ttest_ind(sample1, sample2, equal_var=False)
    else:
        # Equal variances -> Use standard ANOVA
        # Note: For two samples, the result of ANOVA is equivalent to a t-test that assumes equal variances
        print("Levene's test indicates equal variances. Running standard ANOVA.")
        mean_test_name = "Standard ANOVA"
        mean_stat, mean_pvalue = f_oneway(sample1, sample2)

    print(f"\nMean Comparison Test Result ({mean_test_name}):")
    print(f"Statistic = {mean_stat:.4f}")
    print(f"P-value = {mean_pvalue:.4e}")

    if mean_pvalue < alpha:
        print(f"Conclusion: Reject the null hypothesis (p < {alpha}). The means of the two groups are different.")
    else:
        print(f"Conclusion: Fail to reject the null hypothesis (p >= {alpha}). No significant difference in means detected.")

    # Save the results of the mean comparison test
    mean_test_results = {'test_used': mean_test_name, 'statistic': mean_stat, 'p_value': mean_pvalue}
    results_manager.save_json(mean_test_results, "ANOVA_test_result.json", test_name="AONVA_test")
    results_manager.add_to_report(f"ANOVA_Test ({mean_test_name})", mean_pvalue)
    
    # Return all test results
    final_results = {
        "levene_test": levene_results,
        "ANOVA_test": mean_test_results
    }
    
    return final_results