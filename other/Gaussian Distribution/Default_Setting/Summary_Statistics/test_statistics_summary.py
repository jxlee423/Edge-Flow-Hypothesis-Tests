import pandas as pd

file_path = 'G:/Summer 2024/EFHT/Synthetic Data Sets/Code/TestResults(Gaussian)/Scale_Free_Gaussian_M_1s2s3s+u_Defaultsetting.csv'
data = pd.read_csv(file_path)

def calculate_statistics(subset, test, null_val):
    stats = {
        'test': test,
        'null_is_true': null_val,
        'mean': subset['test_result'].mean(),
        'min': subset['test_result'].min(),
        'max': subset['test_result'].max(),
        '25th_percentile': subset['test_result'].quantile(0.25),
        '50th_percentile': subset['test_result'].median(),
        '75th_percentile': subset['test_result'].quantile(0.75)
    }
    return stats

results = []

for test in ['test1', 'test2', 'test3', 'test4']:
    for null_val in [0, 1]:
        subset = data[(data['test'] == test) & (data['null_is_true'] == null_val)]
        if not subset.empty:
            results.append(calculate_statistics(subset, test, null_val))

stats_df = pd.DataFrame(results)

output_file_path = 'G:/Summer 2024/EFHT/Synthetic Data Sets/Code/TestResults(Gaussian)/Scale_Free_M_Test_Statistics_Summary.csv'
stats_df.to_csv(output_file_path, index=False)

