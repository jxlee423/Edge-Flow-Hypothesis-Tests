import pandas as pd
import matplotlib.pyplot as plt
import os

file_paths = [
    r'G:\Summer 2024\EFHT\Synthetic Data Sets\Code\TestResults(Gaussian)\Erdos_Gaussian_M_1s2s3s+u_Defaultsetting.csv',
    r'G:\Summer 2024\EFHT\Synthetic Data Sets\Code\TestResults(Gaussian)\Scale_Free_Gaussian_M_1s2s3s+u_Defaultsetting.csv',
    r'G:\Summer 2024\EFHT\Synthetic Data Sets\Code\TestResults(Gaussian)\Small_World_Gaussian_M_1s2s3s+u_Defaultsetting.csv'
]

# Test labels
tests = ['test1', 'test2', 'test3', 'test4']

# Function to plot and save data
def plot_data(df, file_name, test, null_is_true, save_folder):
    condition_label = "null_is_true" if null_is_true else "null_is_not_true"
    y_label = "TNR" if null_is_true else "TPR"
    
    # Filter the data based on the test and null condition
    filtered_data = df[(df['test'] == test) & (df['null_is_true'] == null_is_true)]
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(filtered_data['subgraphs'], filtered_data['test_result'], marker='o')
    
    # Set plot title and labels
    plt.title(f'{file_name}_{test}_{condition_label}')
    plt.xlabel('Subgraph size (same type)')
    plt.ylabel(y_label)
    
    # Create the save path
    save_path = os.path.join(save_folder, f'{file_name}_{test}_{condition_label}.png')
    
    # Save the plot to a file
    plt.savefig(save_path)
    
    # Close the plot to avoid display
    plt.close()

# Folder to save images
save_folder = r'G:\Summer 2024\EFHT\Synthetic Data Sets\Code\Plot_Results'

# Ensure the save folder exists
os.makedirs(save_folder, exist_ok=True)

# Loop through each file and each test, handling both null_is_true and null_is_not_true
for file_path in file_paths:
    # Extract file name for titles
    file_name = file_path.split('\\')[-1].replace('.csv', '')
    
    # Read the file into a DataFrame
    data = pd.read_csv(file_path)
    
    for test in tests:
        # Plot and save for null_is_true == 1 (TNR)
        plot_data(data, file_name, test, null_is_true=1, save_folder=save_folder)
        
        # Plot and save for null_is_true == 0 (TPR)
        plot_data(data, file_name, test, null_is_true=0, save_folder=save_folder)
