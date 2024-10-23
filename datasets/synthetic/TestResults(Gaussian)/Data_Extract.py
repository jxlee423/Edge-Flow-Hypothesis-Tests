import h5py
import numpy as np
import csv

# Load the .mat file using h5py
file_path = 'G:/Summer 2024/EFHT/Synthetic Data Sets/Code/Test_Results/SmallWorld_Gaussian_M_1s2s3s+u_Defaultsetting.mat'
output_csv_path = 'G:/Summer 2024/EFHT/Synthetic Data Sets/Code/Test_Results/SmallWorld_Gaussian_M_1s2s3s+u_Defaultsetting.csv'

# List of tests to iterate over
test_names = ['test1', 'test2', 'test3', 'test4']

# Open the .mat file
with h5py.File(file_path, 'r') as mat_file:
    # Access the effect size data
    effect_size_data = mat_file['/Examples/Small_World/Effect_sizes']
    subgraph_edges_data = mat_file['/Examples/Small_World/Graphs/subgraph_edges']
    graph_valid_data = mat_file['/Examples/Small_World/Graphs/valid']
    rhos_data = mat_file['/Examples/Small_World/Cov_settings/rhos']
    null_is_true_data = mat_file['/Examples/Small_World/Cov_settings/null_is_true']
    cov_weights_data = mat_file['/Examples/Small_World/Cov_settings/weights']  # Access the weights matrix
    cov_distances_data = mat_file['/Examples/Small_World/Cov_settings/distances']  # Access the distances matrix

    # Prepare to write to CSV
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row, adding new columns
        writer.writerow(['i', 'j', 'k', 'l', 'test', 'test_result', 'test_effect', 'subgraphs', 'graph_valid', 'rho_value', 'null_is_true', 'Cov_weight', 'Cov_distance'])
        
        # Iterate over each test (test1, test2, test3, test4)
        for test_idx, test_name in enumerate(test_names):
            # Access the current test data
            test_data = mat_file[f'/Examples/Small_World/Test_Results/{test_name}']
            
            # Iterate over the actual dimensions as l, k, j, i
            for l in range(test_data.shape[0]):  # 1st dimension, actually l
                for k in range(test_data.shape[1]):  # 2nd dimension, actually k
                    for j in range(test_data.shape[2]):  # 3rd dimension, actually j
                        for i in range(test_data.shape[3]):  # 4th dimension, actually i
                            # Dereference the cell content
                            cell_ref = test_data[l, k, j, i]
                            
                            if isinstance(cell_ref, h5py.Reference):
                                # Dereference to get the actual data
                                cell_data = mat_file[cell_ref]
                                
                                # Access graph validity for the current i, j
                                graph_valid = graph_valid_data[j, i]
                                
                                if graph_valid == 1:  # Graph is valid
                                    if cell_data.size > 0:
                                        value = float(cell_data[0])  # Extract the single value as float
                                    else:
                                        value = 0  # If cell_data is empty, set to 0
                                    
                                    if test_idx < 3:  # Only for test1, test2, and test3
                                        test_effect = effect_size_data[test_idx, l, k]
                                    else:
                                        test_effect = 'NA'  # No effect size for test4
                                    
                                    # Get the subgraph edge matrix row count
                                    subgraph_ref = subgraph_edges_data[j, i]
                                    subgraph_matrix = mat_file[subgraph_ref]
                                    subgraph_rows = subgraph_matrix.shape[-1]  # Row count of the matrix

                                    # Get the rho value from the rhos matrix
                                    rho_value = rhos_data[l, k]

                                    # Get the null_is_true value from the null_is_true vector
                                    null_is_true_value = null_is_true_data[0, l]

                                    # Get the Cov_weight and Cov_distance from the corresponding matrices
                                    cov_weight = cov_weights_data[l, k]
                                    cov_distance = cov_distances_data[l, k]
                                    
                                    # Write the row to the CSV file
                                    writer.writerow([i+1, j+1, k+1, l+1, test_name, value, test_effect, subgraph_rows, graph_valid, rho_value, null_is_true_value, cov_weight, cov_distance])
                                
                                else:  # Graph is invalid, set test_result and subgraphs to 'NA'
                                    # Get the rho value, null_is_true, Cov_weight, and Cov_distance for consistency
                                    rho_value = rhos_data[l, k]
                                    null_is_true_value = null_is_true_data[0, l]
                                    cov_weight = cov_weights_data[l, k]
                                    cov_distance = cov_distances_data[l, k]
                                    writer.writerow([i+1, j+1, k+1, l+1, test_name, 'NA', 'NA', 'NA', graph_valid, rho_value, null_is_true_value, cov_weight, cov_distance])
