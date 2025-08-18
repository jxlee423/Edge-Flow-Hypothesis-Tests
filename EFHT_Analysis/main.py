import argparse
import pandas as pd
from src import config
from src.results_manager import ResultsManager
from src import data_preprocess
from src.tests import ks, bivariate, Independence

def main(input_path, dataset_id, batch_id, run_ks, run_independence, run_bivariate, jobs):
    # --- 1. Initialization ---
    print(f"Starting EFHT analysis - Batch ID: {batch_id}, Dataset ID: {dataset_id}")
    results = ResultsManager(config.OUTPUTS_DIR, batch_id, dataset_id, input_path)

    # --- 2. Load and preprocess data ---
    print("\n--- 2. Loading and preprocessing data for selected tests ---")
    df_input = pd.read_csv(input_path)

    if run_ks or run_bivariate:
        class_data = data_preprocess.Classifying(df_input)

    if run_ks:
        df_class0_KS, df_class1_KS = data_preprocess.KS_Data_Preprocessing(class_data)
    
    if run_independence:
        df_ind_data = data_preprocess.Coloring(df_input, jobs)

    if run_bivariate:
        df0, df1, dfAll = data_preprocess.BEDT_Data_Preprocessing(class_data, df_input)

    # --- 3. Run tests ---
    print("\n>>> Starting statistical tests <<<")

    # K-S Test
    if run_ks:
        print("\n--- 1. Running K-S Test ---")
        ks.run_ks_test(
            df_class0_KS,
            df_class1_KS,
            config,
            results
        )
    
    # Independence Test
    if run_independence:
        print("\n--- 2. Running Independence tests ---")
        Independence.run_independence_test(
            df_ind_data,
            config,
            results
        )

    # Bivariate Equivalence Test
    if run_bivariate:
        print("\n--- 3. Running Bivariate Equivalence Test ---")
        bivariate.run_bivariate_test(
            df0,
            df1,
            dfAll,
            config,
            results
        )
    
    # --- 4. Generate final report ---
    results.compile_report()
    print("\n"+"="*50)
    print("Analysis pipeline finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the EFHT analysis pipeline.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--id', type=str, required=True, help='Unique ID for the dataset to create an output subdirectory.')
    parser.add_argument('--batch-id', type=str, required=True, help='Unique ID for this batch run.')

    parser.add_argument('--no-ks', dest='run_ks', action='store_false', help='Do not run the K-S test.')
    parser.add_argument('--no-independence', dest='run_independence', action='store_false', help='Do not run the Independence test.')
    parser.add_argument('--no-bivariate', dest='run_bivariate', action='store_false', help='Do not run the Bivariate Equivalence test.')
    
    parser.add_argument('--jobs', type=int, default=-1, help='Number of parallel jobs for data preprocessing. -1 uses all available cores (default).')

    args = parser.parse_args()
    
    main(args.input, args.id, args.batch_id, args.run_ks, args.run_independence, args.run_bivariate, args.jobs)