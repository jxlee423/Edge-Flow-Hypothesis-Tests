import argparse
import pandas as pd
import os
import config
from src.results_manager import ResultsManager
from src import data_preprocess
from src.tests import ks, bivariate, Levene_anova, Independence

def main(input_path, dataset_id, batch_id):
    # --- 1. Initialization ---
    print(f"Starting EFHT analysis - Batch ID: {batch_id}, Dataset ID: {dataset_id}")
    results = ResultsManager(config.OUTPUTS_DIR, batch_id, dataset_id, input_path)

    # --- 2. Load and preprocess data ---
    df_input = pd.read_csv(input_path)
    class_data = data_preprocess.Classifying(df_input)
    df_class0_KS, df_class1_KS, df_class0_ANOVA, df_class1_ANOVA = data_preprocess.KS_Data_Preprocessing(class_data)
    df_ind_data = data_preprocess.Coloring(df_input)
    df0, df1, dfAll = data_preprocess.BEDT_Data_Preprocessing(class_data,df_input)

    # --- 3. Run tests ---
    print("\n>>> Starting statistical tests <<<")

    # K-S Test
    print("\n--- 1. Running K-S Test ---")
    ks.run_ks_test(
        df_class0_KS,
        df_class1_KS,
        config,
        results
    )
    
    print("\n--- 2. Running Levene and ANOVA tests ---")
    Levene_anova.run_variance_and_mean_tests(
        df_class0_ANOVA,
        df_class1_ANOVA,
        config,
        results
    )
    
    # Independence Test
    print("\n--- 3. Running Independence tests ---")
    Independence.run_independence_test(
        df_ind_data,
        config,
        results
    )

    #Bivariate Equivalence Test
    print("\n--- 4. Running Bivariate Equivalence Test ---")
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
    # 4. Add the new --batch-id parameter
    parser.add_argument('--batch-id', type=str, required=True, help='Unique ID for this batch run.')
    
    args = parser.parse_args()
    
    # 5. Pass the new batch_id parameter to the main function
    main(args.input, args.id, args.batch_id)