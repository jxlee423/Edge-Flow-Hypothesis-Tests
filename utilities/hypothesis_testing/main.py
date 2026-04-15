import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import pandas as pd
from utilities.hypothesis_testing import config
from utilities.hypothesis_testing.results_manager import ResultsManager
from utilities.hypothesis_testing import data_preprocess
from utilities.hypothesis_testing.tests import ks, bivariate, independence
import os

def main(input_path, dataset_id, batch_id, output_dir, run_ks, run_independence, run_bivariate, jobs, graph_type='SmallWorld', data_type='synthetic', auto_tune=False, user_fraction_top=None, user_fraction_bottom=None):
    # --- 1. Initialization ---
    print(f"Starting EFHT analysis - Batch ID: {batch_id}, Dataset ID: {dataset_id}")
    results = ResultsManager(output_dir, batch_id, dataset_id, input_path)

    # --- 2. Load and preprocess data ---
    print("\n--- 2. Loading and preprocessing data for selected tests ---")
    df_input = pd.read_csv(input_path)

    print("   - Standardizing graph data (ensuring u < v and flip signed flows)...")
    df_input = data_preprocess.standardize_graph_data(df_input)

    if run_ks or run_bivariate:
        mode_str = f"Auto-Tune" if (data_type == 'real' and auto_tune) else f"Manual/Fixed"
        print(f"   - Running Classifying (using {graph_type} graph)...")
        if graph_type == 'SmallWorld':
            fraction_top = 0.5 if user_fraction_top is None else user_fraction_top
            fraction_bottom = 0.25 if user_fraction_bottom is None else user_fraction_bottom
        elif graph_type == 'StochasticBlock':
            fraction_top = 0.5 if user_fraction_top is None else user_fraction_top
            fraction_bottom = 0.20 if user_fraction_bottom is None else user_fraction_bottom
        elif graph_type == 'ScaleFree':
            fraction_top = 0.05 if user_fraction_top is None else user_fraction_top
            fraction_bottom = 0.70 if user_fraction_bottom is None else user_fraction_bottom

        max_iterations = 10 if (auto_tune and data_type == 'real') else 1
        tolerance = 0.4
        loop_iter = 0

        while loop_iter < max_iterations:
            loop_iter += 1
            if data_type == 'real':
                if auto_tune:
                    print(f"\n>>> Auto-Tuning Iteration {loop_iter}/{max_iterations} | fraction_top: {fraction_top:.2f}, fraction_bottom: {fraction_bottom:.2f} <<<")
                else:
                    print(f"\n>>> Manual/Fixed Processing | fraction_top: {fraction_top:.2f}, fraction_bottom: {fraction_bottom:.2f} <<<")
            
            if graph_type == 'SmallWorld':
                class_data = data_preprocess.Classifying_SW(df_input, data_type, fraction_top, fraction_bottom)
            elif graph_type == 'StochasticBlock':
                class_data = data_preprocess.Classifying_SB(df_input, data_type, fraction_top, fraction_bottom)
            elif graph_type == 'ScaleFree':
                class_data = data_preprocess.Classifying_SF(df_input, data_type, fraction_top, fraction_bottom)

            df0, df1, dfAll = data_preprocess.BEDT_Data_Preprocessing(class_data, df_input)
            len0 = len(df0)
            len1 = len(df1)

            if not (auto_tune and data_type == 'real'):
                print(f"   [Bivariate Edge Pairs Counts] Class 0: {len0}, Class 1: {len1} (Fixed thresholds applied)")
                break

            ratio = max(len0, len1) / min(len0, len1)

            if ratio <= 1.0 + tolerance:
                print(f"  Acceptable balance reached (diff < {tolerance*100:.0f}%). Class 0: {len0}, Class 1: {len1}")
                break
                
            print(f"  Imbalance too high (Class 0: {len0}, Class 1: {len1}). Adjusting fractions...")
            
            step_size = 0.1
            if len0 < len1:
                fraction_top = min(0.90, fraction_top + step_size)
                fraction_bottom = max(0.05, fraction_bottom - 0.5*step_size)
            else:
                fraction_bottom = min(0.90, fraction_bottom + step_size)
                fraction_top = max(0.05, fraction_top - 0.5*step_size)

        if loop_iter == max_iterations and auto_tune and data_type == 'real':
            print("  Max iterations reached. Proceeding with the current best fractions.")

        results.set_fractions(fraction_top, fraction_bottom)
        # Save the result of Classifying
        class_data_path = os.path.join(results.run_dir, f"{dataset_id}_preprocessed_class_data.csv")
        print(f"   -> Saving classified data to: {class_data_path}")
        class_data.to_csv(class_data_path, index=False)

    if run_ks:
        print("   - Running KS_Data_Preprocessing...")
        df_class0_KS, df_class1_KS = data_preprocess.KS_Data_Preprocessing(class_data)
        
        # Save the results of KS_Data_Preprocessing
        ks0_path = os.path.join(results.run_dir, f"{dataset_id}_preprocessed_ks_class0.csv")
        ks1_path = os.path.join(results.run_dir, f"{dataset_id}_preprocessed_ks_class1.csv")
        print(f"   -> Saving K-S class 0 data to: {ks0_path}")
        df_class0_KS.to_csv(ks0_path, index=False)
        print(f"   -> Saving K-S class 1 data to: {ks1_path}")
        df_class1_KS.to_csv(ks1_path, index=False)

    if run_independence:
        print("   - Running Coloring...")
        df_ind_data = data_preprocess.Coloring(df_input, jobs)

        # Save the result of Coloring
        coloring_path = os.path.join(results.run_dir, f"{dataset_id}_preprocessed_coloring.csv")
        print(f"   -> Saving colored data to: {coloring_path}")
        df_ind_data.to_csv(coloring_path, index=False)

    if run_bivariate:
        print("   - Running Bivariate Data Preprocessing...")
        # Save the results of BEDT_Data_Preprocessing
        biv0_path = os.path.join(results.run_dir, f"{dataset_id}_preprocessed_bivariate_df0.csv")
        biv1_path = os.path.join(results.run_dir, f"{dataset_id}_preprocessed_bivariate_df1.csv")
        bivAll_path = os.path.join(results.run_dir, f"{dataset_id}_preprocessed_bivariate_dfAll.csv")
        print(f"   -> Saving Bivariate df0 data to: {biv0_path}")
        df0.to_csv(biv0_path, index=False)
        print(f"   -> Saving Bivariate df1 data to: {biv1_path}")
        df1.to_csv(biv1_path, index=False)
        print(f"   -> Saving Bivariate dfAll data to: {bivAll_path}")
        dfAll.to_csv(bivAll_path, index=False)

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
        independence.run_independence_test(
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
    parser.add_argument('--output', type=str, required=True, help='Path to the base output directory for results.')
    parser.add_argument('--no-ks', dest='run_ks', action='store_false', help='Do not run the K-S test.')
    parser.add_argument('--no-independence', dest='run_independence', action='store_false', help='Do not run the Independence test.')
    parser.add_argument('--no-bivariate', dest='run_bivariate', action='store_false', help='Do not run the Bivariate Equivalence test.')
    parser.add_argument('--jobs', type=int, default=-1, help='Number of parallel jobs for data preprocessing. -1 uses all available cores (default).')
    parser.add_argument('--graph-type', type=str, default='SmallWorld', choices=['SmallWorld', 'StochasticBlock', 'ScaleFree'], help="Type of graph model for classification. 'SmallWorld' uses Classifying_SM, 'StochasticBlock' uses Classifying_SB. Default is 'smallworld'.")
    parser.add_argument('--data-type', type=str, default='synthetic', choices=['synthetic', 'real'], help="'synthetic' for fixed thresholds or 'real' for dynamic balancing.")
    parser.add_argument('--auto-tune', action='store_true', help='Enable auto-tuning for fraction thresholds (applies only if data-type is real).')
    parser.add_argument('--fraction-top', type=float, default=None, help='Manually specify the top fraction. Overrides defaults.')
    parser.add_argument('--fraction-bottom', type=float, default=None, help='Manually specify the bottom fraction. Overrides defaults.')
    args = parser.parse_args()
    main(args.input, args.id, args.batch_id, args.output, args.run_ks, args.run_independence, args.run_bivariate, args.jobs, args.graph_type, args.data_type, args.auto_tune, args.fraction_top, args.fraction_bottom)