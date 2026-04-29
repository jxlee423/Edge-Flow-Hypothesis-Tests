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

def main(input_path, dataset_id, batch_id, output_dir, run_ks, run_independence, run_bivariate, jobs, graph_type='SmallWorld', data_type='synthetic', auto_tune=False, user_fraction_top=None, user_fraction_bottom=None, apply_jitter=False):
    # --- 1. Initialization ---
    print(f"Starting EFHT analysis - Batch ID: {batch_id}, Dataset ID: {dataset_id}")
    results = ResultsManager(output_dir, batch_id, dataset_id, input_path)

    # --- 2. Load and preprocess data ---
    print("\n--- 2. Loading and preprocessing data for selected tests ---")
    df_input = pd.read_csv(input_path)

    print("   - Standardizing graph data (ensuring u < v and flip signed flows)...")
    df_input = data_preprocess.standardize_graph_data(df_input)

    df_metrics = None
    if run_ks or run_bivariate:
        print(f"   - Computing Graph Metrics for {graph_type}...")
        if graph_type == 'SmallWorld':
            df_metrics = data_preprocess.Compute_Metrics_SW(df_input)
            init_top = 0.50 if user_fraction_top is None else user_fraction_top
            init_bottom = 0.25 if user_fraction_bottom is None else user_fraction_bottom
        elif graph_type == 'StochasticBlock':
            df_metrics = data_preprocess.Compute_Metrics_SB(df_input)
            init_top = 0.50 if user_fraction_top is None else user_fraction_top
            init_bottom = 0.20 if user_fraction_bottom is None else user_fraction_bottom
        elif graph_type == 'ScaleFree':
            df_metrics = data_preprocess.Compute_Metrics_SF(df_input)
            init_top = 0.05 if user_fraction_top is None else user_fraction_top
            init_bottom = 0.70 if user_fraction_bottom is None else user_fraction_bottom

    ks_results = None
    if run_ks:
        print("   - Running KS Data Preprocessing...")
        class_data_ks, extracted_ks, ks_frac_top, ks_frac_bot = data_preprocess.run_auto_tuning_loop(
            df_metrics, df_input, graph_type, data_type, 'KS', auto_tune, init_top, init_bottom
        )
        
        # Save the results of K-S preprocessing
        class_data_ks.to_csv(os.path.join(results.run_dir, f"{dataset_id}_class_data_KS.csv"), index=False)
        extracted_ks[0].to_csv(os.path.join(results.run_dir, f"{dataset_id}_preprocessed_ks_class0.csv"), index=False)
        extracted_ks[1].to_csv(os.path.join(results.run_dir, f"{dataset_id}_preprocessed_ks_class1.csv"), index=False)
        
        ks_results = {'top': ks_frac_top, 'bot': ks_frac_bot, 'data': extracted_ks}

    bedt_results = None
    if run_bivariate:
        print("   - Running Bivariate Data Preprocessing...")
        # Save the results of BEDT_Data_Preprocessing
        class_data_bedt, extracted_bedt, bedt_frac_top, bedt_frac_bot = data_preprocess.run_auto_tuning_loop(
            df_metrics, df_input, graph_type, data_type, 'BEDT', auto_tune, init_top, init_bottom
        )
        class_data_bedt.to_csv(os.path.join(results.run_dir, f"{dataset_id}_class_data_BEDT.csv"), index=False)
        extracted_bedt[0].to_csv(os.path.join(results.run_dir, f"{dataset_id}_preprocessed_bivariate_df0.csv"), index=False)
        extracted_bedt[1].to_csv(os.path.join(results.run_dir, f"{dataset_id}_preprocessed_bivariate_df1.csv"), index=False)
        extracted_bedt[2].to_csv(os.path.join(results.run_dir, f"{dataset_id}_preprocessed_bivariate_dfAll.csv"), index=False)
        
        bedt_results = {'top': bedt_frac_top, 'bot': bedt_frac_bot, 'data': extracted_bedt}

    if run_independence:
        print("   - Running Coloring...")
        df_ind_data = data_preprocess.Coloring(df_input, jobs)

        # Save the result of Coloring
        coloring_path = os.path.join(results.run_dir, f"{dataset_id}_preprocessed_coloring.csv")
        print(f"   -> Saving colored data to: {coloring_path}")
        df_ind_data.to_csv(coloring_path, index=False)



    # --- 3. Run tests ---
    print("\n>>> Starting statistical tests <<<")

    # K-S Test
    if run_ks:
        print("\n--- 1. Running K-S Test ---")
        df_class0_KS, df_class1_KS = ks_results['data']
        res = ks.run_ks_test(
            df_class0_KS,
            df_class1_KS,
            config,
            results
        )
        if res:
            results.results_summary[-1]['Fraction Top'] = ks_results['top']
            results.results_summary[-1]['Fraction Bottom'] = ks_results['bot']
    
    # Independence Test
    if run_independence:
        print("\n--- 2. Running Independence tests ---")
        res =independence.run_independence_test(
            df_ind_data,
            config,
            results,
            apply_jitter=apply_jitter
        )

    # Bivariate Equivalence Test
    if run_bivariate:
        print("\n--- 3. Running Bivariate Equivalence Test ---")
        df0, df1, dfAll = bedt_results['data']
        res = bivariate.run_bivariate_test(
            df0,
            df1,
            dfAll,
            config,
            results
        )
        if res:
            results.results_summary[-1]['Fraction Top'] = bedt_results['top']
            results.results_summary[-1]['Fraction Bottom'] = bedt_results['bot']
    
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
    parser.add_argument('--apply-jitter', action='store_true', help='Apply tiny random jitter to data in Independence Test to fix KNN density singularity.')
    args = parser.parse_args()
    main(args.input, args.id, args.batch_id, args.output, args.run_ks, args.run_independence, args.run_bivariate, args.jobs, args.graph_type, args.data_type, args.auto_tune, args.fraction_top, args.fraction_bottom, args.apply_jitter)