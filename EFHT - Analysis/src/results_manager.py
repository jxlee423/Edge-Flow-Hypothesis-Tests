import os
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class ResultsManager:
    def __init__(self, base_output_dir, batch_id, dataset_id, input_path):
        """
        Initializes the Results Manager.
        - batch_id: A unique identifier for the current batch run, used as the top-level directory name.
        - dataset_id: The ID of the currently processed dataset, used as the subdirectory name.
        """
        # Top-level batch directory (Folder A)
        self.batch_dir = os.path.join(base_output_dir, batch_id)
        # Dedicated run directory for the current dataset (Folder B_datasetid)
        self.run_dir = os.path.join(self.batch_dir, dataset_id)
        
        # Save IDs for later use
        self.dataset_id = dataset_id
        
        # Create directories
        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        
        print(f"Results for the current dataset will be saved in: {self.run_dir}")
        print(f"The summary report will be saved in: {self.batch_dir}")

        self.results_summary = []
        self._log_metadata(input_path)

    def _log_metadata(self, input_path):
        """Logs metadata for the current run into the dedicated run directory."""
        meta = {
            "dataset_id": self.dataset_id,
            "input_file_path": input_path,
            "run_timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }
        self.save_json(meta, "meta.json")

    def get_test_dir(self, test_name):
        """Creates a subdirectory for a specific test within the dedicated run directory."""
        test_dir = os.path.join(self.run_dir, test_name)
        os.makedirs(test_dir, exist_ok=True)
        return test_dir

    def save_json(self, data, filename, test_name=None):
        """Saves a dictionary as a JSON file."""
        dir_path = self.get_test_dir(test_name) if test_name else self.run_dir
        filepath = os.path.join(dir_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def save_plot(self, fig, filename, test_name=None):
        """Saves a matplotlib figure."""
        dir_path = self.get_test_dir(test_name) if test_name else self.run_dir
        filepath = os.path.join(dir_path, filename)
        fig.savefig(filepath)
        plt.close(fig)

    def save_dataframe(self, df, filename, test_name=None):
        """Saves a DataFrame as a CSV."""
        dir_path = self.get_test_dir(test_name) if test_name else self.run_dir
        filepath = os.path.join(dir_path, filename)
        df.to_csv(filepath, index=False)

    def add_to_report(self, test_name, p_value, params='N/A'):
        """Adds a result row to the in-memory summary list."""
        conclusion = 'Reject H0' if p_value < 0.05 else 'Fail to Reject H0'
        self.results_summary.append({
            'Test Name': test_name,
            'Parameters': str(params), # Ensure parameters are strings
            'P-Value': p_value,
            'Conclusion': conclusion
        })
    
    def compile_report(self):
        """
        Compiles the results of the current run and appends them to the master report in the batch directory.
        """
        if not self.results_summary:
            print("No results to add to the report.")
            return

        report_df = pd.DataFrame(self.results_summary)
        report_df['Dataset ID'] = self.dataset_id
        
        # Reorder columns to make 'Dataset ID' the first column
        cols = ['Dataset ID'] + [col for col in report_df.columns if col != 'Dataset ID']
        report_df = report_df[cols]

        # Define the path for the master report
        master_report_path = os.path.join(self.batch_dir, "summary_report.csv")

        if os.path.exists(master_report_path):
            # If the report already exists, read the old data and append the new data
            try:
                existing_df = pd.read_csv(master_report_path)
                combined_df = pd.concat([existing_df, report_df], ignore_index=True)
                combined_df.to_csv(master_report_path, index=False)
                print(f"Appended results to: {master_report_path}")
            except pd.errors.EmptyDataError:
                # If the file exists but is empty, just write the new data
                report_df.to_csv(master_report_path, index=False)
                print(f"Created new report at: {master_report_path}")
        else:
            # If the report does not exist, create it directly
            report_df.to_csv(master_report_path, index=False)
            print(f"Created new report at: {master_report_path}")