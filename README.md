# Edge Flow Hypothesis Tests

Welcome to the **Edge Flow Hypothesis Tests** repository. This project provides a comprehensive framework for conducting hypothesis tests on network edge flows. It includes Python and MATLAB utilities for data generation, preprocessing, statistical testing, and results visualization.

## 📚 Project Structure

The repository is organized into the following main directories:

  * **`data/`**: Contains all datasets used in the project.

      * `real_wolrd`: Holds real-world network data.

      * `synthetic`: Holds synthetic data. (has been put in `.gitignore`)

  * **`scripts/`**: Includes the main Jupyter notebooks and matlab files for generating synthetic data, running the analysis pipelines, results plotting.

      * `batch_synthetic_data_generator.m`: synthetic data generator.

      * `standard_synthetic_testing_pipeline.ipynb`: The primary workflow for testing synthetic data.

      * `synthetic_results_plotting.ipynb`: Notebook for visualizing the results from the synthetic data tests.

  * **`utilities/`**: Contains the core logic and helper modules for the project.

      * `data_generation/`: A collection of MATLAB scripts (`.m` files) for creating synthetic network datasets.

      * `hypothesis_testing/`: functions with the core implementation of the hypothesis tests, data preprocessing, and results management.

      * `plotting.py`: Python script with utility functions for generating plots.

  * **`plots/`**: Stores the output figures generated from the analyses.
  
  * **`results/`**: Stores all testing results. (has been put in `.gitignore`)

  * **`other/`**: Contains supplementary materials, including older versions of methods, backups.

## 🛠️ Setup Instructions

To get started with this project, clone the repository and install the necessary dependencies.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-repo/Edge-Flow-Hypothesis-Tests.git
    cd Edge-Flow-Hypothesis-Tests
    ```

2.  **Install dependencies:**
    
    Please run `setup.ipynb` for installing dependencies.
