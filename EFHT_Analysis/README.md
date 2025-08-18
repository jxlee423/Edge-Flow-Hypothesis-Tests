# EFHT - Analysis Pipeline

## 1. Project Overview

This project provides a comprehensive analysis pipeline for graph-based datasets. The pipeline reads edge list data from a CSV file, preprocesses it into different forms, runs three distinct statistical tests, and generates structured results including plots and summary reports.

The core analyses include:
- **K-S Test**: Compares the distribution of edge flows between two classes of edges.
- **Levene's and ANOVA/t-test**: Compares the variance and mean of edge flows between two classes. (Just a supplemental test.)
- **Independence Test**: For different "colors" (groups) of edges, tests whether the flows of paired edges are statistically independent.
- **Bivariate Equivalence Test**: Compares the joint distribution of flow pairs between two classes of edges.

---

## 2. Project Structure

The project is organized using a standard, modular structure to ensure clarity and maintainability.

```
EFHT-Analysis/
|
├── data/
│   
|
├── Standard Testing Pipeline.ipynb
│
|
├── outputs/
│   └── (This folder is created automatically to store all results)
|
├── src/
│   ├── __init__.py           # Makes 'src' a Python package.
│   ├── data_preprocess.py    # Contains all data preprocessing and transformation functions.
│   ├── results_manager.py    # A class that handles all file I/O and report generation.
│   └── tests/
│       ├── __init__.py       # Makes 'tests' a Python sub-package.
│       ├── bivariate.py      # Bivariate Equivalence Test logic.
│       ├── Independence.py   # Independence Test logic.
│       ├── ks.py             # K-S Test logic.
│       └── config.py                 # Central configuration file for all parameters.
|
├── main.py                   # The main entry point to run the entire analysis pipeline.
├── requirements.txt          # A list of all required Python packages for the project.
└── README.md                 # This documentation file.
```

---

## 3. Installation

To set up the environment and install all necessary dependencies, follow these steps:

1.  **Clone the repository** ensure you have this EFHT - ANALYSIS folder.
2.  **Navigate to the project root directory**:
    ```bash
    cd path/to/EFHT-Analysis
    ```
3.  **Install all required packages** using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

---

## 4. How to Run the Analysis

There are two primary ways to run the analysis pipeline.

### Method A: Using the Jupyter Notebook (Recommended for Batch Processing)

The `notebooks/Standard Testing Pipeline.ipynb` notebook is designed to run the analysis on multiple dataset files in a loop.

1.  **Place your data**: Move all your input `.csv` files into the `data/` directory.
2.  **Configure the notebook**: Open `notebooks/run_all.ipynb` and modify the `dataset_folder` variable and the file loop logic to match your data files.
3.  **Run the cells**: Execute the cells in the notebook. The script will call `main.py` for each file, and all results will be organized in the `outputs/` directory.

### Method B: Using the Command Line (for a Single Run)

You can run the analysis on a single dataset directly from your terminal.

1.  **Open a terminal** and navigate to the project root directory.
2.  **Execute the `main.py` script** with the required command-line arguments.

**Command Structure:**
```bash
python main.py --input <path_to_your_csv> --id <unique_dataset_id> --batch-id <your_batch_run_id>
```

**Example:**
```bash
python main.py --input "data/0630SW_2000Ns1.csv" --id "0630SW_2000Ns1_final" --batch-id "SmallWorld_Experiment"
```

* `--input`: The path to the input CSV file.
* `--id`: A unique name for this specific dataset's analysis. This will be used to create a results subfolder.
* `--batch-id`: A name for the entire group of runs. This will be the top-level results folder.

---

## 5. Configuration

All parameters for the statistical tests and file paths can be configured in the **`config.py`** file. This includes:
- Significance levels (`ALPHA`)
- Simulation/permutation counts (`N_SIMULATIONS`, `N_PERMUTATIONS`)
- k-NN parameters and grid sizes.

---

## 6. Output Structure

All results are saved within the `outputs/` directory, following this structure:

```
outputs/
└── <batch_id>/
    └── <dataset_id>/
        ├── bivariate_test/
        │   ├── result.json
        │   ├── scatter_plot.png
        │   └── skl_distribution.png
        ├── independence_test/
        │   └── color_X/
        │       ├── result.json
        │       ├── kl_distribution.png
        │       └── scatter_plot.png
        ├── ks_test/
        │   ├── distribution_comparison.png
        │   └── result.json
        ├── levene_test/
        │   └── levene_test_result.json
        │── ANOVA_test/
        │   └── ANOVA_test_result.json
        └── meta.json               # Metadata for this specific run.
    └── summary_report.csv          # A master report summarizing results from all datasets in the batch.
```