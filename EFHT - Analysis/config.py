# config.py

# --- Dataset Configuration ---
# Modify this ID before running a new dataset
DATASET_ID = ""

# Input file path
INPUT_FILE_PATH = ""

# --- Output Configuration ---
# Root directory for results reports
OUTPUTS_DIR = "outputs"

# --- Test Parameters ---
# Significance level parameter
ALPHA = 0.05
EPSILON = 1e-10
RANDOM_STATE = 423

# Independence Test Parameters
INDEPENDENCE_N_SIMULATIONS = 1000
INDEPENDENCE_TOP_N_COLORS = 3
INDEPENDENCE_GRID_SIZE = 1000
# Bivariate Equivalence Test Parameters
BIVARIATE_N_PERMUTATIONS = 1000
BIVARIATE_GRID_SIZE = 1000
BIVARIATE_EXTEND_RATIO = 0.2
