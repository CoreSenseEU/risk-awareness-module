"""
data.paths

The file and directory paths to various datasets.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


# Dataset names
CS_ROBOCUP_2023 = "cs_robocup_2023"

# The dir of the ROS datasets (that need to be converted before using them in ML)
ROS_DATA_DIR = ROOT_DIR / "ros_datasets"

# The dir of datasets used for ML
ML_DATA_DIR = ROOT_DIR / "ml_datasets"

# The ML models
ML_MODELS_DIR = ROOT_DIR / "ml_models"

# The ROS datasets
CS_ROBOCUP_2023_ROS_DIR = ROS_DATA_DIR / "cs_robocup_2023"


# The ML datasets
CS_ROBOCUP_2023_ML_DIR = ML_DATA_DIR / "cs_robocup_2023"
CS_ROBOCUP_2023_ML_RAW_DIR = CS_ROBOCUP_2023_ML_DIR / "raw_dataset"
CS_ROBOCUP_2023_ML_FEAT_DIR = CS_ROBOCUP_2023_ML_DIR / "features"
CS_ROBOCUP_2023_GROUND_TRUTH_PATH = CS_ROBOCUP_2023_ML_DIR / "ground_truth_risk.json"
