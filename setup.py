from setuptools import setup, find_packages
import os 

def parse_requirements(filename):
    here = os.path.dirname(__file__)
    with open(os.path.join(here, filename)) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="riskam",
    version="0.0.1",
    packages=find_packages(),
    # Copied requirements.txt
    install_requires=[
        "accelerate",
        "bitsandbytes",
        "datasets",
        "joblib",
        "matplotlib",
        "mediapipe",
        "opencv-contrib-python-headless",
        "opencv-python",
        "numpy<2.0.0",
        "pillow",
        "PyYAML",
        "seaborn",
        "scikit-learn",
        "setuptools",
        "timm",
        "torch",
        "torchvision",
        "tqdm",
        "transformers",
        "ultralytics"
    ],
)