from setuptools import setup, find_packages
import os 

def parse_requirements(filename):
    here = os.path.dirname(__file__)
    with open(os.path.join(here, filename)) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="riskam",
    version="1.0.0",
    packages=find_packages(),
    # Copied requirements.txt
    install_requires=[
        "joblib",
        "matplotlib",
        "numpy",
        "opencv-contrib-python-headless",
        "opencv-python",
        "pillow",
        "PyYAML",
        "seaborn",
        "scikit-learn",
        "setuptools",
        "torch",
        "torchvision",
        "tqdm",
        "ultralytics",
    ],
)