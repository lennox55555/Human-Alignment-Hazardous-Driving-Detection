'''
This script will setup the project
'''

# setup.py
from setuptools import setup, find_packages

setup(
    name="Human-Alignment-Hazardous-Driving-Detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'boto3',
        'scipy',
        'opencv-python',
        'matplotlib',
        'pymongo',
        'ffmpeg',
        'python-dotenv'
        # add other dependencies
    ]
)