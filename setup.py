from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
        'geopandas',
        'pandas',
        'shapely',
        'scikit-mobility',
        'torch',
        'torchvision',
        'mlflow']

setup(
    name="AdjNet",
    version="0.0.1",
    author="Marco Cardia",
    author_email="m.cardia@studenti.unipi.it",
    description="Python implementation of the thesis AdjNet: a deep learning approach for Crowd Flow Prediction",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/jonpappalord/crowd_flow_prediction",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)