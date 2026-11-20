# astro-notebooks

A collection of small machine-learning experiments exploring astronomy and exoplanet data. Each notebook is designed to run in Google Colab and can be extended as I continue learning about ML and astronomical concepts.

## Experiments

| Notebook | Description | ML Concepts |
|---|---|---|
| [Exoplanet Clustering](./exoplanet_clustering.py) | Downloads confirmed exoplanet data from the NASA Exoplanet Archive and explores groups of similar planets. | K-means clustering, PCA, UMAP, Isolation Forest, silhouette score |
| [Random Forest Classification](./Random_Forest_Classifier.ipynb) | Uses Sloan Digital Sky Survey data to classify astronomical objects as galaxies, stars, or quasars. | Random Forest, train/test split, stratified sampling, classification metrics |

## 1. Exoplanet Family Discovery

The exoplanet experiment downloads confirmed exoplanet data from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) and uses unsupervised learning to discover patterns in planet properties.

It combines planet radius, mass, orbital period, equilibrium temperature, and stellar temperature to identify clusters that may resemble:

- Rocky or super-Earth planets
- Mini-Neptunes
- Neptune-like planets
- Hot Jupiter-like planets
- Warm or cold giants

### Topics covered

- Downloading live astronomical data
- Data cleaning and positive-value filtering
- Log transformations
- Feature scaling
- Choosing cluster count with silhouette scores
- K-means clustering
- 2D visualization with PCA or UMAP
- Detecting unusual planets with Isolation Forest
- Examining the exoplanet radius valley
- Investigating possible observational bias

## 2. Astronomical Object Classification

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gowri369271/astro-notebooks/blob/main/Random_Forest_Classifier.ipynb)

This notebook trains a Random Forest classifier using data from the Sloan Digital Sky Survey.

The model uses astronomical coordinates, photometric measurements, and redshift to predict whether an object is:

- A galaxy
- A star
- A quasar (QSO)

It achieved **97.99% accuracy** on the test data.

### Topics covered

- Loading and exploring an astronomical dataset
- Selecting physical features
- Cleaning invalid filter values
- Creating training and test datasets
- Training a Random Forest classifier
- Evaluating predictions with accuracy and a classification report
- Creating a confusion matrix
- Comparing feature importance
- Visualizing model results

## Running the notebooks

### Option 1: Google Colab

1. Open a notebook from the repository.
2. Click **Open in Colab**.
3. Run the cells from top to bottom.

### Option 2: Local environment

```bash
git clone https://github.com/Gowri369271/astro-notebooks.git
cd astro-notebooks
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook
```

## Dependencies

The experiments use:

- Python 3
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- IPython
- UMAP-learn for optional exoplanet visualization

UMAP is installed automatically by the exoplanet script if it is not already available. If installation fails, the script falls back to PCA.

## Project goals

- Apply machine-learning concepts to real astronomical datasets
- Understand the difference between supervised and unsupervised learning
- Explore how preprocessing changes model behavior
- Connect statistical results to astronomical interpretations
- Build an evolving collection of practical ML experiments
