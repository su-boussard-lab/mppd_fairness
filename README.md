# Group vs. Individual Fairness Evaluation

This repository contains code for evaluating both group and individual fairness in machine learning models. The framework is designed to work with any tabular dataset containing demographic attributes and model predictions.

## Setup Instructions

We can use Anaconda / Miniconda to set up the required environment for this project. Follow these steps:

- Ensure you have Anaconda or Miniconda installed on your system.

- Create the conda environment using the provided `environment.yml` file:
   ```
   conda env create -f environment.yml
   ```
   This will create a new conda environment named `mppd_fairness` with all the necessary dependencies.

- Activate the newly created environment:
   ```
   conda activate mppd_fairness
   ```
- When running Jupyter notebooks, make sure to select the `mppd_fairness` kernel.

## Project Structure
- `notebooks/`: Contains Jupyter notebooks for data analysis
- `src/`: Source code for the project
- `environment.yml`: Conda environment specification

## Data Requirements

To use this fairness evaluation framework, you'll need to prepare your own dataset with the following structure:

### Required Data Format
Your dataset should be a pandas DataFrame with:
- **Features**: Any numerical or categorical features used for your model
- **Target Variable**: The true labels (ground truth) for your prediction task
- **Model Predictions**: The predicted probabilities or binary predictions from your trained model
- **Demographic Attributes**: Sensitive attributes for fairness analysis (e.g., sex, race, ethnicity, age groups, socioeconomic indicators)

### Data Preparation Steps
1. **Load your dataset** into a pandas DataFrame
2. **Ensure demographic columns** are properly encoded (numerical codes or categorical labels)
3. **Prepare your model predictions** as a separate array/Series
4. **Create demographic mappings** (dictionaries mapping numerical codes to readable group names)

### Example Data Structure
```python
# Your dataset should look like:
X_test = pd.DataFrame({
    'feature1': [...],
    'feature2': [...],
    'sex': [0, 1, 0, ...],  # 0=Female, 1=Male
    'race_ethnicity': [1, 2, 1, ...],  # 1=White, 2=Black, etc.
    'insurance_type': [1, 2, 1, ...],  # 1=Private, 2=Public, etc.
    # ... other features
})

y_test = [0, 1, 0, ...]  # True labels
y_pred = [0.2, 0.8, 0.1, ...]  # Model predictions (probabilities)

# Demographic mappings
demographic_mappings = {
    'sex': {0: 'Female', 1: 'Male'},
    'race_ethnicity': {1: 'White', 2: 'Black', 3: 'Hispanic', 4: 'Asian'},
    'insurance_type': {1: 'Private', 2: 'Public', 3: 'Uninsured'}
}
```

## Notebooks
The Jupyter notebooks demonstrate how to perform fairness analysis:
- `group_fairness.ipynb`: Perform group fairness analysis using demographic attributes
- `05.0_ind_fairness.ipynb`: Perform individual fairness analysis
- `05.1_ind_fairness_heatmap.ipynb`: Create heatmaps of individual fairness results
- `05.2_ind_fairness_mppd.ipynb`: Additional individual fairness analysis

## Source Code
Core functionality is implemented in the `src/` directory:
- `constants.py`: Project constants and configuration values
- `util.py`: General utility functions and helper methods
- `group.py`: Group fairness metrics and evaluation functions
- `individual.py`: Individual fairness metrics and evaluation functions

## Usage Workflow
1. **Prepare your data** following the format requirements above
2. **Run group fairness analysis** using the provided functions in `group.py`
3. **Run individual fairness analysis** using the functions in `individual.py`
4. **Visualize results** using the built-in plotting functions
5. **Interpret fairness metrics** to understand potential biases in your model

## Fairness Metrics
The framework provides comprehensive fairness evaluation including:
- **Group Fairness**: Demographic parity, equalized odds, equal opportunity
- **Individual Fairness**: Similarity-based fairness using distance metrics
- **Visualization**: Heatmaps, radar charts, and comparative plots

For detailed usage examples, refer to the notebooks in the `notebooks/` directory.
