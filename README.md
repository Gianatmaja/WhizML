# WhizML

A reusable codebase for fast data science and machine learning experimentation,
integrating various open-source tools to support automatic EDA, ML models experimentation
and tracking, model inference, model explainability, bias, and data drift analysis.

## Quick Examples

Take a look at these repositories below for a quick overview of how WhizML can
help streamline initial data science and machine learning development.
- [WhizML on CVD Analysis Use Case](https://github.com/Gianatmaja/WhizML-CVD_Analysis/tree/main)
- [WhizML on Walmart Sales Analysis Use Case](https://github.com/Gianatmaja/WhizML-Walmart_Sales/tree/main)

## Tech Stack

Some of the main open-source tools used in developing this codebase include:

- [Scikit-Learn](https://scikit-learn.org/stable/): Used for ML model training.
- [D-Tale](https://pypi.org/project/dtale/): Used for automating the EDA process.
- [Weights & Biases](https://docs.wandb.ai/): Used for experiment tracking and ML model management.
- [Explainer-Dashboard](https://explainerdashboard.readthedocs.io/en/latest/): Used for model explainabillity.
- [EvidentlyAI](https://www.evidentlyai.com/): Used for identifying data drift.

The complete set of requirements can be found in the `requirements.txt` file.

## Codebase Structure
The structure of this repository is as follows:

    .
    ├── data/                                   # Data folder
    │  ├── raw/                                 # Raw dataset      
    │  ├── clean/                               # Preprocessed dataset
    │  ├── model_input/                         # Train & test datasets
    │  ├── model/                               # Model files
    │  ├── model_output/                        # Predictions
    │  ├── reporting/                           # Drift, explainers, bias data, etc.
    ├── notebooks/
    ├── src/
    │   ├── pipeline/                           # Pipeline codes
    │   │  ├── eda.py                           # Auto EDA
    │   │  ├── model_experimentation.py         # Model experiments
    │   │  ├── model_explanation.py             # Model explainer
    │   │  ├── model_saving.py                  # Model saving
    │   │  ├── bias_analysis_data_prep.py       # Bias data preparation
    │   │  ├── data_drift.py                    # Data drift detection
    │   ├── tests/
    ├── main.py                                 # Main runner code
    ├── requirements.txt                        # Requirements file
    └── config.yml                              # Config

The main codes are in the `main.py` and `pipeline/` directory, which will be explained in more details in the
next section.

## Pipelines

There are 6 main pipelines in the codebase, namely:

- EDA: This pipeline will launch an auto-EDA dashboard, powered by the Python library D-Tale, allowing users
to get a sense of what their data looks like, as well as observe statistical attributes of each columns.
- Model Experimentation: This pipeline will trigger the model training phase, training several configurations
of Linear Regression, Logistic Regression, Random Forest, and XGBoost models, depending on the problem. Users
will be able to observe the results through the Wandb platform. (Note: Users would need to implement their own
data preprocessing pipeline to return 2 csv files, train.csv and test.csv, as input to the model experimentation
pipeline.)
- Model Finalization: This pipeline will train the best model configuration (from the model experimentation phase)
and save the model as a pickle file.
- Model Explainability: This pipeline will launch an model explainer dashboard, which can be interacted with by
the users.
- Bias Analysis Data Prep: This pipeline will prepare a .csv file, which can be used by the users to analyse
potential model bias, using the [Aequitas web app](http://aequitas.dssg.io/).
- Data Drift Analysis: This pipeline will take as input 2 data files, and check if there is any drift between
the two datasets, using the Python library, EvidentlyAI.

Users can configure each pipeline to serve their data science problem through the `config.yml` file. For more 
information regarding the configurations, refer to [this markdown](https://github.com/Gianatmaja/WhizML/blob/main/Config_Inputs.md).

Furthermore, users can also add additional pipelines to serve their respective needs better.

## Using WhizML in Your Data Science Project
To use WhizML in your data science project, perform the following steps:

1. Download the repository according to the desired release.

2. To install the requirements, run the following command:

```bash
pip install -r requirements.txt
```
3. Place your raw data file in the `data/raw/` directory, and fill-in the `config.yml` file.

4. Run the `eda` pipeline to determine the required data-preprocessing.

```YAML
task:
  eda: True
  model_experimentation: 
  model_finalization:
  model_explainability:
  bias_analysis_data_prep:
  data_drift_analysis:
```

5. Develop the requried data-preprocessing steps to obtain the train and test set data.

6. Run the `model_experimentation` pipeline and determine the best model.

7. Paste the run url (from Wandb) of the best model to the `config.yml` file.

```YAML
task:
wandb:
  user: <Wandb Username>
  project: <Wandb Project Name>
  best_model_url: <Best Run URL from Wandb>
```

8. Run the rest of the pipelines as needed.

```bash
python main.py
```


## What's Next?
WhizML will continuously be improved to serve data science use cases better. Some features planned for the next
releases include:
- More comprehensive artefacts logging in Weights & Biases
- Starter codes for data preprocessing & model inference pipelines
- Support for clustering
- Support for pipeline testing
- Support for different dataset formats