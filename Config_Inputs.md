# WhizML Config Inputs
This markdown will detail the inputs needed for each configuration in the `config.yml` file.

For more examples, refer to the 2 use cases linked in the WhizML Readme.

## task
The first section of the `config.yml` file, `task`, can be filled with Boolean values, or left blank.
If the value for a pipeline key is True, then that pipeline will be run.

```YAML
task:
  eda: 
  model_experimentation: 
  model_finalization:
  model_explainability: True
  bias_analysis_data_prep:
  data_drift_analysis:
```

In the example above, the `model_explainability` pipeline will be the only pipeline executed, when the codebase is run
using the bash command below.

```bash
python main.py
```

## data_path
In this section, data paths to the various dataset used in the data science project is defined. An example can be
found below.

```YAML
data_path:
  raw: data/raw/input.csv
  clean:
  model_input:
  additional_data1: data/raw/add.csv
  additional_data2:
```

## other_directories
Other directories used in the project can be defined in this section.

```YAML
other_directories:
  split:
  model:
  scaler:
  encoder:
  predictions:
  reporting: data/reporting
  others:
```

In the example above, all reporting related files will be stored inside the `data/reporting/` directory.

## problem
In this section, the data science problem is defined.

```YAML
problem:
  classification:
    tag: True
    binary: True
    target: Default
  regression:
    tag:
    target:
  clustering:
    tag:
  exclude:
```

In the example above, the data science problem is a binary classification task, with the target (y-values) being 
the `Default` column in the data.

## preprocessing
The preprocessing section is not used in this release, and therefore will only be elaborated in future releases.

## wandb
Here, the wandb-related configurations are provided.

```YAML
wandb:
  user: azi776
  project: Test_project
  best_model_url: 
```

In the example above, the model experimentation results will be stored inside the `Test_project` project, in the Wandb account of user `azx776`.