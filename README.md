## PaLA: Partial Layer Aggregation for Client-Group Distributed Drift Adaptation

This repository contains the implementation used for the experiments of **Partial Layer Aggregation (PaLA)**, a clustering-free federated learning strategy for adapting to client-group distributed concept drift.

PaLA shares transferable feature-extraction layers through federated aggregation while keeping client-specific classifier layers local. The goal is to reduce cross-group interference under drift while avoiding the overhead of maintaining multiple global models.

### Naming Note

In the codebase, `FEDEX` is used as an internal alias for **PaLA**. Both refer to the same method: Partial Layer Aggregation, which aggregates shared extractor layers while keeping classifier layers local.

### Repository Structure

```text
.
├── data/                    # Dataset-related files and local data storage
├── distance\_metrics/         # Distance and similarity metric utilities
├── drift\_concepts/           # Drift concept definitions and drift schedules
├── federated\_network/        # Federated learning client/server simulation logic
├── jupyter\_notebooks/        # Analysis and plotting notebooks
├── log\_utils/                # Logging utilities
├── models/                   # Neural network model definitions
├── plot\_utils/               # Plotting and visualization utilities
├── strategy/                 # Federated learning strategies, including PaLA, FedAvg, Oracle
├── constants.py              # Global constants and configuration values
├── main.py                   # Main experiment entry point. Includes mainly: (1)drift scenario design,
|                                                                             (2)Federated network systems configuration parameters
|                                                                             (3)Simulation parameters
|                                                                             (4)Ablation experiments (commented)
|
├── requirements.txt          # Python dependencies
└── README.md
```

### Dependencies

The code has been tested with **Python 3.10** and **Python 3.12**.

Install the required Python dependencies with:
 ```bash
pip install -r requirements.txt
```

### Ubuntu System Dependencies

On Ubuntu, the following system packages may be required:

```bash
sudo apt-get install -y libx11-dev python3-tk tk-dev
```

<em>
(Package versions in detail, only if required) 
The following package versions were used on Windows with Python 3.10:

contourpy==1.3.2
cycler==0.12.1
filelock==3.29.0
fonttools==4.62.1
fsspec==2026.4.0
jinja2==3.1.6
kiwisolver==1.5.0
markupsafe==3.0.3
matplotlib==3.10.9
mpmath==1.3.0
networkx==3.4.2
numpy==2.2.6
packaging==26.2
pillow==12.2.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scipy==1.15.3
setuptools==81.0.0
six==1.17.0
sympy==1.14.0
torch==2.11.0
torchvision==0.26.0
typing-extensions==4.15.0
joblib==1.5.3
scikit-learn==1.7.2
threadpoolctl==3.6.0
certifi==2026.4.22
pandas==2.3.3
pytz==2026.2
tzdata==2026.2
ucimlrepo==0.0.7
</em>

### Running Experiments

The main entry point is:
```bash
python main.py
```

### Output logs 
Depending on the experiment configuration, logs and generated outputs may be written to the corresponding logging or output directories.

#### Logging Directory
Logs are stored in:
```
logs/
```
(The users can newly create/ edit or create sub-directories to fit the requirements in the code)

#### Plotting and Analysis
Plots are saved in:
```
plots/
```
(The users can newly create/ edit or create sub-directories to fit the requirements in the code)

#### Read logs
The Jupyter notebook files for reading the logs and creating custom-generated plots are in notebooks located in:
```
jupyter_notebooks/
```

### Documentation and Code explanation
Please refer to the comments, Doctrings, and explanations in the code base.

### License
This repository is licensed under the Apache License 2.0. See the LICENSE file for details.
