# TabPFN

[![PyPI version](https://badge.fury.io/py/tabpfn.svg)](https://badge.fury.io/py/tabpfn)
[![Downloads](https://pepy.tech/badge/tabpfn)](https://pepy.tech/project/tabpfn)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.gg/BHnX2Ptf4j)
[![Documentation](https://img.shields.io/badge/docs-priorlabs.ai-blue)](https://priorlabs.ai/docs)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/tabpfn/)

<img src="https://github.com/PriorLabs/tabpfn-extensions/blob/main/tabpfn_summary.webp" width="80%" alt="TabPFN Summary">

> ⚡ **GPU Recommended**:
> For optimal performance, use a GPU (even older ones with ~8GB VRAM work well; 16GB needed for some large datasets).
> On CPU, only small datasets (≲1000 samples) are feasible.
> No GPU? Use our free hosted inference via [TabPFN Client](https://github.com/PriorLabs/tabpfn-client).

### Installation
Official installation (pip)
```bash
pip install tabpfn
```
OR installation from source
```bash
pip install "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git"
```
OR local development installation
```bash

git clone https://github.com/PriorLabs/TabPFN.git --depth 1
pip install -e "TabPFN[dev]"
```

### Basic Usage

#### Classification
```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize a classifier
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
```

#### Regression
```python
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor

# Load Boston Housing data
df = fetch_openml(data_id=531, as_frame=True)  # Boston Housing dataset
X = df.data
y = df.target.astype(float)  # Ensure target is float for regression

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize the regressor
regressor = TabPFNRegressor()
regressor.fit(X_train, y_train)

# Predict on the test set
predictions = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
```

### Best Results

For optimal performance, use the `AutoTabPFNClassifier` or `AutoTabPFNRegressor` for post-hoc ensembling. These can be found in the [TabPFN Extensions](https://github.com/PriorLabs/tabpfn-extensions) repository. Post-hoc ensembling combines multiple TabPFN models into an ensemble.

**Steps for Best Results:**
1. Install the extensions:
   ```bash
   git clone https://github.com/priorlabs/tabpfn-extensions.git
   pip install -e tabpfn-extensions
   ```

2.
   ```python
   from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier

   clf = AutoTabPFNClassifier(max_time=120, device="cuda") # 120 seconds tuning time
   clf.fit(X_train, y_train)
   predictions = clf.predict(X_test)
   ```
