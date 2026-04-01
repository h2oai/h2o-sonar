# Predictive CLI Hello World Example

This example demonstrates how to use **H2O Sonar CLI** to explain predictions from a scikit-learn model.

## Overview

This example includes:
- **Dataset**: `creditcard.csv` - A credit card default dataset
- **Model**: `creditcard-binomial-sklearn-1.8.0-gbm.pkl` - A pre-trained scikit-learn Gradient Boosting classifier
- **Scripts**: Installation and execution scripts for easy setup


## Prerequisites
Prepare prerequisites:

- **Python 3.11**
- **Java 1.7+** (required for H2O-3 backend)
- **uv** package manager

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For more details, see: https://github.com/h2oai/h2o-sonar#installation

## Quick Start

### 1. Copy the H2O Sonar wheel file

First, obtain the h2o-sonar wheel file and copy it to this directory:

* https://github.com/h2oai/h2o-sonar#installation


### 2. Run the installation script

```bash
cd examples/predictive-cli-hello-world-example
./install-example.sh
```

This script will:
- Verify prerequisites (uv, Java)
- Create a Python 3.11 virtual environment (`.venv/`)
- Install h2o-sonar from the wheel file


### 3. Run the example

```bash
./run-example.sh
```

This will:
- Load the dataset and model
- Run interpretations with selected explainers
- Save results to the `results/` directory
- Generate HTML visualizations

### 4. View the results

Open the HTML files in the `results/` directory to explore the visualizations:

```bash
# Example (adjust path based on your browser)
<YOUR BROWSER> results/h2o-sonar.html
```

## Files

- `creditcard.csv` - Credit card default dataset
- `creditcard-binomial-sklearn-1.8.0-gbm.pkl` - Pre-trained scikit-learn model
- `install-example.sh` - Installation script
- `run-example.sh` - CLI execution script
- `README.md` - This file

## Learn More

- [H2O Sonar Documentation](https://docs.h2o.ai/h2o-sonar/)
- [H2O Sonar GitHub Repository](https://github.com/h2oai/h2o-sonar)

