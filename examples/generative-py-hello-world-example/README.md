# Generative Hello World Example

This example demonstrates how to use **H2O Sonar** to evaluate a RAG (Retrieval-Augmented Generation) system using h2oGPTe.

## Overview

This example includes:
- **Corpus**: `corpus-h2o-ai.txt` - A text document about H2O.ai
- **Test Suite**: `test-suite.json` - A set of questions and expected answers for evaluation
- **Scripts**: Installation and execution scripts for easy setup


## Prerequisites
Prepare prerequisites:

- **Python 3.11**
- **uv** package manager
- **h2oGPTe API Key** - Required for accessing h2oGPTe service

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Get your h2oGPTe API key:
1. Sign up: https://h2ogpte.h2o.ai/
2. Login and navigate to Settings (gear icon) > API Keys
3. Click "Create API Key" button
4. Copy the key and export it: `export H2O_GPTE_API_KEY='your-api-key-here'`

For more details, see: https://github.com/h2oai/h2o-sonar#installation

## Quick Start

### 1. Copy the H2O Sonar wheel file

First, obtain the h2o-sonar wheel file and copy it to this directory:

* https://github.com/h2oai/h2o-sonar#installation


### 2. Run the installation script

```bash
cd examples/generative-py-hello-world-example
./install-example.sh
```

This script will:
- Verify prerequisites (uv, H2O_GPTE_API_KEY)
- Create a Python 3.11 virtual environment (`.venv/`)
- Install h2o-sonar from the wheel file with evaluators extras


### 3. Export your h2oGPTe API key

```bash
export H2O_GPTE_API_KEY='your-api-key-here'
```

### 4. Run the example

```bash
./run-example.sh
```

This will:
- Create a collection in h2oGPTe and upload the corpus
- Run test queries against the collection
- Evaluate the responses using H2O Sonar evaluators
- Save results to the `results/` directory
- Generate HTML visualizations

### 5. View the results

Open the HTML file in the `results/` directory to explore the evaluation results:

```bash
# Example (adjust path based on your browser)
<YOUR BROWSER> results/h2o-sonar.html
```

## Files

- `corpus-h2o-ai.txt` - Test corpus document
- `test-suite.json` - Test suite with questions and expected answers
- `install-example.sh` - Installation script
- `run-example.sh` - Execution script
- `run-example.py` - Python script demonstrating H2O Sonar evaluation
- `README.md` - This file

## Learn More

- [H2O Sonar Documentation](https://docs.h2o.ai/h2o-sonar/)
- [H2O Sonar GitHub Repository](https://github.com/h2oai/h2o-sonar)
- [h2oGPTe Documentation](https://docs.h2o.ai/h2ogpte/)

