H2O Sonar Installation
======================
Prerequisites:

- Operating system: Linux
- Python 3.11
- Pip 25.0+
- CUDA-compatible GPU, NVIDIA drivers (required for certain generative evaluations **only**)
- Java 1.7+ (required for predictive H2O-3 backend **only**)
- Graphviz (required for predictive visualizations **only**)

GPU acceleration (**optional**):

- GPU support accelerates certain evaluators like BERTScore, GPTScore or Perplexity.
- CUDA runtime provided via PyTorch/ONNX dependencies - installed automatically with ``[evaluators]`` extras.
- Configure via environment variable: ``H2O_SONAR_CFG_DEVICE="gpu"`` (default: auto-detect).

Download distribution or Python wheels for your platform and Python version:

- `H2O Sonar releases <https://github.com/h2oai/h2o-sonar/releases>`__

Install Python wheel with **only core dependencies** for your platform:

- ``pip install h2o_sonar-<version>.whl``

Package extras
--------------

Install H2O Sonar with **all** dependencies:

- ``pip install h2o_sonar-<version>.whl[explainers,evaluators]``

Install H2O Sonar with **explainers** dependencies:

- ``pip install h2o_sonar-<version>.whl[explainers]``

Install H2O Sonar with **evaluators** dependencies:

- ``pip install h2o_sonar-<version>.whl[evaluators]``

Install H2O Sonar **core** package only:

- ``pip install h2o_sonar-<version>.whl``


Troubleshooting
---------------

- You may need to upgrade ``pip`` using ``python -m pip install --upgrade pip`` or ``curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11`` in case an H2O Sonar dependency installation fails.
