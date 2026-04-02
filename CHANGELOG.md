# Change Log
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).



## [v3.2.1](https://github.com/h2oai/h2o-sonar/tree/v3.2.1) — 2026-04-01

This is a minor H2O Sonar release.

### Security

* `nltk` upgraded from `3.9.1` to `3.9.4` to fix `GHSA-7p94-766c-hgjp`.

### Fixed

* **Tests**:
    * Fixing failing `test_load_lab_judges` and `test_build_lab_multi_turn` tests by
      updating the h2oGPTe judge LLM to valid Anthropic Sonnet and/or Google Gemini
      models (replacing deprecated Claude Sonnet models no longer available on the
      h2oGPTe test server).



## [v3.2.0](https://github.com/h2oai/h2o-sonar/tree/v3.2.0) — 2026-02-17

H2O Sonar is now open source! This major release grants the four essential
freedoms: the freedom to run the program, to study and change it, and to
redistribute copies of either the original or your modified versions.
Join us in evolving H2O Sonar and helping the AI/ML community thrive!

### Added

* **Evaluators**:
    * **Questions Drift Evaluator** - detects semantic drift in input questions over
      time using mean embedding distance (cosine distance between centroids). Helps
      identify when the space of questions has changed or when untrained users engage in
      unexpected ways. Supports SR 11-7 ongoing monitoring requirements.
* **Documentation**
    * Sphinx theme upgraded to version `3.1.0`.

### Changed

* License changed to `Apache 2.0`.

### Fixed

* Fixed documentation build to dynamically use current version from `version.py`
  instead of hardcoded version in sidebar CSS.
* Fixed H2O-3 OOM in predictive tests by improved clean-up methods (frames,
  models and H2O-3's GC), `pytest` fixtures and 2 patterns - for `Unittest` and
  `pytest` based tests.
* Fixed linter to enforce 88 columns in Python sources again, and fixed source code
  to be compliant again.

### Removed

* H2O Model Validation is no longer tested due to known H2O-3 bug: H2O Model
  Validation wheel uses 'h2o_client' Python package which overwrites 'h2o'
  Python package directory when installed and thus purges H2O-3 JAR which is
  required for H2O Sonar to run. The H2O Model Validation wheel is still
  available in the public S3 bucket for manual installation and testing if
  needed.



## [v3.1.0](https://github.com/h2oai/h2o-sonar/tree/v3.1.0) — 2026-02-16

This is a minor H2O Sonar release.

### Added

* **Enhancements**:
    * Enhanced evaluations comparison HTML and JSON reports with overall comparison
      section that aggregates metrics across all model pairs, providing high-level
      summary before detailed per-model comparisons.
    * Improved HTML report navigation with sidebar submenu for overall comparison
      section, including quick links to Final Recommendation, Evaluations Overview,
      Model Pairs Comparison, Models Comparison, and Technical Performance Metrics
      sections.
    * Added visual highlighting (yellow background) to key values in HTML report:
      number of comparable test cases and sentence comparison method name
      (e.g., COSINE_DISTANCE) for improved readability.



## [v3.0.2](https://github.com/h2oai/h2o-sonar/tree/v3.0.2) — 2026-02-13

This is a patch H2O Sonar release.

### Fixed

* Modernized several type hints, exception handling, and fixed number of other errors
  reported by `ruff check`.

### Changed

* To prevent dependency clashes and reduce security vulnerabilities in products using
  H2O Sonar, the following transitive dependencies - originally added to mitigate
  CVEs - have been removed from `pyproject.toml`: `filelock`, `fonttools`,
  and `urllib3`. This moves responsibility for version selection back to `uv` and
  Python packages dependency requirements.
* Strict Python package dependency requirements enforced with `==` were relaxed
  to `~=` for dependencies and optional-dependencies in `pyproject.toml` to
  simplify / unblock H2O Sonar use by other projects.
* Source code formatting and linting changed from `black`, `isort` and `flake8` to `ruff`.



## [v3.0.1](https://github.com/h2oai/h2o-sonar/tree/v3.0.1) — 2026-02-06

This is a patch H2O Sonar release.

### Added

* **Enhancements**:
    * Enhanced evaluations comparator JSON representation - `test_cases_leaderboard` -
      with all calculated values needed for HTML table generation, including
      `baseline_wins`, `current_wins`, `baseline_rank_avg`, and `current_rank_avg`
      (fields added, 1 deprecated).

### Fixed

* Evaluations comparator leaderboard sorting now sorts by absolute rank difference (most
  significant performance differences first) instead of by changed metrics count. This
  makes the leaderboard more meaningful by highlighting test cases with the largest
  performance gaps.

### Deprecated

* The `wins` field in `test_cases_leaderboard` JSON is deprecated and now equals
  `baseline_wins` for backward compatibility. Use `baseline_wins` or `current_wins`
  directly instead.



## [v3.0.0](https://github.com/h2oai/h2o-sonar/tree/v3.0.0) — 2026-02-06

This major H2O Sonar releases brings **significant modernization and cleanup**.

### Added

* **Enhancements**:
    * Improved Answer Accuracy evaluator robustness by the detection of short actual
      and/or expected answers and fallback to the alternative metrics.
    * Improved evaluations comparator HTML report by adding fields and methods explanation.
   * Branding added to the H2O Sonar configuration for easy global control.
* **Documentation**
    * Added "Hello predictive World!" example.
    * Added "Hello generative World!" example.
    * Removed internal presentations and videos (demo days, meetings, meetups) links
      from `README.md`.

### Fixed

* HMLI imports in the code and tests rewritten from local imports to conditional imports.
* H2O Model Validation imports in the code and tests rewritten from local imports to
  conditional imports.
* Fixed hundreds of warnings and errors of the ReStructuredText documentation build.
* Test and benchmark source code is no longer included in the source distribution wheel.
* All predictive tests stabilized and GitHub Actions workflow for predictive models
  explanations added back.

### Changed

* `[BREAKING]` Branding is newly controlled by H2O Sonar configuration, not by parameters.
* Predictive testing switched from binary to source distribution (FLOSS, resource
  utilization, duration).
* All S3 resources consolidated from many buckets under various AWS profiles
  to single AWS profile with 1 public bucket and a few private ones.
* Test services configuration externalized from `given_generative.py` to JSON
  file which is synchronized from AWS S3 and stored in GitHub actions secrets.
* Examples refactored:
  * `examples/explainers` moved to `examples/predictive`.
  * `examples/methods` moved to `examples/predictive/methods`.
* Test data and models refactored:
  * `data/predictive/*.csv` moved to `data/predictive`.
  * `data/predictive/models` moved to `data/predictive/models`.
  * `data/llm` renamed to `data/generative`.
  * `data/pdf` renamed to `data/generative/corpus`.
* Package `h2o` upgraded to version `3.46.0.9`.
* Package `h2ogpte` upgraded to pypi.org hosted version `1.6.50`.
* Package `daimojo` upgraded to version `2.8.10`.
* Package management, installation and build switched from `pip` to `uv`.

### Removed

* H2O MLInference (`HMLI`) dependency removed - as a consequence the following
  explainers, methods, functionality and other resources have been removed:
    * `[BREAKING]` Methods removed:
        * Leave One Covariate Out (LOCO) method.
        * k-LIME method.
        * LIME-SUP method.
    * `[BREAKING]` Functionality removed:
        * Support for `hmli.H2OFrame` has been removed - frames must be converted
          to `h2o.H2OFrame`, Pandas, datatable, or another supported data type upfront.
    * Other resources removed:
        * Tests, RestructuredText documentation, Jupyter Notebook examples,
          and README.md references removed.
* Datasets cleanup:
    * Incoming raw test datasets removed (CUAD, h2oGPTe, Arabic, Kaggle KGM competition,
      John's NIST and SR 11-7, summarization).
    * OOTB Python dataset with potentially sensitive data removed.
    * Predictive datasets with potentially sensitive data removed.
    * Anonymized support/customer test and repro datasets removed.
* Obsolete examples purged:
    * H2O Sonar in H2O.ai Cloud Notebooks example.
    * H2O Sonar in H2O.ai Hybrid Cloud Notebooks (internal) example.
* `[BREAKING]` macOS platform is no longer supported, as there is no viable version of
  `datatable` Python package for neither x86 nor M-series CPUs. (*)
* Workflows for Driverless AI distribution build, self-hosted build, and release of the
  wheel to S3 for GitHub Actions purged.
* Pytest Beartype plugin removed from the test suite dependencies and Makefile.
* Benchmark and load testing targets, code and datasets removed.
* Obsolete development process documentation removed.
* Superfluous IDEs configuration purged.
* Legacy internal Docker files purged.
* Driverless AI wheel build removed.
* XNN model prototype removed.

### Security

* `filelock` upgraded from 3.20.1 to 3.20.3 to fix `GHSA-qmgc-5h2g-mvrw`.
* `urllib3` upgraded from 2.6.0 to 2.6.3 to fix `GHSA-38jv-5279-wg99`.
* `nbconvert` vulnerability `GHSA-xm59-rqc7-hhvf` (CVE-2025-53000) ignored in security
  audit - Windows-specific issue with no fix available yet (Dec 2025).



## [v2.30.0](https://github.com/h2oai/h2o-sonar/tree/v2.30.0) — 2026-01-14

This is a minor H2O Sonar release.

### Added

* **Features**:
    * Added structured generation to all supported host type clients allowing
      to ensure that the generated content is syntactically correct and complies
      with the expected format (like `JSON`) and (`Pydantic`) type.
* **Enhancements**:
    * Improved Encoding guardrail evaluator description and ReStructuredText
      documentation to correctly describe the exact method used by the evaluator.
    * Expected answer added to the evaluations comparison HTML report.
    * Collection name can be specified on the build of the test suite from JSON
      in order to make it appear in evaluation and comparison reports.

### Fixed

* Fixed the handling of expected (EA) and actual answers (AA) for 25 evaluators
  that require either of them to perform an evaluation.
  If there is no EA and/or AA in the dataset, then evaluator creates new problem
  and makes itself incompatible.
  If a EA and/or AA is missing in the dataset, then evaluator creates new problem
  and performs the evaluation.

### Security

* `filelock` upgraded from 3.20.0 to 3.20.3 to fix `GHSA-w853-jp5j-5j7f` and `GHSA-qmgc-5h2g-mvrw`.
* `fonttools` upgraded from 4.60.1 to 4.60.2 to fix `GHSA-768j-98cg-p3fv`.
* `urllib3` upgraded from 2.5.0 to 2.6.0 to fix `GHSA-gm62-xv2j-4w53` and `GHSA-2xpw-w6gg-jr37`.



## [v2.29.2](https://github.com/h2oai/h2o-sonar/tree/v2.29.2) — 2026-01-13

This is a patch H2O Sonar release.

### Fixed

* Fixed resolution of the CPU vs. GPU device for ONNX model inference in the Fairness
  Bias evaluator.

### Changed

* The Python package `onnxruntime-gpu` is installed by default (instead of
  `onnxruntime`) when the evaluators extra is used. This improves the performance of
  ONNX models when a GPU is available.



## [v2.29.1](https://github.com/h2oai/h2o-sonar/tree/v2.29.1) — 2026-01-05

This is a patch H2O Sonar release.

### Security

* Package `langchain-core` upgraded and fixed to version `0.3.81` to fix the vulnerability `CVE-2025-68664`.



## [v2.29.0](https://github.com/h2oai/h2o-sonar/tree/v2.29.0) — 2025-12-16

This is a minor H2O Sonar release.

### Added

* **Evaluators**:
    * Disabling the preview of the Step alignment and completeness Evaluator - it
      will be back as soon as the implementation reaches production quality.

### Fixed

- Fixing the documentation, description and docstring of the Self-consistency evaluator.



## [v2.28.0](https://github.com/h2oai/h2o-sonar/tree/v2.28.0) — 2025-12-15

This is a minor H2O Sonar release.

### Added

* **Enhancements**:
    * Aligned lower/upper case convention in evaluators display names to use lower
      case names. For instance `Context Mean Reciprocal Rank` evaluator renamed
      to `Context mean reciprocal rank`.

### Fixed

* Added missing test case key(s) to problems created by the evaluators.



## [v2.27.0](https://github.com/h2oai/h2o-sonar/tree/v2.27.0) — 2025-12-11

This is a minor H2O Sonar release.

### Added

* **Features**:
    * Agent/RAG/LLM evaluation results comparison tool - a tool to compare evaluation
      results of different RAG/LLM models, suggest the best model, and generate
      the comparison report in HTML and JSON formats.
      Comparable models are models with non-empty intersection of comparable test
      cases where comparable test cases have the same questions (inputs/prompts)
      and non-empty intersection of metrics scores.
* **Enhancements**:
    * Loader of the LlmEvaluationResults from JSON.

### Fixed

* Fixed evaluation status resolution in case of all failed evaluators.
* Fixed metrics metadata deserialization from JSON to objects.
* Fixed evaluators and explainers numbers in the README.md file.
* Fixed broken logging in the explainer container (legacy or w/o functions).

### Changed

* Contextual misinformation perturbator is newly not listed by default as it is
  slow, expensive and resource intensive. However, it can be enabled using the configuration.
* Renamed JSON to JSON through documentation and code - including `JSONSchemaEvaluator`.

### Testing

* This version of H2O Sonar has been tested with h2oGPTe client version `h2ogpte-1.6.50`
  (the client version is not specified in the `.whl` distribution metadata to give users
  more flexibility).



## [v2.26.7](https://github.com/h2oai/h2o-sonar/tree/v2.26.7) — 2025-12-08

This is a patch H2O Sonar release.

### Fixed

* H2O Sonar agent files downloading configuration detection fixed.

### Changed

* h2oGPTe client upgraded to pypi.org hosted version `h2ogpte-1.6.47`.



## [v2.26.6](https://github.com/h2oai/h2o-sonar/tree/v2.26.6) — 2025-12-08

This is a patch H2O Sonar release.

### Fixed

* H2O Sonar no longer attempts to download agent files if the explainable model
  has `use_agent` configuration item which is set to `False`.



## [v2.26.5](https://github.com/h2oai/h2o-sonar/tree/v2.26.5) — 2025-12-03

This is a patch H2O Sonar release.

### Fixed

* Format specifier of integer metrics fixed in the Agent Sanity Check evaluator.



## [v2.26.4](https://github.com/h2oai/h2o-sonar/tree/v2.26.4) — 2025-12-03

This is a patch H2O Sonar release.

### Fixed

* Format specifier of integer metrics fixed from the float to integer formatting.



## [v2.26.3](https://github.com/h2oai/h2o-sonar/tree/v2.26.3) — 2025-11-26

This is a patch H2O Sonar release.

### Security

* Package `langchain-core` upgraded and fixed to version `0.3.80` to fix the vulnerability `CVE-2025-65106`.



## [v2.26.2](https://github.com/h2oai/h2o-sonar/tree/v2.26.2) — 2025-11-13

This is a patch H2O Sonar release.

### Fixed

* Broken Anthropic Claude build of the test lab fixed.

### Changed

* Row number removed from the agent sanity check flow diagram JSON serialization
  in the evaluation result.



## [v2.26.1](https://github.com/h2oai/h2o-sonar/tree/v2.26.1) — 2025-11-12

This is a patch H2O Sonar release.

### Changed

* Package `setuptools` version requirement relaxed to `>=78.1.1`.



## [v2.26.0](https://github.com/h2oai/h2o-sonar/tree/v2.26.0) — 2025-11-11

This is a minor H2O Sonar release.

### Added

* **Features**:
    * Anthropic Claude host LLMs support - new host type, client and
      test lab integration for Claude models evaluation, which can be
      used as LLM judge and in Model Risk Management (MRM) tools and methods.

### Changed

* Changed Agent Sanity Check evaluator's evaluation result JSON serialization structure
  to enable formal description using JSON Schema and Protocol Buffers `proto`.
* JSon Schema evaluator renamed to JSON Schema evaluator (class name).
* h2oGPTe client upgraded to pypi.org hosted version `h2ogpte-1.6.42`.
* Package `sentence_transformers` upgraded to version `5.1.2`.
* Package `umap-learn` upgraded to version `0.5.9.post2`.
* Package `boto3` upgraded to version `1.39.4`.



## [v2.25.0](https://github.com/h2oai/h2o-sonar/tree/v2.25.0) — 2025-10-30

This is a minor H2O Sonar release.

### Added

* **Evaluators**:
    * Answer Accuracy (Semantic Similarity) evaluator - assesses how closely the
      actual answer matches the expected answer by comparing them using semantic
      similarity at the sentence level with embeddings.
    * Self-consistency evaluator - assesses the consistency of the RAG's/LLM's
      responses to the same question using `ROUGE-L` and `ROUGE-1` metrics.
    * BERTScore evaluator - measures semantic similarity between expected and
      actual answers using contextual embeddings from BERT models.
* **Features**:
    * Copy perturbator - new perturbator which returns the input text unchanged.
      It is meant to be used as a baseline or control in various experiments,
      for instance creation of tests for the Self-consistency evaluator.
* **Enhancements**:
    * Verified reproducibility of the ROUGE evaluator and diverse input types of test cases.
    * CUAD dataset (https://www.atticusprojectai.org/cuad) added to the test suite library
      whose main copy is hosted at https://eval-studio-artifacts.s3.us-east-1.amazonaws.com/h2o-eval-studio-suite-library/index.html
    * Agent Sanity Check evaluator improved to store rendered diagrams to its work/ directory.
* **Documentation**
    * Documentation of regular expressions syntax used by Text Matching evaluator added.
    * Evaluator selection guide allowing to choose evaluator given the requirements and
      available data added.
    * Evaluators (and their metrics) reproducibility documentation added.

### Fixed

* Fixed malfunctioning vector store index deletion on Amazon Bedrock.



## [v2.24.0](https://github.com/h2oai/h2o-sonar/tree/v2.24.0) — 2025-10-10

This is a minor H2O Sonar release.

### Added

* **Enhancements**:
    * Listing of evaluators and getting of evaluator descriptors newly supports portable
      handling of infinite values in parameter, threshold and metric metadata descriptors.

### Changed

* Testing completion API directory aspect aligned with the evaluation API: `user_dir`
  removed in favor of `results_location`.

### Security

* `transformers` upgraded to version `4.53.0` to fix `GHSA-phhr-52qp-3mj4`,
  `GHSA-37mw-44qp-f5jm`, `GHSA-9356-575x-2w9m`, `GHSA-59p9-h35m-wg4g`,
  `GHSA-rcv9-qm8p-9p6j`,  and `GHSA-4w7r-h757-3r74`.
* `h2o` upgraded to version `3.46.0.6` to fix `GHSA-h7xg-cmpp-48hf`.



## [v2.23.1](https://github.com/h2oai/h2o-sonar/tree/v2.23.1) — 2025-10-09

This is a minor H2O Sonar release.

### Fixed

* Fixed aggressive caching of chat session history in Agent Sanity Check evaluator
  which lead to incorrect agent flow diagram data.



## [v2.23.0](https://github.com/h2oai/h2o-sonar/tree/v2.23.0) — 2025-10-09

This is a minor H2O Sonar release.

### Added

* **Enhancements**:
    * Improved loading of evaluations newly involving also explainable models.

### Fixed

* RAG vs. LLM host type classifier updated to correctly handle new types.

### Security

* `setuptools` upgraded to version `78.1.1` to fix `PYSEC-2025-49`.



## [v2.22.2](https://github.com/h2oai/h2o-sonar/tree/v2.22.2) — 2025-10-07

This is a minor H2O Sonar release.

### Fixed

* Fixing test lab completion for agent on LLM (non-RAG) as well as agent resources
  lookup in the Agent Sanity Check evaluator.



## [v2.22.1](https://github.com/h2oai/h2o-sonar/tree/v2.22.1) — 2025-10-02

This is a minor H2O Sonar release.

### Fixed

* Fixed h2oGPTe lab completion for agents and sequential messages category construction
  in particular so that the agent evaluators are able to lookup the artifacts created
  by the agent during the test lab completion.



## [v2.22.0](https://github.com/h2oai/h2o-sonar/tree/v2.22.0) — 2025-10-02

This is a minor H2O Sonar release.

### Fixed

* Fixed h2oGPTe lab completion for agent LLMs (non-RAG) by switch to different h2oGPTe client API call.
* Fixed occurrence of more than one primary metric in case of RAGAs family evaluators.

### Changed

* Source code cythonizer used to build binary wheels upgraded to version `3.1.4`.
* Local imports of optional Python packages rewritten to a Python idiomatic pattern.
* Type hints migrated from Python 3.6 conventions to Python 3.10 which impacts code
  backward compatibility.



## [v2.21.3](https://github.com/h2oai/h2o-sonar/tree/v2.21.3) — 2025-09-29

This is a patch H2O Sonar release.

### Fixed

* Fixed misalignment of the Agent Sanity Check evaluator keywords and metadata.



## [v2.21.2](https://github.com/h2oai/h2o-sonar/tree/v2.21.2) — 2025-09-25

This is a patch H2O Sonar release.

### Fixed

* Fixed automatic caching of the `vectara/hallucination_evaluation_model` from
  Hugging Face models hub by the Hallucination evaluator.



## [v2.21.1](https://github.com/h2oai/h2o-sonar/tree/v2.21.1) — 2025-09-25

This is a patch H2O Sonar release.

### Fixed

* Fixed caching of the `vectara/hallucination_evaluation_model` from Hugging Face
  models hub. The environment variable `HUGGING_FACE_HUB_TOKEN` must be newly set to
  access the model. Account associated with the token must fill in the information
  required by the model author to be able to download the model.



## [v2.21.0](https://github.com/h2oai/h2o-sonar/tree/v2.21.0) — 2025-09-16

This is a minor H2O Sonar release.

### Added

* **Enhancements**:
    * Agent Sanity Check evaluator improved to provide 8 introspection metrics with thresholds,
      problems, and suggestions for improvement. In addition the evaluator generates three
      charts - agentic chat history flow chart, agent tool use/failure bar chart and
      agent script use/failure bart chart (along with JSon data for the chart rendering outside
      of the library).
   * New H2O Sonar package extra - `genaiclient` - to install only client to Gen AI
     hosts (like h2oGPTe, Amazon Bedrock, and OpenAI) and its dependencies.

### Changed

* h2oGPTe client upgraded to pypi.org hosted version `h2ogpte-1.6.40`.
* Fixing `numpy` package to (the last `1.*.*`) version `1.26.4` to support used version of `pandas`.

### Security

* Package `langchain` upgraded to version `0.3.27` to fix the vulnerability `CVE-2025-6984`.
* Package `langchain-community` upgraded to version `0.3.27` to fix the vulnerability `CVE-2025-6984`.



## [v2.20.0](https://github.com/h2oai/h2o-sonar/tree/v2.20.0) — 2025-09-03

This is a minor H2O Sonar release.

### Added

* **Evaluators**:
    * Mean Reciprocal Rank (MRR) evaluator - assesses the performance of
      the retrieval component of a RAG system by measuring the average of the reciprocal
      ranks of the first relevant document retrieved for a set of queries.
    * Agent Sanity Check evaluator - a tool to check the integrity and quality of agent-created artifacts.
    * JSon Schema evaluator - verifies validity of RAG/LLM actual answers against JSon Schema provided
      via evaluator parameter.
* **Features**:
    * Test lab completion newly extracts (meta)data and files - like Python source code, PDFs,
      and JSons - created by the agent when responding to a particular question (prompt).
* **Enhancements**:
    * Explainer / evaluator logs beautified for 3rd party products consumption.

### Fixed

* Guardrail Encoding evaluator fixed to detect encoded leaks even if test case is not categorized
  as perturbed and actual answer has multiple encoded fragments.
* AWS Bedrock test lab build fixed to set collection ID only in the test lab JSon representation.



## [v2.19.2](https://github.com/h2oai/h2o-sonar/tree/v2.19.2) — 2025-08-14

This is a minor H2O Sonar release.

### Security

* Package Hugging Face `transformers` upgraded to version `4.51.3` to fix its regression bug #1407



## [v2.19.1](https://github.com/h2oai/h2o-sonar/tree/v2.19.1) — 2025-08-14

This is a minor H2O Sonar release.

### Security

* Package Hugging Face `transformers` upgraded to version `4.51.0` to fix the vulnerability `CVE-2025-3262`.



## [v2.19.0](https://github.com/h2oai/h2o-sonar/tree/v2.19.0) — 2025-08-13

This is a minor H2O Sonar release.

### Fixed

- Answer Semantic Sentence Similarity evaluator fixed to have compatibility check ensuring
  (apart to other preconditions) that the evaluator will run on valid datasets only.



## [v2.18.0](https://github.com/h2oai/h2o-sonar/tree/v2.18.0) — 2025-08-08

This is a minor H2O Sonar release.

### Fixed

* OpenAPI Assistants client fixed to be compatible with the latest OpenAI Assistants version 2 API changes.



## [v2.17.2](https://github.com/h2oai/h2o-sonar/tree/v2.17.2) — 2025-08-04

This is a minor H2O Sonar release.

### Fixed

* Fixed the reference to Hugging Face model `all-MiniLM-L6-v2` in the Step alignment and completeness evaluator.
* Fixed models caching module to be more robust - fully qualified Hugging Face model names via constants newly
  used through the code base.

### Changed

* Package `pip` upgraded to version `25.2.0`.



## [v2.17.1](https://github.com/h2oai/h2o-sonar/tree/v2.17.1) — 2025-07-31

This is a minor H2O Sonar release.

### Fixed

* Fixed the Hugging Face model `all-MiniLM-L6-v2` caching for the Step alignment and completeness evaluator
  to ensure that the model is cached in the H2O Sonar models online cache location by using the fully
  qualified model name `sentence-transformers/all-MiniLM-L6-v2`.



## [v2.17.0](https://github.com/h2oai/h2o-sonar/tree/v2.17.0) — 2025-07-31

This is a minor H2O Sonar release.

### Added

* **Enhancements**:
    * Test suite and test have new `categories` allowing to indicate the purpose
      and the type of the test suites and tests. Consider for instance `sr-11-7`
      test suites or `stateful` tests.
* **Documentation**
    * Polished math formulas in `docstring` documentation of the Looping, GPT Score family,
      and Classification evaluators.

### Fixed

* Added caching of the Hugging Face model `all-MiniLM-L6-v2` model for the Step alignment and completeness evaluator.

### Changed

* `pyarrow` version fixed to `20.0.0` to keep symbols used by depending libraries which were removed by subsequent versions.



## [v2.16.0](https://github.com/h2oai/h2o-sonar/tree/v2.16.0) — 2025-06-24

This is a minor H2O Sonar release.

### Added

* **Enhancements**:
    * If test lab completion fails to retrieve actual answer from the RAG/LLM host,
      then the metrics values are set to the worst possible values (depending on
      whether higher or lower is better) by all evaluators to ensure that the metric
      score is correct.
* **Documentation**
    * Updated evaluators diagram with Encoding Guardrail evaluator.

### Fixed

* Exponential back-off timeout strategy is not used in case of agentic h2oGPTe test lab build
  in order to avoid the server overload.
* Text Matching evaluator fixed to escape constraints containing apostrophes and quotes.
* Step Alignment and Completeness evaluator preferred model changed to specific up-to-date
  Claude version. Robustness of the LLM model answers parsing improved (Markdown formatting trimming).

### Changed

* h2oGPTe client upgraded to pypi.org hosted version `h2ogpte-1.6.36`.



## [v2.15.0](https://github.com/h2oai/h2o-sonar/tree/v2.15.0) — 2025-05-29

This is a minor H2O Sonar release.

### Added

* **Features**:
    * Added configurable GPU acceleration to the following evaluators:
      * Answer Relevancy (Sentence Similarity) evaluator.
      * Answer Semantic Sentence Similarity evaluator.
      * Context Relevancy (Soft Recall and Precision) evaluator.
      * Fairness Bias evaluator.
      * Groundedness (semantic similarity) evaluator.
      * Hallucination evaluator.
      * Machine Translation (GPTScore) evaluator.
      * Perplexity evaluator.
      * Question Answering (GPTScore) evaluator.
      * Summarization with reference (GPTScore) evaluator.
      * Summarization without reference (GPTScore) evaluator.
      * Step Alignment and Completeness evaluator.
      * Summarization (Completeness and Faithfulness) evaluator.
      * Toxicity evaluator.
* **Enhancements**:
    * `hf-xet` to improve Hugging Face models handling performance.
    * `onnxruntime-gpu` to improve ONNX models performance when GPU is available.

### Changed

* `lmppl` Python dependency `0.0.1` patched with https://github.com/asahi417/lmppl/pull/13 and
  the wheel moved to the public S3 bucket.



## [v2.14.0](https://github.com/h2oai/h2o-sonar/tree/v2.14.0) — 2025-05-22

This is a minor H2O Sonar release.

### Security

* Package `langchain` upgraded to version `0.3.1` to fix the vulnerability `CVE-2024-7042`.
* Package `langchain-community` upgraded to version `0.3.1` to fix the vulnerability `CVE-2024-7042`.
* Package `openai` upgraded to version `1.81.0` as a dependency of `langchain` to fix the vulnerability `CVE-2024-7042`.



## [v2.13.0](https://github.com/h2oai/h2o-sonar/tree/v2.13.0) — 2025-05-20

This is a minor H2O Sonar release.

### Added

* **Evaluators**:
   * Encoding guardrail evaluator - a tool designed to assess the LLM/RAG's ability
     to handle encoding attacks. It evaluates whether the system can be tricked into
     generating incorrect or unexpected outputs through manipulation of the prompt
     encoding, such as encoding the prompt text using Base64 or Base16, which should
     be discarded by the guardrails or the system.
* **Features**:
   * Introducing statefull conversations / multi-turn chats / contextual conversation
     support for the h2oGPTe client - the client can now maintain the context of
     the conversation across multiple turns, allowing for new types of evaluations
     and attacks.
   * Encoding perturbator - a perturbator which encodes the prompt text using base16 encoding.
   * Adding ability to configure and enforce CPU, GPU or automatic device selection for running
     predictive and generative models. Automatic device selection is the default.
   * Added module which calculates various statistics to compare distributions:
     Kolmogorov-Smirnov test, Wasserstein distance, and Jensen-Shannon divergence.
   * H2O Sonar newly automatically uses the shell environment configuration overrides from
     environment variables starting with `H2O_SONAR_CFG_` prefix. The environment variables
     are automatically converted to the H2O Sonar configuration parameters (primitive values only).
* **Enhancements**:
   * Introducing `NaN` tolerance to heatmap leaderboard which brings tolerance for
     `NaN` values on average metric value calculation - it allows to ignore evaluation
     results with `NaN` metric values if the number of evaluation results is lower or
     equal to the given percentage of the total number of evaluation results.
   * RAGAs family evaluators newly support the `NaN` tolerance which can be configured
     using the evaluator parameters.
   * Test lab completion newly supports `auto`, `""` and `None` LLM selectors when
     the test lab is built from h2oGPTe collections. The `auto` selector lets h2oGPTe
     to automatically select the LLM model for the test lab completion; `""` and `None`
     inherit the LLM model from the h2oGPTe collection configuration.

### Fixed

* Classification evaluator was fixed to correctly handle unknown labels in HTML report
  confusion matrices.
* Classification leaderboard explanation improved to provide stable confusion matrices
  in the HTML report with unexpected labels.
* Perplexity evaluator no longer requires Open AI API key.
* Markdown representations texts are newly escaped to ensure formatting and avoid XSS.
* Test suites and labs corpus URLs fixed to reflex AWS S3 bucket migration from `eu-central-1`
  to `us-east-1`.

### Changed

* HMLI moved from public to private S3 bucket, which is accessible only from the H2O.ai
  infrastructure. Therefore, the HMLI wheel dependency must be installed from the private S3 bucket
  before installing H2O Sonar.
* h2oGPTe client upgraded to the custom S3 hosted build `h2ogpte-1.6.28.dev8-py3-none-any.whl`,
  which has been moved from a public to a private S3 bucket.
* Package `pip` upgraded to version `25.1.1`.

### Deprecated

No deprecations.

### Removed

No removals.

### Security

* Package `langchain-community` upgraded to version `0.2.19` to fix the vulnerability `CVE-2024-8309`.



## [v2.12.2](https://github.com/h2oai/h2o-sonar/tree/v2.12.2) — 2025-04-25

This is a minor H2O Sonar release.

### Changed

* Changing an AWS region for the H2O Eval Studio artifacts from `eu-central-1` to `us-east-1`.



## [v2.12.1](https://github.com/h2oai/h2o-sonar/tree/v2.12.1) — 2025-04-04

This is a minor H2O Sonar release.

### Fixed

* Perplexity evaluator no longer requires Open AI API key.

### Changed

* h2oGPTe client upgraded to version `1.6.27.post1`.



## [v2.12.0](https://github.com/h2oai/h2o-sonar/tree/v2.12.0) — 2025-03-27

This is a minor H2O Sonar release.

### Fixed

* Uploaded documents purging fixed for h2oGPTe client.

### Changed

* h2oGPTe client upgraded to version `1.6.25`.
* NLTK upgraded to version `3.9.1` to fix the vulnerability `CVE-2024-39705`.
* Hugging Face Transformers Library upgraded to version `4.50.2`.



## [v2.11.1](https://github.com/h2oai/h2o-sonar/tree/v2.11.1) — 2025-03-17

This is a patch H2O Sonar release.

### Added

* **Enhancements**:
   * Improving performance of test lab completion parallelization in case that the suite
     has less than 20 test cases.



## [v2.11.0](https://github.com/h2oai/h2o-sonar/tree/v2.11.0) — 2025-03-12

This is a minor H2O Sonar release.

### Added

* **Enhancements**:
   * The Text matching evaluator uses the expected answer as the condition (exact match)
     if available, when no condition is specified by the test case.

### Fixed

* Fixed JSon representations of the LLM evaluation result explanation to contain
  evaluator descriptor again.
* LLM evaluation result JSon representation does not include typed structure friendly
  metrics serialization by default.

### Changed

* Tokens presence evaluator renamed to Text matching evaluator.

### Deprecated

No deprecations.

### Removed

No removals.

### Security

No security fixes.



## [v2.10.0](https://github.com/h2oai/h2o-sonar/tree/v2.10.0) — 2025-03-06

This is a minor H2O Sonar release.

### Added

* **Features**:
   * LLM evaluation result JSon representation newly includes metrics serialized which
     is can be described using the proto definitions.
   * Exponential backoff driven timeout added to the h2oGPTe client to better perform
     and report the h2oGPTe timeouts.
* **Enhancements**:
   * Test lab completion parallelization and sharding improved to parallelize also inputs
     assigned to particular RAG/LLM model if the number of the RAG/LLM models is smaller
     than a configurable threshold.

### Fixed

* HTML report performance statistics fixed to handle missing keys.

### Changed

* h2oGPTe client upgraded to version `1.6.23`.

### Deprecated

No deprecations.

### Removed

No removals.

### Security

No security fixes.



## [v2.9.0](https://github.com/h2oai/h2o-sonar/tree/v2.9.0) — 2025-02-17

This is a minor H2O Sonar release.

### Added

* **Enhancements**:
   * Improved Step alignment and completeness evaluator - better step extraction from
     the retrieved context and model answer, propagation of the dynamic programming metrics and
     alignment matrix to the HTML report, and new ability to combine multiple steps into one if the
     reference or the generated text contains compound step (left combined, right without the step
     combination).
   * Markdown summary of the evaluation newly includes statistics for response times per LLM model.
   * Identical insights reported by different evaluators are newly deduplicated and reported as a single
     insight.

### Fixed

No fixes.

### Changed

* The default threshold for the Toxicity evaluator has been changed from `0.75` to `0.25` based
  on empirical observations and feedback from users.
* h2oGPTe client upgraded to version `1.6.22`.

### Deprecated

No deprecations.

### Removed

No removals.

### Security

* Upgrading `scikit-learn` to version `1.5.2` to fix the vulnerability `CVE-2024-5206`.



## [v2.8.2](https://github.com/h2oai/h2o-sonar/tree/v2.8.2) — 2025-02-05

This is a minor H2O Sonar release.

### Changed

* h2oGPTe client upgraded to version `1.6.18.post1`.



## [v2.8.1](https://github.com/h2oai/h2o-sonar/tree/v2.8.1) — 2025-01-13

This is a minor H2O Sonar release.

### Changed

* HMLI wheel dependency location changed to the H2O Eval Studio AWS account.



## [v2.8.0](https://github.com/h2oai/h2o-sonar/tree/v2.8.0) — 2025-01-10

This is a minor H2O Sonar release.

### Added

* **Evaluators**:
   * Step alignment and completeness evaluator (preview) - a tool for evaluating the steps of procedures, sequences, or process descriptions.
* **Features**:
   * Support for agent-based and LLM-based perturbators.
   * New Contextual misinformation perturbator.
* **Evaluation data**:
   * Test suite evaluation library with 1M+ test cases published at
     https://eval-studio-artifacts.s3.us-east-1.amazonaws.com/h2o-eval-studio-suite-library/index.html
     and
     https://eval-studio-artifacts.s3.us-east-1.amazonaws.com/h2o-eval-studio-suite-library/index.json
     Makefile targets to maintain the test suite evaluation library added to the project.
* **Documentation**
   * ReStructuredText documentation for the new Step alignment and completeness evaluator.
   * Added ReStructuredText documentation for Fact-check evaluator evaluator parameters.

### Fixed

No fixes.

### Changed

No changes.

### Deprecated

No deprecations.

### Removed

No removals.

### Security

No security fixes.



## [v2.7.0](https://github.com/h2oai/h2o-sonar/tree/v2.7.0) — 2024-12-16

This is a minor H2O Sonar release.

### Added

* **Evaluators**:
   * Fact-check evaluator (agent-based).
* **Enhancements**:
   * Enhanced test lab prompt cache which is meant for testing/demo purposes: improved
     configuration (environment variable and H2O Sonar configuration), added retrieved
     context caching.
* **Documentation**
   * New evaluator documentation for the Fact-check evaluator.
   * Added documentation for new perturbators which were added in H2O Sonar 2.6.0.

### Fixed

* Heatmap leaderboard explanation no longer shows empty most difficult prompts section.



## [v2.6.0](https://github.com/h2oai/h2o-sonar/tree/v2.6.0) — 2024-12-05

This is a minor H2O Sonar release.

### Added

* **Evaluators**:
   * Answer Semantic Sentence Similarity evaluator.
* **Features**:
   * New character level perturbators - insert/delete random character(s), QWERTY keyboard typos,
     and common OCR errors.

### Fixed

* Keywords of RAGAs evaluator, Classification evaluator and GPTScore Q&A evaluator fixed.
* RAGAs evaluator metadata in leaderboard serializations fixed to include exactly the metrics
  it calculates.
* Fixed the escaping of special characters in classification class names for
  the multi-class Classification evaluator.
* `httpx` Python dependency fixed to `0.27.0` to avoid `openai` Python library issues with unexpected
  proxy parameter.
* Resolved random hangs that occurred during h2oGPTe RAG retrieved context fetching when using
  a session connection managed by the resource manager.

### Changed

* H2O Sonar models online cache location has been moved from root to H2O EvalStudio tenant to download
  the models from the right location in case of the deployments with the internet access (and to cache
  the models from the right location in case of the air-gapped deployments).




## [v2.5.4](https://github.com/h2oai/h2o-sonar/tree/v2.5.4) — 2024-10-14

This is a patch H2O Sonar release.

### Fixed

* Fixed missing and non-float bool metrics in BYOP evaluators.
* Fixed `punkt` caching in Context relevancy (soft recall and precision) and
  Answer relevancy (sentence similarity) evaluators.
* Fixed keywords metadata in multiple evaluators.



## [v2.5.3](https://github.com/h2oai/h2o-sonar/tree/v2.5.3) — 2024-11-13

This is a minor H2O Sonar release

### Added

* **Enhancements**:
   * Perturbators can newly work without raising the exceptions - instead they gather
     the errors and return them in the passed lists.



## [v2.5.2](https://github.com/h2oai/h2o-sonar/tree/v2.5.2) — 2024-10-12

This is a patch H2O Sonar release.

### Fixed

* Fixed missing taglines in the evaluator descriptors.
* Fixed singular/plural in classification evaluator metadata.



## [v2.5.1](https://github.com/h2oai/h2o-sonar/tree/v2.5.1) — 2024-10-09

This is a patch H2O Sonar release.

### Added

* **Enhancements**:
   * Improved - shorter and concise - taglines in evaluators.



## [v2.5.0](https://github.com/h2oai/h2o-sonar/tree/v2.5.0) — 2024-10-08

This is a minor H2O Sonar release.

### Added

* **Enhancements**:
   * Tagline added to all evaluators to provide a brief description of the evaluator.

### Fixed

* Fixed bugs / inconsistencies between evaluator metadata and keywords like LLM vs. RAG
  compatibility.
* Fixed Classification evaluator metrics values included in the evaluation result to be consistent
  with the declared metrics in the evaluator metadata.
* Ensured caching of the `punkt` tokenizer for the Fairness Bias evaluator, Groundedness evaluator,
  and Hallucination evaluator to work correctly in air-gapped deployments.
* Fixed Groundedness evaluator AVID error codes and tokenization unpacking.

### Changed

* Summarization (Completeness and Faithfulness) evaluator excluded from the explainer container
  as it is resource intensive, expensive, difficult to interpret, and not suitable for the use
  without GPU HW support.
* h2oGPTe client upgraded to version `1.5.26`.

### Deprecated

No deprecations.

### Removed

No removals.

### Security

No security fixes.



## [v2.4.0](https://github.com/h2oai/h2o-sonar/tree/v2.4.0) — 2024-10-14

This is a minor H2O Sonar release.

### Added

* **Features**:
   * Amazon Bedrock RAG newly supports creation of the knowledge bases (collections)
     from test suites as a part of the test lab build and completion.
* **Enhancements**:
   * The following evaluators newly report metrics values in the evaluation results
     on the sentence granularity as actual answer metadata and they highlight problems
     in the HTML report:
      * Groundedness evaluator
      * Toxicity evaluator
      * Fairness Bias evaluator
      * Answer Relevancy (sentence similarity) evaluator
      * Hallucination evaluator
      * PII evaluator
      * Sensitive Data evaluator
   * Token presence evaluator reports which part of the condition caused the evaluation failure.
     The error message is provided in the `meta` section of the actual answer metadata and
     highlighted in the HTML report (error message section).
   * Summarization evaluator error messages improved to indicate the root cause of the
     summarization evaluation failure.
   * Test lab is newly accepting custom HTTP headers for the document caching when building
     the test lab or synchronizing the documents.
* **Documentation**
   * Added generative AI section to the introduction of the ReStructuredText documentation.
   * Completed licenses for all Python dependencies, used tools and test datasets in `library/` directory.
   * Added missing licenses to the ReStructuredText documentation.

### Fixed

* AVID problem taxonomy fixed to report codes in the problems.
* Failure to get LLM statistics in the h2oGPTe client no longer causes the evaluation to fail
  (it is optional to get the statistics).
* Fixed case in the names of evaluators to ensure the naming consistency.

### Changed

No changes.

### Deprecated

No deprecations.

### Removed

No removals.

### Security

No security fixes.



## [v2.3.0](https://github.com/h2oai/h2o-sonar/tree/v2.3.0) — 2024-10-07

This is a minor H2O Sonar release.

### Added

* **Enhancements**:
    * Text matching evaluator reports result parsing failures in the evaluation results.
    * Text matching evaluator ability to evaluate
      both actual answer and retrieved context is newly configurable - default is to actual answer only.
    * PII evaluator and Sensitive data evaluator ability to evaluate
      both actual answer and retrieved context is newly configurable - default is to evaluate both
      actual answer and retrieved context.

### Fixed

* Evaluation result JSon representation fixed to correctly serialize infinity and NaN values.
* Generation/Retrieval/Generation+Retrieval prefix of model failure errors in the HTML report
  fixed to be visible again.
* Passed and failed test cases counts in the test lab completion progress report fixed to
  be correctly calculated (when retrieved context failures are not considered).
* Fixed missing resolved test cases when building lab using the parallel job completion - if
  resolution of all test cases fails in the job, then the result is not discarded, but
  kept.

### Changed

* Boolean leaderboard (JSon, Markdown, dataset) results changed to fail the test case evaluation
  if the generation fails, and/or retrieval fails, and/or generation+retrieval fails. Previously,
  retrieval failures were not considered as a failure of the test case evaluation which lead to
  confusing results. Users can enable/disable the retrieval checks in Text matching evaluator,
  PII evaluator, and Sensitive data evaluator.
* h2oGPTe client upgraded to version `1.5.22`.

### Deprecated

No deprecations.

### Removed

No removals.

### Security

No security fixes.



## [v2.2.0](https://github.com/h2oai/h2o-sonar/tree/v2.2.0) — 2024-09-26

This is a minor H2O Sonar release.

### Added

* **Enhancements**:
   * h2oGPTe LLM performance statistics - like cost, input tokens, output tokens and time to
     the first token - added to the explainable model and Markdown boolean leaderboard explanation.
   * Markdown report newly includes h2oGPTe LLM vision model associated with the evaluated model.
   * Conditional evaluation by the Text matching evaluator newly reports sub-condition
     which caused the evaluation failure.
   * All row keys and all test cases added to problems reporting that model didn't pass
     a metric threshold check
   * Evaluator descriptor added to LLM result JSon.

### Fixed

No fixes.

### Changed

* Added LLM model metadata to the explainable (LLM and RAG) model,
  which changes the serialization and deserialization of the model metadata,
  test labs (impact H2O Eval Studio) and test results.
* Problem attribute `test_case_key` renamed to `test_case_keys` and type changed to list.

### Deprecated

No deprecations.

### Removed

No removals.

### Security

No security fixes.



## [v2.1.0](https://github.com/h2oai/h2o-sonar/tree/v2.1.0) — 2024-09-25

This minor H2O Sonar release brings **looping detection** evaluator and smaller enhancements.

### Added

* **Evaluators**:
   * Looping Detection evaluator.
* **Enhancements**:
   * Amazon Bedrock RAG client models listing speed up.
   * Evaluated models added to the JSon representation of the evaluation results.
   * Test case key added to the JSon representation of the evaluation results.
* **Documentation**
   * Added reStructuredText documentation of the evaluators.
   * Added prompts documentation for LLM judge-based evaluators.

### Fixed

* Row key(s), test case keys and model keys added to the problems and insights
  (where applicable) to simplify the mapping of the evaluation results to the
  original data.

### Changed

* Metrics column names of boolean leaderboard evaluators changed from ad hoc names
  to actual boolean metrics names.

### Deprecated

No deprecations.

### Removed

No removals.

### Security

No security fixes.



## [v2.0.0](https://github.com/h2oai/h2o-sonar/tree/v2.0.0) — 2024-09-18

This major H2O Sonar releases brings **generative AI** evaluation.

### Added

* **Evaluators**
   * Generation evaluation
      * Answer Correctness evaluator.
      * Answer Relevancy evaluator.
      * Answer Relevancy (Sentence Similarity) evaluator.
      * Answer Semantic Similarity evaluator.
      * Bring Your Own Prompt (BYOP) evaluator.
      * Faithfulness evaluator.
      * Groundedness (semantic similarity) evaluator.
      * Hallucination evaluator.
      * Language Mismatch evaluator.
      * Machine Translation (GPTScore) evaluator.
      * Perplexity evaluator.
      * Question Answering (GPTScore) evaluator.
      * RAGAS evaluator.
      * Text matching evaluator.
   * Retrieval evaluation
      * Context Precision evaluator.
      * Context Recall evaluator.
      * Context Relevancy evaluator.
      * Context Relevancy (Soft Recall and Precision) evaluator.
   * Privacy evaluation
      * Contact Information evaluator.
      * PII evaluator.
      * Sensitive Data evaluator.
   * Fairness evaluation
      * Fairness Bias evaluator.
      * Sexism evaluator.
      * Stereotype evaluator.
      * Toxicity evaluator.
   * Summarization evaluation
      * BLEU evaluator.
      * ROUGE evaluator.
      * Summarization (Completeness and Faithfulness) evaluator.
      * Summarization (Judge) evaluator.
      * Summarization with reference (GPTScore) evaluator.
      * Summarization without reference (GPTScore) evaluator.
   * Classification evaluation
      * Classification evaluator.
* **Features**
   * Introducing `Evaluators` as a new type of explainers which are able to evaluate
     the quality of Retrieval-Augmented Generations (RAG) products.
   * New evaluator API - `evaluate` module to run evaluators,
    `evaluators` module  to implement new evaluators and Bring Your Own Evaluator (BYOE).
   * New evaluator specific datasets based on `LlmDataset`, models `ExplainableRagModel`
     with implementations for `h2oGPTe` and OpenAI Assistants with retrieval.
   * New evaluator `testing` module with the test support bringing test suites, test cases
     tests and test labs.
   * New `genai` module with LLM/RAG host clients:
      * H2O Enterprise `h2oGPTe`
      * H2O GPT
      * H2O LLMOps
      * OpenAI Chat
      * Open AI Assistants with Retrieval tool (version 1) or File Search tool (version 2)
      * Microsoft Azure hosted OpenAI Chat
      * Open AI Chat compatible endpoints
      * Amazon Bedrock
      * ollama
   * HTML report branding for the EvalStudio.
   * Insights - new feature allowing explainers and explanations to provide insights into the
     evaluation results and suggest actions to be taken.
* **Explanations and formats**
   * New leaderboard (heatmap and bool) explanations with support for multiple
     evaluation metrics along with HTML, JSon and Markdown formats.
   * New (normalized) evaluator result (`EvalResult`) and explanation formats (JSon, Markdown).
* **Enhancements**
   * Installation of H2O Sonar using package extras - install only what you need: core,
     `explainers` and/or `evaluators`.
   * `ragas` library integration (license).
* **Testing**
   * New `llm` pytest label for LLM and RAG tests.
   * Test suites, test labs and test datasets for the LLM and RAG evaluation:
     `h2oGPTe` benchmark, Kaggle LLM Data Science competition, Talk to report and evalgpt.ai.
* **Changes**
   * `Cython` Python dependency upgraded from `0.29.32` to `0.29.37`.
* **Backward compatibility breaking changes:**
   * Python 3.8 is no longer officially supported.
   * Python 3.9 is no longer officially supported.
   * Python 3.10 is no longer officially supported.
   * JSon file with interpretation parameters which was stored in the interpretation
     directory is no longer persisted as it contained duplicate information which can be
     found in the `interpretation.json` file.
* **Documentation**
   * Updated documentation of new features and enhancements.

### v2.0.0 Release Candidates
List of 2.0.0 release candidates with the detailed description of the changes:

* **RC 68** - 2024-09-13
   * Evaluators:
      * The new Answer Relevancy (Sentence Similarity) evaluator assesses how relevant
        the actual answer is by computing the semantic similarity between the question and
        the actual answer sentences.
      * The new Context Relevancy (Soft Recall and Precision) evaluator measures the relevancy
        of the retrieved context based on the question and context sentences sentences
        semantic similarity.
   * Enhancements:
      * Toxicity evaluator improved to calculate the toxicity metrics on the sentence
        granularity and report the maximum of the toxicity metrics values.
        This enhancement makes the evaluator results more valuable as it
        can detect the toxic content in the generated text regardless its length
        (toxic content can no longer hide in long(er) actual answers).
      * Amazon Bedrock model host newly checks the accessibility of the LLM models
        supported by the RAG and filters out the inaccessible models.
* **RC 67** - 2024-09-06
   * Fixes:
      * Summarization (Completeness and Faithfulness) evaluator fixed to safely use MD5
        for the metrics calculation.
   * Changes:
      * `h2oGPTe` client downgraded to version `1.5.16` to integrate with old(er) servers.
* **RC 66** - 2024-09-06
   * Fixes:
      * Evaluated model ID added to the HTML report to simplify mapping of model IDs in the
        evaluation results (JSon, CSV, frame) to human readable model metadata.
   * Changes:
      * `h2oGPTe` client upgraded to version `1.6.0.dev3`.
      * H2O Eval Studio leaderboards Markdown representation title heading level changed to `H2`.
* **RC 65** - 2024-09-05
   * Features:
      * Amazon Bedrock model host support - evaluation of Amazon Bedrock RAG -
        knowledge bases (collections) and configured LLM models.
   * Fixes:
      * Perturbation flip detection fixed - it didn't consider answers created by the
        different RAG/LLM models and reported false negatives.
* **RC 64** - 2024-09-03
   * Fixes:
      * h2oGPTe LLM models listing retries fixed to avoid the flakiness and ensure it will
        be performed at least once.
   * Documentation:
      * Comprehensive update of evaluator documentation: formulas, methods, prompts,
        links to used models, and fixes.
* **RC 63** - 2024-08-29
   * Enhancements:
      * H2O Eval Studio Markdown representation revamp - new header section for
        bool/heat/class based leaderboard summaries, model/prompt/... failure sections
        truncated to at most 3 entries to scale the UI in case of many failures.
      * Model `vectara/hallucination_evaluation_model`, which is used by the Hallucination
        evaluator, updated to `HHEM-2.1-Open` and is frozen to avoid the model changes.
      * Added retries to the `h2oGPTe` client to avoid the flakiness when listing
        base LLM models.
      * Improved rendering of the multinomial classification confusion matrices in
        the HTML report.
* **RC 62** - 2024-08-27
   * Fixes:
      * Fixed broken retrieval and generation error messages construction in the Text matching
        evaluator.
      * Model and prompt leaderboard in the HTML report/Markdown/JSon representations - result
        failures are shown based on **generation** failures (not union of retrieval
        and generation failures) which ensures that failures and passes give 100%.
      * Model failure entries colors in the HTML report fixed - if the problem is in retrieval,
        only the context is in red. If the problem is in a generation, then the actual answer
        is in red.
      * Input field in the model failure list of the H2O Eval Studio markdown de-duplicated.
        Missing fields added to be on par with the HTML.
* **RC 61** - 2024-08-22
   * Enhancements:
      * Groundedness (semantic similarity) evaluator documentation updated.
      * Improved Hallucination evaluator error reporting on too long retrieved context chunks.
      * More robust perturbation flip direction detection.
* **RC 60** - 2024-08-21
   * Evaluators:
      * The new Groundedness evaluator assesses the groundedness of the generated
        text by considering the retrieved context - measuring hallucinations and
        fabricated text. It reports problems on sentence granularity
        in order to identify the hallucinations and fabricated root causes.
   * Enhancements:
      * Added infrastructure to detect the low number of evaluation examples
        in evaluators and report it as a problem.
      * Problems are newly categorized using the AVID taxonomy:
        https://docs.avidml.org/taxonomy/effect-sep-view/security
   * Fixes:
      * Threshold consistency between evaluator thresholds and
        metrics threshold defaults fixed.
      * Propagating of actual threshold values to the JSon leaderboard
        representation fixed.
      * Exception handling in the test lab completion on the parallel job
        failure fixed.
* **RC 59** - 2024-08-20
   * Enhancements:
      * Test lab completion progress reporting is now more detailed - it includes prompt, LLM, and RAG/LLM
        host names.
   * Fixes:
      * Rounding of metrics values in insights, problems and Markdown representations aligned
        to 4 decimal places. Percentage values are rounded to 1 decimal place.
   * Changes:
      * `h2oGPTe` client upgraded to version `1.5.11`.
* **RC 58** - 2024-08-14
   * Fixes:
      * Links to explanation data in the HTML report changed from directories to files
        in case of H2O Eval Studio branding as (S3) directories cannot be listed in
        case of the H2O Eval Studio deployment.
   * Changes:
      * Rollback to vulnerable NLTK `3.8.1` (`CVE-2024-39705`) Python dependency as `3.8.2`
        has been purged from pypi.org
* **RC 57** - 2024-08-14
   * Enhancements:
      * Default h2oGPTe client timeout to get answer from the LLM or RAG collection
        is newly 420s (was 1000s).
      * Metrics values in Markdown are newly rounded to 4 decimal places.
   * Fixes:
      * Perturbation of perturbed test suites is newly cloned when not perturbing in place.
      * GPTScore threshold parameter description fixed in the evaluator metadata.
      * Hiding H2O Sonar specific texts in the HTML report in case of H2O Eval Studio branding.
* **RC 56** - 2024-08-12
   * Enhancements:
      * Added detection of Summarization evaluator failures on all dataset rows and fail fast
        via raising an exception.
      * Added precondition check on empty evaluation results to all leaderboard types.
      * Evaluator metadata lookup made possible for incompatible evaluators in the HTML report.
      * Test lab completion no longer uses "shard" terminology, but "parallel job" instead.
      * English variant of `punkt` from NLTK is newly cached as the model used by the evaluators.
   * Changes:
      * Updated vulnerable NLTK `3.8.1` (`CVE-2024-39705`) Python dependency to fixed version `3.8.2`.
* **RC 55** - 2024-08-09
   * Fixes:
      * Minor robustness fix in the handling of extra argument passed to the h2oGPTe client.
* **RC 54** - 2024-08-08
   * Fixes:
      * Fixed problem detection on Answer semantic similarity evaluator flip detection.
        RAGAs evaluator fixed to declare all metrics it calculates in the metadata.
        Also RAGAs evaluator docstring changed to announce RAGAs metrics only in
        the documentation.
* **RC 53** - 2024-08-08
   * Fixes:
      * Perturbation of a test suite using multiple perturbators no longer creates
        exponential number of perturbed test cases. Instead, there are original tests
        with their test cases and perturbed tests with their perturbed test cases.
        Thus the number of test cases is 2x the original number of test cases.
   * Changes:
      * Internal perturbation API of test suites, tests and test cases changed to
        support multiple perturbators so that the perturbations can be created in place
        and relationships properly set.
* **RC 52** - 2024-08-07
   * Enhancements:
      * Test lab completion newly fails fast - raises exception - in case that completion
        of all test lab's test cases fail.
      * Evaluations, interpretations and their JSon representations has new `error`
        field which contains the error message in case of the evaluation/interpretation
        failure.
   * Changes:
      * `h2oGPTe` client upgraded to version `1.5.11-dev2`.
* **RC 51** - 2024-08-06
   * Enhancements:
      * Missing expected answer in the test case is reported as a problem by the evaluators.
   * Fixes:
      * The HTML report generator doesn't fail on an invalid explainer ID when getting
        the display name, but returns the ID with a prefix. An error message is logged.
* **RC 50** - 2024-08-02
   * Enhancements:
      * Progress report in the test lab completion no longer includes a full prompt,
        but just a prefix.
* **RC 49** - 2024-08-02
   * Enhancements:
      * Brief evaluators descriptions were shortened - newly contain just the first
        paragraph of the full description.
      * Evaluators check whether actual answers in test cases/suites/labs has correct type
        and if not, they generate the corresponding problems.
      * Air-gapped deployment support improved - 3rd party models used by
        the evaluators/evaluation libraries are newly frozen (where possible) to prevent
        model changes.
   * Fixes:
      * In an attempt to complete the test lab for exactly one model in parallel, the test lab
        automatically switches to the serial mode.
      * Insights about the fastest/slowest/cheapest/most expensive models are not generated
        for the evaluations with exactly one model.
   * Changes:
      * Progress reports generated by evaluators newly start with display names of the
        evaluators rather then IDs.
* **RC 48** - 2024-07-30
   * Enhancements:
      * Brief evaluator description added to the public API - `list_evaluators()` and
        `describe_evaluator()` newly return it.
* **RC 47** - 2024-07-30
   * Features:
      * Added support of the Open AI RAG version 2.0 - Assistants with File Search tool.
      * New conditions in Token Presence evaluator - new syntax which brings support of
        ``NOT`` and parentheses for the complex conditions.
      * Red teaming test suite with various LLM/RAG attacks added to the repository. This
        test suite can be used for penetration testing of the LLM/RAG models.
   * Enhancements:
      * Improved test lab API allows to complete test labs of RAG system using given
        (existing) collections instead of creating new ones. This API allows user to create,
        configure and customize the collections, upload corpus and documents, and then use
        them in the test lab completion.
      * Evaluator container newly detects invalid LLM dataset rows which contain RAG/LLM host
        error messages instead of the actual data and reports them as problems.
      * Evaluators newly provide brief description apart to full description.
      * Perturbators are newly ensuring that the perturbed data are not equal to the original data
        and fail if the perturbation did not change the data.
      * Connection configuration has new `extra_params` dictionary field which can be used to pass
        additional parameters to the connection client. For example, setting the `timeout` parameter
        on the h2oGPTe connection will apply the timeout parameter to all requests (that support it)
        made by the h2oGPTe client.
      * Versions of cached/downloaded models - like `vectara/hallucination_evaluation_model` or
        `gpt2-medium` - used by evaluators are newly frozen to avoid the model changes.
   * Fixes:
      * Negative (RAG/LLM) cost of the prompt is reported as a problem by evaluators which create
        boolean leaderboards. The cost is also set to `0.0` in the evaluation results to minimize
        the impact of the cost on the evaluation.
   * Changes:
      * `h2oGPTe` client upgraded to version `1.5.8`.
      * Perturbation probability intensity increased in Qwerty and Antonym perturbators to
        ensure sufficient perturbation of the data.
   * Security:
      * `setuptools` upgraded to `70.0.0` to fix vulnerability `CVE-2024-6345`.
      * Open AI RAG version 2.0 support brings upgrade of the `openai` Python library
        from version `1.20.0` to the version `1.35.13`, which fixes LangChain community
        vulnerability `CVE-2024-2965`.
* **RC 46** - 2024-06-28
   * Evaluators:
      * Four new GPTScore-based evaluators for the evaluation of the summarizations
        with the reference summaries, evaluation of the summarizations without
        the reference summaries, evaluation of machine translations and evaluation
        of the question answering.
   * Features:
      * Evaluation / interpretation API can list all and incompatible evaluators / explainers.
   * Enhancements:
      * Evaluators assessing `boolean` metrics, such as token presence or PII leakage,
        now have the ability to use custom metric names and descriptions to make reports
        and evaluation data more comprehensive.
      * Evaluators newly have keywords indicating whether they require LLM judge,
        prompt, expected answer, actual answer, retrieved context or constraints.
      * Significantly improved descriptions of all evaluators - descriptions are mostly
        generated from the evaluator class metadata.
      * Problems are newly sorted by severity (from highest to lowest).
      * Insights are sorted by type (alphabetically).
      * All and incompable evaluators/explainers newly shown in the evaluation report.
   * Fixes:
      * Missing threshold added to parametrizable BYOP evaluator.
   * Breaking changes:
      * Evaluator keyword `sr-11-7-ongoing-analysis` has been fixed
        to the correct `sr-11-7-ongoing-monitoring` keyword.
   * Documentation:
      * reStructuredText documentation of the evaluators rewritten - every evaluator has brief
        description, requirements, evaluation method, evaluation metrics, insights,
        and problems sections.
* **RC 45** - 2023-06-25
   * Enhancements:
      * New Random character type perturbator.
   * Fixes:
      * Integrity checks and validation of the model configuration
        (like embeddings, tokenization, temperature, token limits) used to build the test lab.
   * Changes:
      * Interpretation/evaluation is marked as successful if at least one evaluator successfully finishes.
      * h2oGPTe client upgraded to version `1.5.1-dev7`.
      * Python 3.11 dependencies upgraded: `cryptography` to version `42.0.8`, `scikit-learn` to version `1.5.0`,
        and `toml` to version `0.10.2`.
* **RC 44** - 2023-06-14
   * Enhancements:
      * h2oGPTe client upgraded to version `1.5.0-dev21` to support the upcoming H2O Enterprise h2oGPTe release.
      * Colorized evaluation status added to the HTML report.
      * Crash of an evaluator is newly reported as a high severity problem, and makes the evaluation to be marked as failed.
        However, the evaluation continues with the other evaluators.
      * An attempt to run non-registered evaluator is newly reported as a high severity problem,
        and makes the evaluation to be marked as failed. However, the evaluation continues with
        the other evaluators.
      * Improved measurements of the LLM latency in the GenAI client.
   * Fixes:
      * Fixed duplicate prompts in the model weak points (the most difficult prompts) section of the HTML report.
* **RC 43** - 2024-06-11
   * Features:
      * Ability to configure h2oGPTe, h2oGPT, H2O LLMOps, ollama,
        OpenAI chat, OpenAI RAG, and Microsoft Azure hosted OpenAI clients to control
        the evaluation of LLM models (for instance `temperature`)
        and RAG systems (for instance `embeddings provider`, `system prompt` or `prompt template`).
   * Enhancements:
      * All perturbators are newly deterministic for improved robustness and testability
        (except synonym and antonym pertubators which are deterministic in testing only).
      * Synonym and antonym perturbators improved with eager synonym/antonym swap which tries to
        match the percentage of words swapped (prior the fix perturbators tried only x times, and
        if the new synonym/antonym was the same word, it would not swap anything).
   * Fixes:
      * Fixed all perturbators for issues with special tokens in de/tokenization like undesired spaces
        around expressions in parenthesis after detokenization.
   * Security:
      * Upgraded `scikit-learn` library to version `1.5.0` to solve vulnerabilities detected by SNYK.
      * Upgraded `cryptography` library to version `42.0.8` to mitigate vulnerabilities detected by SNYK.
   * Documentation:
      * reStructuredText documentation of the evaluation and new features (host configuration) with
        configuration prototypes examples.
* **RC 42** - 2024-05-31
   * Enhancements:
      * Keyword groups for grouping of keywords which are used to tag evaluators.
      * H2O Eval Studio *purpose* keyword group which organizes evaluators into disjunct sets.
* **RC 41** - 2024-05-30
   * Evaluators:
      * New perplexity evaluator for LLMs which calculates the perplexity - "measure of uncertainty" - of
        the generated text.
   * Enhancements:
      * Save JSon data decoder for NaN and infinities.
      * H2O Sonar can be configured whether to use GPU or CPU for the evaluation.
   * Fixes:
      * HTML report generation fixed in case that evaluation of all rows in the dataset fails.
* **RC 40** - 2024-05-29
   * Evaluators:
     * New summary evaluator provides completeness and faithfulness metrics for LLM summarization
       tasks evaluation without the need for a reference summary.
   * Features:
      * Insights - new feature allowing explainers and explanations to provide insights into the
        evaluation results and suggest actions to be taken.
   * Enhancements:
      * Evaluation JSon and HTML result includes overall evaluation result represented as one value
        which is based on the severity of the problems detected in the evaluation. It is represented
        as traffic light colors (green, yellow, red) in the HTML report.
      * All evaluators report insights about the evaluation results and suggest actions to be taken
        via insight enhancements in bool, heatmap and classification leaderboards explanations.
      * Text matching, PII and Sensitive data leakage evaluators report apart problems and
        accuracy related insights also insights about cost and performance (speed) of evaluated
        models.
      * Models section in the HTML report rewritten to contain model details, insights, and problems.
      * Example PIIs (emails, credit cards, SSNs) in the PII evaluator are no longer reported as problems.
        These false positives are now marked as `False` in the evaluation results.
      * Test lab statistics.
   * Fixes:
      * Hallucination evaluator fixed to correctly handle low values as
        hallucinations (not vice versa).
   * Changes:
      * Bool leaderboard JSon representation values (and metrics metadata) changed from percentages to
        `[0.0, 1.0]` float range.
* **RC 39** - 2024-05-06
   * Enhancements:
      * `ragas` library upgrade to version ``0.1.7``.
   * Fixes:
      * Added on-demand caching of `tiktoken`'s BLOBs which are used by `ragas` library.
      * Fixed Faithfullness evaluator and RAGAs evaluator flakiness (`NaN`) by `ragas` library upgrade.
* **RC 38** - 2024-05-03
   * Features:
      * ``ollama`` (https://ollama.com/) hosted LLMs support - new connection, client and test lab builder.
   * Enhancements:
      * All evaluators detect flip of metrics and report the flip in the evaluation results
        as problems.
        In case of boolean metrics, the flip is detected as change from `True` to `False` and
        vice versa. In case of numeric metrics, the flip is detected as change from above to
        below the threshold and vice versa. In case of the classification, the flip is detected
        as change from the correct to incorrect classification and vice versa.
   * Changes:
      * Introducing relationships among test cases which adds new `relationships` key to
        test case, test suite and test lab as well as column `relationsihps` to
        LLM dataset and LLM evaluation result.
        JSon representations (key) and CSV representations are extended (column).
        Old JSon files are deserialized in loosely coupled way to avoid the backward compatibility
        breaking changes.
      * Added `key` field inputs in the test lab.
      * Added `key` field/column to LLM dataset inputs (rows).
      * Added `key` field/column to evaluation result inputs (rows).
   * Fixes:
      * Fixed undesired retries in the RAG/LLM test lab completion of h2oGPTe LLM and H2O LLMOps hosts
        in case of the successful completion of the test cases.
      * Fixed `NaN` (not a number) handling in leaderboard pallette color lookup.
* **RC 37** - 2024-04-25
   * Evaluators:
      * New Classification evaluator for RAGs/LLMs used for classification problems. The evaluator
        calculates common metrics used in case of binomial and multinomial classification problems
        like accuracy, precision, recall and F1.
        The Classification evaluator is also bringing new classification leaderboard explanation.
   * Features:
      * New perturbations module with the ability to perturb the input data (5 perturbation methods)
        in order to test the robustness of the RAGs/LLMs and the quality of the data:
        comma, word swap, QWERTY, synonym and antonym.
      * New public perturbations API with list, filtering and (multiple) perturbation methods
        application to string, test case, test suite or LLM dataset prompts.
      * 3 new summarization tests for evaluation of summaries both with and without reference summary
        (Frank, SamSum and SummEval).
   * Enhancements:
      * Format specifier in evaluation metrics metadata changed from Python f-strings to
        JavaScript D3 format strings.
   * Fixes:
      * Ranges in evaluation metrics metadata fixed - [0, 1] vs. [0, 100].
   * Testing:
      * RAG/LLM test suite can finish successfully even if OpenAI API key is not set
        (auto reconfiguration to 3rd party judges; tests which use OpenAI endpoints are skipped).
* **RC 36** - 2024-04-18
   * Fixes:
      * OpenAI client fixed to version 1.20.0 to keep version 1 API compatibility (OpenAI Assistants
        code in H2O Sonar must be rewritten to version 2 to move from retrieval tool to file search).
* **RC 35** - 2024-04-18
   * Feature:
      * New metrics metadata - all evaluators newly declare the metrics they calculate
        with the metadata (name, description, type, unit, range, scale, ...).
        Metrics metadata are used in the evaluator (descriptor, evaluation, results),
        in the leaderbords (JSon representation, HTML report generation), and explanation/evaluation
        formats (JSon, HTML, Markdown).
      * Loosely coupled serialization and deserialization of object/JSon data structures:
        `ExplainerDescriptor`, `ExplanationDescriptor`, `ConfigItem` and `FilterEntry`.
      * Caching of the models used (internally) by evaluators and explainers: public API,
        caching module, and caching configuration enabling air gapped evaluators deployment.
   * Backward compatibility breaking changes:
      * `data` key added to heatmap and bool leaderboards JSon representations.
* **RC 34** - 2024-04-12
   * Fixes:
      * `NaN` (not a number) handling/encoding in the heatmap leaderboard JSon "all metrics" data file.
* **RC 33** - 2024-04-12
   * Features:
      * Microsoft Azure hosted OpenAI LLMs support - new connection, client and test lab builder.
      * H2O LLMOps hosted LLMs support - new connection, client and test lab builder.
   * Security:
      * HTTPS requests SSL certificate verification configuration: H2O Sonar configuration
        controls the SSL certificate verification process/level in requests library,
        LLM hosts client libraries and other HTTP(S) clients.
   * Changes:
      * H2O GPT client rewritten to OpenAI API client (please update server port and base URL).
      * H2O LLMOps client rewritten to OpenAI API client (no configuration changes needed).
      * Base URL parameter removed from OpenAI API client constructor (connection configuration is used).
* **RC 32** - 2024-04-10
   * Enhancements:
      * Constants for keys in `datasets.py` Python modules.
   * Documentation:
      * BLEU and ROUGE evaluators .rst documentation.
* **RC 31** - 2024-04-08
   * Evaluators:
      * BLEU evaluator.
      * ROUGE evaluator.
   * Enhancements:
      * New keywords for the most important ML problem types solved by RAGs/LLMs:
        question answering, information retrieval, summarization, classification
        (binomail and multinomial) and regression. All evaluators were decorated
        with relevant keywords.
      * New keyword for the referential user role: regulator.
   * Fixes:
      * NaN (not a number) handling in the evaluator results, formats and leaderboard.
   * Security:
      * `nltk` added as evaluators Pytho extras dependency.
      * `rouge-score` added as evaluators Pytho extras dependency.
      * `punkt` is new cached NLTK model for text to sentence tokenization.
* **RC 30** - 2024-03-27
   * Enhancements:
      * Toxicity evaluator reimplemented to directly use the `toxicity` library
        and show several metrics which explain what type of toxic content has been
        detected in the answer.
      * Fairness bias evaluator reimplemented to directly use bias detection model
        (in ONNX format) for the evaluation.
      * Hallucinations evaluator reimplemented to use LLM judge for the hallucination
        detection.
   * Security:
      * `deepeval` Python dependency removed.
        `Evaluators` based on `deepeval` were rewritten to use underlying libraries without
        relying on `deepeval`.
      * `TensorFlow` and `DBias` Python dependencies removed.
        Fairness Bias evaluator newly does not rely on `DBias` Python library as
        the underlying model was ported from `TensorFlow` to `ONNX`.
      * `HMLI` moved from the core H2O Sonar dependencies to the `explainers` package extras.
        in order to avoid the CVE vulnerabilities which must be fixed for H2O Eval Studio
        cloud deployment certification.
      * `H2O-3` moved from the core H2O Sonar dependencies to the `explainers` package extras
        in order to avoid the CVE vulnerabilities which must be fixed for H2O Eval Studio
        cloud deployment certification.
* **RC 29** - 2024-03-21
   * Security:
      * HMLI upgraded to MLI version 1.10.26 to mitigate CVE-2023-39013 (HMLI's Duke dependency vulnerability).
* **RC 28** - 2024-03-17
   * Features:
      * Bring Your Own Judge (BYOJ) - ability to configure H2O Sonar so that evaluators use
        custom LLM judges. For instance in order to ensure privacy and avoid sending of the
        sensitive data to a 3rd party.
        This feature includes reconfiguration of embeddings provider from the same reasons.
        Custom judges can be either forced from the H2O Sonar configuration or specified
        in the evaluator parameters.
      * Bring Your Own Prompt (BYOP) - ability to easily run evaluation just be providing a prompt
        template or implement a new evaluator just by inheriting from the BYOP abstract class and
        specifying prompt which returns the boolean value.
      * OpenAI LLM client (only Assistants with retrieval tool was supported before).
        The client supports both OpenAI service (no base URL specified) and OpenAI compatible
        endpoints (base URL specified).
   * Evaluators:
      * Contact Information evaluator (BYOP).
      * Language Mismatch evaluator (BYOP).
      * Parametrizable BYOP evaluator with the ability to specify the prompt template in the evaluator parameters.
      * Sexism evaluator (BYOP).
      * Stereotype evaluator which detects undesired gender/race content in the answer (BYOP).
      * Summarization evaluator (BYOP).
   * Enhancements:
      * `ragas` library upgrade to version ``0.1.3``.
* **RC 27** - 2024-03-14
   * Security:
      * Fairness Bias evaluator removed as it used `dbias` library which depends on vulnerable TensorFlow version. This change ensures there is no TensorFlow, un-registers the evaluator and skips all evaluator tests (code is kept in the codebase).
   * Enhancements:
      * Problems are newly loaded with the load of the evaluation from the JSon representation.
   * QA:
      * MMC builds disabled (it was extra cost in addition to GH Actions build; MMC has old Python version)
* **RC 26** - 2024-03-08
  * Features and enhancements:
      * progress reporting:
          * end to end, evaluation, all evaluators, lab (build and completion)
          * callback or file-system
      * evaluators can be filtered by labels for:
          * SR 11-7
          * NIST AI MRM
      * HTML report refactored
          * sections shuffled by importance
          * new evaluation (details) group added
          * dataset section content reordering
          * explanation and title added to insight leaderboards
      * Markdown representation robustness
          * input/output escaping
          * LLM vs. RAG failures listing fixed
          * ES summary .md redesigned
      * improved text matching regexp error messages and docstring (ES UI)
      * improved .rst documentation
      * evaluator parameters refactored from standalone file to i/e.json
  * Changes:
      * evaluation result is stored on the file system (no longer discarded)
  * Fixed:
      * 3x faster lab completion (fixed duplicate requests)
      * hangs/deadlocks in the lab completion (configurable multiprocessing)
  * QA:
      * new GH Actions test suite in GREEN
        since 09117c3ce891e410e68e62361d109f179ed4c79f
      * GHA builds and test H2O EvalStudio deployment runtime configuration only
      * improved h2oGPT/h2oGPTe test server selection (config switch)
      * method to purge h2oGPTe relics
      * new h2oGPT servers
* **RC 25** - 2024-02-05
   * Evaluators:
      * Fairness bias evaluator (`deepeval` based).
* **RC 24** - 2024-02-02
   * Evaluators:
      * Toxicity evaluator (`deepeval` based).
   * Fixed:
      * PII and sensitive data leakages (regexps).
* **RC 23** - 2024-01-26
   * Features:
      * LLM/RAG clients telementry.
      * Prompt cache: LLM/RAG responses can be cached on building a test lab.
        The cache can be build from existing test lab and used in RD only mode.
   * Enhancements:
      * LLM/RAG client retries (3 by default).
      * Evaluators which require OpenAI key are tagged using keywords.
      * ...
* **RC 22** - 2024-01-22
   * Enhancements:
      * Changing h2oGPTe dependency to the last Python package version.
   * Fixes:
      * Hiding retrieval errors in the bool leaderboard.
   * Evaluation tests:
      * Removal of constraints OR expressions from test suite/labs for Atlanta event as H2O Eval Studio
        does not support it yet.
* **RC 21** - 2024-01-21
   * Fixes:
      * Fixed RAGAs leaderboard calculation.
      * Retrieved context builder enhancements.
   * Tests:
      * OpenAI end to end CI test which runs all evaluators.
   * Evaluation tests:
      * Polished, fixed (duplicate prompts) and extended SR 11-7 and Bank teller test suites.
* **RC 20** - 2024-01-18
   * Evaluators:
      * Sensitive data leakage evaluator.
   * Enhancements:
      * Test lab build fallbacks: dummy doc for RAG.
   * Fixes:
      * OpenAI test lab build (missing arguments).
   * Evaluation tests:
      * SR 11-7 test suite w/ 171 prompts.
* **RC 19** - 2024-01-18
   * Evaluators:
      * PII evaluator.
   * Fixes:
      * Asynchronous interpretation execution fixed (inconsistent method signatures).
   * Changes:
      * `datatable` upgraded from AWS S3 hosted version, to `1.1.0` pypi.org hosted version.


## [v1.1.1](https://github.com/h2oai/h2o-sonar/tree/v1.1.1) — 2023-10-09

A patch release bringing minor fixes and enhancements.

### Added

* Both CLI and Python API accept library configuration and encryption key parameters in
  case that the the interpretation arguments are provided as JSon.

### Fixed

* HTML interpretation report path in the CLI output fixed (it was pointing to the interpretation HTML index).
* False positive feature importance leak detection is no longer reported in case of multinomial problems.
* Morris Sensitivity Analysis no longer fails in case of non-numeric boolean columns presence in the training dataset.



## [v1.2.0](https://github.com/h2oai/h2o-sonar/tree/v1.2.0) — 2023-10-31

Talk to H2O Sonar report - upload your interpretation report to the [Enterprise h2oGPT](https://h2o.ai/platform/enterprise-h2ogpt)
in order to find out more about your model, data, problems, insights and suggested (mitigation) actions.

### Added

* **Features**
   * Ability to upload your interpretation report to [Enterprise h2oGPT](https://h2o.ai/platform/enterprise-h2ogpt) either using
     Python API (``run_interpretation()`` method parameter, ``upload_interpretation()`` method), or CLI.
     The feature is supported with Python 3.10 and Python 3.11 only.
* **Documentation**
  * H2O.ai documentation theme.

### Fixed

* Wheels are no longer built with the legacy pip resolver which was causing dependency conflicts
  in some cases on certain platforms.
* Test and validation dataset details are newly shwon in the HTML report.
* Opened port / Driverless AI server port check is no longer verbose.



## [v1.1.2](https://github.com/h2oai/h2o-sonar/tree/v1.1.2) — 2023-10-13

A patch release bringing minor fixes and enhancements.

### Added

No additions.

### Fixed

* SHAP library version fixed to `shap>=0.40.0,<=0.42.5` as new version is causing instability in feature importance explainers.

### Changed

* H2O Model Validation upgraded to `0.16.3` with updated `h2osteam` and H2O MLOps clients which avoid version clashes in upcoming H2O.ai Cloud notebook kernels.



## [v1.1.0](https://github.com/h2oai/h2o-sonar/tree/v1.1.0) — 2023-10-03

Integration of H2O Sonar and H2O Model Validation projects.

### Added

* **New explainers**
   * Adversarial Similarity explainer.
   * Backtesting explainer.
   * Drift Detection explainer (reports exceeded PSI threshold as a problem).
   * Size Dependency explainer.
   * Segment Performance explainer.
   * Calibration Score explainer.
* **Features**
   * H2O Model Validation based explainers are able to use H2O AIEM hosted Driverless AI,
     H2O Enterprise Steam hosted Driverless AI or any H2O Driverless AI which uses
     username/password authentication.
   * Ability of H2O Sonar to run with or without H2O Model Validation library installed.
     If H2O Model Validation is not available, then H2O Model Validation based explainers
     just indicate incompatibility and do not cause the interpretation to fail.
   * Portable export and import of ``MVTest`` related instances like settings, results,
     artifacts, and logs. The implementation is based on JSon, CSV, and directory hierarchy.
     Therefore it can be used by a wide range of tools, programming languages, and runtimes.
   * ``RemoteHandle``s bring support for remote (Driverless AI) datasets and models. Apart from
     the data structure, it is a part of explainers metadata and compatibility checks.
   * Model is no longer required when running a new interpretation which allows to run
     explainers on datasets only.
   * Automatic fallback guess of the model metadata - like problem type, labels, and used
     features - in case the model does not provide them.
* **Enhancements**
   * Attributes (dictionary) added to ``ProblemAndAction`` class which enables explainers
     to pass machine-processable data from problems to actions for further actionability.
   * Connections and licenses are newly identified by unique keys (identifiers) in
     the H2O Sonar configuration and through the runtime.
   * Python 3.10 support.
   * Python 3.11 support - H2O Model Validation explainers not available as transitive library
     dependencies do not support Python 3.11.
   * ``daimojo`` library pre-heat prediction to activate the MOJO models introspection.
   * Interpretations index HTML path added to the CLI interpretation output.
   * Completion of the ``testset`` and ``validset`` handling implementation
     in the explainer container - datasets are newly passed to explainers along
     with their metadata.
  * The following configuration keys were added to the H2O Sonar library configuration:
     * `server_id`
     * `environment_url`
     * `token_use_type`
   * Shapley Values for Original Features (Kernel SHAP Method) explainer is approximately
     3x faster case of multinomial problems (the speed up is proportional to the number
     of classes - more classes, more speed up).
* **Utilities**
   * Shapley contributions sorter which can be used by all Shapley-based explainers whenever
     multi-class contributions are reported within the same frame - makes the code cleaner
     and simpler.
* **Documentation**
   * Library configuration CLI API ``reStructuredText`` documentation.
   * Jupyter Notebook with examples of how to run H2O Model Validation explainers using
     the Python API and CLI.
   * ``reStructuredText`` documentation of all new H2O Model Validation based explainers.
   * New explainers overview table with per-explainer features and requirements
     added to both ``README.md`` and ``reStructuredText``.
   * Explainers overview diagram is newly organized according to the functional architecture
     of explainers.
* **Tests**
   * Python and CLI tests of all H2O Model Validation explainers.

### Fixed

* Shapley Values for Original Features (Kernel SHAP Method) explainer reports per-class
  contributions in the case of multinomial problems (contributions were mixed together).
* Morris Sensitivity Analysis explainer fixed to work with ``InterpretML`` 0.1.20.
* Pseudocode and Python code generated by the Decision Tree explainer is consistent again.
* HTML report fixed to properly handle if no explainer is run within the interpretation.
* Thread safe interpretation executor shutdown.

### Changed

* The following configuration key was changed in H2O Sonar library configuration:
  * `client_refresh_token` has been renamed to `token`.

### Removed

* Test suites which were replaced by Pytest markers.
* Tests of legacy Driverless AI models (``Makefile`` targets, S3 archives).



## [v1.0.0](https://github.com/h2oai/h2o-sonar/tree/v1.0.0) — 2023-06-30

The first stable H2O Sonar release.

### Added

* **Enhancements**
   * Multiple sampling methods for the explainer dataset (stratified, random, head).
   * Configurable out-of-memory (OOM) protection.
   * Improved ability of the interpretable model to extract `scikit-learn` models metadata.
* **Utilities**
   * Random attack utility that tests H2O Sonar on many datasets and models: it gets
     a directory with datasets as a parameter, trains a (`scikit-learn`) model for
     a random dataset and its column, and finally runs all the explainers to test the H2O Sonar.
* **Documentation**
   * Explainers overview diagram indicates whether the explainer reports problem(s).
   * Configuration management documentation (including encryption).
   * Per-explainer problem reporting capabilities documentation.

### Fixed

* Summary Shapley explainer and Original feature importance explainer fixed to properly
  use SHAP library to get Shapley values for regression vs. multinomial (experiment
  type detection).
* Disparate Impact Analysis calculation fixes (comparisons in metrics) in case of
  string features.
* Decision tree Python code and pseudo-code generator fixed.
* HTML report fixed to properly display explanations type and format(s).
* Division by zero fixed in the progress reporting runtime.

### Changed

* CLI, JSon and Python parameter names were unified - this change breaks backward
  compatibility and was intentionally done before the first stable release.

### Security

* Added encryption of sensitive fields in the H2O Sonar configuration (config, CLI, documentation).



## [v0.11.2](https://github.com/h2oai/h2o-sonar/tree/v0.11.2) — 2023-07-26

A patch release bringing minor fixes and enhancements.

### Fixed

* Fix .py/pseudo code generated by DT: > vs. >=

### Changed

* Upgrade MLI jar to 1.10.23



## [v0.11.1](https://github.com/h2oai/h2o-sonar/tree/v0.11.1) — 2023-05-22

Handle missing value bins for PD when OOR is enabled and output histogram
data to PD results.

### Added

* **Enhancements**
   * Output previously missing histogram data to PD results.

### Fixed

* Correctly handle missing value bins for PD when OOR is enabled.



## [v0.11.0](https://github.com/h2oai/h2o-sonar/tree/v0.11.0) — 2023-04-24

Leak detection added to feature importance explainers.

### Added

* **Enhancements**
   * Leak detection added to feature importance explainers: Shapley
     Values for Original Features (naive method) explainer, Morris
     Sensitivity Analysis explainer, Shapley Values for Original
     Features (Kernel SHAP method) explainer.
   * Missing values are treated as a separate bin in the PD explainer.
   * H2O Sonar CLI can read arguments from JSon file.

### Fixed

* Fixed display of plots in Jupyter notebooks.



## [v0.10.1](https://github.com/h2oai/h2o-sonar/tree/v0.10.1) — 2023-03-09

Patch release bringing `Result` (documentation) enhancements.



## [v0.10.0](https://github.com/h2oai/h2o-sonar/tree/v0.10.0) — 2023-02-02

New Dataset and Model Insights explainer and fixes of bugs found by a new random attack.

### Added

* **Explainers**
   * New Dataset and Model Insights explainer.
* **Enhancements**
   * Residual Decision Tree explainer newly highlights the whole path to the highest
     residuum in the visualized tree.
   * DIA result API help related to the reference level improved.

### Fixed

* Surrogate Decision Tree Python code generator fixed: added missing `(` `)` in boolean
  expressions, features can have any characters in their names.
* Move from `os.rename` to `shutil.move` in order to ensure that the operation will not
  fail if the source and target are on different file systems.
* Missing `isna` symbol used in the Disparate Impact Analysis explainer.
* Comparison of `string`s and `bool`s in the ICE method.
* Float division by zero in the Residual Decision Tree explainer.



## [v0.9.0](https://github.com/h2oai/h2o-sonar/tree/v0.9.0) — 2023-01-13

Minor H2O Sonar release which brings asynchronous interpretation execution.

### Added

* **Features**
   * New option allowing to run interpretations asynchronously.
* **Enhancements**
   * New introspection API for Result classes (method parameters).

### Fixed

* Sqrt MSE to get RMSE in the Surrogate Decision Tree explainer.
* Handling of date, time and date time features in the PD/ICE explainer.



## [v0.8.0](https://github.com/h2oai/h2o-sonar/tree/v0.8.0) — 2022-12-08

New Partial Dependence for 2 Features explainer and enhancements for H2O Sonar
explainer container implementation for Driverless AI.

### Added

* **Explainers**
    * New Partial Dependence for 2 Features explainer.
* **Features**
    * New Global 3D Data result, explanation and associated formats (JSon, CSV).
* **Enhancements**
   * Command-line interface with pretty-printed listing of explainers,
     improved formatting of explainer descriptions and H2O Sonar version `show` action.
   * Residual PD/ICE for multinomial problems added.
   * Improved explainer container resolution and creation (identifier, instance).
   * Model agnostic API to indicate the ability to provide/calculate Shapley values added.
   * Improved compatibility checks and new compatibility error type.
   * Explainable model's features metadata simplification, completion and consolidation.
   * Explainable dataset's metadata simplification, completion and consolidation.
   * Improved HTML report highlights failed explainers, brings a comprehensive
     overview section, shows new modal and dataset metadata fields.
* **Documentation**
   * Added Jupyter Notebook documentation of how to run H2O Sonar in
     the Internal H2O.ai Cloud.

### Fixed

* Disparate Impact Analysis explanations completed to be 100% binary compatible
  with Driverless AI's Grammar of MLI (entities).
* Disparate Impact Analysis explainer feature resolution for DIA calculation rewritten.
* Disparate Impact Analysis explainer and PD/ICE explainer fixed to work on
  a dataset with string (target) column(s).
* Residual PD/ICE no longer returns regular PD/ICE as the default representation
  (and residual as an extension), but the residual PD/ICE.
* Residual PD/ICE HTML fragment representation path to images fixed so that
  it no longer renders the same charts for all classes.
* Summary Shapley explainer name correctly indicates SHAP method
  (not wrong naive Shapley method).

### Changed

* Features metadata class of the explainable model has been refactored to the
  `h2o_sonar.methods.core.method` module and all constant references consolidated
  to this class.
* Operating system version to build Linux distribution and wheels has been changed
  from `Ubuntu 20.04` to `Ubuntu 18.04` to ensure that H2O Sonar wheels will work
  both on this and new Ubuntu versions.

### Security

* MLI upgrade to 1.10.21 to mitigate CVE-2022-2048 and CVE-2022-25647.


## [v0.7.0](https://github.com/h2oai/h2o-sonar/tree/v0.7.0) — 2022-10-18

H2O Sonar **beta** release with Bring Your Own Explainer based extensibility,
reporting of model problems,  new Residual PD/ICE explainer, new Morris sensitivity
analysis and various smaller enhancements.

### Added

* **Features**
   * BYOE - Bring Your Own Explainer.
   * Model problems and actions.
* **Explainers**
   * New Residual Partial Dependence/Individual Conditional Expectations explainer.
   * New Morris Sensitivity Analysis explainer.
   * Residual Decision Tree explainer reports problems and actions.
* **Explanations**
   * New interpretation report - structure, content, and theme in H2O.ai colors.
   * Organization of explainers to functional groups.
* **Utilities**
   * Improved label encoder to simplify the use of 3rd party libraries that require
     numeric (non-categorical) features. Label encoder is integrated into both
     explainable dataset and explainable model APIs.
* **Command-line interface**
   * All Python API's interpretation parameters are newly available on CLI.
* **Documentation**
   * Added Getting started with BYOE.

### Fixed

* HTML report paths to images and explanations are relative and valid regardless
  of the results directory location.
* Explainer container runtime and explainers stabilized to work on raw (non-sanitized)
  datasets.
* Explainers listing action help fixed on the command line interface.

### Changed

* `list_explainers()` method on both Python API and CLI lists **all** explainers
  by default (it listed only basic explainers with `run-by-default` keywords before
* this change).
* Logging consolidated to single module `h2o_sonar.loggers` and loggers
  renamed/refactored so that it can be used both in methods and explainers.
* Migration of explainer container runtime from `HMLI` to `h2o` wheel dependency.
* Parameter `path` of `zip()` method used by explainer's `Result` class has been
  changed to `file_path` to make it consistent with other `Result` parameters.
* `Result` classes refactoring from explainer implementations into consolidated
  and reusable results classes for main supported explanation types.
* The `summary()` method's functionality is moved to `params()` and the new
  `summary()` method returns the summary of the explanation (content of
  `result_descriptor.json`)




## [v0.6.0](https://github.com/h2oai/h2o-sonar/tree/v0.6.0) — 2022-09-08

New Friedman's H-statistic and Residual Surrogate Decision Tree explainers,
Driverless AI REST interface model support and improved HTML interpretation
representation.

### Added

* **Explainers**
    * Friedman's H-statistic explainer for feature behavior explanations.
    * Residual Surrogate Decision Tree for model debugging (new default explainer).
* **Model support**
   * Added Driverless AI REST interface model support.
* **Explanations**
   * Significantly improved HTML interpretation representation with new explanation
     charts for every explainer, interpretation parameters and explainers parameters.
* **Command line interface**
   * Added parameter to run all explainers (not just basic explainers).
   * Interpretation listing including HTML representation.
* **Documentation**
   * Bring Your Own Explainer templates and examples added to distributions.

### Fixed

* Improved scikit-learn multinomial models support with labels lookup.
* Compatibility check function gets all available parameters for more advanced checks.
* DIA HTML fragment representation path to images.
* In-memory persistence store (keys) stabilization.
* Logging names and interpretation and explainer logging keys consistency.

### Changed

* `hmli` and `daimojo` dependencies updated.
* Source distribution - `tarball` - build changed so that doesn't contain `.whl`.
* Binary distributions are built for every supported platform.



## [v0.5.0](https://github.com/h2oai/h2o-sonar/tree/v0.5.0) — 2022-08-16

Fix release which brings binary distribution with improved documentation
and Jupyter Notebook examples.

### Added

* **Documentation**
   * Improved `ReStructuredText` documentation with getting started, library
     documentation (interpretation, configuration, explainers), licenses and change log.
   * New and improved Jupyter Notebook examples.
* **Model support**
   * Added pickled (Scikit-learn) models interpretability.
* **Command line interface**
   * Added parameters to specify features used by the model and per-explainer parameters.

### Fixed

* Summary Shapley explainer stabilization: scatter plot feature values fixed,
  main chart includes all features, regression/binomial/multinomial labels fixed,
  `max_features` parameter honored, per-class multinomial explanations
  are generated in all supported formats.
* Fixed the simple mock model prediction function and added SHAP method support
  for mock models.

### Changed

* Models and datasets - used by examples, demos and tests - consolidated and refactored
  to indicate dataset and model type.



## [v0.4.2](https://github.com/h2oai/h2o-sonar/tree/v0.4.2) — 2022-11-29

Fix of the following MLI Java backend security issues: CVE-2022-2048 and CVE-2022-25647.

### Security

* MLI upgrade to 1.10.17.2 to mitigate CVE-2022-2048 and CVE-2022-25647.



## [v0.4.1](https://github.com/h2oai/h2o-sonar/tree/v0.4.1) — 2022-11-17

Fix of the following MLI Java backend security issues: CVE-2022-2048 and CVE-2022-25647.

### Security

* MLI upgrade to 1.10.17.1 to mitigate CVE-2022-2048 and CVE-2022-25647.


## [v0.4.0](https://github.com/h2oai/h2o-sonar/tree/v0.4.0) — 2022-06-29

New Transformed Feature Importance explainer for Driverless AI MOJO models
and preparation for H2O Sonar integration to Driverless AI.

### Added

* **Explainers**
    * Transformed Feature Importance explainer for Driverless AI MOJO models.
* **Explainer container API and CLI**
   * H2O Sonar version available in runtime.
* **Documentation**
   * Jupyter Notebook with interpretation result API for the new explainer.
   * H2O Sonar explainers overview diagram updated.

### Fixed

* All [MLI-2](https://github.com/h2oai/mli-2) fixes between
  [H2O Sonar](https://github.com/h2oai/h2o-sonar) fork and now ported to this repository.
* Naive Shapley Feature Importance explainer multinomial explanations fixed
  and the performance improved.

### Changed

* Core H2O Sonar dependencies updated to be aligned with Driverless AI 1.10.4,
  two separate builds will be available going forward - regular and Driverless AI.

### Security

* MLI upgrade to 0.10.17 to mitigate CVE-2022-25647.



## [v0.3.0](https://github.com/h2oai/h2o-sonar/tree/v0.3.0) — 2022-06-22

New Kernel SHAP feature Importance explainer.

### Added

* **Explainers**
   * Kernel SHAP Feature Importance explainer for all supported interpretable models.
* **Explainer container API and CLI**
   * H2O-3 is automatically started (or reused) - based on H2O-3 configuration.
   * CLI rewrite to provide more accurate help, error reporting and robust execution.
* **Documentation**
   * Jupyter Notebook with interpretation result API for the new explainer.

### Fixed

* Interpretation HTML representation links are no longer broken on the use of the relative path.
* Explainers' summary method returns the correct (non-empty) parameters of the explainer run.
* Disparate Impact Analysis explainer core dump on invalid target column specification.



## [v0.2.0](https://github.com/h2oai/h2o-sonar/tree/v0.2.0) — 2022-06-03

New Feature Importance explainer for Driverless AI MOJO models.

### Added

* **Explainers**
   * Naive Shapley Feature Importance explainer for Driverless AI MOJO models.
* **Explainer container API and CLI**
   * list explainers to get available explainer IDs or descriptors.
* **Documentation**
   * Jupyter Notebook with interpretation result API for the new explainer.

### Fixed

* CLI: log level specification case insensitivity.
* macOS: Driverless AI MOJO import made local.



## [v0.1.0](https://github.com/h2oai/h2o-sonar/tree/v0.1.0) — 2022-05-27

Initial H2O Sonar internal MVP release.

### Added

* **Explainers**
   * Partial dependence/Individual Conditional Expectations explainer (PD/ICE)
   * Shapley summary plot explainer
   * Decision tree explainer
   * Disparate Impact Analysis explainer (DIA)
* **Explainer container with public explainer APIs**
   * Interpretation, model, dataset, explainer and persistence API.
   * Explainer container (runtime).
   * File-system and in-memory persistence.
   * Easy to use API for retrieval of explainer results.
* **Model vendor support**
   * Scikit-learn models.
   * H2O-3 models.
   * Driverless AI MOJO models.
* **Command line interface**
   * CLI support of MOJO and pickled models interpretations.
* **Documentation**
   * Per-explainer Jupyter Notebook with interpretation result API.
   * Installation, Getting Started and Reference Guide (Sphinx/HTML).
