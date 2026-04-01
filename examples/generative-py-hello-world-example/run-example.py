#!/usr/bin/env python3
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
"""
Generative Hello World Example

This example demonstrates how to use H2O Sonar to evaluate a RAG system
using h2oGPTe with a simple test suite.
"""
import json
import os
import sys

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import rag_groundedness_evaluator
from h2o_sonar.lib.api import models
from h2o_sonar.utils import testing


def setup():
    """Setup paths and check prerequisites."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_suite_path = os.path.join(script_dir, "test-suite.json")
    results_dir = os.path.join(script_dir, "results")

    api_key = os.environ.get("H2O_GPTE_API_KEY")
    if not api_key:
        print("ERROR: H2O_GPTE_API_KEY environment variable is not set.")
        print("Please export your h2oGPTe API key:")
        print("  export H2O_GPTE_API_KEY='your-api-key-here'")
        sys.exit(1)

    return test_suite_path, results_dir, api_key


def run_evaluation(test_suite_path, results_dir, api_key):
    """Run RAG evaluation using H2O Sonar."""
    logger = loggers.SonarPrintLogger()

    connection = h2o_sonar_config.ConnectionConfig(
        name="h2oGPTe Public",
        description="Connection to h2oGPTe Public instance",
        connection_type=h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name,
        server_url="https://h2ogpte.h2o.ai",
        token=api_key,
        token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
    )
    print(f"Using h2oGPTe connection - check the URL of the instance: {connection.server_url}")

    test_suite = testing.RagTestSuiteConfig.load_from_json(test_suite_path)
    logger.info(f"Loaded {len(test_suite.test_cases)} test cases from {test_suite_path}")

    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=connection,
        rag_test_suite=test_suite,
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=["auto"],
    )

    logger.info("Building test lab and running queries (this may take a few minutes)...")
    test_lab.build()
    test_lab.complete_dataset(parallelize=False)
    test_lab_path = "test_lab.json"
    with open(test_lab_path, "w") as f:
        json.dump(test_lab.to_dict(), f, indent=4)
    print(f"Completed test lab saved to: {test_lab_path}")

    logger.info("Running evaluation...")
    evaluation = evaluate.run_evaluation(
        dataset=test_lab.dataset,
        models=test_lab.evaluated_models.values(),
        evaluators=[rag_groundedness_evaluator.RagGroundednessEvaluator.evaluator_id()],
        results_location=results_dir,
        log_level=loggers.INFO,
    )

    if evaluation.is_evaluator_successful():
        logger.info("OK Evaluation completed successfully!")
        logger.info(f"Results: {results_dir}")
        logger.info(
            f"HTML report: file://{evaluation.result.get_html_report_location()}"
        )
    else:
        logger.warning(
            f"ERROR Some evaluators failed: {evaluation.get_failed_evaluator_ids()}"
        )

    return 0 if evaluation.is_evaluator_successful() else 1


def main():
    """Run H2O Sonar RAG evaluation example."""
    test_suite_path, results_dir, api_key = setup()
    return run_evaluation(test_suite_path, results_dir, api_key)


if __name__ == "__main__":
    sys.exit(main())
