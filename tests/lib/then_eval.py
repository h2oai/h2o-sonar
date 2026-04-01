# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib

import pytest

from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import evaluations
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api.explanations import LlmLeaderboardExplanation


def then_leaderboard_json(
    evaluation: evaluations.Evaluation,
    evaluator_id: str,
    model_names_to_assert: list[str] | None = None,
    metric_ids_to_assert: list[str] | None = None,
    threshold: float | None = None,
) -> dict:
    """Evaluate the leaderboard JSon."""
    model_names_to_assert = model_names_to_assert or []
    metric_ids_to_assert = metric_ids_to_assert or []

    evaluator_jobs = evaluation.get_jobs_for_explainer_id(evaluator_id)
    if not evaluator_jobs:
        pytest.skip(f"No explainer jobs found for the given evaluator: {evaluator_id}")
    evaluator_job = evaluator_jobs[0]

    ep = persistences.ExplainerPersistence(
        data_dir=evaluation.result.results_location,
        mli_key=evaluation.key,
        username=commons.DEFAULT_USER,
        explainer_id=evaluator_id,
        explainer_job_key=evaluator_job.key,
    )

    json_leaderboard_idx_path = None
    for llm_ldb_expl in LlmLeaderboardExplanation.__subclasses__():
        json_leaderboard_idx_path = ep.get_explanation_file_path(
            explanation_type=llm_ldb_expl.explanation_type(),
            explanation_format=f5s.LlmLeaderboardJSonFormat.mime,
        )
        if pathlib.Path(json_leaderboard_idx_path).exists():
            break

    with open(json_leaderboard_idx_path) as f:
        json_leaderboard_idx_dict = json.load(f)
    print(f"JSon leaderboard index:\n{json_leaderboard_idx_dict}")

    # get all metrics data file & assert
    assert f5s.ExplanationFormat.KEY_METRICS in json_leaderboard_idx_dict
    assert (
        e10s.AbcHeatmapExplanation.METRIC_ALL
        in json_leaderboard_idx_dict[f5s.ExplanationFormat.KEY_METRICS]
    )

    json_leaderboard_data_file = json_leaderboard_idx_dict[
        f5s.ExplanationFormat.KEY_FILES
    ][e10s.AbcHeatmapExplanation.METRIC_ALL]

    json_leaderboard_data_path = (
        pathlib.Path(json_leaderboard_idx_path).parent / json_leaderboard_data_file
    )
    with open(json_leaderboard_data_path) as f:
        json_leaderboard_data_dict = json.load(f)
    print(f"JSon leaderboard data:\n{json_leaderboard_data_dict}")

    # assert: DATA & METADATA
    assert f5s.ExplanationFormat.KEY_DATA in json_leaderboard_data_dict
    model_names_to_assert = model_names_to_assert or list(
        json_leaderboard_data_dict[f5s.ExplanationFormat.KEY_DATA].keys()
    )
    for model_name in model_names_to_assert:
        assert model_name in json_leaderboard_data_dict[f5s.ExplanationFormat.KEY_DATA]

        metric_ids_to_assert = metric_ids_to_assert or list(
            json_leaderboard_data_dict[f5s.ExplanationFormat.KEY_METADATA].keys()
        )
        for metric_id in metric_ids_to_assert:
            assert (
                metric_id
                in json_leaderboard_data_dict[f5s.ExplanationFormat.KEY_DATA][
                    model_name
                ]
            ), f'Metric "{metric_id}" is not present!'

            metric_value = json_leaderboard_data_dict[f5s.ExplanationFormat.KEY_DATA][
                model_name
            ][metric_id]

            # assert: METADATA threshold in metrics
            if threshold:
                for m in json_leaderboard_data_dict[f5s.ExplanationFormat.KEY_METADATA]:
                    actual_threshold = json_leaderboard_data_dict[
                        f5s.ExplanationFormat.KEY_METADATA
                    ][m][commons.MetricMeta.KEY_THRESHOLD]
                    assert actual_threshold == threshold, (
                        f"Expected {threshold} vs. actual {actual_threshold}"
                    )

            # assert: DATA range withing METADATA range
            metrics_meta = commons.MetricsMeta.from_dict(
                json_leaderboard_data_dict[f5s.ExplanationFormat.KEY_METADATA]
            )
            data_range_min = metrics_meta.get_metric(metric_id).value_range[0]
            data_range_max = metrics_meta.get_metric(metric_id).value_range[1]

            assert isinstance(data_range_min, (int, float)), (
                f"Invalid data range min value {data_range_min} type for metric "
                f"'{metric_id}': {type(data_range_min)}."
            )
            assert isinstance(data_range_max, (int, float)), (
                f"Invalid data range max value '{data_range_max}' type for metric "
                f"'{metric_id}': {type(data_range_max)}."
            )
            assert isinstance(metric_value, (int, float)), (
                f"Invalid metric value {metric_value} type for metric '{metric_id}': "
                f"{type(metric_value)}."
            )
            assert data_range_min <= metric_value <= data_range_max, (
                f"Metric value {metric_value} out of range "
                f"[{data_range_min}, {data_range_max}] for metric '{metric_id}'."
            )

    return json_leaderboard_data_dict
