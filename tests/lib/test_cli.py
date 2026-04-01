# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import logging
import os
import pathlib
import pprint
import subprocess

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import interpret
from h2o_sonar.explainers import dia_explainer
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers
from h2o_sonar.utils import io
from tests import conftest
from tests import test_utils
from tests.lib import test_containers


try:
    import h2o  # noqa: F401

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


TEST_CONFIG_FILE = "test_config.json"


def given_base_cli_cmd() -> tuple[list[str], dict[str, str]]:
    cli_cmd = (
        ["h2o-sonar"]
        if os.system("which h2o-sonar") == 0
        else ["python", "h2o_sonar/h2o_sonar_cli.py"]
    )
    child_env = os.environ.copy()
    # add the root of the repo to Python path so that the CLI can load model class
    child_env["PYTHONPATH"] = "."

    return cli_cmd, child_env


@pytest.mark.cli
@pytest.mark.h2o_sonar
def test_cli_help():
    #
    # GIVEN
    #
    (cli_cmd, child_env) = given_base_cli_cmd()

    #
    # WHEN
    #
    p = subprocess.Popen(cli_cmd + ["--help"], env=child_env)

    #
    # THEN
    #
    assert p.wait() == 0


@pytest.mark.cli
@pytest.mark.h2o_sonar
def test_cli_describe_explainer():
    #
    # GIVEN
    #
    (cli_cmd, child_env) = given_base_cli_cmd()

    #
    # WHEN
    #
    p = subprocess.Popen(
        cli_cmd
        + [
            "describe",
            "explainer",
            f"--explainer={dia_explainer.DiaExplainer.explainer_id()}",
        ],
        env=child_env,
    )

    #
    # THEN
    #
    assert p.wait() == 0


@pytest.mark.parametrize(
    "sonar_command",
    [
        # test: missing dataset
        [
            "run",
            "interpretation",
            "--model",
            "model.mojo",
            "--target-col",
            "AGE",
            "--results-location",
            "./results",
        ],
        # test: missing model
        [
            "run",
            "interpretation",
            "--dataset",
            "dataset.csv",
            "--target-col",
            "AGE",
            "--results-location",
            "./results",
        ],
        # NO test: model is not REQUIRED (might be missing)
        # test: missing target
        [
            "run",
            "interpretation",
            "--dataset",
            "dataset.csv",
            "--model",
            "model.mojo",
            "--results-location",
            "./results",
        ],
        # test: missing results
        [
            "run",
            "interpretation",
            "--dataset",
            "dataset.csv",
            "--model",
            "model.mojo",
            "--target-col",
            "AGE",
        ],
    ],
)
@pytest.mark.cli
@pytest.mark.h2o_sonar
def test_cli_interpret_negative(sonar_command):
    #
    # GIVEN
    #
    (cli_cmd, child_env) = given_base_cli_cmd()

    #
    # WHEN
    #
    p = subprocess.Popen(cli_cmd + sonar_command, stdin=subprocess.PIPE)

    #
    # THEN
    #
    assert p.wait() == 1


def given_h2o_sonar_base_config(
    tmpdir, h2o_auto_start=True, custom_explainers: list | None = None
):
    # create and save H2O Sonar configuration
    h2o_sonar_config_dict = {
        h2o_sonar_config.H2o3Config.KEY_PORT: (
            conftest.get_h2o3_config().get(h2o_sonar_config.H2o3Config.KEY_PORT, "")
            if conftest.get_h2o3_config()
            else h2o_sonar_config.config.h2o_port
        ),
        h2o_sonar_config.H2o3Config.KEY_AUTO_START: h2o_auto_start,
        h2o_sonar_config.H2oSonarConfig.KEY_CUSTOM_EXPLAINERS: custom_explainers,
    }
    h2o_sonar_config_path = os.path.join(tmpdir, TEST_CONFIG_FILE)
    # save
    h2o_sonar_config.config.save(
        config_path=h2o_sonar_config_path,
        config_data=h2o_sonar_config_dict,
    )

    return h2o_sonar_config_path


def given_cli(tmpdir, h2o_auto_start=True, custom_explainers: list | None = None):
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "default payment next month"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path,
        target_col=target_col,
    )
    # pickle model
    model_pickle_path = os.path.join(tmpdir, "simple_mock_model.pickle")
    mock_model.save(model_pickle_path, update=True)
    print(f"Pickled mock model: {mock_model}")
    # check unpickled mock model
    unpickled_model = test_containers.SimpleMockModel.load(model_pickle_path)
    print(f"CHECK: un-pickled mock model: {unpickled_model}")
    assert unpickled_model
    assert unpickled_model.meta
    assert unpickled_model.meta.used_features

    h2o_sonar_config_path = given_h2o_sonar_base_config(
        tmpdir=tmpdir,
        h2o_auto_start=h2o_auto_start,
        custom_explainers=custom_explainers,
    )

    return dataset_path, target_col, model_pickle_path, h2o_sonar_config_path


@pytest.mark.parametrize(
    "run_all_explainers,model_path,h2o_auto_start",
    [
        # mock model
        (False, "", False),
        (True, "", True),
        # MOJO model @ selected explainer
        pytest.param(
            False,
            "data/predictive/models/creditcard-binomial.mojo",
            False,
            marks=pytest.mark.skipif(
                not test_utils.is_mojo_supported(), reason="MOJO is not supported"
            ),
        ),
        # MOJO model @ all explainers
        pytest.param(
            True,
            "data/predictive/models/creditcard-binomial.mojo",
            False,
            marks=pytest.mark.skipif(
                not test_utils.is_mojo_supported(), reason="MOJO is not supported"
            ),
        ),
    ],
)
@pytest.mark.cli
@pytest.mark.h2o_sonar
def test_cli(tmpdir, run_all_explainers, model_path, h2o_auto_start):
    #
    # GIVEN
    #
    (
        dataset_path,
        target_col,
        mock_model_path,
        config_path,
    ) = given_cli(tmpdir=tmpdir, h2o_auto_start=h2o_auto_start)

    model_path = model_path or mock_model_path

    (cli_cmd, child_env) = given_base_cli_cmd()

    #
    # WHEN
    #
    cmd = cli_cmd + [
        "run",
        "interpretation",
        "--dataset",
        dataset_path,
        "--target-col",
        target_col,
        "--model",
        model_path,
        "--results-location",
        tmpdir,
        "--config-path",
        config_path,
    ]
    if not run_all_explainers:
        cmd.extend(["--explainers", dia_explainer.DiaExplainer.explainer_id()])

    print(f"\nRunning interpretation via CLI:\n{cmd}\n")
    p = subprocess.Popen(cmd, env=child_env)
    p.wait()

    #
    # THEN
    #
    p_tree = str(os.popen(f"find {tmpdir}").read())
    print(p_tree)
    # DIA
    assert "global_disparate_impact_analysis" in p_tree
    assert "dia_entity.json" in p_tree
    assert "disparity.jay" in p_tree
    if run_all_explainers:
        # PD
        assert "global_partial_dependence" in p_tree
        assert "pd_feature_0_class_0.json" in p_tree
        assert "ice_feature_0_class_0.jay" in p_tree
        # Shapley
        assert "global_summary_feature_importance" in p_tree
        assert "feature_0_class_0.png" in p_tree
        assert "summary_feature_importance_class_0_offset_0.json" in p_tree
        assert "shapley-class-0.png" in p_tree
        # DT (requires H2O3 - only check if available)
        if HAS_H2O:
            assert "dtModel.json" in p_tree
            assert "dtPathsFrame.csv" in p_tree
    else:
        # PD
        assert "global_partial_dependence" not in p_tree
        # Shapley
        assert "global_summary_feature_importance" not in p_tree
        # DT
        assert "dtModel.json" not in p_tree


@pytest.mark.skip(reason="New sklearn pickles are not backward compatible")
@pytest.mark.parametrize(
    "model_path,h2o_auto_start,args_as_json",
    [
        # pickled sklearn model (<= Python 3.9 or >= Python 3.11 is resolved in runtime)
        (
            "data/predictive/models/creditcard-binomial-sklearn-gbm.pkl",
            True,
            True,
        ),
    ],
)
@pytest.mark.cli
@pytest.mark.h2o_sonar
def test_cli_sklearn(tmpdir, model_path, h2o_auto_start, args_as_json):
    """Test scikit-learn model interpretation via CLI + passing of interpretation
    arguments as JSon file.

    """
    #
    # GIVEN
    #
    (
        dataset_path,
        target_col,
        mock_model_path,
        config_path,
    ) = given_cli(tmpdir=tmpdir, h2o_auto_start=h2o_auto_start)

    if model_path:
        model_path = test_utils.get_version_specific_scikit_model(
            test_utils.find_locally(model_path)
        )
    else:
        model_path = mock_model_path

    (cli_cmd, child_env) = given_base_cli_cmd()

    #
    # WHEN run interpretation
    #
    if args_as_json:
        # args
        json_args_str = io.to_run_interpretation_args_json(
            dataset=dataset_path,
            target_col=target_col,
            model=model_path,
            results_location=str(tmpdir),
        )
        args_json_path = pathlib.Path(tmpdir, "run_interpretation_args.json")
        with open(args_json_path, "w") as file_handle:
            file_handle.write(json_args_str)
        # command
        cmd = cli_cmd + [
            "run",
            "interpretation",
            "--args-as-json-location",
            str(args_json_path),
            "--config-path",
            config_path,
        ]
    else:
        cmd = cli_cmd + [
            "run",
            "interpretation",
            "--dataset",
            dataset_path,
            "--target-col",
            target_col,
            "--model",
            model_path,
            "--results-location",
            tmpdir,
            "--config-path",
            config_path,
        ]

    print(f"\nRunning interpretation via CLI:\n{cmd}\n")
    p = subprocess.Popen(cmd, env=child_env)
    p.wait()

    # THEN run interpretation
    p_tree = str(os.popen(f"find {tmpdir}").read())
    print(p_tree)
    assert "global_summary_feature_importance" in p_tree
    assert "feature_0_class_0.png" in p_tree
    assert "summary_feature_importance_class_0_offset_0.json" in p_tree
    assert "shapley-class-0.png" in p_tree

    #
    # WHEN list interpretations
    #
    cmd = cli_cmd + [
        "list",
        "interpretations",
        "--results-location",
        tmpdir,
    ]

    print(f"\nRunning interpretations listing via CLI:\n{cmd}\n")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=child_env)
    output = process.communicate()[0]

    # THEN
    print(f"\nInterpretations list:\n{output}")
    assert 0 == process.wait()
    assert output
    output_str = output.decode("utf-8")
    assert output_str.startswith("['")
    assert "']" in output_str


@pytest.mark.h2o_sonar
def test_api(tmpdir):
    #
    # GIVEN
    #
    (
        dataset_path,
        target_col,
        model_path,
        _,
    ) = given_cli(tmpdir)

    #
    # WHEN
    #
    print("Running interpretation via CLI API...")
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=model_path,
        target_col=target_col,
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    #
    # THEN
    #
    print(interpretation)
    print(f"FINISHED explainers: {interpretation.get_finished_explainer_ids()}")
    print(f"FAILED explainers: {interpretation.get_failed_explainer_ids()}")
    assert not interpretation.get_failed_explainer_ids()


@pytest.mark.cli
@pytest.mark.h2o_sonar
@pytest.mark.parametrize("detailed_listing", [False, True])
def test_cli_list_explainers(detailed_listing):
    #
    # GIVEN
    #
    (cli_cmd, child_env) = given_base_cli_cmd()

    #
    # WHEN
    #
    cmd = cli_cmd + [
        "list",
        "explainers",
    ]
    if detailed_listing:
        cmd.append("--detailed")
    print(f"\nRunning CLI:\n{cmd}\n")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=child_env)
    output = process.communicate()[0]

    #
    # THEN
    #
    print(f"\nExplainers:\n{output}")
    assert 0 == process.wait()
    assert output


@pytest.mark.parametrize(
    "json_args,expected_explainers",
    [
        (
            io.to_list_explainers_args_json(),
            "all",
        ),
        (
            io.to_list_explainers_args_json(
                experiment_types=[commons.ExperimentType.regression.name]
            ),
            "regression",
        ),
        (
            io.to_list_explainers_args_json(
                explanation_scopes=[commons.ExplanationScope.local_scope.name],
            ),
            "local",
        ),
        (
            io.to_list_explainers_args_json(
                keywords=[explainers.Explainer.KEYWORD_EXPLAINS_FAIRNESS]
            ),
            "fair",
        ),
    ],
    ids=["all", "regression", "local", "fair"],
)
@pytest.mark.cli
@pytest.mark.h2o_sonar
def test_cli_list_explainers_json_args(tmp_path, json_args, expected_explainers):
    #
    # GIVEN
    #
    print(f"JSon args:\n{json_args}")
    args_json_path = pathlib.Path(tmp_path, "list_explainers_args.json")
    with open(args_json_path, "w") as file_handle:
        file_handle.write(json_args)
    # CLI cmd
    (cli_cmd, child_env) = given_base_cli_cmd()

    #
    # WHEN
    #
    cmd = cli_cmd + [
        "list",
        "explainers",
        "--args-as-json-location",
        str(args_json_path),
    ]
    if expected_explainers == "fair":
        cmd.append("--detailed")
    print(f"\nRunning {expected_explainers} CLI:\n{cmd}\n")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=child_env)
    output = process.communicate()[0]

    #
    # THEN
    #
    print(f"Explainers:\n{output}")
    assert 0 == process.wait()
    assert output
    output_str = output.decode("utf-8")
    print(output_str)
    # assert JSon parsing
    output_dict = json.loads(output_str)
    assert output_dict
    print(f"{expected_explainers} count: {len(output_dict['explainers'])}")
    expected_count = {
        "all": 13,
        "regression": 13,  # filtering not implemented yet
        "local": 13,  # filtering not implemented yet
        "fair": 1,
    }
    assert len(output_dict["explainers"]) >= expected_count.get(
        expected_explainers, 1000
    )


@pytest.mark.cli
@pytest.mark.h2o_sonar
def test_cli_used_features_explainer_pars(tmpdir):
    #
    # GIVEN
    #
    target_col = "default payment next month"
    (
        dataset_path,
        _,
        mock_model_path,
        config_path,
    ) = given_cli(tmpdir=tmpdir, h2o_auto_start=False)

    (cli_cmd, child_env) = given_base_cli_cmd()

    #
    # WHEN
    #
    cmd = cli_cmd + [
        "run",
        "interpretation",
        "--dataset",
        dataset_path,
        "--target-col",
        target_col,
        "--model",
        mock_model_path,
        "--used-features",
        "PAY_1,EDUCATION",
        "--results-location",
        tmpdir,
        "--config-path",
        config_path,
        "--explainers",
        dia_explainer.DiaExplainer.explainer_id(),
        "--explainers-pars",
        f"{{'{dia_explainer.DiaExplainer.explainer_id()}':{{'cut_off': 0.5}}}}",
    ]

    print(f"\nRunning interpretation via CLI:\n{cmd}\n")
    p = subprocess.Popen(cmd, env=child_env)
    p.wait()

    #
    # THEN
    #
    p_tree = str(os.popen(f"find {tmpdir}").read())
    i_params = {}
    for line in p_tree.splitlines():
        if line and line.endswith("/interpretation.json"):
            print(f"Path to interpretation parameters:\n{line}")
            with open(line) as file:
                i_params = json.load(file)
            print(
                f"Used features parameters:\n"
                f"{i_params['interpretation_parameters']['used_features']}"
            )
            assert "['PAY_1', 'EDUCATION']" == str(
                i_params["interpretation_parameters"]["used_features"]
            )
    if not i_params:
        raise AssertionError(
            "Unable to find interpretation JSon file!"
            f"\nTest directory:\n{tmpdir}"
            f"\nProcess output:\n{p_tree}"
        )


@pytest.mark.cli
@pytest.mark.h2o_sonar
def test_bug_128(tmpdir):
    """Test DIA w/ wrong target column."""

    #
    # GIVEN
    #
    target_col = "INVALID"
    model_path = "data/predictive/models/creditcard-binomial.mojo"
    h2o_auto_start = False
    (
        dataset_path,
        _,
        mock_model_path,
        config_path,
    ) = given_cli(tmpdir=tmpdir, h2o_auto_start=h2o_auto_start)

    model_path = model_path or mock_model_path

    (cli_cmd, child_env) = given_base_cli_cmd()

    #
    # WHEN
    #
    cmd = cli_cmd + [
        "run",
        "interpretation",
        "--dataset",
        dataset_path,
        "--target-col",
        target_col,
        "--model",
        model_path,
        "--results-location",
        tmpdir,
        "--config-path",
        config_path,
    ]
    cmd.extend(["--explainers", dia_explainer.DiaExplainer.explainer_id()])

    print(f"\nRunning interpretation via CLI:\n{cmd}\n")
    p = subprocess.Popen(cmd, env=child_env)
    p.wait()

    #
    # THEN
    #
    p_tree = str(os.popen(f"find {tmpdir}").read())
    print(p_tree)
    # assert files which must present in the explainer sandbox
    assert "global_disparate_impact_analysis" not in p_tree
    assert "dia_entity.json" not in p_tree
    assert "disparity.jay" not in p_tree


#
# CLI/interpret.py args as JSon file
#


@pytest.mark.cli
@pytest.mark.cli_json_args
@pytest.mark.h2o_sonar
def test_list_explainers_args_default():
    #
    # GIVEN
    #
    expected = (
        '{"experiment_types": [], "explanation_scopes": [], "model_meta": '
        'null, "keywords": [], "explainer_filter": [], "extra_params": {}}'
    )

    #
    # WHEN
    #
    json_str = io.to_list_explainers_args_json()

    #
    # THEN
    #
    print("RESULT:")
    print(json_str)
    print("EXPECTED:")
    print(expected)
    assert json_str
    assert expected == json_str


@pytest.mark.cli
@pytest.mark.cli_json_args
@pytest.mark.h2o_sonar
def test_list_explainers_args_all():
    #
    # GIVEN
    #
    expected = (
        '{"experiment_types": ["regression", "binomial", "multinomial"], '
        '"explanation_scopes": ["global_scope", "local_scope"], "model_meta": null, '
        '"keywords": ["KEYWORD-1", "KEYWORD-2", "KEYWORD-3"], '
        '"explainer_filter": ['
        '{"filter_by": "FILTER-BY-1", "value": "VALUE-1"}, '
        '{"filter_by": "FILTER-BY-2", "value": "VALUE-2"}, '
        '{"filter_by": "FILTER-BY-3", "value": "VALUE-3"}], '
        '"extra_params": {"CUSTOM-ARG-1": "custom-value-1"}}'
    )

    #
    # WHEN serialize
    #
    json_str = io.to_list_explainers_args_json(
        experiment_types=[
            commons.ExperimentType.regression.name,
            commons.ExperimentType.binomial.name,
            commons.ExperimentType.multinomial.name,
        ],
        explanation_scopes=[
            commons.ExplanationScope.global_scope.name,
            commons.ExplanationScope.local_scope.name,
        ],
        model_meta=None,
        keywords=[
            "KEYWORD-1",
            "KEYWORD-2",
            "KEYWORD-3",
        ],
        explainer_filter=[
            commons.FilterEntry(filter_by="FILTER-BY-1", value="VALUE-1"),
            commons.FilterEntry(filter_by="FILTER-BY-2", value="VALUE-2"),
            commons.FilterEntry(filter_by="FILTER-BY-3", value="VALUE-3"),
        ],
        extra_params={"CUSTOM-ARG-1": "custom-value-1"},
    )

    #
    # THEN serialize
    #
    print("RESULT:")
    print(json_str)
    print("EXPECTED:")
    print(expected)
    assert json_str
    assert expected == json_str

    #
    # WHEN deserialize
    #
    from_dict = io.from_list_explainers_args_json(json_str)

    #
    # THEN deserialize
    #
    pprint.pprint(from_dict)
    assert from_dict
    assert isinstance(from_dict["explainer_filter"][0], commons.FilterEntry)
    assert from_dict["explainer_filter"][0].filter_by == "FILTER-BY-1"
    assert from_dict["explainer_filter"][0].value == "VALUE-1"


@pytest.mark.cli
@pytest.mark.cli_json_args
@pytest.mark.h2o_sonar
def test_run_interpretation_args_default():
    #
    # GIVEN
    #
    expected = (
        '{"dataset": "", "model": "", "target_col": "", "explainers": [], '
        '"explainer_keywords": [], "validset": "", "testset": "", '
        '"use_raw_features": true, "used_features": null, "weight_col": "", '
        '"prediction_col": "", "drop_cols": [], "sample_num_rows": null, '
        '"log_level": 30, "results_location": null, "persistence_type": "file_system", '
        '"run_asynchronously": false, "run_explainers_in_parallel": false, '
        '"extra_params": {}}'
    )

    #
    # WHEN
    #
    json_str = io.to_run_interpretation_args_json()

    #
    # THEN
    #
    print("RESULT:")
    print(json_str)
    print("EXPECTED:")
    print(expected)
    assert json_str
    assert expected == json_str


@pytest.mark.cli
@pytest.mark.cli_json_args
@pytest.mark.h2o_sonar
def test_run_interpretation_args_all():
    #
    # GIVEN
    #
    expected = (
        '{"dataset": "DATASET.csv", "model": "MODEL.csv", "target_col": "TARGET_COL", '
        '"explainers": ["EXPLAINER-ID-1", {"id": "EXPLAINER-ID-2", "params": "{'
        '\'explainer-id-2-param\': 1}", "extra_params": null}, "EXPLAINER-ID-3"], '
        '"explainer_keywords": ["KEYWORD-1", "KEYWORD-2", "KEYWORD-3"], '
        '"validset": "VALIDSET.csv", "testset": "TESTSET.csv", '
        '"use_raw_features": false, '
        '"used_features": ["USED_FEATURE_1", "USED_FEATURE_2", "USED_FEATURE_3"], '
        '"weight_col": "WEIGHT_COL_1", "prediction_col": "PREDICTION_COL_1", '
        '"drop_cols": ["DROP_COL_1", "DROP_COL_2"], "sample_num_rows": 12345, '
        '"log_level": 35, "results_location": null, '
        '"persistence_type": "file_system", "run_asynchronously": false, '
        '"run_explainers_in_parallel": false, '
        '"extra_params": {"EXTRA-PARAM-1": 1}}'
    )

    #
    # WHEN serialize
    #
    json_str = io.to_run_interpretation_args_json(
        dataset="DATASET.csv",
        model="MODEL.csv",
        target_col="TARGET_COL",
        explainers=[
            "EXPLAINER-ID-1",
            commons.ExplainerToRun(
                explainer_id="EXPLAINER-ID-2",
                params="{'explainer-id-2-param': 1}",
            ),
            "EXPLAINER-ID-3",
        ],
        explainer_keywords=[
            "KEYWORD-1",
            "KEYWORD-2",
            "KEYWORD-3",
        ],
        validset="VALIDSET.csv",
        testset="TESTSET.csv",
        use_raw_features=False,
        used_features=[
            "USED_FEATURE_1",
            "USED_FEATURE_2",
            "USED_FEATURE_3",
        ],
        weight_col="WEIGHT_COL_1",
        prediction_col="PREDICTION_COL_1",
        drop_cols=["DROP_COL_1", "DROP_COL_2"],
        sample_num_rows=12345,
        log_level=35,
        extra_params={"EXTRA-PARAM-1": 1},
    )

    #
    # THEN serialize
    #
    print("RESULT:")
    print(json_str)
    print("EXPECTED:")
    print(expected)
    assert json_str
    assert expected == json_str

    #
    # WHEN deserialize
    #
    from_dict = io.from_run_interpretation_args_json(args_str=json_str)

    #
    # THEN deserialize
    #
    pprint.pprint(from_dict)
    assert from_dict
    assert isinstance(from_dict["explainers"][1], commons.ExplainerToRun)
    assert from_dict["explainers"][1].id == "EXPLAINER-ID-2"
    assert from_dict["explainers"][1].params == "{'explainer-id-2-param': 1}"
