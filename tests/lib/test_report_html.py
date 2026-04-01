# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import pathlib

import airium
import datatable
import pytest

from h2o_sonar import interpret
from h2o_sonar import loggers
from h2o_sonar.explainers import dia_explainer
from h2o_sonar.explainers import pd_ice_explainer
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import interpretations
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from tests import test_utils
from tests.lib import test_containers


@pytest.mark.h2o_sonar
def test_airium_merges():
    #
    # GIVEN
    #
    merged_aurium = airium.Airium()
    with merged_aurium.p():
        merged_aurium("PRELUDE 1")

    airium1 = airium.Airium()
    airium1.h1(_t="Header 1")
    with airium1.p():
        airium1("Paragraph 1")

    airium2 = airium.Airium()
    airium2.h2(_t="Header 2")
    with airium2.p():
        airium2("Paragraph 1")

    #
    # WHEN
    #
    merged_aurium.append(str(airium1))
    with merged_aurium.p():
        merged_aurium("PRELUDE 2")
    merged_aurium.append(str(airium2))

    #
    # THEN
    #
    print("\n=== Airium 1 ===")
    print(str(airium1))
    print("=== Airium 2 ===")
    print(str(airium2))
    print("=== MERGED ===")
    print(str(merged_aurium))
    print("================")

    assert airium1
    assert airium2
    assert merged_aurium
    assert len(str(merged_aurium)) > len(str(airium1))
    assert len(str(merged_aurium)) > len(str(airium2))


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "result_path_param",
    [
        None,
        "",
        ".",
        "d1/..",
        "d1/d2/../..",
        "d1/../d1/../d1/..",
        "d1/d2/d3",
    ],
)
def test_result_paths(tmp_path, result_path_param):
    #
    # GIVEN
    #
    dataset_path = "./data/predictive/creditcard.csv"
    dataset = datatable.fread(dataset_path)
    target_col = "default payment next month"
    model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )
    original_cwd = os.getcwd()
    try:
        # CHANGE PYTHON WORKING DIRECTORY & return it back after test finishes
        print(f"Changing working directory from '{original_cwd}' to '{tmp_path}'")
        os.chdir(tmp_path)

        # prepare test directory structure
        pathlib.Path(tmp_path / "d1" / "d2" / "d3").mkdir(parents=True, exist_ok=True)

        #
        # WHEN
        #
        print(f"Running interpretation w/ result path: '{result_path_param}'")
        interpretation = interpret.run_interpretation(
            dataset=dataset,
            model=model,
            target_col=target_col,
            results_location=result_path_param,
            log_level=loggers.DEBUG,
            explainers=[
                commons.ExplainerToRun(
                    explainer_id=dia_explainer.DiaExplainer.explainer_id(),
                    params="",
                ),
                commons.ExplainerToRun(
                    explainer_id=pd_ice_explainer.PdIceExplainer.explainer_id(),
                    params="",
                ),
            ],
        )

        #
        # THEN
        #
        assert interpretation
        print(f"Interpretation finished: {interpretation.key=}")
        assert interpretation.key

        # ASSERT validity of all INTERPRETATION paths

        results_dir = tmp_path / (result_path_param or "")
        assert results_dir.is_dir()
        sonar_log_file = results_dir / loggers.SonarFileLogger.FILE_NAME_H2O_SONAR_LOG
        assert sonar_log_file.is_file()
        interpretations_file = (
            results_dir / persistences.InterpretationPersistence.FILE_H2O_SONAR_HTML
        )
        assert interpretations_file.is_file()
        sonar_dir = results_dir / "h2o-sonar"
        assert sonar_dir
        i_dir = sonar_dir / (
            f"{persistences.InterpretationPersistence.DIR_MLI_EXPERIMENT}"
            f"{interpretation.key}"
        )
        assert i_dir
        i_html_file = (
            i_dir / persistences.InterpretationPersistence.FILE_INTERPRETATION_HTML
        )
        assert i_html_file
        i_json_file = (
            i_dir / persistences.InterpretationPersistence.FILE_INTERPRETATION_JSON
        )
        assert i_json_file

        # FILE: h2o-sonar.html
        # ASSERT validity of path from interpretations HTML index to interpret. report
        interpretations_str = interpretation.persistence.store.load(
            key=interpretations_file, data_type=persistences.PersistenceDataType.text
        )
        assert interpretation.key in interpretations_str
        is_to_i_path = (
            f"h2o-sonar"
            f"/{persistences.InterpretationPersistence.DIR_MLI_EXPERIMENT}"
            f"{interpretation.key}"
            f"/{persistences.InterpretationPersistence.FILE_INTERPRETATION_HTML}"
        )
        assert is_to_i_path in interpretations_str
        # ASSERT validity of path from interpretations HTML index to H2O Sonar log
        assert (
            f"./{loggers.SonarFileLogger.FILE_NAME_H2O_SONAR_LOG}"
            in interpretations_str
        )

        # FILE: interpretation.html
        i_html_str = interpretation.persistence.store.load(
            key=i_html_file, data_type=persistences.PersistenceDataType.text
        )
        # ASSERT validation of path images in interpretation HTML report
        explainer_ids = [
            pd_ice_explainer.PdIceExplainer.explainer_id(),
            dia_explainer.DiaExplainer.explainer_id(),
        ]
        for e_id in explainer_ids:
            # DIRS & CONFIG SECTION
            e_job = interpretation.get_jobs_for_explainer_id(e_id)[0]
            e_dir = (
                f"{persistences.ExplainerPersistence.DIR_EXPLAINER}"
                f"{persistences.InterpretationPersistence.to_alphanum_name(e_id)}"
                f"_"
                f"{e_job.key}"
            )
            # ASSERT HTML report (self)
            assert (
                f"href="
                f'"./{persistences.InterpretationPersistence.FILE_INTERPRETATION_HTML}"'
                f">Interpretation summary in HTML format" in i_html_str
            )
            # ASSERT JSon report (relative)
            assert (
                f"href="
                f'"./{persistences.InterpretationPersistence.FILE_INTERPRETATION_JSON}"'
                f">Interpretation summary in JSon format" in i_html_str
            )
            # ASSERT interpretation dir (relative)
            assert 'href="./">This model interpretation directory' in i_html_str
            # ASSERT H2O Sonar results directory (relative)
            assert 'href="../..">H2O Sonar library results directory' in i_html_str
            # ASSERT H2O Sonar log
            assert 'href="../../h2o-sonar.log">H2O Sonar library log' in i_html_str

            # EXPLAINER SECTION
            # explainer log path (relative)
            assert (
                f'href="'
                f"{e_dir}"
                f"/{persistences.ExplainerPersistence.DIR_LOG}"
                f"/{persistences.ExplainerPersistence.EXPLAINER_LOG_PREFIX}"
                f'{e_job.key}.log"'
                f">explainer.log"
            ) in i_html_str

            if e_id == pd_ice_explainer.PdIceExplainer.explainer_id():
                rel_paths = [
                    # ASSERT image paths @ explainer section (relative)
                    (
                        f"{e_dir}"
                        f"/global_html_fragment"
                        f"/text_html"
                        f"/pd-feature-0-class-0.png"
                    ),
                    # ASSERT JSon/dt explanations paths @ explainer section (relative)
                    (
                        f"{e_dir}"
                        f"/global_html_fragment"
                        f"/text_html"
                        f"/pd-feature-5-class-0.png"
                    ),
                    f"{e_dir}/global_partial_dependence/application_json",
                    (
                        f"{e_dir}"
                        f"/local_individual_conditional_explanation"
                        f"/application_vnd_h2oai_json_datatable_jay"
                    ),
                ]
                for rel_path in rel_paths:
                    # ASSERT path @ HTML
                    assert f'="{rel_path}"' in i_html_str
                    # ASSER file / dir existence
                    if rel_path.endswith(".png"):
                        assert os.path.isfile(os.path.join(str(i_dir), rel_path))
                    else:
                        assert os.path.isdir(os.path.join(str(i_dir), rel_path))
            elif e_id == dia_explainer.DiaExplainer.explainer_id():
                # ASSERT image paths @ explainer section (relative)
                rel_paths = [
                    f"{e_dir}/global_html_fragment/text_html/dia-0-n.png",
                    (
                        f"{e_dir}"
                        f"/global_html_fragment"
                        f"/text_html"
                        f"/dia-0-adverse_impact.png"
                    ),
                    # ASSERT JSon/dt explanations paths @ explainer section (relative)
                    f"{e_dir}/global_disparate_impact_analysis/text_plain",
                ]
                for rel_path in rel_paths:
                    # ASSERT path @ HTML
                    assert f'="{rel_path}"' in i_html_str
                    # ASSER file / dir existence
                    if rel_path.endswith(".png"):
                        assert os.path.isfile(os.path.join(str(i_dir), rel_path))
                    else:
                        assert os.path.isdir(os.path.join(str(i_dir), rel_path))
            else:
                raise RuntimeError(f"Invalid explainer ID: {e_id}")

    finally:
        if original_cwd:
            os.chdir(original_cwd)


@pytest.mark.h2o_sonar
def test_parse_1st_img():
    #
    # GIVEN
    #
    expected_img_path = "images/dt-class-0.png"
    html_representation = (
        "<p>Approximate model behavior for the class '1':</p>"
        "<div>"
        f'  <img width="1000" src="{expected_img_path}" '
        "       alt=\"Decision tree for class '1'\">"
        "</div>"
    )

    #
    # WHEN
    #
    img_path = interpretations.HtmlInterpretationFormat._html_parse_1st_img_path(
        html_representation
    )

    #
    # THEN
    #
    print(f"Image path: '{img_path}'")
    assert expected_img_path == img_path


@pytest.mark.skip("Explainable dataset introspection")
@pytest.mark.h2o_sonar
def test_explainable_dataset():
    #
    # GIVEN
    #
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    explainable_dataset = datasets.DatasetApi().create_dataset(
        dataset_src=dataset_path,
        dataset_type=datasets.ExplainableDatasetType.filesystem,
        target_col="default payment next month",
    )

    #
    # WHEN
    #
    print(f"Explainable dataset:\n{explainable_dataset}")


@pytest.mark.skip("Explainable model @ MOJO introspection")
@pytest.mark.h2o_sonar
def test_dai_mojo_explainable_model():
    import daimojo

    #
    # GIVEN
    #
    mojo_path = test_utils.find_locally(
        "data/predictive/models/creditcard-binomial.mojo"
    )
    model = daimojo.model(mojo_path)
    explainable_model = models.ModelApi().create_model(
        model_src=model,
        target_col="default payment next month",
        used_features=list(model.feature_names),
    )

    #
    # WHEN
    #
    print(f"Explainable model:\n{explainable_model}")


@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
@pytest.mark.h2o_sonar
def test_no_explainers_paths(tmp_path):
    """Assert that in case that no explainers were run, the interpretation path
    points to the index of all interpretations (which indicates that 0 explainers
    were run).

    """
    #
    # GIVEN
    #
    model_path = test_utils.find_locally(
        "data/predictive/models/creditcard-multinomial-raw.mojo"
    )
    dataset_path = test_utils.find_locally(
        "data/predictive/creditcard_train_BAD_COL_NAMES_BAD_EDU_VALUES.csv"
    )
    target_col = "E.D.U.C.A.T.I.O.N"

    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=model_path,
        target_col=target_col,
        results_location=str(tmp_path),
        log_level=loggers.DEBUG,
        explainers=[dia_explainer.DiaExplainer.explainer_id()],
    )

    #
    # THEN
    #
    assert interpretation
    print(f"Interpretation finished: {interpretation.get_finished_explainer_ids()}")
    assert interpretation.get_finished_explainer_ids() == []
    assert interpretation.get_failed_explainer_ids() == []
    print(f"Results path: {interpretation.result.get_results_dir_location()}")
    print(f"Results property: {interpretation.result.results_location}")
    print(
        f"Interpretation path: "
        f"{interpretation.result.get_interpretation_dir_location()}"
    )
    print(f"Interpretation property: {interpretation.result.interpretation_location}")
    print(f"HTML report path: {interpretation.result.get_html_report_location()}")
    print(f"HTML property: {interpretation.result.html_location}")
    print(f"JSon report path: {interpretation.result.get_json_report_location()}")
    print(f"JSon property: {interpretation.result.json_location}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
