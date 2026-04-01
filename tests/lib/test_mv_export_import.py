# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import filecmp
import pathlib
import pprint
import uuid

import datatable
import pytest

from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.integrations import mv_adapter
from tests import test_utils


FOO_DICT = {
    "k1": "value1",
    "k2": "value2",
    "k3": "value3",
}
FOO_FRAME_DATA = {
    "c1": ["value1", "value1", "value1"],
    "c2": ["value2", "value1", "value1"],
    "c3": ["value3", "value1", "value1"],
}
FOO_FRAME_FI_DATA = {
    "feature_importance": [1.1, 1.2, 1.3, 1.4, 1.5],
}
FOO_PD_FRAME = datatable.Frame(FOO_FRAME_DATA).to_pandas()


class MockMvObjStorage:
    @staticmethod
    def download(mvid, dst_path: pathlib.Path):
        with open(dst_path, "w") as f:
            f.write(f"FOO DATA of artifact w/ MV ID {mvid}")

    @staticmethod
    def upload(src_path: pathlib.Path, mvid: str):
        print(f"MV object storage is uploading file {src_path} to MV ID {mvid}")
        assert src_path.exists()
        assert mvid


class MockMvDb:
    @property
    def obj_storage(self):
        return MockMvObjStorage()


class MockMvClient:
    @property
    def mvdb(self):
        return MockMvDb()


def _given_adversarial_results():
    if test_utils.are_python_modules_installed({mv_adapter.PACKAGE_NAME}):
        from h2o_mv.platforms import summaries
        from h2o_mv.recipes import adversarial

        return adversarial.AdversarialResults(
            feature_importance=datatable.Frame(FOO_FRAME_FI_DATA).to_pandas(),
            exp_settings=FOO_DICT,  # dict[str, Any] = None
            exp_metrics=FOO_DICT,  # dict[str, Union[str, float]]
            val_score="1.23",
            primary_oof_summary=summaries.TabularPredictionsSummary(),
            secondary_oof_summary=summaries.TabularPredictionsSummary(),
        )

    return None


def _given_drift_results():
    if test_utils.are_python_modules_installed({mv_adapter.PACKAGE_NAME}):
        from h2o_mv.recipes import drift

        return drift.DriftResults(
            psi_scores=FOO_DICT,
            psi_scores_df=FOO_PD_FRAME,
            report=FOO_DICT,
        )

    return None


def _given_adversarial_settings():
    if test_utils.are_python_modules_installed({mv_adapter.PACKAGE_NAME}):
        from h2o_mv.core import mv_dataset
        from h2o_mv.platforms import summaries
        from h2o_mv.recipes import adversarial

        primary_dataset = mv_dataset.MVDataset(mvid="mvid-primary", name="PRIMARY")
        primary_dataset._summary = summaries.DataSummary(
            column_summaries=None,
            sample_table=datatable.Frame(
                {
                    "SAMPLE-1": [1, 2, 3, 4, 5],
                    "SAMPLE-2": [1, 2, 3, 4, 5],
                }
            ).to_pandas(),
            column_histogram_stats={"my-stat": 0.42},
        )

        return adversarial.AdversarialSettings(
            primary_dataset=primary_dataset,
            secondary_dataset=mv_dataset.MVDataset(
                mvid="mvid-secondary", name="SECONDARY"
            ),
            shapley_values=True,
            drop_columns=["drop-col-1", "drop-col-2", "drop-col-3"],
        )
    return None


def _given_drift_settings():
    if test_utils.are_python_modules_installed({mv_adapter.PACKAGE_NAME}):
        from h2o_mv.core import mv_dataset
        from h2o_mv.platforms import summaries
        from h2o_mv.recipes import drift

        primary_dataset = mv_dataset.MVDataset(mvid="mvid-primary", name="PRIMARY")
        primary_dataset._summary = summaries.DataSummary(
            column_summaries=None,
            sample_table=datatable.Frame(
                {
                    "SAMPLE-1": [1, 2, 3, 4, 5],
                    "SAMPLE-2": [1, 2, 3, 4, 5],
                }
            ).to_pandas(),
            column_histogram_stats={"my-stat": 0.42},
        )

        return drift.DriftSettings(
            primary_dataset=primary_dataset,
            secondary_dataset=mv_dataset.MVDataset(
                mvid="mvid-secondary", name="SECONDARY"
            ),
            drop_columns=["drop-col-1"],
            drift_threshold=0.42,
        )
    return None


def _given_adversarial_artifacts():
    if test_utils.are_python_modules_installed({mv_adapter.PACKAGE_NAME}):
        from h2o_mv.core.mv_test import ArtifactInfo

        return {
            "ALPHA": ArtifactInfo(
                name="ALPHA",
                fpath="alpha-file.extension",
                obj_mvid="aaa979c4-50a5-11ee-971b-10828613f8ad",
            ),
            "BETA": ArtifactInfo(
                name="BETA",
                fpath="beta-file.extension",
                obj_mvid="bbb979c4-50a5-11ee-971b-10828613f8ad",
            ),
            "GAMA": ArtifactInfo(
                name="GAMA",
                fpath="gama-file.extension",
                obj_mvid="ccc979c4-50a5-11ee-971b-10828613f8ad",
            ),
        }

    return []


def _given_mv_log():
    if test_utils.are_python_modules_installed({mv_adapter.PACKAGE_NAME}):
        from h2o_mv.core import mv_test

        log = mv_test.MVTestLog()
        log.warning("My warning")
        log.error("My error")
        log.info("My info")
        log.status("My status")
        log.debug("My debug")

        return log
    return None


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({mv_adapter.PACKAGE_NAME}),
    reason="H2O Model Validation Python package is not installed",
)
@pytest.mark.parametrize(
    "mv_test_results,mv_test_settings,mv_test_artifacts,mv_test_log,pathlib_path",
    [
        (
            _given_drift_results(),
            _given_drift_settings(),
            None,  # Drift does not generates artifacts
            _given_mv_log(),
            True,
        ),
        (
            _given_adversarial_results(),
            _given_adversarial_settings(),
            _given_adversarial_artifacts(),
            _given_mv_log(),
            False,
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.h2o_model_validation
def test_export(
    tmp_path,
    mv_test_results,
    mv_test_settings,
    mv_test_artifacts,
    mv_test_log,
    pathlib_path,
):
    #
    # GIVEN
    #
    from h2o_mv.core import mv_test
    from h2o_mv.recipes import drift

    mv_test_artifacts = {
        "alpha": mv_test.ArtifactInfo(
            name="APHA",
            fpath="alpha-file.extension",
            obj_mvid="aaa979c4-50a5-11ee-971b-10828613f8ad",
        ),
        "beta": mv_test.ArtifactInfo(
            name="BETA",
            fpath="beta-file.extension",
            obj_mvid="bbb979c4-50a5-11ee-971b-10828613f8ad",
        ),
        "gama": mv_test.ArtifactInfo(
            name="GAMA",
            fpath="gama-file.extension",
            obj_mvid="ccc979c4-50a5-11ee-971b-10828613f8ad",
        ),
    }

    mv_client_mock = MockMvClient()

    #
    # WHEN
    #
    mv_persistence = mv_adapter.MvResultPersistence(
        target_dir_path=tmp_path if pathlib_path else str(tmp_path),
        mv_client=mv_client_mock,  # MOCK object to persist the data
    )
    result = mv_persistence.export_mv_test(
        mv_test_type=f"{drift.Drift}",
        mv_test_name="MV Test Name @ unit test",
        mv_test_id="abcdefc4-50a5-11ee-971b-10828613f8ad",
        mv_test_results=mv_test_results,
        mv_test_settings=mv_test_settings,
        mv_test_artifacts=mv_test_artifacts,
        mv_test_log=mv_test_log,
        fail_fast=True,
    )

    #
    # THEN
    #
    print("Result:")
    pprint.pprint(result)
    assert result


def get_mvresult_zip_path_for_interpretation(
    interpretation,
    explainer_id: str,
    explainer_job_id: str,
) -> pathlib.Path:
    # get path to the ZIP w/ MVResult archive
    interpretation_path = interpretation.result.interpretation_location
    explainer_dir_name = (
        f"{persistences.ExplainerPersistence.DIR_EXPLAINER}"
        f"{persistences.InterpretationPersistence.to_alphanum_name(explainer_id)}"
        f"_{explainer_job_id}"
    )
    zip_rel_path = (
        f"{explainer_dir_name}"
        f"/global_model_validation_result/application_zip/explanation.zip"
    )
    zip_path = pathlib.Path(interpretation_path) / zip_rel_path
    assert zip_path.exists()
    assert zip_path.is_file()

    return zip_path


def do_test_import_export(
    tmp_path: pathlib.Path,
    zip_or_dir_path: pathlib.Path,
    keep_zip: bool = False,
    raise_exception: bool = False,
) -> bool:
    """Test function which is meant to be called from H2O MV tests to test import
    and export of MV results. The function first loads MV result ZIP archive created
    as explainer's format. Thus, it gets in memory MV result object. Finally, it
    exports the MV result instance to the filesystem and compares both ZIP archive
    contents:

    1. load MV result ZIP archive
    2. import MV result to runtime
    3. export MV result to filesystem
    4. compare original ZIP archive and newly exported ZIP archive

    """
    #
    # GIVEN
    #
    test_dir_name = f"mv-result-import-export-{uuid.uuid1()}"
    tmp_path = tmp_path / test_dir_name

    mv_client_mock = MockMvClient()

    if zip_or_dir_path.is_file():
        # in:
        #   a ZIP archive path
        # out:
        #   <tmp_path>/mv_result_...<UUID>/original-extracted-zip/MVTest (ZIP extract)
        #   <tmp_path>/mv_result_...<UUID>/MVTest (EXPORT of imported MV result
        zip_path = zip_or_dir_path
        export_path = tmp_path / "original-extracted-zip"
        export_path.mkdir(parents=True, exist_ok=True)
    else:
        # in:
        #   a directory w/ MVTest subdirectory
        # out:
        #   <in directory>/MVTest (input directory)
        #   <tmp_path>/mv_result_...<UUID>/MVTest (EXPORT of imported MV result
        zip_path = ""
        export_path = zip_or_dir_path
        tmp_path.mkdir(parents=True, exist_ok=True)

    #
    # WHEN: import
    #
    mv_persistence = mv_adapter.MvResultPersistence(
        target_dir_path=export_path,
        mv_client=mv_client_mock,
    )
    import_result = mv_persistence.import_mv_test(
        zip_path=zip_path, fail_fast=raise_exception
    )

    #
    # THEN: import
    #
    print(f"Import return value: {import_result}")
    pprint.pprint(import_result)
    assert import_result
    assert isinstance(import_result, dict)
    assert import_result[mv_adapter.MvResultPersistence.KEY_DATA][
        mv_adapter.MvResultPersistence.DIR_MVRESULTS
    ]
    assert import_result[mv_adapter.MvResultPersistence.KEY_DATA][
        mv_adapter.MvResultPersistence.DIR_MVSETTINGS
    ]
    assert import_result[mv_adapter.MvResultPersistence.KEY_DATA][
        mv_adapter.MvResultPersistence.DIR_MVLOG
    ]
    if "Adversarial" in import_result["name"]:
        assert import_result[mv_adapter.MvResultPersistence.KEY_DATA][
            mv_adapter.MvResultPersistence.DIR_MVARTIFACTS
        ]

    #
    # WHEN: export (imported MV result)
    #
    mv_persistence = mv_adapter.MvResultPersistence(
        target_dir_path=tmp_path, mv_client=mv_client_mock
    )
    export_result = mv_persistence.export_mv_test(
        mv_test_type=import_result[mv_adapter.MvResultPersistence.KEY_TYPE],
        mv_test_name=import_result[mv_adapter.MvResultPersistence.KEY_NAME],
        mv_test_id=import_result[mv_adapter.MvResultPersistence.KEY_MV_ID],
        mv_test_results=import_result[mv_adapter.MvResultPersistence.KEY_DATA][
            mv_adapter.MvResultPersistence.DIR_MVRESULTS
        ][mv_adapter.MvResultPersistence.KEY_DATA],
        mv_test_settings=import_result[mv_adapter.MvResultPersistence.KEY_DATA][
            mv_adapter.MvResultPersistence.DIR_MVSETTINGS
        ][mv_adapter.MvResultPersistence.KEY_DATA],
        mv_test_artifacts=import_result[mv_adapter.MvResultPersistence.KEY_DATA]
        .get(mv_adapter.MvResultPersistence.DIR_MVARTIFACTS, {})
        .get(mv_adapter.MvResultPersistence.KEY_DATA, {}),
        mv_test_log=import_result[mv_adapter.MvResultPersistence.KEY_DATA][
            mv_adapter.MvResultPersistence.DIR_MVLOG
        ][mv_adapter.MvResultPersistence.KEY_DATA],
        fail_fast=False,
    )

    #
    # THEN: export (compare with original explainer format export)
    #
    pprint.pprint(export_result)
    assert export_result

    # compare directories
    if not keep_zip:
        original_path = export_path / mv_adapter.MvResultPersistence.DIR_MVTEST
        exported_path = export_result[0]
        print("COMPARING DIRECTORIES of original and exported MV result:")
        print(f"  Original: {original_path}")
        print(f"  Exported: {exported_path}")
        cmp_result = filecmp.dircmp(original_path, exported_path)
        print("COMPARING REPORT:")
        cmp_result.report_full_closure()

        print("COMPARING SUMMARY:")
        print(f"Diff files : {cmp_result.diff_files}")
        print(f"Left only  : {cmp_result.left_only}")
        print(f"Right only : {cmp_result.right_only}")
        print(f"Funny files: {cmp_result.right_only}")

        if cmp_result.diff_files:
            if raise_exception:
                assert not cmp_result.diff_files, (
                    f"There are different files in the exported MV result: "
                    f"{cmp_result.diff_files}"
                )
            return False
        return True
    else:
        return True


def _given_import_drift(tmp_path) -> pathlib.Path:
    from tests.lib import test_explainer_drift

    interpretation = test_explainer_drift.test_explainer(
        tmp_path=tmp_path,
        dai_connection=None,
        dataset_path="data/predictive/creditcard.csv",
        another_dataset_path="",
        target_col="default payment next month",
    )

    explainer_job_id = next(iter(interpretation.result.explainers.keys()))
    explainer_id = interpretation.result.explainers[
        explainer_job_id
    ].explainer_descriptor.id
    zip_path = get_mvresult_zip_path_for_interpretation(
        interpretation=interpretation,
        explainer_id=explainer_id,
        explainer_job_id=explainer_job_id,
    )

    return zip_path


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({mv_adapter.PACKAGE_NAME}),
    reason="H2O Model Validation Python package is not installed",
)
@pytest.mark.h2o_sonar
@pytest.mark.h2o_model_validation
def test_zip_import_export_assert(tmp_path):
    zip_path = _given_import_drift(tmp_path)
    do_test_import_export(tmp_path, zip_path, keep_zip=True, raise_exception=True)


_DIR_AD_HOC_IE_BASE_DIR = "/home/user/tmp/mv-import-export-002-20230912"


@pytest.mark.skip(
    reason=(
        "This is ad hoc export test - it requires a prepared set of MV result "
        "ZIP archives"
    )
)
@pytest.mark.parametrize(
    "export_dir_path",
    [
        # adversarial similarity
        f"{_DIR_AD_HOC_IE_BASE_DIR}/as",
        # adversarial similarity ZIP
        f"{_DIR_AD_HOC_IE_BASE_DIR}/as.zip",
        # backtesting
        f"{_DIR_AD_HOC_IE_BASE_DIR}/b",
        # calibration score
        f"{_DIR_AD_HOC_IE_BASE_DIR}/cs",
        # drift
        f"{_DIR_AD_HOC_IE_BASE_DIR}/d",
        # segment performance
        f"{_DIR_AD_HOC_IE_BASE_DIR}/sp",
        # size dependency
        f"{_DIR_AD_HOC_IE_BASE_DIR}/sd",
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.h2o_model_validation
def test_ad_hoc_import_export_assert(tmp_path, export_dir_path):
    do_test_import_export(
        tmp_path=tmp_path,
        zip_or_dir_path=pathlib.Path(export_dir_path),
        raise_exception=True,
    )


@pytest.mark.skip(reason="This is ad hoc Python reflection test.")
def test_python_inspect_methods(inspect_methods: bool = False):
    #
    # GIVEN
    #

    # from h2o_mv.core import mv_model
    # instance = mv_model.MVModel()
    # from h2o_mv import platforms
    # instance = platforms.LocalDataset()
    from h2o_mv.platforms import summaries

    instance = summaries.ColumnSummaries()

    #
    # WHEN
    #

    # methods
    if inspect_methods:
        import inspect

        print("\nMethods:")
        for m in inspect.getmembers(instance):
            if inspect.ismethod(m[1]):
                print(f"--- {m}")
                print(m)
                print(type(m))

    # CHOSEN: attributes @ __dict__
    print("\nAttributes: (__dict__)")
    for a in instance.__dict__.keys():
        print(f"--- {a}")
        print(type(a))

    # attributes @ vars()
    print("\nAttributes: (vars)")
    for a in vars(instance):
        print(f"--- {a}")
        print(type(a))

    #
    # THEN
    #
    print("Done")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
