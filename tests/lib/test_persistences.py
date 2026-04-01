# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import os.path
import pathlib

import pytest

from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import persistences


@pytest.mark.h2o_sonar
def test_in_memory_persistence():
    #
    # GIVEN
    #
    path_root = pathlib.Path("/")
    persistence_api = persistences.PersistenceApi()

    path_log = path_root / "sonar" / "sonar.log"
    data_log = "now: H2O Sonar log"

    path_1_json = path_root / "sonar" / "interpretation" / "explainers" / "file_1.json"
    data_1_json = "{ 'data_json': 1 }"
    path_2_json = path_root / "sonar" / "interpretation" / "explainers" / "file_2.json"
    data_2_json = "{ 'data_json': 2 }"
    path_3_json = path_root / "sonar" / "interpretation" / "explainers" / "file_3.json"
    data_3_json = "{ 'data_json': 3 }"

    path_interpretation = path_root / "sonar" / "interpretation.json"
    data_interpretation = "{ 'interpretation': 'result' }"

    paths = [
        path_log,
        path_1_json,
        path_2_json,
        path_3_json,
        path_interpretation,
    ]
    datas = [
        data_log,
        data_1_json,
        data_2_json,
        data_3_json,
        data_interpretation,
    ]

    #
    # WHEN
    #
    persistence: persistences.InMemoryPersistence = persistence_api.create_persistence(
        persistence_type=persistences.PersistenceType.in_memory
    )
    for i in range(0, len(paths)):
        persistence.save(
            key=paths[i],
            data_type=persistences.PersistenceDataType.json,
            data=datas[i],
        )

    #
    # THEN
    #
    print(persistence)
    assert persistence
    assert len(persistence.memory_store) == len(paths)
    for i in range(0, len(paths)):
        assert persistence.exists(paths[i])
    assert data_interpretation == persistence.load(key=path_interpretation)
    assert data_1_json == persistence.load(key=path_1_json)
    # work/ directory and logs are always stored on the filesystem
    print(f"Internal store: '{persistence.internal.base_path}'")
    assert (
        persistences.Persistence.PREFIX_INTERNAL_STORE in persistence.internal.base_path
    )
    # dir listing
    files = persistence.list_dir(
        key=path_root / "sonar" / "interpretation" / "explainers"
    )
    print(f"Files: {files}")
    assert ["file_1.json", "file_2.json", "file_3.json"] == files
    # wildcard filtered listing
    wild_files = persistence.list_files_by_wildcard(
        key=path_root / "sonar" / "interpretation" / "explainers",
        wildcard="*3.json",
    )
    print(f"Wildcard filtered files: {wild_files}")
    assert ["/sonar/interpretation/explainers/file_3.json"] == wild_files


@pytest.mark.h2o_sonar
def test_in_memory_interpretation_persistence():
    #
    # GIVEN
    #
    data_dir = "/tmp/data-dir"
    in_memory_persistence = persistences.InMemoryPersistence()

    #
    # WHEN
    #
    interpretation_persistence = persistences.InterpretationPersistence(
        data_dir=data_dir,
        username="john",
        mli_key="12345678-1234-1234-1234-123456789012",
        store_persistence=in_memory_persistence,
    )
    interpretation_persistence.make_interpretation_sandbox()

    #
    # THEN
    #
    print(
        f"In-memory store: internal_base_path="
        f"{in_memory_persistence.internal.base_path}"
    )
    assert (
        persistences.Persistence.PREFIX_INTERNAL_STORE
        in in_memory_persistence.internal.base_path
    )


@pytest.mark.h2o_sonar
def test_in_memory_explainer_persistence():
    #
    # GIVEN
    #
    data_dir = "/tmp/data-dir"
    in_memory_persistence = persistences.InMemoryPersistence()

    #
    # WHEN
    #
    explainer_persistence = persistences.ExplainerPersistence(
        data_dir=data_dir,
        username="john",
        explainer_id="h2o_sonar.explainers.FooExplainerId",
        explainer_job_key="12345678-eeee-1234-1234-123456789012",
        mli_key="12345678-1111-1234-1234-123456789012",
        store_persistence=in_memory_persistence,
    )
    explainer_persistence.make_explainer_sandbox()

    #
    # THEN
    #
    print(
        f"In-memory store: internal_base_path="
        f"{in_memory_persistence.internal.base_path}"
    )
    assert (
        persistences.Persistence.PREFIX_INTERNAL_STORE
        in in_memory_persistence.internal.base_path
    )


@pytest.mark.h2o_sonar
def test_file_system_persistence(tmpdir):
    #
    # GIVEN
    #
    path_root = pathlib.Path(tmpdir)
    interpretation_key = "12345678-1111-1234-1234-123456789012"
    explainer_id = "test.TestExplainer"
    explainer_job_key = "12345678-eeee-1234-1234-123456789012"

    persistence_api = persistences.PersistenceApi()

    path_log = path_root / "sonar" / "sonar.log"
    data_log = "now: H2O Sonar log"

    path_foo_explainer = path_root / "sonar" / "interpretation" / "explainers"
    path_1_json = path_foo_explainer / "file_1.json"
    data_1_json = "{ 'data_json': 1 }"
    path_2_json = path_foo_explainer / "file_2.json"
    data_2_json = "{ 'data_json': 2 }"
    path_3_json = path_foo_explainer / "file_3.json"
    data_3_json = "{ 'data_json': 3 }"

    path_interpretation = path_root / "sonar" / "interpretation.json"
    data_interpretation = "{ 'interpretation': 'result' }"

    paths = [
        path_log,
        path_1_json,
        path_2_json,
        path_3_json,
        path_interpretation,
    ]
    datas = [
        data_log,
        data_1_json,
        data_2_json,
        data_3_json,
        data_interpretation,
    ]

    #
    # WHEN: store
    #
    persistence = persistence_api.create_persistence(
        persistence_type=persistences.PersistenceType.file_system,
        base_path=str(tmpdir),
    )
    persistence.make_dir(path_foo_explainer)
    for i in range(0, len(paths)):
        persistence.save(
            key=paths[i],
            data_type=persistences.PersistenceDataType.json,
            data=datas[i],
        )

    #
    # THEN: store
    #
    print(persistence)
    assert persistence
    files = persistence.list_dir(path_root / "sonar")
    print(f"Store root files: {files}")
    for f in ["sonar.log", "interpretation.json"]:
        assert f in files
    files = persistence.list_dir(path_foo_explainer)
    print(f"Store explainer files: {files}")
    for f in ["file_2.json", "file_3.json", "file_1.json"]:
        assert f in files
    persistence.delete_tree(path_root / "sonar")
    assert not os.path.isdir(str(path_root / "sonar"))

    #
    # WHEN: interpretation persistence
    #
    i_persistence = persistence_api.create_interpretation_persistence(
        store_persistence=persistence,
        base_path=str(tmpdir),
        interpretation_key=interpretation_key,
    )
    i_persistence.make_interpretation_sandbox()

    #
    # THEN: interpretation persistence
    #
    print(
        f"Interpretation persistence: "
        f"\n  root={i_persistence.data_dir} "
        f"\n  user={i_persistence.user_dir}"
        f"\n  base={i_persistence.base_dir} "
    )
    assert interpretation_key in i_persistence.base_dir

    #
    # WHEN: explainer persistence
    #
    r_persistence = persistence_api.create_explainer_persistence(
        store_persistence=persistence,
        base_path=str(tmpdir),
        interpretation_key=interpretation_key,
        explainer_id=explainer_id,
        explainer_job_key=explainer_job_key,
    )
    r_persistence.make_explainer_sandbox()
    zip_path = path_root / "explainer_zip_test.zip"
    r_persistence.make_dir_zip_archive(
        src_dir_path=r_persistence.get_explainer_working_dir(),
        zip_path=zip_path,
    )

    #
    # THEN: explainer persistence
    #
    print(
        f"Explainer persistence: "
        f"\n  root      ={r_persistence.data_dir} "
        f"\n  user      ={r_persistence.user_dir}"
        f"\n  base      ={r_persistence.base_dir} "
        f"\n  explainers={r_persistence.get_explainer_dir()}"
        f"\n  work      ={r_persistence.get_explainer_working_dir()}"
        f"\n  log       ={r_persistence.get_explainer_log_dir()}"
    )
    assert r_persistence
    assert os.path.isfile(str(zip_path))
    assert os.path.isdir(r_persistence.get_explainer_working_dir())
    assert os.path.isdir(r_persistence.get_explainer_log_dir())


@pytest.mark.parametrize(
    "data, expected",
    [
        (float("nan"), f'"{commons.SafeJavaScript.NAN}"'),
        (float("inf"), f'"{commons.SafeJavaScript.INF}"'),
        ([float("nan")], json.dumps([commons.SafeJavaScript.NAN])),
        ([float("inf")], json.dumps([commons.SafeJavaScript.INF])),
        ([float("-inf")], json.dumps([commons.SafeJavaScript.NEG_INF])),
        (
            {
                "float_nan": float("nan"),
                "list_nan": [float("nan")],
                "dict_nan": {"key": float("nan")},
                "tuple_nan": (float("nan"), float("inf"), float("-inf")),
                "float_inf": float("inf"),
                "float_neg_inf": float("-inf"),
            },
            (
                '{"float_nan": "NaN", "list_nan": ["NaN"], "dict_nan": {"key": "NaN"}, '
                '"tuple_nan": ["NaN", "Infinity", "-Infinity"], '
                '"float_inf": "Infinity", "float_neg_inf": "-Infinity"}'
            ),
        ),
    ],
)
@pytest.mark.h2o_sonar
def test_nan_json(data, expected):
    #
    # GIVEN
    #
    print(f"\nData:\n{data}")

    #
    # WHEN
    #
    result = json.dumps(data, cls=persistences.NanEncoder)

    #
    # THEN
    #
    print(f"\nResult:\n{result}")
    assert expected == result


@pytest.mark.parametrize(
    "value,expected_value",
    [
        (commons.SafeJavaScript.NAN, float("nan")),
        (commons.SafeJavaScript.INF, float("inf")),
        (commons.SafeJavaScript.NEG_INF, float("-inf")),
    ],
)
@pytest.mark.h2o_sonar
def test_json_float_decode(value, expected_value):
    #
    # WHEN
    #
    result = commons.SafeJavaScript.decode_to_float(value)

    #
    # THEN
    #
    assert expected_value, result


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
