# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import pathlib
import pickle
import subprocess

import pytest

from h2o_sonar.methods.core import _data
from h2o_sonar.methods.core._mli import MLI
from h2o_sonar.methods.surrogates import _abstract_tree_traverser
from h2o_sonar.methods.surrogates import _tree_traverser_h2o
from h2o_sonar.methods.surrogates._decision_tree_h2o import DecisionTreeH2O
from h2o_sonar.methods.surrogates.rules import rules
from tests import test_utils
from tests.conftest import get_h2o3_config


try:
    from h2o.tree import H2OTree

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


def _pickle_h2o_tree(tree, test_data_dir: str):
    with open(os.path.join(test_data_dir, "h2otree_tree.pickle"), "wb") as file_handle:
        pickle.dump(tree.root_node, file_handle)
    with open(
        os.path.join(test_data_dir, "h2otree_features.pickle"), "wb"
    ) as file_handle:
        pickle.dump(tree.features, file_handle)


def _unpickle_h2o_tree(test_data_dir: pathlib.Path, test_data_set: str) -> tuple:
    with open(
        test_data_dir / f"{test_data_set}_h2otree_tree.pickle", "rb"
    ) as file_handle:
        tree = pickle.load(file_handle)
    with open(
        test_data_dir / f"{test_data_set}_h2otree_features.pickle", "rb"
    ) as file_handle:
        features = pickle.load(file_handle)

    return tree, features


def test_txt_code_generator():
    # GIVEN
    test_data_dir = pathlib.Path(test_utils.find_locally("data/predictive/models"))
    (root_node, features) = _unpickle_h2o_tree(
        test_data_dir=test_data_dir, test_data_set="01"
    )
    print(f"H2OTree.features ({type(features)}:\n'{features}'")
    print(f"H2OTree.tree ({type(root_node)}):\n'{root_node}'")
    h2o_traverser = _tree_traverser_h2o.H2OTreeTraverser(
        root_node=root_node, features=features
    )

    # WHEN
    txt = h2o_traverser.extract_rules_from_tree_as_txt(
        rules.CodeStyle.DICT_ROW
    ).__str__()

    # THEN
    print(f"Pseudocode:\n---\n{txt}\n---")


def test_py_code_generator(tmpdir):
    # GIVEN
    test_data_dir = pathlib.Path(test_utils.find_locally("data/predictive/models"))
    (root_node, features) = _unpickle_h2o_tree(
        test_data_dir=test_data_dir, test_data_set="01"
    )
    print(f"H2OTree.features ({type(features)}:\n'{features}'")
    print(f"H2OTree.tree ({type(root_node)}):\n'{root_node}'")
    h2o_traverser = _tree_traverser_h2o.H2OTreeTraverser(
        root_node=root_node, features=features
    )

    # WHEN
    py_code = h2o_traverser.extract_rules_from_tree_as_py_code(
        rules.CodeStyle.DICT_ROW
    ).__str__()

    # THEN
    print(f"Python:\n---\n{py_code}\n---")
    # run the Python code
    input_row = {feature: 3_333 for feature in features}
    with open(os.path.join(tmpdir, "tree_traversal.py"), "a") as file_handle:
        print(
            f"{py_code}\n\nrow = {input_row}\n\nprint(dt_surrogate(row))\n",
            file=file_handle,
        )

    # THEN (This should return status 0. Otherwise will fail.)
    subprocess.run(
        ["python", os.path.join(tmpdir, "tree_traversal.py")],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )


#
# RAW variables TXT / Python code generator
#


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
@pytest.mark.parametrize(
    "code_style,expected_file",
    [
        (
            _abstract_tree_traverser.CodeStyle.RAW_VARS,
            "data/predictive/models/tree_traversal_raw_vars.txt",
        ),
        (
            _abstract_tree_traverser.CodeStyle.DICT_ROW,
            "data/predictive/models/tree_traversal_dict_row.txt",
        ),
    ],
)
def test_raw_var_tree_traversal_output(
    tmpdir, h2o3_cleanup_fixture, code_style, expected_file
):
    # GIVEN
    mli = MLI(work_dir=str(tmpdir), seed=1234, config=get_h2o3_config())

    # Model as data
    model = mli.wrap(
        "test_model_data",
        data=_data.PersistedData(
            test_utils.find_locally("data/predictive/cc_cat_num.csv")
        ),
    )

    # WHEN
    dt = DecisionTreeH2O(max_depth=3, nfolds=0)
    dt.fit(model, response_column="DEFAULT_PAYMENT_NEXT_MONTH")
    tree = H2OTree(dt.estimator, tree_number=0, tree_class=None)
    h2o_traverser = _tree_traverser_h2o.H2OTreeTraverser(
        root_node=tree.root_node,
        features=tree.features,
    )

    #
    with open(test_utils.find_locally(expected_file)) as tree_traversal_from_file:
        tree_traversal_from_file_string = tree_traversal_from_file.read().strip()

    tree_traversal_from_impl = (
        "H2O Tree as Text:\n===\n"
        + h2o_traverser.extract_rules_from_tree_as_txt(code_style).__str__()
        + "\n===\nH2O Tree as Py Code:\n===\n"
        + h2o_traverser.extract_rules_from_tree_as_py_code(code_style).__str__()
        + "\n==="
    )

    # THEN
    print(f"{tree_traversal_from_impl}")
    assert (
        tree_traversal_from_file_string.split() == tree_traversal_from_impl.split()
    ), (
        f"Saved tree traversal output does not match current implementation. "
        f"Saved tree traversal output is\n\n "
        f"{tree_traversal_from_file_string} "
        f"\n\nand implementation is\n\n "
        f"{tree_traversal_from_impl}"
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
@pytest.mark.skip(
    reason=(
        "This test is repro of the RAW VAR code generator bug. The test be removed "
        "with the RAW VAR code generator removal"
    )
)
def test_raw_var_tree_traversal_py_code(tmpdir, h2o3_cleanup_fixture):
    # GIVEN
    mli = MLI(work_dir=str(tmpdir), seed=1234, config=get_h2o3_config())

    # Model as data
    model = mli.wrap(
        "test_model_data",
        data=_data.PersistedData(
            test_utils.find_locally("data/predictive/cc_cat_num.csv")
        ),
    )

    # WHEN
    dt = DecisionTreeH2O(max_depth=3, nfolds=0)
    dt.fit(model, response_column="DEFAULT_PAYMENT_NEXT_MONTH")
    tree = H2OTree(dt.estimator, tree_number=0, tree_class=None)
    h2o_traverser = _tree_traverser_h2o.H2OTreeTraverser(
        root_node=tree.root_node,
        features=tree.features,
    )
    tree_traversal_py_code = h2o_traverser.extract_rules_from_tree_as_py_code(
        rules.CodeStyle.RAW_VARS
    ).__str__()
    with open(str(tmpdir) + "/tree_traversal.py", "a") as f:
        print(
            tree_traversal_py_code + "\n\nprint(dt_surrogate(np.nan, np.nan, np.nan))",
            file=f,
        )

    # THEN (this should return status 0, otherwise will fail.)
    subprocess.run(
        ["python", str(tmpdir) + "/tree_traversal.py"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
