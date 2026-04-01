# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import loggers
from h2o_sonar.lib.container import explainer_container
from tests.lib import test_containers


@pytest.mark.h2o_sonar
def test_byoe_hot_deploy():
    # GIVEN
    c = test_containers.ExplainerExamplesAndTemplatesTestContainer()
    explainer_descriptors = [
        "tests.explainers.doc.example_morris_sa_explainer"
        "::ExampleMorrisSensitivityAnalysisExplainer"
    ]

    # WHEN
    for explainer_descriptor in explainer_descriptors:
        explainer_type = c.hot_deploy_explainer(explainer_descriptor)

        # THEN
        assert explainer_type
        explainer = explainer_type()
        explainer_d = explainer.as_descriptor()
        print(f"Explainer descriptor:\n{explainer_d}")
        assert explainer_d


@pytest.mark.h2o_sonar
def test_byoe_container_registration(tmpdir):
    """This test case tests whether custom explainers in H2O Sonar configuration
    are registered in the resolved explainer container.

    """

    # GIVEN
    from h2o_sonar import config

    explainer_class_name = "ExampleMorrisSensitivityAnalysisExplainer"
    config.config.custom_explainers = [
        f"tests.explainers.doc.example_morris_sa_explainer::{explainer_class_name}"
    ]

    container = explainer_container.LocalExplainerContainer()
    container.setup(
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    # WHEN
    try:
        # register_configured_explainers() is called by LocalExplainerContainer

        # THEN
        explainers_list = container.list_explainers()
        print("Registered explainers:")
        for e in explainers_list:
            print(f"  {e.id}")
        for explainer_descriptor in config.config.custom_explainers:
            (_, class_name) = explainer_descriptor.split("::")
            found = ""
            for e in explainers_list:
                if class_name in e.id:
                    found = class_name
                    break
            if not found:
                raise AssertionError(
                    f"BYOE not registered: '{explainer_descriptor}' w/  {class_name}"
                )
    finally:
        for explainer_descriptor in config.config.custom_explainers:
            (module_name, class_name) = explainer_descriptor.split("::")
            e_id = f"{module_name}.{class_name}"
            try:
                print(f"Un-registering explainer ID '{e_id}'")
                unreg_status = container.unregister_explainer(e_id)
                print(f"Unregistered explainer ID {e_id} w/ status '{unreg_status}'")
            except Exception as ex:
                print(f"Un-registration of explainer ID '{e_id}': failed with {ex}")

        # clear configuration > avoid explainer re-registration
        config.config.custom_explainers.clear()

    post_test_explainers = container.list_explainers()
    print(f"POST test list_explainers(): {post_test_explainers}")
    assert explainer_class_name not in str(post_test_explainers)
    print(f"Post test H2O Sonar configuration: {config.config.custom_explainers}")
    assert not config.config.custom_explainers


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
