#!/usr/bin/env python3
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.

def generate_rst_for_package(
    package: str,
    subpackages: list[str],
    submodules: list[str],
) -> str:
    # title
    s = f"{package} package"
    s = f"\n{s}\n{gen_title_underline(s, '=')}\n"

    # subpackages
    if subpackages:
        s = f"{s}Subpackages\n-----------\n\n.. toctree::\n\n"
        for subpackage in subpackages:
            s = f"{s}    {subpackage}\n"
    s = f"{s}\n"

    # submodules
    if submodules:
        s = f"{s}Submodules\n----------\n\n"
        for submodule in submodules:
            submodule = f"{package}.{submodule}"
            title = f"{submodule} module"
            s = f"\n{s}{title}\n{gen_title_underline(title)}\n\n"
            s = f"{s}.. automodule:: {submodule}\n"
            s = f"{s}    :members:\n"
            s = f"{s}    :undoc-members:\n"
            s = f"{s}    :show-inheritance:\n\n"

    # modules contents
    s = f"{s}Module contents\n---------------\n\n"
    s = f"{s}.. automodule:: {package}\n"
    s = f"{s}    :members:\n"
    s = f"{s}    :undoc-members:\n"
    s = f"{s}    :show-inheritance:\n\n"

    return s


def gen_title_underline(title: str, underline_char: str = "^") -> str:
    u = ""
    for _ in range(0, len(title)):
        u += f"{underline_char}"
    return u


def gen_explainer_examples() -> str:
    return generate_rst_for_package(
        package="h2o_sonar.explainers.examples",
        subpackages=[],
        submodules=[
            "example_compatibility_check_explainer",
            "example_custom_explanation_explainer",
            "example_eda_explainer",
            "example_hello_world_explainer",
            "example_logging_explainer",
            "example_metadata_explainer",
            "example_params_explainer",
            "example_persistence_explainer",
            "example_score_explainer",
        ],
    )


def gen_explainer_templates() -> str:
    return generate_rst_for_package(
        package="h2o_sonar.explainers.templates",
        subpackages=[],
        submodules=[
            "template_dt_explainer",
            "template_featimp_explainer",
            "template_md_explainer",
            "template_md_featimp_summary_explainer",
            "template_md_vega_explainer",
            "template_pd_explainer",
            "template_scatter_plot_explainer",
            "template_summary_featimp_explainer",
        ],
    )


def gen_method_utils() -> str:
    return generate_rst_for_package(
        package="h2o_sonar.methods.utils",
        subpackages=[],
        submodules=[
            "data_utils",
            "fairness_utils",
            "h2o_utils.py",
            "histogram.py",
        ],
    )


def gen_lib_api() -> str:
    return generate_rst_for_package(
        package="h2o_sonar.lib.api",
        subpackages=[],
        submodules=[
            "commons",
            "datasets",
            "explainers",
            "explanations",
            "formats",
            "interpretations",
            "models",
            "persistences",
            "plots",
        ],
    )


if __name__ == "__main__":
    print(gen_lib_api())
