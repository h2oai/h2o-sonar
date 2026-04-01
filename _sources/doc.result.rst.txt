Report and Results
==================
This section describes the directory structure that is created by H2O Sonar when it runs an interpretation/evaluation.

An **example** of a directory structure with an interpretation run that includes the Decision Tree Explainer explainer is:

.. code-block:: text

    .
    ├── h2o-sonar
    │   └── mli_experiment_6fccd3f9-4350-401c-b29a-cf2ab5b9c2c9
    │       ├── explainer_h2o_sonar_explainers_dt_surrogate_explainer_DecisionTreeSurrogateExplainer_89c7869d-666d-4ba2-ba84-cf1aa1b903bd
    │       │   ├── global_custom_archive
    │       │   │   ├── application_zip
    │       │   │   │   └── explanation.zip
    │       │   │   └── application_zip.meta
    │       │   ├── global_decision_tree
    │       │   │   ├── application_json
    │       │   │   │   ├── dt_class_0.json
    │       │   │   │   └── explanation.json
    │       │   │   └── application_json.meta
    │       │   ├── local_decision_tree
    │       │   │   ├── application_json
    │       │   │   │   └── explanation.json
    │       │   │   └── application_json.meta
    │       │   ├── log
    │       │   │   └── explainer_run_89c7869d-666d-4ba2-ba84-cf1aa1b903bd.log
    │       │   ├── result_descriptor.json
    │       │   └── work
    │       │       ├── dtModel.json
    │       │       ├── dtpaths_frame.bin
    │       │       ├── dtPathsFrame.csv
    │       │       ├── dtsurr_mojo.zip
    │       │       ├── dtSurrogate.json
    │       │       └── dt_surrogate_rules.zip
    │       ├── explainers_parameters.json
    │       ├── interpretation.html
    │       └── interpretation.json
    └── h2o-sonar.log

Directory structure documentation:

- ``$SONAR_HOME``
   - Directory specified in ``results_location`` parameter of H2O Sonar Python API,
     ``--results-location`` on command line or the current directory if no results directory is provided.
- ``$SONAR_HOME/h2o_sonar.html``
   - HTML file with the list of H2O Sonar interpretations.
- ``$SONAR_HOME/h2o_sonar.log``
   - H2O Sonar library log of all interpretation runs within this results directory.
- ``$SONAR_HOME/h2o_sonar``
   - Directory with all interpretation runs where an interpretation directory
     name is ``mli_experiment_<UUID>``
     e.g. ``mli_experiment_6fccd3f9-4350-401c-b29a-cf2ab5b9c2c9``.
- ``$SONAR_HOME/h2o_sonar/mli_experiment_<UUID>``
   - The interpretation directory which contains:
      - interpretation overview in HTML and JSon representation
      - interpretation and explainers parameters
      - sub-directory for each explainer which was run within this interpretation.
        The name of explainer directory is ``<explainer ID>_<UUID>``, for example:

.. code-block:: text

    explainer_h2o_sonar_explainers_dt_surrogate_explainer_DecisionTreeSurrogateExplainer_89c7869d-666d-4ba2-ba84-cf1aa1b903bd

- ``$SONAR_HOME/h2o_sonar/mli_experiment_<UUID>/interpretation.html``
   - Interpretation overview (list of executed/failed/successful explainers, parameters,
     descriptions, links to explanations and artifacts in various representations) in
     HTML format.
- ``$SONAR_HOME/h2o_sonar/mli_experiment_<UUID>/interpretation.json``
   - Interpretation overview (list of executed/failed/successful explainers, parameters,
     descriptions, links to explanations and artifacts in various representations) in
     machine-processable ``JSon`` format.
- ``$SONAR_HOME/h2o_sonar/mli_experiment_<UUID>/explainers_parameters.json``
   - The interpretation and explainers' parameters in JSon format.
- ``$SONAR_HOME/h2o_sonar/mli_experiment_<UUID>/<explainer ID>_<UUID>``
   - Explainer directory (sandbox) with the artifacts created by the explainer during
     its run.
- ``$SONAR_HOME/h2o_sonar/mli_experiment_<UUID>/<explainer ID>_<UUID>/work/``
   - Explainer working directory with (transient) files and directories which were used
     to create actual explanations.
- ``$SONAR_HOME/h2o_sonar/mli_experiment_<UUID>/<explainer ID>_<UUID>/log/``
   - Directory with the explainer log.
- ``$SONAR_HOME/h2o_sonar/mli_experiment_<UUID>/<explainer ID>_<UUID>/result_descriptor.json``
   - Machine-processable overview of explanations created by the explainer in JSon
     format.
- ``$SONAR_HOME/h2o_sonar/mli_experiment_<UUID>/<explainer ID>_<UUID>/*``
   - All other directories with the explainer run directory are explanations e.g.
     ``global_decision_tree`` with decision tree in various formats or
     ``global_custom_archive`` with ZIP archive of explainer artifacts.
   - Explainers typically create explanations which are different explainer to
     explainer. Particular explainer explanations are documented in explainer
     sections.
   - Explanation directory always contains directories named according to MIME types
     with the representations (formats) of the explanations e.g. ``application_zip``
     (for ``application/zip``) directory contains ZIP archive.

