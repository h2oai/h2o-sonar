Interpreting Models
===================
H2O Sonar provides interpretability of machine learning models. Model interpretations
can be run either from the command line interface or using the Python API.

Currently supported **models** include:

- `H2O Driverless AI MOJO <https://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/index.html>`_
- `H2O-3 <https://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html>`_,
- `scikit-learn <https://scikit-learn.org/>`_
- `H2O Driverless AI models exposed as REST service <https://h2oai.github.io/dai-deployment-templates/local-rest-scorer/>`_
- ``h2o_sonar.lib.api.models.ExplainableModelHandle`` instance
- ``h2o_sonar.lib.api.models.ExplainableModel`` instance

Jupyter Notebooks with examples for all supported models:

   - :ref:`H2O Sonar Examples`



Explainable Model
-----------------
``h2o_sonar.lib.api.models.ExplainableModel`` is typically used when there is a need
to explain the model which is not **yet** supported by the library. In this case,
the explainable model instance (implemented by the library user) provides model's
metadata and prediction function. Explainable model instance is model agnostic
and can be used with any model type so that it can be explainable by the library.



Explainable Model Handle
~~~~~~~~~~~~~~~~~~~~~~~~
``h2o_sonar.lib.api.models.ExplainableModelHandle`` represents a remote model
**hosted** e.g. by a Driverless AI server. For example, it is used by
`H2O Model Validation <https://docs.h2o.ai/wave-apps/h2o-model-validation/>`_ explainers,
which use Driverless AI servers as workers to explain models. Explainable model handle
string serialization (used, for example, on the command line) has the following format:

.. code-block:: text

    resource:connection:<connection ID>:key:<model ID>

where:

- ``connection ID``
   - ... is a unique identifier of the Driverless AI connection specified in the H2O Sonar :ref:`configuration<Library Configuration>`.
- ``model ID``
   - ... is a unique identifier of the model hosted by the Driverless AI server (typically UUID).

Example:

.. code-block:: text

    resource:connection:local-driverless-ai-server:key:7965e2ea-f898-11ed-b979-106530ed5ceb



Interpreting Models using the Command Line Interface
----------------------------------------------------
Use ``--help`` option to determine interpretation options:

.. code-block:: text

    $ h2o-sonar --help
        H2O Sonar Python library for Responsible AI.

        ...

        optional arguments per action and entity:

          ...

          run interpretation:
            ...
            --model          path to the serialized model or model handle.
            --model-type     model type: 'pickle' or 'mojo'

- ``--model``
   - Path to the model persisted on the filesystem or :ref:`model handle<Explainable Model Handle>`.
- ``--model-type``
   - Optional model type specification (typically auto detected):
      - ``mojo``
          - Driverless AI model saved as MOJO (``.mojo`` file extension).
      - ``pickle``
          - Instance of the :ref:`ExplainableModel<h2o_sonar.lib.api.models module>`
            class or a `Scikit-learn <https://scikit-learn.org/>`_ model saved using
            pickle.

Example of pickled `Scikit-learn <https://scikit-learn.org/>`_ model interpretation:

.. code-block:: text

    $ h2o-sonar run interpretation
        --dataset=dataset.csv
        --target-col=TARGET
        --results-location=./results
        --model-type=pickle
        --model=model.pkl

Example of running selected explainers to interpret the model - first find out what
are the available explainers, then run selected:

.. code-block:: text

    $ h2o-sonar list explainers

    [
        'h2o_sonar.explainers.fi_naive_shapley_explainer.NaiveShapleyMojoFeatureImportanceExplainer',
        'h2o_sonar.explainers.summary_shap_explainer.SummaryShapleyExplainer',
        'h2o_sonar.explainers.dt_surrogate_explainer.DecisionTreeSurrogateExplainer',
        'h2o_sonar.explainers.pd_ice_explainer.PdIceExplainer',
        'h2o_sonar.explainers.dia_explainer.DiaExplainer',
        'h2o_sonar.explainers.transformed_fi_shapley_explainer.ShapleyMojoTransformedFeatureImportanceExplainer'
    ]

.. code-block:: text

    $ h2o-sonar run interpretation
        --explainers=h2o_sonar.explainers.summary_shap_explainer.SummaryShapleyExplainer,h2o_sonar.explainers.pd_ice_explainer.PdIceExplainer
        --dataset=dataset.csv
        --target-col=TARGET
        --results-location=./results
        --model-type=pickle
        --model=model.pkl

Example of `H2O Driverless AI model <https://h2oai.github.io/dai-deployment-templates/local-rest-scorer/>`_ model interpretation over REST:

.. code-block:: text

    $ h2o-sonar run interpretation
        --dataset=dataset.csv
        --target-col=TARGET
        --results-location=./results
        --model=http://localhost:8080/model

See also Jupyter Notebook Examples:

   - :ref:`H2O Sonar Examples`
