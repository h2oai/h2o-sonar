Interpreting Datasets
=====================
Datasets are used either to interpret models (for example by the Decision Tree explainer)
or to be interpreted themselves (for example the Drift Detection Explainer).
Datasets can be provided as:

* path to the dataset stored as CSV or ``.jay`` file
* ``datatable.Frame`` instance
* ``pandas.DataFrame`` instance
* ``h2o.H2OFrame`` instance
* ``h2o_sonar.lib.api.datasets.ExplainableDatasetHandle`` instance
* ``h2o_sonar.lib.api.datasets.ExplainableDataset`` instance

See also:

- :ref:`h2o_sonar.lib.api.datasets module`



Explainable Dataset
--------------------
``h2o_sonar.lib.api.datasets.ExplainableDataset`` is typically used when there is a need
to specify dataset **metadata** or when ``h2o_sonar.lib.api.datasets.DatasetApi::create_dataset()`` method
is used to create ``ExplainableDataset``. In the latter case, metadata - like shape, columns and unique
column values frequencies - are constructed automatically.

.. code-block:: python

    dataset: datasets.ExplainableDataset = (
        self.container.dataset_api.create_dataset(
            dataset_src= ... path to dataset or frame instance
        )
    )



Explainable Dataset Handle
~~~~~~~~~~~~~~~~~~~~~~~~~~
``h2o_sonar.lib.api.datasets.ExplainableDatasetHandle`` represents a remote dataset
**hosted** e.g. by a Driverless AI server. For instance it is
used by `H2O Model Validation <https://docs.h2o.ai/wave-apps/h2o-model-validation/>`_ based
explainers which use Driverless AI servers as workers to explain the models. Explainable dataset handle
string serialization (used for instance on the command line) has the following format:

.. code-block:: text

    resource:connection:<connection ID>:key:<dataset ID>

where:

- ``connection ID``
   - ... is a unique identifier of the Driverless AI connection specified in the H2O Sonar configuration.
- ``dataset ID``
   - ... is a unique identifier of the dataset hosted by the Driverless AI server (typically UUID).

Example:

.. code-block:: text

    resource:connection:local-driverless-ai-server:key:7965e2ea-f898-11ed-b979-106530ed5ceb
