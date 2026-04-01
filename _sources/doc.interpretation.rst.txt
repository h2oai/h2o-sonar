Interpretation Parameters
=========================
The following parameters can be specified when starting a new interpretation using
the Python API:

- ``dataset : Union[str, Path, ExplainableDataset, datatable.Frame, Any]``
   - Dataset source: explainable dataset instance, datatable frame, string
     (path to CSV, .jay or any other file type supported by datatable),
     handle to a remote dataset hosted by Driverless AI or
     a dictionary (used to construct frame).
- ``model : Union[str, Path, ExplainableModel, Any]``
   - Path to model (str, Path), explainable model (``ExplainableModel``) ,
     handle to a remote model hosted by Driverless AI or
     an instance of 3rd party model (like Scikit) to interpret.
- ``target_col : str``
   - Target column name - must be valid dataset column name.
- ``explainers : Optional[List[Union[str, commons.ExplainerToRun]]]``
   - Explainer IDs to run within the interpretation or ``ExplainerToRun`` instances
     with explainer parameters. In case of ``None`` or empty list are run all
     compatible explainers.
- ``explainer_keywords: Optional[List[str]]``
   - Run compatible explainers which have given keyword (AND). This setting is used
     *only* in case that ``explainers`` parameter is empty list (or ``None``).
- ``validset : Optional[Union[src, Path, ExplainableDataset, datatable.Frame, Any]]``
   - Optional validation dataset - sources might the same as in case of ``dataset``.
- ``testset : Optional[Union[src, Path, ExplainableDataset, datatable.Frame, Any]]``
   - Optional test dataset - sources might the same as in case of ``dataset``.
- ``use_raw_features : bool``
   - ``True`` to use original features, ``False`` to use transformed features.
- ``used_features : Optional[List]``
   - Optional parameter specifying features (dataset columns) used by the model.
     This parameter is used in case that an instance of the model (not
     ``ExplainableModel``) is provided by the user - therefore ``ExplainableModel``'s
     metadata are not available.
- ``weight_col : str``
   - Name of the weight column to be used by explainers.
- ``prediction_col : str``
   - Name of the predictions column - in case of 3rd party model (standalone MLI).
- ``drop_cols : Optional[List]``
   - List of the columns to drop from the interpretation i.e. columns names which
     should not be explained.
- ``sample_num_rows : Optional[int]``
   - Sample the ``dataset`` to given number of rows. By default, the dataset is
     sampled based on the RAM size (or to 25000 rows). Use ``0`` to disable sampling or
     use an integer value greater than ``0`` to sample to the specified number of rows.
- ``sampler : Optional[DatasetSampler]``
      Sampling method (implementation) to be used - see ``h2o_sonar.utils.sampling``
      module (documentation) for available sampling methods. Use a sampler instance
      to use the specific sampling method.
- ``container : Optional[Union[str, explainer_container.ExplainerContainer]]``
   - Optional explainer container name (str) or container instance to be used
     to run the interpretation.
- ``results_location : Optional[Union[str, pathlib.Path, Dict, Any]]``
   - Where to store interpretation results - filesystem (path as string or ``Path``),
     memory (dictionary) or DB. If ``None``, then results are stored to the current
     directory.
- ``persistence_type : persist.PersistenceType``
   - Optional choice of the persistence type: file-system (default), in-memory
     or database. This option does not override persistence type in case that
     container is provided.
- ``args_as_json_location : Optional[Union[str, pathlib.Path]]``
   - Load all positional arguments and keyword arguments from JSon file.
     This is useful when input is generated, persisted, repeated and used from CLI
     (which doesn't support all the options).
     IMPORTANT: if this argument is specified, then all other function parameters
     are ignored.
- ``upload_to : Union[str, config.ConnectionConfig]``
   - Upload the interpretation report to the H2O GPT Enterprise in order to talk to the report.
- ``log_level : int``
   - Optional container and explainers log level.

See also:

- :ref:`h2o_sonar.interpret module`
