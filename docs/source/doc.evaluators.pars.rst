Evaluator Parameters
--------------------
Evaluators can be parameterized using the parameters documented in the sections below.
These parameters are evaluator specific and can be passed to evaluators using
the Python API as follows:

.. code-block:: python

    evaluation = evaluate.run_evaluation(
        dataset=dataset_path,
        models=[mock_llm_model],
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=PiiLeakageEvaluator.evaluator_id(),
                params={
                    PiiEvaluator.PARAM_METRIC_THRESHOLD: 0.75,
                },
            )
        ],
        ...
    )


In the example above, ``metric_threshold`` parameter is passed to the :ref:`PII Leakage Evaluator`.
An instance of the ``EvaluatorToRun`` class is used to specify evaluator ID and its parameters,
which are passed in a dictionary, where the key is string identifier of the parameter and
value is the parameter value.


