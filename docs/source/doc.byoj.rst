BYOJ: Bring Your Own Judge
==========================
H2O Sonar can be configured to use custom evaluation LLM judges.
For instance in order to ensure privacy and avoid sending of
the sensitive data to a 3rd party and/or to the cloud.

There are two ways how to configure the custom judges:

- **Forced from the H2O Sonar configuration**
  - the custom judge is forced from the H2O Sonar configuration in order to be used by evaluators which need an LLM judge to evaluate the model.

- **Specified in the evaluator parameters**
  - the custom judge is specified in the evaluator parameters.

Custom LLM judge can be configured for the following evaluators:

- :ref:`Answer Correctness Evaluator`
- :ref:`Answer Semantic Similarity Evaluator`
- :ref:`Context Relevancy Evaluator`
- :ref:`RAGAS Evaluator`
- :ref:`Context Precision Evaluator`
- :ref:`Faithfulness Evaluator`
- :ref:`Context Recall Evaluator`
- :ref:`Answer Relevancy Evaluator`

This feature includes also reconfiguration of embeddings provider from the same reasons.



Force Custom Judge from H2O Sonar Configuration
-----------------------------------------------
In order to force the custom judge from the H2O Sonar configuration, the custom judge
must be configured in H2O Sonar configuration and the ``force_eval_judge`` must be set
either to its key or to ``true``.

.. code-block:: python

   from h2o_sonar import config as h2o_sonar_config

   # create connection where judge's LLM is hosted
   my_h2ogpt_connection = h2o_sonar_config.ConnectionConfig(
      connection_type=h2o_sonar_config.ConnectionConfigType.H2O_GPT.name,
      name="H2O GPT",
      description="My H2O GPT server.",
      server_url="https://gpt.host.ai/",
      token=os.getenv(KEY_H2OGPT_API_KEY),
      token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
   )

   # register the connection in H2O Sonar configuration
   h2o_sonar_config.config.add_connection(my_h2ogpt_connection)

   # create custom judge configuration - note LLM model name
   judge_config = h2o_sonar_config.EvaluationJudgeConfig(
         name="My custom judge",
         description="My custom LLM judge to be used by evaluators.",
         judge_type=judge_type,
         connection=judge_connection,
         llm_model_name="h2oai/h2ogpt-4096-llama2-70b-chat",
   )

   # register the custom judge in H2O Sonar configuration
   h2o_sonar_config.config.add_evaluation_judge(judge_config)

   # force the custom judge in H2O Sonar configuration
   h2o_sonar_config.config.force_eval_judge = "true"

   ...

   # run evaluation

   ...

In the example above will be used the first LLM evaluation judge from the H2O Sonar configuration.
In order to use particular LLM evaluation judge custom judge key must be specified instead of ``true``.



LLM Judge Specified in the Evaluator Parameters
-----------------------------------------------
In order to specify the custom judge in the evaluator parameters, the custom judge must be
configured in H2O Sonar configuration and the judge key must be specified in the evaluator parameter.

.. code-block:: python

   from h2o_sonar import config as h2o_sonar_config

   # create connection where judge's LLM is hosted
   my_h2ogpt_connection = h2o_sonar_config.ConnectionConfig(
      connection_type=h2o_sonar_config.ConnectionConfigType.H2O_GPT.name,
      name="H2O GPT",
      description="My H2O GPT server.",
      server_url="https://gpt.host.ai/",
      token=os.getenv(KEY_H2OGPT_API_KEY),
      token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
   )

   # register the connection in H2O Sonar configuration
   h2o_sonar_config.config.add_connection(my_h2ogpt_connection)

   # create custom judge configuration - note LLM model name
   judge_config = h2o_sonar_config.EvaluationJudgeConfig(
         name="My custom judge",
         description="My custom LLM judge to be used by evaluators.",
         judge_type=judge_type,
         connection=judge_connection,
         llm_model_name="h2oai/h2ogpt-4096-llama2-70b-chat",
   )

   # register the custom judge in H2O Sonar configuration
   h2o_sonar_config.config.add_evaluation_judge(judge_config)

   # use evaluator parameter to specify the custom judge

   evaluators = [
         commons.EvaluatorToRun(
            evaluator_id=ContextRelevancyEvaluator.evaluator_id(),
            params={
               ContextRelevancyEvaluator.PARAM_EVAL_JUDGE_CFG_KEY: judge_config.key,
            },
         )
   ]

   # run evaluation
   evaluation = evaluate.run_evaluation(
        # dataset w/ prompts, conditions and model keys
        dataset=test_lab.dataset,
        # models to be evaluated / compared to get leaderboard
        models=list(test_lab.evaluated_models.values()),
        # evaluators
        evaluators=evaluators,
        # where to save the report
        results_location=result_dir,
    )

   ...

In the example above will be used the custom judge specified in the evaluator parameters.
If the custom judge is specified, then ``ragas`` evaluators are also reconfigured to use
privacy safe embeddings provider which is run on the same machine as the custom judge.



Privacy Safe LLM Models
-----------------------
The decision which LLM model to use for the evaluation is crucial. The user must consider
the quality of the model, the privacy, the cost, the availability and performance. Different
models may have different trade-offs. The user can choose the most suitable model for
the evaluation of the particular problem or task which is being solved.

The following leaderboard of LLM models gives an rough overview of the quality of the models
and how much precision / performance might be sacrificed when changing the judge for
instance in order to ensure privacy.

+--------------+-----------------------------+-------------+-------+
| Privacy safe | LLM                         | Correlation | MAE   |
+==============+=============================+=============+=======+
| ❌           | gpt-4-1106-preview          | 0.964       | 0.024 |
+--------------+-----------------------------+-------------+-------+
| ❌           | claude-3-opus-20240229      | 0.891       | 0.078 |
+--------------+-----------------------------+-------------+-------+
| ❌           | mistral-large-latest        | 0.827       | 0.160 |
+--------------+-----------------------------+-------------+-------+
| ❌           | mistral-medium              | 0.823       | 0.128 |
+--------------+-----------------------------+-------------+-------+
| ✓            | Mixtral-8x7B-Instruct-v0.1  | 0.819       | 0.179 |
+--------------+-----------------------------+-------------+-------+
| ❌           | claude-3-sonnet-20240229    | 0.795       | 0.155 |
+--------------+-----------------------------+-------------+-------+
| ✓            | Nous-Capybara-34B           | 0.794       | 0.109 |
+--------------+-----------------------------+-------------+-------+
| ✓            | Mistral-7B-Instruct-v0.2    | 0.784       | 0.190 |
+--------------+-----------------------------+-------------+-------+
| ❌           | claude-2.1                  | 0.779       | 0.184 |
+--------------+-----------------------------+-------------+-------+
| ✓            | openchat-3.5-1210           | 0.736       | 0.221 |
+--------------+-----------------------------+-------------+-------+
| ✓            | gemma-7b-it                 | 0.696       | 0.226 |
+--------------+-----------------------------+-------------+-------+
| ❌           | gpt-3.5-turbo-0613          | 0.630       | 0.258 |
+--------------+-----------------------------+-------------+-------+
| ✓            | h2ogpt-4096-llama2-70b-chat | 0.604       | 0.347 |
+--------------+-----------------------------+-------------+-------+
| ✓            | zephyr-7b-beta              | 0.513       | 0.378 |
+--------------+-----------------------------+-------------+-------+
| ✓            | h2ogpt-4096-llama2-13b-chat | 0.343       | 0.441 |
+--------------+-----------------------------+-------------+-------+

The comparison of LLM models in the above table is based on a H2O.ai KGM benchmark
as of 2024/3/15. The benchmark uses a custom prompt with detailed description of how
to construct the LLM model score. The score is then used to calculate the metrics in
the table. The table shows both commercial and non-commercial (open) models side-by-side
to compare the quality of the models. Leaderboards which compare LLM models from various
perspectives - like `Chatbot Arena Leaderboard <https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard>`_ - are
available on the internet and can be used to choose the most suitable model for the evaluation.

+--------------+-----------------------------+-------------+-------+
| Privacy safe | LLM                         | Correlation | MAE   |
+==============+=============================+=============+=======+
+--------------+-----------------------------+-------------+-------+
| ✓            | Mixtral-8x7B-Instruct-v0.1  | 0.819       | 0.179 |
+--------------+-----------------------------+-------------+-------+
| ✓            | Nous-Capybara-34B           | 0.794       | 0.109 |
+--------------+-----------------------------+-------------+-------+
| ✓            | Mistral-7B-Instruct-v0.2    | 0.784       | 0.190 |
+--------------+-----------------------------+-------------+-------+
| ✓            | openchat-3.5-1210           | 0.736       | 0.221 |
+--------------+-----------------------------+-------------+-------+
| ✓            | gemma-7b-it                 | 0.696       | 0.226 |
+--------------+-----------------------------+-------------+-------+
| ✓            | h2ogpt-4096-llama2-70b-chat | 0.604       | 0.347 |
+--------------+-----------------------------+-------------+-------+
| ✓            | zephyr-7b-beta              | 0.513       | 0.378 |
+--------------+-----------------------------+-------------+-------+
| ✓            | h2ogpt-4096-llama2-13b-chat | 0.343       | 0.441 |
+--------------+-----------------------------+-------------+-------+

The table above shows only privacy safe LLM models. The privacy safe LLM models are
models which can be hosted on-premise / within the organization. As was described above
the privacy safe LLM models are suitable for the evaluation of the sensitive data in
order to ensure privacy and avoid sending of the sensitive data to a 3rd party and/or
to the cloud.

See also:

- :ref:`Library Connections Configuration<Library Configuration>`
