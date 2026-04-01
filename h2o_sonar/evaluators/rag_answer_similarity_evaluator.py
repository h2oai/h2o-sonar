# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.evaluators import rag_ragas_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import results
from h2o_sonar.utils import caching


class AnswerSemanticSimilarityEvaluator(evaluators.Evaluator):
    _display_name = "Answer semantic similarity"
    _tagline = "Evaluate the resemblance between the actual and expected answers."

    # IMPLEMENTATION: evaluator implementation details
    # - uses Open AI embeddings blobs in tiktoken transitive dependency - for example:
    #   https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken

    # COMPATIBILITY: LLM/RAG evaluation
    _rag = True
    _llm = True  # evaluator does NOT require context

    # GLOBAL: metric value for all dataset rows
    _global_explanation = True
    # LOCAL: metric value for particular row
    _local_explanation = True
    _explanation_types = [
        e10s.LlmEvalResultsExplanation,
        e10s.LlmHeatmapLeaderboardExplanation,
        e10s.WorkDirArchiveExplanation,
    ]

    _keywords = [
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_EA,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_REQUIRES_OPENAI_KEY,
        evaluators.KEYWORD_SR_11_7_CS,
        evaluators.KEYWORD_SR_11_7_OA,
        evaluators.KEYWORD_NIST_AI_RMF_S,
        evaluators.KEYWORD_PROBLEM_TYPE_IR,
        evaluators.KEYWORD_PROBLEM_TYPE_SUM,  # needs reference summary as actual answer
        evaluators.KEYWORD_PROBLEM_TYPE_QA,
        evaluators.KEYWORD_ES_GENERATE,
        evaluators.KEYWORD_METHOD_SEMANTIC_SIMILARITY,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
    ]

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta.clone(
                rag_ragas_evaluator.RagasEvaluator.METRIC_META_ANSWER_SIMILARITY
            )
        ]
    )

    _parameters = [
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        ),
        evaluators.Evaluator._PARAM_NAN_TOLERANCE,
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.Evaluator._PARAM_EVAL_JUDGE,
        evaluators.Evaluator._get_custom_param_min_test_case(),
    ]

    _modules_needed_by_name = ["ragas==0.1.3"]

    # models used by the evaluator
    _e_model_baai_bge = caching.MODEL_BAAI_BGE_SMALL_EN_V1_5

    _brief_description = """Answer Semantic Similarity Evaluator assesses the semantic
resemblance between the generated answer and the expected answer (ground truth).

- Cross-encoder model or embeddings + cosine similarity.
- Compatibility: RAG and LLM evaluation.
- Based on
  [RAGAs library](https://docs.ragas.io/en/latest/concepts/metrics/index.html)"""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- Evaluator utilizes a **cross-encoder model** to calculate the semantic similarity
  score between the actual answer and expected answer. A cross-encoder model takes two
  text inputs and generates a score indicating how similar or relevant they are to
  each other.
- Method is configurable, and the evaluator defaults to **embeddings**
  [{_e_model_baai_bge}](https://huggingface.co/{_e_model_baai_bge}) (where BGE
  stands for "BAAI General Embedding" which refers to a suite of open-source text
  embedding models developed by the Beijing Academy of Artificial Intelligence (BAAI))
  and [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) as
  the similarity metric. In this case, evaluator does vectorization of the
  ground truth and generated answers and calculates the cosine similarity between them.
- In general, cross-encoder models (like
  [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers>))
  tend to have higher accuracy in complex tasks, but are slower. Embeddings with
  cosine similarity tend to be faster, more scalable, but less accurate for nuanced
  similarities.

```math
   answer similarity = cosine_similarity(emb(expected answer), emb(actual answer))
```

- Where:
    - `emb(expected answer)` is the embedding of the expected answer.
    - `emb(actual answer)` is the embedding of the actual answer.

See also:

- Paper *"Semantic Answer Similarity for Evaluating Question Answering Models"*:
  https://arxiv.org/pdf/2108.06130.pdf
- 3rd party metric documentation:
  https://docs.ragas.io/en/latest/concepts/metrics/index.html
- 3rd party library used: https://github.com/explodinggradients/ragas""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = "Answer similarity"

        self.ragas = rag_ragas_evaluator.RagasEvaluator()

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not evaluators.Evaluator._check_llm_dataset_compatibility(
            self,
            params=params,
            evaluator_keywords=self.keywords,
            check_empty_contexts=evaluators.KEYWORD_RQ_RC in self.keywords,
            fail_on_all_empty_contexts=evaluators.KEYWORD_RQ_RC in self.keywords,
        ):
            return False

        # check that at least one row has both actual answer and expected answer
        if not self._check_llm_dataset_field_presence(
            params=params,
            require_actual_answer=True,
            require_expected_answer=True,
        ):
            return False

        return True

    def setup(self, model, persistence, **kwargs):
        evaluators.Evaluator.setup(self, model, persistence, **kwargs)

        self._resolve_evaluator_params()

        self.ragas.progress_callback = self.progress_callback
        self.ragas.setup(model, persistence, **kwargs)

    def evaluate(self, llm_testset, explanations_types=None, **kwargs) -> list:
        return self.ragas.eval_custom_metrics(
            llm_testset,
            metrics_threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
            ),
            save_llm_result=self.args.get(
                evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
                evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
            ),
            custom_eval_judge_cfg_key=self._resolve_judge_key(),
            metrics_to_run=AnswerSemanticSimilarityEvaluator._metrics_meta,
            evaluator=self,
            nan_tolerance=self.args.get(
                evaluators.Evaluator.PARAM_NAN_TOLERANCE,
                evaluators.Evaluator.DEFAULT_NAN_TOLERANCE,
            ),
        )

    # PROBLEMS: set by RAGAs evaluator on this evaluator instance
    # INSIGHTS: set by RAGAs evaluator on this evaluator instance

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=AnswerSemanticSimilarityEvaluator.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
