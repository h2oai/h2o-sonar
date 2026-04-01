# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.evaluators import abc_byop_evaluator


class HallucinationByopEvaluator(abc_byop_evaluator.AbcByopEvaluator):
    _display_name = "Hallucination BYOP Evaluator"
    _CLASSES = abc_byop_evaluator.AbcByopEvaluator.Classes(
        failure="hallucinated", success="factual"
    )
    _llm = False
    _rag = True

    def _initialize_prompt_template(self) -> str:
        return """In this task, you will be presented with a query,
a reference text and an answer. The answer is generated to the question based on the
reference text. The answer may contain false information, you must use the reference
text to determine if the answer to the question contains false information, if the
answer is a hallucination of facts. Your objective is to determine whether the
reference text contains factual information and is not a hallucination.
A 'hallucination' in this context refers to an answer that is not based on the
reference text or assumes information that is not available in the reference text.
Your response should be a single word: either "factual" or "hallucinated", and it
should not include any other text or characters. "hallucinated" indicates that the
answer provides factually inaccurate information to the query based on the reference
text. "factual" indicates that the answer to the question is correct relative to the
reference text, and does not contain made up information.
Please read the query and reference text carefully before determining your response.

# Query: {INPUT}
# Reference text: {CONTEXT}
# Answer: {ACTUAL_OUTPUT}
Is the answer above factual or hallucinated based on the query and reference text?
Your response must be single word, either "factual" or "hallucinated"."""
