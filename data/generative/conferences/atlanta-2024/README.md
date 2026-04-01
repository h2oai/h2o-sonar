# Demo Data for H2O GenAI World Atlanta '24
This directory contains labs and suites for GenAI World event:

* **Bank teller**: (LLM)
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/eval_llm/bank_teller_test_lab.json
    * Name:
        - Bank Teller exam
    * Description:    
        - LLMs evaluation in the Bank Teller exam.
    * Evaluator:
        - Text matching
    * Connection:
        - LLM
* **PII leakage**: (LLM)
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/eval_llm/pii_test_lab.json
    * Name:
        * PII leakage
    * Description:
        * LLMs PII leakage evaluation (credit cards, SSNs, emails).
    * Evaluator:
        - Text matching
    * Connection:
        - LLM
* **Sensitive data leakage**: (LLM)
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/eval_llm/sensitive_data_test_lab.json
    * Name:
        * Sensitive data leakage
    * Description:
        * LLMs sensitive data leakage evaluation - certificates (SSL/TLS certs in PEM format), API keys (H2O.ai and OpenAI), activation keys (Windows).
    * Evaluator:
        - Text matching
    * Connection:
        - LLM
* **SR 11-7 small**: (RAG)
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/conferences/atlanta-2024/sr1107_test_lab_small.json
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/conferences/atlanta-2024/sr1107_test_suite_small.json
    * Name:
        * SR 11-7 RAG evaluation (small)
    * Description
        - SR 11-7 model risk management document related RAG Q&As.
    * Evaluator:
        - Text matching
    * Connection:
        - RAG
* **SR 11-7 large**: (RAG)
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/conferences/atlanta-2024/sr1107_test_suite_large.json
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/conferences/atlanta-2024/sr1107_test_suite_large.json
    * Name:
        * SR 11-7 RAG evaluation (large)
    * Description
        - SR 11-7 model risk management document related RAG Q&As.
    * Evaluator:
        - Text matching
    * Connection:
        - RAG
* **RAGAS**: (RAG)
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/conferences/atlanta-2024/ragas_no_or_h2ogpte_test_lab.json
    * Name:
        * RAGAS
    * Description
        - RAGAS metrics suite (Answer relevancy, Context precision, Faithfulness, Context recall) RAGs evaluation.
    * Evaluator:
        - RAGAS
    * Connection:
        - RAG




# Leaderboard Backlog
* **Answer correctness** (RAG and LLM)
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/conferences/atlanta-2024/ragas_no_or_h2ogpte_test_lab.json
    * Name:
        * Answer correctness
    * Description
        - Answer correctness can assess whether the answer is correct given the expected answer (ground truth).
    * Evaluator:
        - Answer correctness
    * Connection:
        - RAG

* **Answer relevance** (RAG and LLM)
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/conferences/atlanta-2024/ragas_no_or_h2ogpte_test_lab.json
    * Name:
        * Answer relevance
    * Description
        - Answer relevance (RAGAS metric) can assess whether the answer is (in)complete and does not contain redundant information which was not asked - noise.
    * Evaluator:
        - Answer relevance
    * Connection:
        - RAG

* **Answer semantic similarity** (RAG and LLM)
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/conferences/atlanta-2024/ragas_no_or_h2ogpte_test_lab.json
    * Name:
        * Answer semantic similarity
    * Description
        - Answer semantic similarity can assess semantic similarity of the answer and expected answer.
    * Evaluator:
        - Answer semantic similarity
    * Connection:
        - RAG

* **Context precision** (RAG)
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/conferences/atlanta-2024/ragas_no_or_h2ogpte_test_lab.json
    * Name:
        * Context precision
    * Description
        - Context precision (RAGAS metric) can assess the quality of the retrieved context considering order and relevance of the text chunks on the context stack.
    * Evaluator:
        - Context precision
    * Connection:
        - RAG

* **Context recall** (RAG)
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/conferences/atlanta-2024/ragas_no_or_h2ogpte_test_lab.json
    * Name:
        * Context recall
    * Description
        - Context recall (RAGAS metric) can assess how much of the ground truth is represented in the retrieved context.
    * Evaluator:
        - Context recall
    * Connection:
        - RAG

* **Context relevance** (RAG)
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/conferences/atlanta-2024/ragas_no_or_h2ogpte_test_lab.json
    * Name:
        * Context relevance
    * Description
        - Context relevance can assess whether the context is (in)complete and does not contain redundant information which is not needed - noise.
    * Evaluator:
        - Context relevance
    * Connection:
        - RAG

* **Faithfulness**: (RAG)
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/conferences/atlanta-2024/ragas_no_or_h2ogpte_test_lab.json
    * Name:
        * Faithfulness
    * Description
        - Faithfulness (RAGAS metric) can assess whether answer claims can be inferred from the context i.e. factual consistency of the answer given the context (hallucinations).
    * Evaluator:
        - Faithfulness
    * Connection:
        - RAG

* **Hallucination** (RAG)
    * https://github.com/h2oai/h2o-sonar/blob/mvp/eval-studio/data/generative/conferences/atlanta-2024/ragas_no_or_h2ogpte_test_lab.json
    * Name:
        * Hallucination
    * Description
        - Hallucination metric can asses the RAG’s base LLM model hallucination.
    * Evaluator:
        - Hallucination
    * Connection:
        - RAG


