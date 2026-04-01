# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from abc import abstractmethod

import airium

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import problems


KEYWORD_LLM = explainers.Explainer.KEYWORD_LLM
# evaluator targets
KEYWORD_EVALUATES_LLM = explainers.Explainer.KEYWORD_EVALUATES_LLM
KEYWORD_EVALUATES_RAG = explainers.Explainer.KEYWORD_EVALUATES_RAG
# evaluator requirements
KEYWORD_RQ_J = explainers.Explainer.KEYWORD_RQ_J
KEYWORD_RQ_P = explainers.Explainer.KEYWORD_RQ_P
KEYWORD_RQ_EA = explainers.Explainer.KEYWORD_RQ_EA
KEYWORD_RQ_RC = explainers.Explainer.KEYWORD_RQ_RC
KEYWORD_RQ_AA = explainers.Explainer.KEYWORD_RQ_AA
KEYWORD_RQ_C = explainers.Explainer.KEYWORD_RQ_C
# evaluator capabilities
KEYWORD_CAP_AH = f"{explainers.Explainer.KEYWORD_PREFIX_CAPABILITY}-answer-highlight"
KEYWORD_CAP_CH = f"{explainers.Explainer.KEYWORD_PREFIX_CAPABILITY}-condition-highlight"

KEYWORD_REQUIRES_OPENAI_KEY = explainers.Explainer.KEYWORD_REQUIRES_OPENAI_KEY

# hardware requirements
# explainer requires GPU
KEYWORD_GPU_RQ: str = "hardware-gpu-required"
# explainer can use GPU if available
KEYWORD_GPU_OPT: str = "hardware-gpu-optional"

# KEYWORD GROUP prefixes
PREFIX_PROBLEM_TYPE: str = "problem-type-"
PREFIX_NIST_AI_RMF: str = "nist-ai-rmf-"
PREFIX_SR_11_7: str = "sr-11-7-"
PREFIX_EVAL_ROLE: str = "evaluator-role-"
PREFIX_ES_PURPOSE: str = "es-purpose-"
PREFIX_EVAL_METHOD: str = "evaluation-method-"
PREFIX_EVAL_METHOD_TYPE: str = "evaluation-type-"

# ML problem type keywords
# RAG/LLM response is a regression value only
KEYWORD_PROBLEM_TYPE_REG = f"{PREFIX_PROBLEM_TYPE}regression"
# RAG/LLM response is class identifier only
KEYWORD_PROBLEM_TYPE_CLS = f"{PREFIX_PROBLEM_TYPE}classification"
# RAG/LLM response is binomial class identifier only
KEYWORD_PROBLEM_TYPE_BIN = f"{PREFIX_PROBLEM_TYPE}binary-classification"
# RAG/LLM response is multinomial class identifier only
KEYWORD_PROBLEM_TYPE_MUL = f"{PREFIX_PROBLEM_TYPE}multiclass-classification"
# RAG/LLM response is a text retrieved from a document / LLM context / RAG
KEYWORD_PROBLEM_TYPE_IR = f"{PREFIX_PROBLEM_TYPE}information-retrieval"
# RAG/LLM response is a text summarization
KEYWORD_PROBLEM_TYPE_SUM = f"{PREFIX_PROBLEM_TYPE}summarization"
# RAG/LLM response is a text with an answer to a question from the prompt
KEYWORD_PROBLEM_TYPE_QA = f"{PREFIX_PROBLEM_TYPE}question-answering"

# NIST AI RMF keywords
# - https://www.nist.gov/itl/ai-risk-management-framework
KEYWORD_NIST_AI_RMF_S = f"{PREFIX_NIST_AI_RMF}safe"
KEYWORD_NIST_AI_RMF_SR = f"{PREFIX_NIST_AI_RMF}secure-and-resilient"
KEYWORD_NIST_AI_RMF_EI = f"{PREFIX_NIST_AI_RMF}explainable-and-interpretable"
KEYWORD_NIST_AI_RMF_PE = f"{PREFIX_NIST_AI_RMF}privacy-enhanced"
KEYWORD_NIST_AI_RMF_F = f"{PREFIX_NIST_AI_RMF}fair"
KEYWORD_NIST_AI_RMF_AT = f"{PREFIX_NIST_AI_RMF}accountable-and-transparent"
KEYWORD_NIST_AI_RMF_VR = f"{PREFIX_NIST_AI_RMF}valid-and-reliable"

# SR 11-7 keywords
# - https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
KEYWORD_SR_11_7_CS = f"{PREFIX_SR_11_7}conceptual-soundness"
KEYWORD_SR_11_7_OGA = f"{PREFIX_SR_11_7}ongoing-monitoring"
KEYWORD_SR_11_7_OA = f"{PREFIX_SR_11_7}outcomes-analysis"
KEYWORD_SR_11_7_BT = f"{PREFIX_SR_11_7}backtesting"

# evaluator roles keywords
KEYWORD_EVALUATOR_ROLE_REGULATOR = f"{PREFIX_EVAL_ROLE}regulator"

# H2O Eval Studio functional keywords
# ~ structure evaluators to disjunctive sets
# ~ every evaluator can belong to exactly one set
KEYWORD_ES_GENERATE = f"{PREFIX_ES_PURPOSE}generation"
KEYWORD_ES_RETRIEVE = f"{PREFIX_ES_PURPOSE}retrieval"
KEYWORD_ES_PRIVACY = f"{PREFIX_ES_PURPOSE}privacy"
KEYWORD_ES_FAIRNESS = f"{PREFIX_ES_PURPOSE}fairness"
KEYWORD_ES_SUMMARIZE = f"{PREFIX_ES_PURPOSE}summarization"
KEYWORD_ES_CLASSIFY = f"{PREFIX_ES_PURPOSE}classification"

# Evaluation Method
KEYWORD_METHOD_NLI = f"{PREFIX_EVAL_METHOD}nli"
KEYWORD_METHOD_NGRAM = f"{PREFIX_EVAL_METHOD}ngram"
KEYWORD_METHOD_SEMANTIC_SIMILARITY = f"{PREFIX_EVAL_METHOD}semantic-similarity"
KEYWORD_METHOD_JUDGE = f"{PREFIX_EVAL_METHOD}judge"
KEYWORD_METHOD_AGENTS = f"{PREFIX_EVAL_METHOD}agents"
KEYWORD_METHOD_RULE_BASED = f"{PREFIX_EVAL_METHOD}rule-based"

# Evaluation Method Type
KEYWORD_METHOD_TYPE_DETERMINISTIC = f"{PREFIX_EVAL_METHOD_TYPE}deterministic"
KEYWORD_METHOD_TYPE_NON_DETERMINISTIC = f"{PREFIX_EVAL_METHOD_TYPE}non-deterministic"


KEYWORD_GROUPS = commons.KeywordGroups(
    [
        commons.KeywordGroup(
            prefix=PREFIX_PROBLEM_TYPE,
            name="Problem Type",
            description="Evaluators by problem type.",
            keywords=[
                commons.Keyword(
                    key=KEYWORD_PROBLEM_TYPE_REG,
                    name="Regression",
                    description=(
                        "Model evaluation involves assessing how well a machine "
                        "learning model predicts continuous outcomes. Metrics like "
                        "mean squared error (MSE), mean absolute error (MAE), "
                        "and R-squared are commonly used to measure the model's "
                        "performance. These metrics quantify the difference between "
                        "the predicted values and the actual values, providing "
                        "insights into the accuracy and reliability of the regression "
                        "model."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_PROBLEM_TYPE_CLS,
                    name="Classification",
                    description=(
                        "Model evaluation is about assessing how well a machine "
                        "learning model categorizes data into predefined classes. It "
                        "involves measuring metrics like accuracy, precision, recall, "
                        "and F1 score to understand the model's performance. Accuracy "
                        "tells us the proportion of correctly classified instances, "
                        "precision focuses on the accuracy of positive predictions, "
                        "recall assesses how well the model identifies positive "
                        "instances, and F1 score combines precision and recall into a "
                        "single metric for balanced evaluation. These metrics help in "
                        "gauging the model's effectiveness in classification tasks."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_PROBLEM_TYPE_BIN,
                    name="Binary Classification",
                    description=(
                        "Binary classification in model evaluation involves assessing "
                        "the performance of a machine learning model that categorizes "
                        "data into two distinct classes or categories. Commonly used "
                        "metrics for evaluating binary classification models include "
                        "accuracy, precision, recall, F1 score, and the receiver "
                        "operating characteristic (ROC) curve with its associated "
                        "area under the curve (AUC). These metrics help measure the "
                        "model's ability to distinguish between the two classes and "
                        "make correct predictions, providing insights into its "
                        "effectiveness for binary classification tasks."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_PROBLEM_TYPE_MUL,
                    name="Multiclass Classification",
                    description=(
                        "Focuses on assessing how accurately a machine learning model "
                        "assigns instances to multiple predefined classes. Metrics "
                        "like accuracy, precision, recall, and F1 score are utilized "
                        "to measure the model's performance across all classes. These "
                        "metrics provide insights into the model's ability to "
                        "correctly classify instances belonging to each class, "
                        "helping to gauge its overall effectiveness in multiclass "
                        "classification tasks."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_PROBLEM_TYPE_IR,
                    name="Information Retrieval",
                    description=(
                        "Information retrieval techniques evaluate how well a system "
                        "retrieves relevant data in response to user queries from a "
                        "large dataset. Metrics like precision, recall, and F1 score "
                        "are used to measure retrieval effectiveness. Precision "
                        "assesses the accuracy of retrieved results, recall measures "
                        "the completeness of retrieval, and the F1 score balances "
                        "both aspects for a comprehensive evaluation."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_PROBLEM_TYPE_SUM,
                    name="Summarization",
                    description=(
                        "Summarization technique in model evaluation assesses the "
                        "ability of a system to generate concise and informative "
                        "summaries from large pieces of text or data. Metrics like "
                        "ROUGE (Recall-Oriented Understudy for Gisting Evaluation) "
                        "scores are commonly used to evaluate the quality of "
                        "summaries by measuring overlap with reference summaries. "
                        "Higher ROUGE scores indicate better performance in capturing "
                        "important information and maintaining coherence in the "
                        "generated summaries."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_PROBLEM_TYPE_QA,
                    name="Question Answering",
                    description=(
                        "Model evaluation involves testing how effectively a system "
                        "can understand questions posed in natural language and "
                        "provide accurate answers based on the information available "
                        "to it. This evaluation typically measures the system's "
                        "ability to comprehend the questions, locate relevant "
                        "information, and generate responses that are correct and "
                        "contextually appropriate. It often utilizes datasets "
                        "containing questions paired with their corresponding answers "
                        "to assess the system's performance using metrics such as "
                        "accuracy, precision, recall, and F1 score."
                    ),
                ),
            ],
        ),
        commons.KeywordGroup(
            prefix=PREFIX_NIST_AI_RMF,
            name="NIST AI RMF",
            description="Evaluators by NIST AI RMF.",
            keywords=[
                commons.Keyword(
                    key=KEYWORD_NIST_AI_RMF_S,
                    name="Safe",
                    description=(
                        "AI systems should not under defined conditions, lead to a "
                        "state in which human life, health, property, or the "
                        "environment is endangered. Safe operation of AI systems is "
                        "improved through: responsible design, development, "
                        "and deployment practices, clear information to deployers on "
                        "responsible use of the system, responsible decision-making "
                        "by deployers and end users; and explanations and "
                        "documentation of risks based on empirical evidence of "
                        "incidents."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_NIST_AI_RMF_SR,
                    name="Secure and Resilient",
                    description=(
                        "AI systems, as well as the ecosystems in which they are "
                        "deployed, may be said to be resilient if they can "
                        "withstand unexpected adverse events or unexpected changes "
                        "in their environment or use - or if they can maintain "
                        "their functions and structure in the face of internal and "
                        "external change and degrade safely and gracefully when this "
                        "is necessary."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_NIST_AI_RMF_EI,
                    name="Explainable and Interpretable",
                    description=(
                        "Explainable and interpretable AI systems are those that can "
                        "provide clear and understandable explanations for their "
                        "decisions or actions. This means that users can understand "
                        "why a system made a particular choice, rather than just "
                        "knowing what the choice was."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_NIST_AI_RMF_PE,
                    name="Privacy Enhanced",
                    description=(
                        "Privacy refers generally to the norms and practices that "
                        "help to safeguard human autonomy, identity, and dignity. "
                        "These norms and practices typically address freedom from "
                        "intrusion, limiting observation, or individuals' agency to "
                        "consent to disclosure or control of facets of their "
                        "identities (e.g., body, data, reputation). Privacy values "
                        "such as anonymity, confidentiality, and control generally "
                        "should guide choices for AI system design, development, "
                        "and deployment."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_NIST_AI_RMF_F,
                    name="Fair",
                    description=(
                        "Fairness in AI includes concerns for equality and equity by "
                        "addressing issues such as harmful bias and discrimination. "
                        "Standards of fairness can be complex and difficult to define "
                        "because perceptions of fairness differ among cultures and "
                        "may shift depending on application. Organizations' risk "
                        "management efforts will be enhanced by recognizing and "
                        "considering these differences. Systems in which harmful "
                        "biases are mitigated are not necessarily fair. For example, "
                        "systems in which predictions are somewhat balanced across "
                        "demographic groups may still be inaccessible to individuals "
                        "with disabilities or affected by the digital divide or may "
                        "exacerbate existing disparities or systemic biases."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_NIST_AI_RMF_AT,
                    name="Accountable and Transparent",
                    description=(
                        "Trustworthy AI depends upon accountability. Accountability "
                        "presupposes transparency. Transparency reflects the extent "
                        "to which information about an AI system and its outputs is "
                        "available to individuals interacting with such a system - "
                        "regardless of whether they are even aware that they are "
                        "doing so. Meaningful transparency provides access to "
                        "appropriate levels of information based on the stage of the "
                        "AI lifecycle and tailored to the role or knowledge of AI "
                        "actors or individuals interacting with or using the AI "
                        "system. By promoting higher levels of understanding, "
                        "transparency increases confidence in the AI system."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_NIST_AI_RMF_VR,
                    name="Valid and Reliable",
                    description=(
                        "Validity and reliability for deployed AI systems are often "
                        "assessed by ongoing testing or monitoring that confirms a "
                        "system is performing as intended. Measurement of validity, "
                        "accuracy, robustness, and reliability contribute to "
                        "trustworthiness and should take into consideration that "
                        "certain types of failures can cause greater harm."
                    ),
                ),
            ],
        ),
        commons.KeywordGroup(
            prefix=PREFIX_SR_11_7,
            name="SR 11-7",
            description="SR-11-7 evaluators.",
            keywords=[
                commons.Keyword(
                    key=KEYWORD_SR_11_7_CS,
                    name="Conceptual Soundness",
                    description=(
                        "Involves assessing the quality of the model design and "
                        "construction. It entails review of documentation and "
                        "empirical evidence supporting the methods used and variables "
                        "selected for the model."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_SR_11_7_OGA,
                    name="Ongoing Monitoring",
                    description=(
                        "Emphasizes the continuous evaluation of a model's "
                        "performance after deployment. This involves tracking the "
                        "model's outputs against real-world data, identifying any "
                        "deviations or unexpected results, and assessing if the "
                        "model's underlying assumptions or market conditions have "
                        "changed. This ongoing process ensures the model remains "
                        "reliable and trustworthy for decision-making."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_SR_11_7_OA,
                    name="Outcomes Analysis",
                    description=(
                        "Comparison of model outputs to corresponding actual "
                        "outcomes. Outcomes analysis typically relies on statistical "
                        "tests or other quantitative measures. It can also include "
                        "expert judgment to check the intuition behind the outcomes "
                        "and confirm that the results make sense."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_SR_11_7_BT,
                    name="Backtesting",
                    description=(
                        "Backtesting is a method used to evaluate the performance of "
                        "an AI system on historical data."
                    ),
                ),
            ],
        ),
        commons.KeywordGroup(
            prefix=PREFIX_EVAL_ROLE,
            name="User Role",
            description="Evaluators by relevance for specific user roles.",
            keywords=[
                commons.Keyword(
                    key=KEYWORD_EVALUATOR_ROLE_REGULATOR,
                    name="Regulator",
                    description=(
                        "Regular persona evaluates the compliance of RAGs/LLMs with "
                        "regulatory requirements related to data privacy, bias, "
                        "and ethical considerations The persona is RAG/LLM end user "
                        "who is directly interacting with the system and "
                        "exploring/defining different approaches to regulate "
                        "RAGs/LLMs use in regulated institutions - like banks, "
                        "insurance companies, brokerage firms, hedge funds, private "
                        "equity firms, exchanges, pension funds, and foundations - "
                        "and domains - like securities (stocks, bonds, options, "
                        "futures), derivatives (swaps, forwards, futures), credit ("
                        "loans, mortgages, credit cards), commodities (oil, gas, "
                        "metals), money and banking (interest rates, exchange rates), "
                        "insurance (property and casualty, life, health), investments "
                        "(portfolio management, financial planning), pensions and "
                        "employee benefits. Some are taking a hands-off approach, "
                        "while others are developing specific rules and regulations."
                    ),
                ),
            ],
        ),
        commons.KeywordGroup(
            prefix=PREFIX_ES_PURPOSE,
            name="Purpose",
            description="Evaluators by H2O Eval Studio purpose.",
            keywords=[
                commons.Keyword(
                    key=KEYWORD_ES_GENERATE,
                    name="Generation",
                    description=(
                        "AI systems designed for generation tasks are evaluated based "
                        "on their ability to produce human-like outputs, such as "
                        "text, images, or music. Evaluation metrics for generative it "
                        "includes metrics like RAGAS, perplexity, or answer "
                        "similarity to assess the quality and coherence of the "
                        "generated content These metrics help in understanding the "
                        "model’s performance in generating realistic and contextually "
                        "relevant outputs."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_ES_RETRIEVE,
                    name="Retrieval",
                    description=(
                        "Retrieval-based AI systems are assessed on their capacity to "
                        "retrieve relevant information from a large dataset in "
                        "response to user queries or prompts. Evaluation metrics for "
                        "retrieval include context relevancy or context precision. "
                        "These metrics help in evaluating the system's effectiveness "
                        "in retrieving relevant information for user queries."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_ES_PRIVACY,
                    name="Privacy",
                    description=(
                        "Privacy-focused AI systems are evaluated based on their "
                        "ability to protect sensitive user data and maintain "
                        "confidentiality. Evaluation criteria for privacy-enhancing "
                        "models include data anonymization, encryption techniques, "
                        "and compliance with privacy regulations. These criteria help "
                        "in assessing the system’s capacity to safeguard user privacy "
                        "and prevent unauthorized access to personal information."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_ES_FAIRNESS,
                    name="Fairness",
                    description=(
                        "Fairness evaluation in AI systems examines the presence of "
                        "bias, discrimination, or disparities in model predictions "
                        "and outcomes across different demographic groups. A typical "
                        "method in the category is the Fairness Bias evaluator."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_ES_SUMMARIZE,
                    name="Summarization",
                    description=(
                        "Summarization metrics evaluate models on their ability to "
                        "condense large volumes of text or data into concise and "
                        "coherent summaries. Evaluation metrics for summarization "
                        "tasks include ROUGE or BLEU scores to measure the quality, "
                        "relevance, and coherence of the generated summaries. These "
                        "metrics help in assessing the system’s performance in "
                        "producing informative and well-structured summaries."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_ES_CLASSIFY,
                    name="Classification",
                    description=(
                        "Classification metrics assess models on their capacity to "
                        "categorize input data into predefined classes or categories. "
                        "Evaluation metrics for classification tasks include "
                        "accuracy, precision, recall, or F1 score to measure the "
                        "model’s performance in classifying instances correctly."
                    ),
                ),
            ],
        ),
        commons.KeywordGroup(
            prefix=PREFIX_EVAL_METHOD,
            name="Evaluation Method",
            description="Evaluators by the evaluation method.",
            keywords=[
                commons.Keyword(
                    key=KEYWORD_METHOD_NLI,
                    name="Natural Language Inference",
                    description=(
                        "Natural Language Inference (NLI) is a method used to "
                        "evaluate the ability of a language model (LM) to understand "
                        "and reason about text. In the context of LLM evaluation, "
                        "it involves presenting the model with a pair of sentences: a "
                        "premise and a hypothesis. The model must then determine if "
                        "the hypothesis is a logical consequence of the premise."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_METHOD_NGRAM,
                    name="N-gram",
                    description=(
                        "N-grams are sequences of n words that appear consecutively "
                        "in a text. In the context of LLM evaluation, n-grams are "
                        "often used to measure the fluency and coherence of the "
                        "generated text. By comparing the n-grams in the generated "
                        "text to those in a high-quality reference corpus, "
                        "we can assess how well the model has learned the statistical "
                        "patterns of natural language. A higher overlap between the "
                        "n-grams in the generated text and the reference corpus "
                        "generally indicates a more fluent and coherent output."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_METHOD_SEMANTIC_SIMILARITY,
                    name="Semantic Similarity",
                    description=(
                        "Semantic similarity is a measure of how closely related two "
                        "pieces of text are in terms of their meaning. In the context "
                        "of the evaluation, semantic similarity is often assessed "
                        "using embeddings and cosine similarity."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_METHOD_JUDGE,
                    name="Judge",
                    description=(
                        "LLM judge is a method of evaluating by having it interact "
                        "with LLM or RAG judges. These judges assess the quality of "
                        "the evaluated model's responses using well designed "
                        "prompts to assess various criteria."
                    ),
                ),
                commons.Keyword(
                    key=KEYWORD_METHOD_RULE_BASED,
                    name="Rule Based",
                    description=(
                        "Rule-based evaluation is a method of evaluation by comparing "
                        "the system output to a set of predefined rules or criteria. "
                        "These rules can be based on linguistic principles, "
                        "domain-specific knowledge, or other relevant factors."
                    ),
                ),
            ],
        ),
        commons.KeywordGroup(
            prefix=PREFIX_EVAL_METHOD_TYPE,
            name="Evaluation Method Type",
            description="Evaluators by the evaluation method type.",
            keywords=[
                commons.Keyword(
                    key=KEYWORD_METHOD_TYPE_DETERMINISTIC,
                    name="Deterministic",
                    description="",
                ),
                commons.Keyword(
                    key=KEYWORD_METHOD_TYPE_NON_DETERMINISTIC,
                    name="Non-deterministic",
                    description="",
                ),
            ],
        ),
    ]
)


class EvaluatorParam(explainers.ExplainerParam):
    def __init__(
        self,
        param_name: str,
        param_type: commons.EvaluatorParamType,
        description: str = "",
        comment: str = "",
        default_value: bool | str | float = "",
        value_min: float = 0.0,
        value_max: float = 0.0,
        predefined: list | None = None,
        tags: list | None = None,
        category: str = "",
        src: str = "",
    ):
        explainers.ExplainerParam.__init__(
            self,
            param_name=param_name,
            param_type=param_type,
            description=description,
            comment=comment,
            default_value=default_value,
            value_min=value_min,
            value_max=value_max,
            predefined=predefined,
            tags=tags,
            category=category,
            src=src,
        )


class EvaluatorDescriptor(explainers.ExplainerDescriptor):
    def __init__(
        self,
        id: str,
        name: str = "",
        display_name: str = "",
        tagline: str = "",
        description: str = "",
        brief_description: str = "",
        model_types: list[str] | None = None,
        can_explain: list[str] | None = None,
        explanation_scopes: list[str] | None = None,
        explanations: list[e10s.ExplanationDescriptor] | None = None,
        parameters: list[commons.ConfigItem] | None = None,
        keywords: list[str] | None = None,
        metrics_meta: commons.MetricsMeta | None = None,
    ):
        explainers.ExplainerDescriptor.__init__(
            self,
            id=id,
            name=name,
            display_name=display_name,
            tagline=tagline,
            description=description,
            brief_description=brief_description,
            model_types=model_types,
            can_explain=can_explain,
            explanation_scopes=explanation_scopes,
            explanations=explanations,
            parameters=parameters,
            keywords=keywords,
            metrics_meta=metrics_meta,
        )


class Evaluator(explainers.Explainer):
    """Evaluator abstract class."""

    # abstract method implementation:
    #
    # - evaluator.evaluate() ... ABC
    # - evaluator.explain() -> evaluator.evaluate() ... bridge for container
    #

    PARAM_SAVE_LLM_RESULT = "save_llm_result"
    DEFAULT_SAVE_LLM_RESULT = True
    _PARAM_SAVE_LLM_RESULT = EvaluatorParam(
        param_name=PARAM_SAVE_LLM_RESULT,
        description=(
            "Control whether to save LLM result which contains input LLM dataset "
            "and all metrics calculated by the evaluator."
        ),
        param_type=commons.EvaluatorParamType.bool,
        default_value=DEFAULT_SAVE_LLM_RESULT,
        src=EvaluatorParam.SRC_EVALUATOR_PARAMS,
    )

    PARAM_METRIC_THRESHOLD = "metric_threshold"
    DEFAULT_METRIC_THRESHOLD = 0.75

    PARAM_MIN_TEST_CASES = "min_test_cases"

    PARAM_EVAL_JUDGE_CFG_KEY = "custom_eval_judge_config_key"
    DEFAULT_EVAL_JUDGE_CFG_KEY = ""
    _PARAM_EVAL_JUDGE = EvaluatorParam(
        param_name=PARAM_EVAL_JUDGE_CFG_KEY,
        description=(
            "Configuration key of the custom (LLM) judge to be used for the evaluation."
        ),
        param_type=commons.EvaluatorParamType.str,
        default_value=DEFAULT_EVAL_JUDGE_CFG_KEY,
        src=EvaluatorParam.SRC_EVALUATOR_PARAMS,
    )

    PARAM_SENTENCE_LEVEL_METRICS = "sentence_level_metrics"
    DEFAULT_SENTENCE_LEVEL_METRICS = True
    _PARAM_SENTENCE_LEVEL_METRICS = EvaluatorParam(
        param_name=PARAM_SENTENCE_LEVEL_METRICS,
        description="Controls whether sentence level metrics are generated.",
        param_type=commons.EvaluatorParamType.bool,
        default_value=DEFAULT_SENTENCE_LEVEL_METRICS,
        src=EvaluatorParam.SRC_EVALUATOR_PARAMS,
    )

    PARAM_NAN_TOLERANCE = "nan_tolerance"
    DEFAULT_NAN_TOLERANCE = 0.2  # 20% of NaN values are allowed
    _PARAM_NAN_TOLERANCE = EvaluatorParam(
        param_name=PARAM_NAN_TOLERANCE,
        description=(
            "Control whether to allow filtering out NaN values from the evaluation "
            "results for the average metrics score calculation. If the number of NaN "
            "values is lower than the specified tolerance, the NaN values are filtered"
            "out and ignored when the average score is calculated. If the tolerance "
            "is exceeded, the NaN values are not filtered out and the average score "
            "may be NaN."
        ),
        param_type=commons.EvaluatorParamType.float,
        default_value=DEFAULT_NAN_TOLERANCE,
        src=EvaluatorParam.SRC_EVALUATOR_PARAMS,
    )

    def __init__(self):
        explainers.Explainer.__init__(self)

        # integrity check: metadata vs. keywords (H2O Sonar products users requirements)
        assert self._llm == (KEYWORD_EVALUATES_LLM in self.keywords), (
            f"LLM evaluator must have '{KEYWORD_EVALUATES_LLM}' keyword in the "
            f"keywords list in {self.display_name} ({self.evaluator_id()})."
        )
        if self._llm:
            assert KEYWORD_RQ_RC not in self.keywords, (
                f"Retrieved context (keyword) is not allowed for LLM evaluators in "
                f"{self.display_name} ({self.evaluator_id()})."
            )
        assert self._rag == (KEYWORD_EVALUATES_RAG in self.keywords), (
            f"RAG evaluator must have '{KEYWORD_EVALUATES_RAG}' keyword in the "
            f"keywords list in {self.display_name} ({self.evaluator_id()})."
        )

    def get_evaluation_metrics(self) -> commons.MetricsMeta:
        return self._metrics_meta

    def _resolve_evaluator_params(self):
        self._resolve_explainer_params()

    def _resolve_judge_key(self):
        judge_key = self.args.get(
            Evaluator.PARAM_EVAL_JUDGE_CFG_KEY,
            Evaluator.DEFAULT_EVAL_JUDGE_CFG_KEY,
        )
        if not judge_key and h2o_sonar_config.config.force_eval_judge:
            if isinstance(h2o_sonar_config.config.force_eval_judge, str):
                if str(h2o_sonar_config.config.force_eval_judge).lower() == "true":
                    # find the first judge in the configuration
                    judge_cfg = h2o_sonar_config.config.get_evaluation_judge()
                    if judge_cfg:
                        judge_key = judge_cfg.key
                    else:
                        raise ValueError(
                            "Cannot use a default custom (LLM) judge for "
                            "the evaluation as no evaluation judge configuration "
                            "found in the H2O Sonar configuration."
                        )
                elif str(h2o_sonar_config.config.force_eval_judge).lower() == "false":
                    judge_key = ""
                else:
                    judge_key = h2o_sonar_config.config.force_eval_judge
            else:
                raise ValueError(
                    f"Invalid value type for force_eval_judge: "
                    f"{h2o_sonar_config.config.force_eval_judge} "
                    f"({type(h2o_sonar_config.config.force_eval_judge)})"
                )

        return judge_key

    @staticmethod
    def _eval_row_progress_msg(
        metric_name: str,
        device,
        row: int,
        total_rows: int,
    ) -> str:
        device_str = f"({device}) " if device else ""
        return (
            f"Build > config > run '{metric_name}' {device_str} evaluation for input "
            f"{row}/{total_rows} "
        )

    @staticmethod
    def _is_internal_err_answer(answer: str) -> bool:
        """Detect whether the provided answer is an internal error message
        indicating that the test lab was unable to get actual answer from the
        RAG/LLM model host.

        Parameters
        ----------
        answer : str
            Answer to be checked.

        Returns
        -------
        bool
            `True` if the answer is an internal error message, `False` otherwise.

        """
        return answer and (
            (commons.ERROR_MODEL_HOST in answer)
            or answer.startswith(commons.ERROR_LLM_HOST)
            or answer.startswith(commons.LEGACY_LLM_HOST)
        )

    @staticmethod
    def _internal_err_answer_msg(err_msg: str) -> str:
        """Get the internal error message indicating that the test lab was unable
        to get actual answer from the RAG/LLM model host.

        Returns
        -------
        str
            Internal error message.

        """
        return (
            f"The test lab was unable to get actual answer from the evaluated RAG/LLM "
            f"model host: {err_msg}"
        )

    @staticmethod
    def _internal_err_answer_msg_html(err_msg: str) -> airium.Airium:
        """Get the internal error message HTML indicating that the test lab was
        unable to get actual answer from the RAG/LLM host.

        Returns
        -------
        airium.Airium
            Internal error message HTML.

        """
        html = airium.Airium()
        html(
            "The test lab was unable to get actual answer from the evaluated RAG/LLM "
            "model host: "
        )
        with html.code():
            html(f"{err_msg}")
        return html

    def _problem_aa_with_err(
        self,
        error_msg: str,
        prompt: str,
        llm_model_name: str,
        check_error_messages: bool,
        t_explanation,
        row_key: str,
        model_key: str,
    ):
        html = airium.Airium()
        html(
            "Evaluation data contain error message from the RAG/LLM host "
            "which indicates that the test lab was "
        )
        with html.b(klass="w3-black"):
            html("&nbsp;unable to get actual answer&nbsp;")
        if llm_model_name:
            html("&nbsp;from the model ")
            with html.code():
                html(f"{llm_model_name}")
        html("for the prompt '")
        with html.i():
            html(f"{prompt}")
        html("'. ")
        if check_error_messages:
            with html.details():
                with html.summary():
                    html("Error message")
                with html.code():
                    html(f"{error_msg}")

        problems.ProblemAndAction(
            description=(
                f"Evaluation data - actual answer - contains error message from the "
                f"RAG/LLM host which indicates that the test lab was "
                f"unable to get actual answer "
                f"{'from the model ' if llm_model_name else ''}"
                f"{llm_model_name if llm_model_name else ''} "
                f"for the prompt '{prompt}'. "
                f"{'Error message:' if check_error_messages else ''}"
                f"{error_msg if check_error_messages else ''}"
            ),
            description_html=html,
            severity=problems.ProblemSeverity.high,
            problem_type="dataset",
            problem_attrs={
                problems.ProblemAndAction.ATTR_ROW_KEYS: [(row_key, model_key)],
                # input dataset ~ test lab ~ key is the test case key
                problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [row_key],
            },
            actions_description=(
                "Provide a valid LLM dataset containing actual answers "
                "generated by the evaluated RAG/LLM model."
            ),
            evaluator_id=self.evaluator_id(),
            evaluator_name=self._display_name,
            explanation_type=t_explanation.explanation_type(),
            explanation_name=t_explanation.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
            resources=[],
        )

    def _problem_ea_missing(
        self,
        evaluator_name: str,
        prompt: str,
        t_explanation,
        row_key: str,
        model_key: str,
    ):
        html = airium.Airium()
        html(
            f"{evaluator_name} evaluator cannot evaluate the provided LLM dataset as "
            f"it does "
        )
        with html.b(klass="w3-black"):
            html("&nbsp;not contain expected answers&nbsp;")
        html("&nbsp;for the prompt: ")
        with html.i():
            html(f"'{prompt}'")
        html("&nbsp;which is required by the evaluator.")

        return problems.ProblemAndAction(
            description=(
                f"{evaluator_name} evaluator cannot evaluate "
                f"the provided LLM dataset as it does not contain "
                f"expected answers which are required by "
                f"the evaluator."
            ),
            description_html=html,
            severity=problems.ProblemSeverity.high,
            problem_type="dataset",
            problem_attrs={
                problems.ProblemAndAction.ATTR_ROW_KEYS: [(row_key, model_key)],
                # input dataset ~ test lab ~ key is the test case key
                problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [row_key],
            },
            actions_description=(
                "Provide a valid LLM dataset containing expected answers."
            ),
            evaluator_id=self.evaluator_id(),
            evaluator_name=self._display_name,
            explanation_type=t_explanation.explanation_type(),
            explanation_name=t_explanation.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
            resources=[],
        )

    def _problem_aa_type_mismatch(
        self,
        actual_output: str,
        prompt: str,
        llm_model_name: str,
        t_explanation,
        row_key: str,
        model_key: str,
    ) -> problems.ProblemAndAction:
        wrong_type = f"{type(actual_output)}".replace("<", "").replace(">", "")

        html = airium.Airium()
        html("Actual answer provided by evaluated RAG/LLM has  ")
        with html.b(klass="w3-black"):
            html("&nbsp;wrong type&nbsp;")
        html(
            f"&nbsp;- it must be string, but it is of '{wrong_type}' type. "
            f"Generated for prompt "
        )
        with html.i():
            html(f"'{prompt}'")
        if llm_model_name:
            html("&nbsp;by LLM ")
            with html.code():
                html(f"{llm_model_name}")
        html(".")

        return problems.ProblemAndAction(
            description=(
                f"One of the actual answers provided by evaluated RAG/LLM has wrong "
                f"type which cannot be used by most of the evaluators - the actual "
                f"answer must be string, but it is of '{type(actual_output)}' type. "
                f"Generated for prompt '{prompt}'"
                f"{' by LLM ' if llm_model_name else '.'}"
                f"{llm_model_name if llm_model_name else ''}"
            ),
            description_html=html,
            severity=problems.ProblemSeverity.high,
            problem_type="dataset",
            problem_code=problems.AVIDProblemCode.P0100_DATA,
            problem_attrs={
                problems.ProblemAndAction.ATTR_ROW_KEYS: [(row_key, model_key)],
                # input dataset ~ test lab ~ key is the test case key
                problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [row_key],
            },
            actions_description=(
                "Provide a valid LLM dataset containing actual answers "
                "generated by the evaluated RAG/LLM model."
            ),
            evaluator_id=self.evaluator_id(),
            evaluator_name=self._display_name,
            explanation_type=t_explanation.explanation_type(),
            explanation_name=t_explanation.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
            resources=[],
        )

    def _check_llm_dataset_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        evaluator_keywords: list[str] | None = None,
        check_error_messages: bool = True,
        check_empty_contexts: bool = False,
        fail_on_all_empty_contexts: bool = False,
    ) -> bool:
        """Check whether provided dataset is LLM dataset and whether the data
        in the dataset are RAG/LLM actual answers or error messages indicating that
        test lab was unable to complete the test suite.

        Parameters
        ----------
        params: commons.CommonInterpretationParams | None
            Common interpretation parameters.
        evaluator_keywords: Optional[list[str]
            Keywords of the evaluator.
        check_error_messages: bool
            Whether to check that actual answers contain H2O Sonar host clients error
            messages instead of the actual data.
        check_empty_contexts: bool
            Whether to check that the dataset rows have empty retrieval contexts.
            Problems for rows with empty contexts are reported, but the method
            always returns `True` - may return `False` if `fail_on_all_empty_contexts`
            is set to `True`.
        fail_on_all_empty_contexts: bool
            Whether to return ``False`` if the dataset contains ALL rows with empty
            retrieval contexts.

        Returns
        -------
        bool
            `True` if the dataset is compatible with the evaluator, `False` otherwise.

        """
        if (
            not params
            or not isinstance(params, commons.CommonInterpretationParams)
            or not params.dataset
            or not isinstance(params.dataset, datasets.LlmDataset)
        ):
            return True

        self.logger = self.logger or loggers.SonarPrintLogger()
        healthy_inputs_count = non_empty_ctxs_count = len(params.dataset.inputs)
        empty_ctxs_rows = []
        dataset_problems = []
        try:
            t_explanation = e10s.LlmEvalResultsExplanation

            key_to_model = {m.key: m for m in params.models}

            for i in params.dataset.inputs:
                if check_empty_contexts and i and not i.context:
                    non_empty_ctxs_count -= 1
                    empty_ctxs_rows.append(i)

                if i and not i.expected_output and KEYWORD_RQ_EA in evaluator_keywords:
                    self.add_problem(
                        self._problem_ea_missing(
                            evaluator_name=self._display_name,
                            prompt=i.i,
                            t_explanation=t_explanation,
                            row_key=i.key,
                            model_key=i.model_key,
                        )
                    )
                    self.logger.error(
                        f"Compatibility check - evaluator {self._display_name} is "
                        f"INCOMPATIBLE: provided LLM dataset does not contain "
                        f"expected answer(s) which are required by the evaluator for "
                        f"prompt '{i.i}'."
                    )
                    return False

                if i and i.actual_output:
                    if not isinstance(i.actual_output, str):
                        llm_model_name = (
                            key_to_model[i.model_key].llm_model_name
                            if key_to_model.get(i.model_key, {})
                            else ""
                        )

                        self.add_problem(
                            self._problem_aa_type_mismatch(
                                actual_output=i.actual_output,
                                prompt=i.i,
                                llm_model_name=llm_model_name,
                                t_explanation=t_explanation,
                                row_key=i.key,
                                model_key=i.model_key,
                            )
                        )

                        # fail the evaluation as evaluators which need actual answer
                        # will crash
                        self.logger.error(
                            f"Compatibility check - evaluator {self._display_name} is "
                            f"INCOMPATIBLE: provided LLM dataset does not contain "
                            f"actual answer(s) which are required by the evaluator for "
                            f"prompt '{i.i}'."
                        )
                        return False

                    if Evaluator._is_internal_err_answer(i.actual_output):
                        healthy_inputs_count -= 1
                        error_msg = i.actual_output
                        llm_model_name = (
                            key_to_model[i.model_key].llm_model_name
                            if key_to_model.get(i.model_key, {})
                            else ""
                        )

                        dataset_problems.append(
                            self._problem_aa_with_err(
                                error_msg=error_msg,
                                prompt=i.i,
                                llm_model_name=llm_model_name,
                                check_error_messages=check_error_messages,
                                t_explanation=t_explanation,
                                row_key=i.key,
                                model_key=i.model_key,
                            )
                        )
        except Exception as e:
            self.logger.error(
                f"Compatibility check - evaluator {self._display_name} - error while "
                f"checking LLM dataset compatibility: {str(e)}"
            )

        t_eval_results = e10s.LlmEvalResultsExplanation

        if non_empty_ctxs_count <= 0:
            self.add_problem(
                problems.ProblemAndAction(
                    description=(
                        f"{self._display_name} evaluator cannot evaluate the provided "
                        f"LLM dataset as it does not contain retrieved contexts which "
                        f"is required by the evaluator."
                    ),
                    severity=problems.ProblemSeverity.high,
                    problem_type="dataset",
                    actions_description=(
                        "Provide a valid LLM dataset containing retrieved contexts. "
                        "Check whether RAG/LLM host (client) supports retrieval "
                        "context introspection, whether there were any errors during "
                        "retrieval context generation and test lab build."
                    ),
                    evaluator_id=self.evaluator_id(),
                    evaluator_name=self._display_name,
                    explanation_type=t_eval_results.explanation_type(),
                    explanation_name=t_eval_results.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                    resources=[],
                )
            )
            if fail_on_all_empty_contexts:
                self.logger.error(
                    f"Compatibility check - evaluator {self._display_name} is "
                    f"INCOMPATIBLE: provided LLM dataset does not contain "
                    f"retrieved contexts (all test cases) which are required by the "
                    f"evaluator."
                )
                return False

        # ensure the worst problem is the first
        if healthy_inputs_count <= 0:
            self.add_problem(
                problems.ProblemAndAction(
                    description=(
                        f"{self._display_name} evaluator cannot evaluate the provided "
                        f"LLM dataset as it does not contain any healthy inputs with "
                        f"actual answers generated by the evaluated RAG/LLM model. "
                        f"Actual answers contain error messages instead."
                    ),
                    severity=problems.ProblemSeverity.high,
                    problem_type="dataset",
                    actions_description=(
                        "Provide a valid LLM dataset containing actual answers "
                        "generated by the evaluated RAG/LLM model. Check also other "
                        "problems for more details."
                    ),
                    evaluator_id=self.evaluator_id(),
                    evaluator_name=self._display_name,
                    explanation_type=t_eval_results.explanation_type(),
                    explanation_name=t_eval_results.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                    resources=[],
                )
            )
        elif healthy_inputs_count != len(params.dataset.inputs):
            self.add_problem(
                problems.ProblemAndAction(
                    description=(
                        f"Provided LLM dataset contains "
                        f"{len(params.dataset.inputs) - healthy_inputs_count} out of "
                        f"{len(params.dataset.inputs)} invalid inputs which do not "
                        f"contain actual answers generated by the evaluated RAG/LLM "
                        f"model, but error messages instead. This will negatively "
                        f"impact the evaluation results of the {self._display_name} "
                        f"evaluator."
                    ),
                    severity=problems.ProblemSeverity.high,
                    problem_type="dataset",
                    actions_description=(
                        "Provide a valid LLM dataset containing actual answers "
                        "generated by the evaluated RAG/LLM model. Check also other "
                        "problems for more details."
                    ),
                    evaluator_id=self.evaluator_id(),
                    evaluator_name=self._display_name,
                    explanation_type=t_eval_results.explanation_type(),
                    explanation_name=t_eval_results.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                    resources=[],
                )
            )

        if empty_ctxs_rows:
            for i in empty_ctxs_rows:
                self.add_problem(
                    problems.ProblemAndAction(
                        description=(
                            f"Provided LLM dataset contains {len(empty_ctxs_rows)} "
                            f"out of {len(params.dataset.inputs)} rows with empty "
                            f"retrieval contexts. Retrieval context for the prompt "
                            f"'{i.i}' was not generated. This will negatively impact "
                            f"the evaluation results of the {self._display_name} "
                            f"evaluator and it may provide incorrect metrics values."
                        ),
                        severity=problems.ProblemSeverity.high,
                        problem_type="dataset",
                        problem_attrs={
                            problems.ProblemAndAction.ATTR_ROW_KEYS: [
                                (i.key, i.model_key)
                            ],
                            # input dataset ~ test lab ~ key is the test case key
                            problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [i.key],
                        },
                        actions_description=(
                            "Provide a valid LLM dataset containing retrieved "
                            "contexts. Check whether RAG/LLM host (client) supports "
                            "retrieval context introspection, whether there were any "
                            "errors during retrieval context generation and test lab "
                            "build."
                        ),
                        evaluator_id=self.evaluator_id(),
                        evaluator_name=self._display_name,
                        explanation_type=t_eval_results.explanation_type(),
                        explanation_name=t_eval_results.__name__,
                        explanation_mime=f5s.HtmlFormat.mime,
                        resources=[],
                    )
                )

        for p in dataset_problems:
            self.add_problem(p)

        return bool(healthy_inputs_count > 0)

    def _check_llm_dataset_field_presence(
        self,
        params: commons.CommonInterpretationParams | None = None,
        require_actual_answer: bool = False,
        require_expected_answer: bool = False,
    ) -> bool:
        """Check whether the provided LLM dataset has at least one row with
        the required fields (actual answer and/or expected answer).

        This function verifies that the dataset contains at least one valid row
        where both required fields are present and non-empty. It generates
        compatibility problems if no valid rows are found.

        Parameters
        ----------
        params: commons.CommonInterpretationParams | None
            Common interpretation parameters containing the dataset.
        require_actual_answer: bool
            Whether to require at least one row with a non-empty actual answer.
        require_expected_answer: bool
            Whether to require at least one row with a non-empty expected answer.

        Returns
        -------
        bool
            `True` if the dataset has at least one row with the required fields,
            `False` otherwise.

        """
        if (
            not params
            or not isinstance(params, commons.CommonInterpretationParams)
            or not params.dataset
        ) or not isinstance(params.dataset, datasets.LlmDataset):
            return True

        if not require_actual_answer and not require_expected_answer:
            # no requirements specified, return True
            return True

        self.logger = self.logger or loggers.SonarPrintLogger()

        # check if at least one row has both required fields
        has_valid_row = False
        for i in params.dataset.inputs:
            actual_ok = not require_actual_answer or (
                i and i.actual_output and isinstance(i.actual_output, str)
            )
            expected_ok = not require_expected_answer or (
                i and i.expected_output and isinstance(i.expected_output, str)
            )

            if actual_ok and expected_ok:
                has_valid_row = True
                break

        if not has_valid_row:
            # generate problem and return False
            t_explanation = e10s.LlmEvalResultsExplanation

            # construct description based on requirements
            missing_fields = []
            if require_actual_answer:
                missing_fields.append("actual answers")
            if require_expected_answer:
                missing_fields.append("expected answers")
            missing_fields_str = " and ".join(missing_fields)

            description = (
                f"{self._display_name} evaluator cannot evaluate the provided "
                f"LLM dataset as it does not contain at least one row with "
                f"{missing_fields_str}, which are required by the evaluator."
            )

            html = airium.Airium()
            html(f"{self._display_name} evaluator cannot evaluate the provided ")
            html("LLM dataset as it does ")
            with html.b(klass="w3-black"):
                html("&nbsp;not contain at least one row&nbsp;")
            html(
                f"&nbsp;with {missing_fields_str} which are required by the evaluator."
            )

            self.add_problem(
                problems.ProblemAndAction(
                    description=description,
                    description_html=html,
                    severity=problems.ProblemSeverity.high,
                    problem_type="dataset",
                    actions_description=(
                        f"Provide a valid LLM dataset with at least one row "
                        f"containing {missing_fields_str}."
                    ),
                    evaluator_id=self.evaluator_id(),
                    evaluator_name=self._display_name,
                    explanation_type=t_explanation.explanation_type(),
                    explanation_name=t_explanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                    resources=[],
                )
            )

            self.logger.error(
                f"Compatibility check - evaluator {self._display_name} is "
                f"INCOMPATIBLE: provided LLM dataset does not contain at least "
                f"one row with {missing_fields_str} which are required by the "
                f"evaluator."
            )

            return False

        return True

    @abstractmethod
    def evaluate(self, llm_testset, explanations_types: list = None, **kwargs) -> list:
        raise NotImplementedError()

    def explain(self, X, y=None, explanations_types: list = None, **kwargs) -> list:
        return self.evaluate(X, explanations_types=explanations_types, **kwargs)

    def _diagnose_perturbation_problems(
        self,
        eval_results: datasets.LlmEvalResults,
        key_2_evaluated_model: dict,
        metrics_meta: commons.MetricsMeta | None = None,
    ):
        metrics_meta = metrics_meta or self._metrics_meta

        # perturbation flips
        #   map: row ID -> metric ID -> FlippedPerturbedTestCase
        perturbation_flips = e10s.diagnose_perturbation_flips(
            eval_results=eval_results,
            metrics_meta=metrics_meta,
            key_2_evaluated_model=key_2_evaluated_model,
            logger=self.logger,
        )

        t_flip_explanation = e10s.LlmEvalResultsExplanation
        for row_key in perturbation_flips:
            for flip in perturbation_flips[row_key].values():
                flip_txt = (
                    "from failed to passed"
                    if flip.good_to_bad is False
                    else "from passed to failed"
                )
                flip_txt = (
                    f"{flip_txt} (from {flip.orig_metric_value_str} "
                    f"to {flip.perturbed_metric_value_str}"
                )
                if flip.heat_threshold is not None:
                    flip_txt += f" with threshold {flip.heat_threshold}"
                flip_txt += ")"

                html = airium.Airium()
                html("Model&nbsp;")
                with html.b(klass="w3-black"):
                    html("&nbsp;robustness problem")
                html("&nbsp;detected in case of prompt perturbation: metric ")
                with html.code():
                    html(f"{flip.metric_meta.display_name}")
                html(" value&nbsp;")
                with html.b(klass="w3-black"):
                    html("&nbsp;flipped&nbsp;")
                html(
                    f"&nbsp;{flip_txt} in case of answers generated by the model&nbsp;"
                )
                with html.code():
                    html(f"{flip.llm_model_name}")
                html(".&nbsp;")
                with html.b(klass="w3-black"):
                    html("&nbsp;Original")
                html("&nbsp;prompt: ")
                with html.i():
                    html(f"'{flip.orig_row.i}'")
                html(", ")
                with html.b(klass="w3-black"):
                    html("&nbsp;Perturbed")
                html("&nbsp;prompt: ")
                with html.i():
                    html(f"'{flip.perturbed_row.i}'")
                # html(".") ... was always on a new line

                problem = problems.ProblemAndAction(
                    description=(
                        f"Model robustness problem detected in case of prompt "
                        f"perturbation: metric '{flip.metric_meta.display_name}' "
                        f"value flipped {flip_txt} in case of answers generated by "
                        f"the model '{flip.llm_model_name}'. "
                        f"ORIGINAL prompt: '{flip.orig_row.i}', "
                        f"PERTURBED prompt: '{flip.perturbed_row.i}'."
                    ),
                    description_html=html,
                    severity=problems.ProblemSeverity.high,
                    problem_type="robustness",
                    problem_code=problems.AVIDProblemCode.P0200_MODEL,
                    problem_attrs={
                        problems.ProblemAndAction.ATTR_MODEL_NAME: (
                            flip.llm_model_name
                        ),
                        problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                            self._display_name
                        ),
                        problems.ProblemAndAction.ATTR_ROW_KEYS: [
                            (flip.perturbed_row.key, flip.perturbed_row.model_key)
                        ],
                        # input dataset ~ test lab ~ key is the test case key
                        problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [
                            flip.perturbed_row.key
                        ],
                    },
                    actions_description=(
                        "Perform sensitivity analysis on various perturbation types "
                        "and intensities to explore the model's robustness with regard "
                        "to the specified perturbations. Please refer to "
                        "the explanation for more details."
                    ),
                    evaluator_id=self.evaluator_id(),
                    evaluator_name=self._display_name,
                    explanation_type=t_flip_explanation.explanation_type(),
                    explanation_name=t_flip_explanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                    resources=[],
                )
                self.add_problem(problem)

    def _diagnose_low_test_case_problem(
        self,
        eval_results: datasets.LlmEvalResults,
        models: list,
        test_case_minimum: int,
    ):
        res_pd = eval_results.to_datatable().to_pandas()
        for m in models:
            test_case_len = len(res_pd.loc[res_pd["model_key"] == m.key])
            model_name = m.llm_model_name
            if not test_case_len == 0 and test_case_len < test_case_minimum:
                html = airium.Airium()
                html("The number of evaluated test cases for the model ")
                with html.code():
                    html(f"{model_name}")
                html(" is ")
                with html.b(klass="w3-black"):
                    html("&nbsp;too low&nbsp;")
                html(
                    f"&nbsp;({test_case_len}&nbsp;<&nbsp;{test_case_minimum}). "
                    "Therefore the evaluation results may not be conclusive."
                )

                problem = problems.ProblemAndAction(
                    description=(
                        f"The number of evaluated rows is too low "
                        f"({test_case_len} < {test_case_minimum}). Therefore the "
                        f"evaluation results may not be conclusive."
                    ),
                    description_html=html,
                    severity=problems.ProblemSeverity.high,
                    problem_type="accuracy",
                    problem_attrs={
                        problems.ProblemAndAction.ATTR_MODEL_NAME: model_name,
                        problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                            self._display_name
                        ),
                        "test_cases": test_case_len,
                    },
                    actions_description=(
                        "Consider increasing the number of evaluated rows to get "
                        "more reliable results."
                    ),
                    evaluator_id=self.evaluator_id(),
                    evaluator_name=self._display_name,
                    explanation_type=(
                        e10s.GlobalHtmlFragmentExplanation.explanation_type()
                    ),
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                    resources=[],
                )

                self.add_problem(problem)

    def _diagnose_judge_answer_parsing_problem(
        self,
        result_row: datasets.LlmEvalResults.LlmEvalResultRow,
        key_2_evaluated_model: dict,
        agent_answer: str,
        problem_type: str = "evaluation",
    ):
        """Diagnose the problem with parsing the agent or judge answer."""

        prompt = result_row.dataset_row.i
        evaluated_model = key_2_evaluated_model[result_row.dataset_row.model_key]
        llm_model_name = evaluated_model.llm_model_name if evaluated_model else ""

        description = (
            f"Agent which evaluated the actual answer to the prompt '{prompt}' "
            f"returned the evaluation report which cannot be parsed. Agent's "
            f"evaluation report is expected to contain JSON section with the answer, "
            f"metric score and brief justification of the metric score, but instead"
            f"it contains: [BEGIN-AGENT-ANSWER]{agent_answer}[END-AGENT-ANSWER]"
        )

        description_html = airium.Airium()
        description_html("Agent which evaluated the actual answer to the prompt '")
        with description_html.code():
            description_html(f"{prompt}")
        description_html("' returned the evaluation report which ")
        with description_html.b():
            description_html("cannot be parsed")
        description_html(
            ". Agent's evaluation report is expected to contain JSON section with the "
            "answer, metric score and brief justification of the metric score, but "
            "instead it contains: '"
        )
        with description_html.i():
            with description_html.b():
                description_html(f"{agent_answer}")
        description_html("'.")

        problem = problems.ProblemAndAction(
            description=description,
            description_html=description_html,
            severity=problems.ProblemSeverity.high,
            problem_type=problem_type,
            problem_attrs={
                problems.ProblemAndAction.ATTR_MODEL_NAME: llm_model_name,
                problems.ProblemAndAction.ATTR_EVALUATOR_NAME: self._display_name,
            },
            actions_description=(
                "Contact the agent-based evaluator vendor and report this problem."
            ),
            evaluator_id=self.evaluator_id(),
            evaluator_name=self._display_name,
            explanation_type=(e10s.GlobalHtmlFragmentExplanation.explanation_type()),
            explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
            resources=[],
        )

        self.add_problem(problem)

    @staticmethod
    def _get_custom_param_metric_threshold(
        metric: commons.MetricMeta,
    ) -> EvaluatorParam:
        customized_param = EvaluatorParam(
            param_name=Evaluator.PARAM_METRIC_THRESHOLD,
            description=(
                "Evaluated metric threshold - values "
                f"{'below' if metric.higher_is_better else 'above'} this threshold"
                " are considered problematic."
            ),
            param_type=commons.EvaluatorParamType.float,
            default_value=metric.threshold,
            src=EvaluatorParam.SRC_EVALUATOR_PARAMS,
        )
        return customized_param

    @staticmethod
    def _get_custom_param_min_test_case(minimum: int = 0) -> EvaluatorParam:
        return EvaluatorParam(
            param_name=Evaluator.PARAM_MIN_TEST_CASES,
            description=(
                "Minimum number of test cases, which produces useful results."
            ),
            param_type=commons.EvaluatorParamType.int,
            default_value=minimum,
            src=EvaluatorParam.SRC_EVALUATOR_PARAMS,
        )
