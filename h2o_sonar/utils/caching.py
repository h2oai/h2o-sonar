# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
"""Caching module provides functionality to download and cached models used
for evaluation upfront, to avoid downloading the models in the runtime.

"""

import os
import shutil
import zipfile

from h2o_sonar import loggers
from h2o_sonar.lib.api import commons


try:
    from tiktoken_ext import openai_public

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    import detoxify

    HAS_DETOXIFY = True
except ImportError:
    HAS_DETOXIFY = False

try:
    import nltk

    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from transformers import AutoConfig
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# environment variable names
ENV_HUGGING_FACE_HUB_TOKEN = "HUGGING_FACE_HUB_TOKEN"

# Model constants: preferably use FULLY QUALIFIED names, not short names - as HF client
# can find any model with the short name in any repository, which can lead to
# unexpected results, crashes and authentication issues.
MODEL_ALBERT_XLARGE_VITAMINC = "tals/albert-xlarge-vitaminc-mnli"
MODEL_BAAI_BGE_M3 = "BAAI/bge-m3"
MODEL_BAAI_BGE_SMALL_EN = "BAAI/bge-small-en"
MODEL_BAAI_BGE_SMALL_EN_V1_5 = "BAAI/bge-small-en-v1.5"
MODEL_BERT_BASE_UNCASED = "google-bert/bert-base-uncased"
MODEL_DISTILGPT2 = "distilbert/distilgpt2"
MODEL_ELEUTHERAI_GPT_J_6B = "EleutherAI/gpt-j-6B"
MODEL_FACEBOOK_OPT_125M = "facebook/opt-125m"
MODEL_FACEBOOK_OPT_13B = "facebook/opt-13b"
MODEL_FACEBOOK_OPT_1_3B = "facebook/opt-1.3b"
MODEL_FACEBOOK_OPT_2_7B = "facebook/opt-2.7b"
MODEL_FACEBOOK_OPT_350M = "facebook/opt-350m"
MODEL_FACEBOOK_OPT_66B = "facebook/opt-66b"
MODEL_FACEBOOK_OPT_6_7B = "facebook/opt-6.7b"
MODEL_GOOGLE_FLAN_T5_BASE = "google/flan-t5-base"
MODEL_GOOGLE_FLAN_T5_LARGE = "google/flan-t5-large"
MODEL_GOOGLE_FLAN_T5_SMALL = "google/flan-t5-small"
MODEL_GOOGLE_FLAN_T5_XL = "google/flan-t5-xl"
MODEL_GOOGLE_FLAN_T5_XXL = "google/flan-t5-xxl"
MODEL_GPT2_LARGE = "openai-community/gpt2-large"
MODEL_GPT2_MEDIUM = "openai-community/gpt2-medium"
MODEL_GPT2_XL = "openai-community/gpt2-xl"
MODEL_HKUNLP_INSTRUCTOR_LARGE = "hkunlp/instructor-large"
MODEL_SENTENCE_TRANSFORMERS_ALL_MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_VECTARA_HALLUCINATION = "vectara/hallucination_evaluation_model"

CACHE_HOME = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
HF_HOME = os.environ.get("HF_HOME", f"{CACHE_HOME}/huggingface")
NLTK_DATA = os.environ.get("NLTK_DATA", os.path.expanduser("~/nltk_data"))

DEFAULT_SONAR_CACHE_MODEL_DIR = os.path.expanduser(f"{CACHE_HOME}/h2o_sonar/models")
DEFAULT_TORCH_CHECKPOINTS_DIR = os.path.expanduser(
    f"{CACHE_HOME}/torch/hub/checkpoints"
)
DEFAULT_HF_CACHE_DIR = os.path.expanduser(f"{HF_HOME}/hub")

# Model revisions mapping
REVISIONS_FOR_MODEL = {
    MODEL_BAAI_BGE_M3: "5617a9f61b028005a4858fdac845db406aefb181",
    MODEL_BAAI_BGE_SMALL_EN: "2275a7bdee235e9b4f01fa73aa60d3311983cfea",
    MODEL_BAAI_BGE_SMALL_EN_V1_5: "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    MODEL_BERT_BASE_UNCASED: "86b5e0934494bd15c9632b12f734a8a67f723594",
    MODEL_DISTILGPT2: "2290a62682d06624634c1f46a6ad5be0f47f38aa",
    MODEL_ELEUTHERAI_GPT_J_6B: "47e169305d2e8376be1d31e765533382721b2cc1",
    MODEL_FACEBOOK_OPT_125M: "27dcfa74d334bc871f3234de431e71c6eeba5dd6",
    MODEL_FACEBOOK_OPT_13B: "e515202d1e7750da62d245fbccb2723b9c1790f5",
    MODEL_FACEBOOK_OPT_1_3B: "3f5c25d0bc631cb57ac65913f76e22c2dfb61d62",
    MODEL_FACEBOOK_OPT_2_7B: "905a4b602cda5c501f1b3a2650a4152680238254",
    MODEL_FACEBOOK_OPT_350M: "08ab08cc4b72ff5593870b5d527cf4230323703c",
    MODEL_FACEBOOK_OPT_66B: "7259969061237fe940036d22bea0fd349e4485e9",
    MODEL_FACEBOOK_OPT_6_7B: "a45aa65bbeb77c1558bc99bedc6779195462dab0",
    MODEL_GOOGLE_FLAN_T5_BASE: "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    MODEL_GOOGLE_FLAN_T5_LARGE: "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
    MODEL_GOOGLE_FLAN_T5_SMALL: "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab",
    MODEL_GOOGLE_FLAN_T5_XL: "7d6315df2c2fb742f0f5b556879d730926ca9001",
    MODEL_GOOGLE_FLAN_T5_XXL: "ae7c9136adc7555eeccc78cdd960dfd60fb346ce",
    MODEL_GPT2_LARGE: "32b71b12589c2f8d625668d2335a01cac3249519",
    MODEL_GPT2_MEDIUM: "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
    MODEL_GPT2_XL: "15ea56dee5df4983c59b2538573817e1667135e2",
    MODEL_HKUNLP_INSTRUCTOR_LARGE: "54e5ffb8d484de506e59443b07dc819fb15c7233",
    MODEL_SENTENCE_TRANSFORMERS_ALL_MINILM_L6_V2: (
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    ),
    MODEL_VECTARA_HALLUCINATION: "8f6b0b5865cb7a6a09c299be3f788f91e22313b0",
}


def cache_all_models(logger: loggers.SonarLogger):
    """Cache all the models used in the Sonar package."""
    cache_eval_studio_models(logger)
    cache_vectara_hallucination_model(logger)
    cache_bert_base_uncased(logger)
    cache_baai_bge_small_env15(logger)
    cache_detoxify_models(logger)
    cache_nltk(logger)
    cache_tiktoken_blobs(logger)
    cache_summac_vitc(logger)
    cache_lmppl_perplexity_evaluator_model(logger)
    cache_gptscore_evaluator_model(logger)
    cache_bge_m3(logger)
    cache_hkunlp_instructor(logger)
    cache_baai_bge_small_en(logger)
    cache_all_minilm_l6_v2(logger)


def cache_tiktoken_blobs(logger: loggers.SonarLogger):
    """Cache the TikToken blobs."""
    if not HAS_TIKTOKEN:
        commons.raise_opt_import_err("tiktoken_ext")

    logger.info("Caching TikToken blobs...")

    logger.info("  Caching TikToken blobs GPT2...")
    openai_public.gpt2()
    logger.info("  Caching TikToken blobs R50K...")
    openai_public.r50k_base()
    logger.info("  Caching TikToken blobs P50K base...")
    openai_public.p50k_base()
    logger.info("  Caching TikToken blobs P50K edit...")
    openai_public.p50k_edit()
    logger.info("  Caching TikToken blobs CL100K base...")
    openai_public.cl100k_base()

    logger.info("DONE caching TikToken blobs")


def cache_eval_studio_models(logger: loggers.SonarLogger):
    """Download the Eval Studio models from the S3"""
    logger.info("Caching Eval Studio models...")

    cache_dir = DEFAULT_SONAR_CACHE_MODEL_DIR
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    model_name = "bias-detection-model-onnx"
    url = (
        "https://s3.us-east-1.amazonaws.com/"
        f"eval-studio-artifacts/models/{model_name}.zip"
    )
    tmp_zip = os.path.join(cache_dir, f"{model_name}.zip")
    target_dir = os.path.join(cache_dir, model_name)

    logger.info(f"  BEGIN caching {model_name} model from {url} ...")

    if os.path.exists(tmp_zip) or os.path.exists(target_dir):
        # NOTE: Skip if the model is already cached or being cached
        logger.info("  SKIPPING: model already cached or being cached.")
        return

    _download(url, tmp_zip)
    with zipfile.ZipFile(tmp_zip, "r") as zip_ref:
        zip_ref.extractall(cache_dir)

    os.remove(tmp_zip)

    logger.info("DONE caching Eval Studio models")


def cache_vectara_hallucination_model(logger: loggers.SonarLogger):
    """Cache the Vectara hallucination evaluation model."""
    logger.info(f"BEGIN caching {MODEL_VECTARA_HALLUCINATION} evaluation model...")

    repo = MODEL_VECTARA_HALLUCINATION
    _cache_hf_file(repo, tokenizer=MODEL_GOOGLE_FLAN_T5_BASE)

    logger.info(f"DONE caching {MODEL_VECTARA_HALLUCINATION} evaluation model")


def cache_bert_base_uncased(logger: loggers.SonarLogger):
    """Cache the BERT base uncased model."""
    logger.info(f"BEGIN caching {MODEL_BERT_BASE_UNCASED} model...")

    repo = MODEL_BERT_BASE_UNCASED
    _cache_hf_file(repo)

    logger.info(f"DONE caching {MODEL_BERT_BASE_UNCASED} model")


def cache_baai_bge_small_env15(logger: loggers.SonarLogger):
    """Cache the BAAI BGE small environment v1.5 model."""
    logger.info(f"BEGIN caching {MODEL_BAAI_BGE_SMALL_EN_V1_5} model...")

    repo = MODEL_BAAI_BGE_SMALL_EN_V1_5
    _cache_hf_file(repo)

    logger.info(f"DONE caching {MODEL_BAAI_BGE_SMALL_EN_V1_5} model")


def cache_baai_bge_small_en(logger: loggers.SonarLogger):
    """Cache the BAAI BGE small en"""
    logger.info(f"BEGIN caching {MODEL_BAAI_BGE_SMALL_EN} model...")

    repo = MODEL_BAAI_BGE_SMALL_EN
    _cache_hf_file(repo)

    logger.info(f"DONE caching {MODEL_BAAI_BGE_SMALL_EN} model")


def cache_bge_m3(logger: loggers.SonarLogger):
    """Cache the BGE m3"""
    logger.info(f"BEGIN caching {MODEL_BAAI_BGE_M3}")
    repo = MODEL_BAAI_BGE_M3
    _cache_hf_file(repo)
    logger.info(f"DONE caching {MODEL_BAAI_BGE_M3} model")


def cache_hkunlp_instructor(logger: loggers.SonarLogger):
    """Cache hkunlp Instructor"""
    logger.info(f"BEGIN caching {MODEL_HKUNLP_INSTRUCTOR_LARGE} model")
    repo = MODEL_HKUNLP_INSTRUCTOR_LARGE
    _cache_hf_file(repo)
    logger.info(f"DONE caching {MODEL_HKUNLP_INSTRUCTOR_LARGE} model")


def cache_detoxify_models(logger: loggers.SonarLogger):
    """Download and cache the Detoxify models."""
    if not HAS_DETOXIFY:
        commons.raise_opt_import_err("detoxify")

    logger.info("BEGIN caching Detoxify models...")

    detoxify.toxic_bert()

    logger.info("DONE caching Detoxify models")


def cache_all_minilm_l6_v2(logger: loggers.SonarLogger):
    """Cache the all-MiniLM-L6-v2 model."""
    logger.info(
        f"BEGIN caching {MODEL_SENTENCE_TRANSFORMERS_ALL_MINILM_L6_V2} model..."
    )

    repo = MODEL_SENTENCE_TRANSFORMERS_ALL_MINILM_L6_V2
    _cache_hf_file(repo)

    logger.info(f"DONE caching {MODEL_SENTENCE_TRANSFORMERS_ALL_MINILM_L6_V2}")


def cache_nltk(logger: loggers.SonarLogger):
    """Cache the NLTK models.

    - Punkt - used in BLEU and perturbations
    - averaged_perceptron_tagger - used in perturbations
    - wordnet - used in perturbations

    """
    logger.info("BEGIN caching NLTK models...")
    cache_nltk_punkt(logger)
    cache_nltk_wordnet(logger)
    cache_nltk_averaged_perceptron_tagger(logger)
    logger.info("DONE caching NLTK models")


def cache_nltk_averaged_perceptron_tagger(logger: loggers.SonarLogger | None = None):
    if not HAS_NLTK:
        commons.raise_opt_import_err("nltk")

    if logger is None:
        logger = loggers.SonarPrintLogger()

    for perceptron in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
        try:
            if not os.path.exists(f"{NLTK_DATA}/taggers/{perceptron}.zip"):
                logger.info(f"BEGIN caching NLTK {perceptron}...")
                nltk.download(perceptron, download_dir=NLTK_DATA, quiet=True)
                logger.info(f"DONE caching NLTK {perceptron} corpus")
            else:
                logger.info(f"NLTK {perceptron} is already cached...")
        except Exception as e:
            logger.error(f"Error caching NLTK {perceptron}: {e}")


def cache_nltk_wordnet(logger: loggers.SonarLogger | None = None):
    if not HAS_NLTK:
        commons.raise_opt_import_err("nltk")

    if logger is None:
        logger = loggers.SonarPrintLogger()
    if not os.path.exists(f"{NLTK_DATA}/corpora/wordnet.zip"):
        logger.info("BEGIN caching NLTK wordnet corpus...")
        nltk.download("wordnet", download_dir=NLTK_DATA, quiet=True)
        logger.info("DONE caching NLTK wordnet corpus")
    else:
        logger.info("NLTK wordnet corpus is already cached...")


def cache_nltk_punkt(logger: loggers.SonarLogger | None = None):
    if not HAS_NLTK:
        commons.raise_opt_import_err("nltk")

    if logger is None:
        logger = loggers.SonarPrintLogger()

    if not os.path.exists(f"{NLTK_DATA}/tokenizers/punkt/english.pickle"):
        logger.info("BEGIN caching NLTK punkt tokenizers...")
        nltk.download("punkt", download_dir=NLTK_DATA, quiet=True)
        nltk.download("punkt_tab", download_dir=NLTK_DATA, quiet=True)
        logger.info("DONE caching NLTK punkt tokenizers")
    else:
        logger.info("NLTK punkt tokenizer is already cached...")


def cache_lmppl_perplexity_evaluator_model(logger: loggers.SonarLogger):
    """Cache default model for perplexity evaluator"""

    logger.info(f"BEGIN caching {MODEL_DISTILGPT2} model...")

    repo = MODEL_DISTILGPT2
    _cache_hf_file(repo)

    logger.info(f"DONE caching {MODEL_DISTILGPT2} model")


def cache_summac_vitc(logger: loggers.SonarLogger):
    """Cache the summac used for summarization"""
    logger.info("BEGIN caching SummaC vitc model...")

    cache_path = os.path.expanduser(DEFAULT_SONAR_CACHE_MODEL_DIR)

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    _download(
        "https://github.com/tingofurro/summac/raw/"
        "9e4f35722b635402c6fd5a1399d987bc80b45b43/summac_conv_vitc_sent_perc_e.bin",
        f"{cache_path}/summac_conv_vitc_sent_perc_e.bin",
    )

    logger.info("DONE caching SummaC vitc model")


def cache_gptscore_evaluator_model(logger: loggers.SonarLogger):
    """Cache default model for gptscore evaluator"""
    # import h2o_sonar.evaluators.gptscore_evaluator as gse
    # MODELS = gse.GptScoreEvaluator._ALLOWED_MODELS

    models = [
        MODEL_GOOGLE_FLAN_T5_SMALL,  # used in tests
        # MODEL_GOOGLE_FLAN_T5_BASE,
        # MODEL_GOOGLE_FLAN_T5_LARGE,
        # MODEL_GOOGLE_FLAN_T5_XL,
        # MODEL_GOOGLE_FLAN_T5_XXL,
        # MODEL_FACEBOOK_OPT_125M,
        # MODEL_FACEBOOK_OPT_350M,
        # MODEL_FACEBOOK_OPT_1_3B,
        # MODEL_FACEBOOK_OPT_2_7B,
        # MODEL_FACEBOOK_OPT_6_7B,
        # MODEL_FACEBOOK_OPT_13B,
        # MODEL_FACEBOOK_OPT_66B,
        MODEL_GPT2_MEDIUM,
        # MODEL_GPT2_LARGE,
        # MODEL_GPT2_XL,
        # MODEL_ELUTHERAI_GPTJ_6B,
    ]

    for repo in models:
        logger.info(f"BEGIN caching {repo} model...")
        _cache_hf_file(repo)
        logger.info(f"DONE caching {repo} model")


def _download(url: str, target: str, verify: bool | str = True):
    """Download the model from the provided URL to the target location.

    Parameters
    ----------
    url : str
        Source URL of the file
    target : str
        Target (local) path, where the downloaded file will be saved
    """
    import requests

    if os.path.exists(target):
        return

    with requests.get(url, stream=True, verify=verify) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            # NOTE: Use of `copyfileobj` is more memory efficient and faster.
            shutil.copyfileobj(r.raw, f)


def _cache_hf_file(repo: str, tokenizer: str | None = None):
    """Cache a Hugging Face files from the provided repository.

    Parameters
    ----------
    repo : str
        Repository URL
    """
    if not HAS_TRANSFORMERS:
        commons.raise_opt_import_err("transformers")

    hf_token = os.getenv(ENV_HUGGING_FACE_HUB_TOKEN, None)
    if not hf_token:
        raise ValueError(
            f"Environment variable {ENV_HUGGING_FACE_HUB_TOKEN} is not set. "
            "Please set it to a valid HuggingFace Hub read access token so that "
            "the models can be cached."
        )

    tokenizer = tokenizer or repo
    revision = REVISIONS_FOR_MODEL[repo]

    AutoTokenizer.from_pretrained(
        tokenizer,
        trust_remote_code=True,
        revision=REVISIONS_FOR_MODEL[tokenizer],
        force_download=True,
        token=hf_token,
    )
    AutoModelForSequenceClassification.from_pretrained(
        repo,
        trust_remote_code=True,
        revision=revision,
        force_download=True,
        token=hf_token,
    )
    AutoConfig.from_pretrained(
        repo,
        trust_remote_code=True,
        revision=revision,
        force_download=True,
        token=hf_token,
    )


if __name__ == "__main__":
    cache_all_models(loggers.SonarPrintLogger())
