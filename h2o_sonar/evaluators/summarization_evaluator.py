# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import collections
import os
import traceback
from typing import Any

import airium
import numpy as np
import pandas as pd

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results
from h2o_sonar.utils import caching
from h2o_sonar.utils import resource_mgmt


try:
    import nltk
    import sentence_transformers
    import torch
    import transformers

    HAS_REQUIRED_PACKAGES = True
except ImportError:
    HAS_REQUIRED_PACKAGES = False


_CONFIG_MD5_SUCCESSFUL = None


# MD5 hashing for joblib
try:
    import hashlib
    import io
    import pickle

    import joblib.hashing

    def __init__(self, hash_name="md5"):
        Pickler = pickle._Pickler
        self.stream = io.BytesIO()
        # by default we want a pickle protocol that only changes with
        # the major python version and not the minor one
        protocol = 3
        Pickler.__init__(self, self.stream, protocol=protocol)
        # initialise the hash obj
        self._hash = hashlib.new(hash_name, usedforsecurity=False)

    joblib.hashing.Hasher.__init__ = __init__

    _CONFIG_MD5_SUCCESSFUL = True
except ImportError:
    _CONFIG_MD5_SUCCESSFUL = False


# Several functions used in the coverage calculations
def segment_calc(distances: Any, n: int) -> float:
    # Calculate a one segment or three segment coverage measure
    if len(distances) > 0:
        if n == 0:
            answer = np.max(distances)
        else:
            d1 = np.max(distances)
            # If there are enough points, calculate the longest 3 segment distance
            if len(distances) > 2:
                ind = np.unravel_index(np.argmax(distances), distances.shape, order="C")
                d2 = sorted(distances[ind[0], :])[-2]
                d3 = sorted(distances[:, ind[1]])[-2]
            else:
                # Else return just the largest two point segment
                d2 = 0
                d3 = 0
            answer = d1 + d2 + d3
    else:
        answer = 0
    return answer


def pairwise_distances_wrapper(points):
    from sklearn import metrics

    answer = []
    if len(points) > 0:
        answer = metrics.pairwise_distances(points)
    return answer


#
# START excerpt from summac (https://github.com/tingofurro/summac) (Apache 2 license)
#


def load_summac():
    model_map = {
        "vitc": {
            "model_card": SummarizationEvaluator.e_model_vitamin_c,
            "entailment_idx": 0,
            "contradiction_idx": 1,
        },
    }

    def name_to_card(name):
        if name in model_map:
            return model_map[name]["model_card"]
        return name

    def get_neutral_idx(ent_idx, con_idx):
        return list({0, 1, 2} - {ent_idx, con_idx})[0]

    def batcher(iterator, batch_size=4):
        batch = []
        for elem in iterator:
            batch.append(elem)
            if len(batch) == batch_size:
                final_batch = batch
                batch = []
                yield final_batch
        if len(batch) > 0:  # Leftovers
            yield batch

    class SummaCImager:
        def __init__(
            self,
            model_name="mnli",
            granularity="paragraph",
            max_doc_sents=100,
            device="cuda",
            **kwargs,
        ):
            self.grans = granularity.split("-")

            assert (
                all(
                    gran in ["paragraph", "sentence", "document", "2sents", "mixed"]
                    for gran in self.grans
                )
                and len(self.grans) <= 2
            ), "Unrecognized `granularity` %s" % granularity
            assert model_name in model_map.keys(), "Unrecognized model name: `%s`" % (
                model_name
            )

            self.model_name = model_name
            if model_name != "decomp":
                self.model_card = name_to_card(model_name)
                self.entailment_idx = model_map[model_name]["entailment_idx"]
                self.contradiction_idx = model_map[model_name]["contradiction_idx"]
                self.neutral_idx = get_neutral_idx(
                    self.entailment_idx, self.contradiction_idx
                )

            self.granularity = granularity

            self.max_doc_sents = max_doc_sents
            self.max_input_length = 500
            self.device = device
            self.cache = {}
            self.model = None  # Lazy loader

        def load_nli(self):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_card,
                trust_remote_code=True,
                revision=caching.REVISIONS_FOR_MODEL.get(self.model_card, "main"),
            )
            self.model = (
                transformers.AutoModelForSequenceClassification.from_pretrained(
                    self.model_card,
                    trust_remote_code=True,
                    revision=caching.REVISIONS_FOR_MODEL.get(self.model_card, "main"),
                ).eval()
            )
            self.model.to(self.device)
            if self.device == "cuda":
                self.model.half()

        @staticmethod
        def split_sentences(text):
            sentences = nltk.tokenize.sent_tokenize(text)
            sentences = [sent for sent in sentences if len(sent) > 10]
            return sentences

        @staticmethod
        def split_2sents(text):
            sentences = nltk.tokenize.sent_tokenize(text)
            sentences = [sent for sent in sentences if len(sent) > 10]
            two_sents = [
                " ".join(sentences[i : (i + 2)]) for i in range(len(sentences))
            ]
            return two_sents

        @staticmethod
        def split_paragraphs(text):
            if text.count("\n\n") > 0:
                paragraphs = [p.strip() for p in text.split("\n\n")]
            else:
                paragraphs = [p.strip() for p in text.split("\n")]
            return [p for p in paragraphs if len(p) > 10]

        def split_text(self, text, granularity="sentence"):
            if granularity == "document":
                return [text]
            elif granularity == "paragraph":
                return self.split_paragraphs(text)
            elif granularity == "sentence":
                return self.split_sentences(text)
            elif granularity == "2sents":
                return self.split_2sents(text)
            elif granularity == "mixed":
                return self.split_sentences(text) + self.split_paragraphs(text)

        def build_chunk_dataset(self, original, generated, pair_idx=None):
            if len(self.grans) == 1:
                gran_doc, gran_sum = self.grans[0], self.grans[0]
            else:
                gran_doc, gran_sum = self.grans[0], self.grans[1]

            original_chunks = self.split_text(original, granularity=gran_doc)[
                : self.max_doc_sents
            ]
            generated_chunks = self.split_text(generated, granularity=gran_sum)

            N_ori, N_gen = len(original_chunks), len(generated_chunks)
            dataset = [
                {
                    "premise": original_chunks[i],
                    "hypothesis": generated_chunks[j],
                    "doc_i": i,
                    "gen_i": j,
                    "pair_idx": pair_idx,
                }
                for i in range(N_ori)
                for j in range(N_gen)
            ]
            return dataset, N_ori, N_gen

        def build_image(self, original, generated):
            cache_key = (original, generated)
            if cache_key in self.cache:
                cached_image = self.cache[cache_key]
                cached_image = cached_image[:, : self.max_doc_sents, :]
                return cached_image

            dataset, N_ori, N_gen = self.build_chunk_dataset(original, generated)

            if len(dataset) == 0:
                return np.zeros((3, 1, 1))

            image = np.zeros((3, N_ori, N_gen))

            if self.model is None:
                self.load_nli()

            for batch in batcher(dataset, batch_size=20):
                batch_prems = [b["premise"] for b in batch]
                batch_hypos = [b["hypothesis"] for b in batch]
                batch_tokens = self.tokenizer.batch_encode_plus(
                    list(zip(batch_prems, batch_hypos, strict=False)),
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_length,
                    return_tensors="pt",
                    truncation_strategy="only_first",
                )
                with torch.no_grad():
                    model_outputs = self.model(
                        **{k: v.to(self.device) for k, v in batch_tokens.items()}
                    )

                batch_probs = torch.nn.functional.softmax(
                    model_outputs["logits"], dim=-1
                )
                batch_evids = batch_probs[:, self.entailment_idx].tolist()
                batch_conts = batch_probs[:, self.contradiction_idx].tolist()
                batch_neuts = batch_probs[:, self.neutral_idx].tolist()

                for b, evid, cont, neut in zip(
                    batch, batch_evids, batch_conts, batch_neuts, strict=False
                ):
                    image[0, b["doc_i"], b["gen_i"]] = evid
                    image[1, b["doc_i"], b["gen_i"]] = cont
                    image[2, b["doc_i"], b["gen_i"]] = neut

            return image

        def build_images(self, originals, generateds, batch_size=128):
            todo_originals, todo_generateds = [], []
            for ori, gen in zip(originals, generateds, strict=False):
                cache_key = (ori, gen)
                if cache_key not in self.cache:
                    todo_originals.append(ori)
                    todo_generateds.append(gen)

            total_dataset = []
            todo_images = []
            for pair_idx, (ori, gen) in enumerate(
                zip(todo_originals, todo_generateds, strict=False)
            ):
                dataset, N_ori, N_gen = self.build_chunk_dataset(
                    ori, gen, pair_idx=pair_idx
                )
                if len(dataset) == 0:
                    image = np.zeros((3, 1, 1))
                else:
                    image = np.zeros((3, N_ori, N_gen))
                todo_images.append(image)
                total_dataset += dataset
            if (
                len(total_dataset) > 0 and self.model is None
            ):  # Can't just rely on the cache
                self.load_nli()

            for batch in batcher(total_dataset, batch_size=batch_size):
                batch_prems = [b["premise"] for b in batch]
                batch_hypos = [b["hypothesis"] for b in batch]
                batch_tokens = self.tokenizer.batch_encode_plus(
                    list(zip(batch_prems, batch_hypos, strict=False)),
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_length,
                    return_tensors="pt",
                    truncation_strategy="only_first",
                )
                with torch.no_grad():
                    model_outputs = self.model(
                        **{k: v.to(self.device) for k, v in batch_tokens.items()}
                    )

                batch_probs = torch.nn.functional.softmax(
                    model_outputs["logits"], dim=-1
                )
                batch_evids = batch_probs[:, self.entailment_idx].tolist()
                batch_conts = batch_probs[:, self.contradiction_idx].tolist()
                batch_neuts = batch_probs[:, self.neutral_idx].tolist()

                for b, evid, cont, neut in zip(
                    batch, batch_evids, batch_conts, batch_neuts, strict=False
                ):
                    image = todo_images[b["pair_idx"]]
                    image[0, b["doc_i"], b["gen_i"]] = evid  # noqa
                    image[1, b["doc_i"], b["gen_i"]] = cont  # noqa
                    image[2, b["doc_i"], b["gen_i"]] = neut  # noqa

            for pair_idx, (ori, gen) in enumerate(
                zip(todo_originals, todo_generateds, strict=False)
            ):
                cache_key = (ori, gen)
                self.cache[cache_key] = todo_images[pair_idx]

            images = [
                self.cache[(ori, gen)]
                for ori, gen in zip(originals, generateds, strict=False)
            ]
            return images

    class SummaCConv(torch.nn.Module):
        def __init__(
            self,
            models=None,
            bins="even50",
            granularity="sentence",
            nli_labels="e",
            device="cuda",
            start_file=None,
            agg="mean",
            **kwargs,
        ):
            # `bins` should be `even%d` or `percentiles`
            if models is None:
                models = ["mnli", "anli", "vitc"]
            assert nli_labels in ["e", "c", "n", "ec", "en", "cn", "ecn"], (
                "Unrecognized nli_labels argument %s" % nli_labels
            )

            super().__init__()
            self.device = device
            self.models = models

            self.imagers = []
            for model_name in models:
                self.imagers.append(
                    SummaCImager(
                        model_name=model_name,
                        granularity=granularity,
                        device=self.device,
                        **kwargs,
                    )
                )
            assert len(self.imagers) > 0, "Imager names were empty or unrecognized"

            if "even" in bins:
                n_bins = int(bins.replace("even", ""))
                self.bins = list(np.arange(0, 1, 1 / n_bins)) + [1.0]
            elif bins == "percentile":
                self.bins = [
                    0.0,
                    0.01,
                    0.02,
                    0.03,
                    0.04,
                    0.07,
                    0.13,
                    0.37,
                    0.90,
                    0.91,
                    0.92,
                    0.93,
                    0.94,
                    0.95,
                    0.955,
                    0.96,
                    0.965,
                    0.97,
                    0.975,
                    0.98,
                    0.985,
                    0.99,
                    0.995,
                    1.0,
                ]  # Based on the percentile of the distribution on some large
                # number of summaries

            self.nli_labels = nli_labels
            self.n_bins = len(self.bins) - 1
            self.n_rows = 10
            self.n_labels = 2
            self.n_depth = len(self.imagers) * len(self.nli_labels)
            self.full_size = self.n_depth * self.n_bins

            self.agg = agg

            self.mlp = torch.nn.Linear(self.full_size, 1).to(device)
            self.layer_final = torch.nn.Linear(3, self.n_labels).to(device)

            if start_file == "default":
                start_file = (
                    f"{os.path.expanduser(caching.DEFAULT_SONAR_CACHE_MODEL_DIR)}/"
                    f"summac_conv_vitc_sent_perc_e.bin"
                )
                if not os.path.isfile(start_file):
                    caching.cache_summac_vitc(loggers.SonarPrintLogger())
                    assert bins == "percentile", (
                        "bins mode should be set to percentile if using the default 1-d"
                        " convolution weights."
                    )
            if start_file is not None:
                self.load_state_dict(torch.load(start_file, weights_only=True))

        def build_image(self, original, generated):
            images = [
                imager.build_image(original, generated) for imager in self.imagers
            ]
            image = np.concatenate(images, axis=0)
            return image

        def compute_histogram(self, original=None, generated=None, image=None):
            # Takes the two texts, and generates a (n_rows, 2*n_bins)

            if image is None:
                image = self.build_image(original, generated)

            N_depth, N_ori, N_gen = image.shape

            full_histogram = []
            for i_gen in range(N_gen):
                histos = []

                for i_depth in range(N_depth):
                    if (
                        (i_depth % 3 == 0 and "e" in self.nli_labels)
                        or (i_depth % 3 == 1 and "c" in self.nli_labels)
                        or (i_depth % 3 == 2 and "n" in self.nli_labels)
                    ):
                        histo, X = np.histogram(
                            image[i_depth, :, i_gen],
                            range=(0, 1),
                            bins=self.bins,
                            density=False,
                        )
                        histos.append(histo)

                histogram_row = np.concatenate(histos)
                full_histogram.append(histogram_row)

            n_rows_missing = self.n_rows - len(full_histogram)
            full_histogram += [[0.0] * self.full_size] * n_rows_missing
            full_histogram = full_histogram[: self.n_rows]
            full_histogram = np.array(full_histogram)
            return image, full_histogram

        def forward(self, originals, generateds, images=None):
            if images is not None:
                # In case they've been pre-computed.
                histograms = []
                for image in images:
                    _, histogram = self.compute_histogram(image=image)
                    histograms.append(histogram)
            else:
                images, histograms = [], []
                for original, generated in zip(originals, generateds, strict=False):
                    image, histogram = self.compute_histogram(
                        original=original, generated=generated
                    )
                    images.append(image)
                    histograms.append(histogram)

            N = len(histograms)
            histograms = torch.FloatTensor(histograms).to(self.device)

            non_zeros = (torch.sum(histograms, dim=-1) != 0.0).long()  # noqa
            seq_lengths = non_zeros.sum(dim=-1).tolist()

            mlp_outs = self.mlp(histograms).reshape(N, self.n_rows)
            features = []

            for mlp_out, seq_length in zip(mlp_outs, seq_lengths, strict=False):
                if seq_length > 0:
                    Rs = mlp_out[:seq_length]
                    if self.agg == "mean":
                        features.append(
                            torch.cat(
                                [
                                    torch.mean(Rs).unsqueeze(0),
                                    torch.mean(Rs).unsqueeze(0),
                                    torch.mean(Rs).unsqueeze(0),
                                ]
                            ).unsqueeze(0)
                        )
                    elif self.agg == "min":
                        features.append(
                            torch.cat(
                                [
                                    torch.min(Rs).unsqueeze(0),
                                    torch.min(Rs).unsqueeze(0),
                                    torch.min(Rs).unsqueeze(0),
                                ]
                            ).unsqueeze(0)
                        )
                    elif self.agg == "max":
                        features.append(
                            torch.cat(
                                [
                                    torch.max(Rs).unsqueeze(0),
                                    torch.max(Rs).unsqueeze(0),
                                    torch.max(Rs).unsqueeze(0),
                                ]
                            ).unsqueeze(0)
                        )
                    elif self.agg == "all":
                        features.append(
                            torch.cat(
                                [
                                    torch.min(Rs).unsqueeze(0),
                                    torch.mean(Rs).unsqueeze(0),
                                    torch.max(Rs).unsqueeze(0),
                                ]
                            ).unsqueeze(0)
                        )
                else:
                    features.append(
                        torch.FloatTensor([0.0, 0.0, 0.0]).unsqueeze(0)
                    )  # .cuda()
            features = torch.cat(features)
            logits = self.layer_final(features)
            histograms_out = [histogram.cpu().numpy() for histogram in histograms]
            return logits, histograms_out, images

        def score(self, originals, generateds, **kwargs):
            with torch.no_grad():
                logits, histograms, images = self.forward(originals, generateds)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                batch_scores = probs[:, 1].tolist()
            return {
                "scores": batch_scores
            }  # , "histograms": histograms, "images": images

    class SummaCZS:
        def __init__(
            self,
            model_name="mnli",
            granularity="paragraph",
            op1="max",
            op2="mean",
            use_ent=True,
            use_con=True,
            device="cuda",
            **kwargs,
        ):
            assert op2 in ["min", "mean", "max"], "Unrecognized `op2`"
            assert op1 in ["max", "mean", "min"], "Unrecognized `op1`"
            self.device = device
            self.imager = SummaCImager(
                model_name=model_name,
                granularity=granularity,
                device=self.device,
                **kwargs,
            )
            self.op2 = op2
            self.op1 = op1
            self.use_ent = use_ent
            self.use_con = use_con

        def score_one(self, original, generated):
            image = self.imager.build_image(original, generated)
            score = self.image2score(image)
            return {"image": image, "score": score}

        def image2score(self, image):
            ent_scores = np.max(image[0], axis=0)
            co_scores = np.max(image[1], axis=0)
            if self.op1 == "mean":
                ent_scores = np.mean(image[0], axis=0)
                co_scores = np.mean(image[1], axis=0)
            elif self.op1 == "min":
                ent_scores = np.min(image[0], axis=0)
                co_scores = np.min(image[1], axis=0)

            if self.use_ent and self.use_con:
                scores = ent_scores - co_scores
            elif self.use_ent:
                scores = ent_scores
            elif self.use_con:
                scores = 1.0 - co_scores
            else:
                raise NotImplementedError(
                    "use_ent, use_con or both must be set to True."
                )

            final_score = np.mean(scores)
            if self.op2 == "min":
                final_score = np.min(scores)
            elif self.op2 == "max":
                final_score = np.max(scores)
            return final_score

        def score(self, sources, generateds, batch_size=128, **kwargs):
            images = self.imager.build_images(
                sources, generateds, batch_size=batch_size
            )
            scores = [self.image2score(image) for image in images]
            return {"scores": scores, "images": images}

    return SummaCZS, SummaCConv


#
# END excerpt from summac (https://github.com/tingofurro/summac) (Apache 2 license)
#


class SummarizationEvaluator(evaluators.Evaluator):
    _display_name = "Summarization (completeness and faithfulness)"
    _tagline = (
        "Evaluate summaries for completeness and faithfulness without reference "
        "summaries."
    )

    # COMPATIBILITY: LLM model explanations only
    _llm = True
    _rag = True

    # GLOBAL: leaderboard as global explanation
    _global_explanation = True

    # EXPLANATION TYPES created by the evaluator
    _explanation_types = [
        e10s.LlmEvalResultsExplanation,
        e10s.LlmHeatmapLeaderboardExplanation,
    ]

    _keywords = [
        evaluators.KEYWORD_GPU_OPT,
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_P,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_SR_11_7_CS,
        evaluators.KEYWORD_SR_11_7_OA,
        evaluators.KEYWORD_NIST_AI_RMF_S,
        evaluators.KEYWORD_PROBLEM_TYPE_IR,
        evaluators.KEYWORD_PROBLEM_TYPE_QA,
        evaluators.KEYWORD_PROBLEM_TYPE_SUM,
        evaluators.KEYWORD_ES_SUMMARIZE,
        evaluators.KEYWORD_METHOD_SEMANTIC_SIMILARITY,
        evaluators.KEYWORD_METHOD_TYPE_NON_DETERMINISTIC,
    ]

    KEY_COMPLETENESS = "completeness"
    KEY_FAITHFULNESS_CONV = "faithfulness_conv"
    KEY_FAITHFULNESS_ZS = "faithfulness_zs"

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=KEY_COMPLETENESS,
                display_name="Completeness",
                description=(
                    "Completeness metric is calculated using distance of "
                    "embeddings between the reference and faithful parts of summary."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=True,
            ),
            commons.MetricMeta(
                key=KEY_FAITHFULNESS_CONV,
                display_name="Faithfulness (SummaC Conv)",
                description=(
                    "The faithfulness metric measures how well the summary preserves "
                    "the meaning and factual content of the original text. "
                    "SummaC Conv is a trained model consisting of a single learned "
                    "convolution layer compiling the distribution of entailment scores "
                    "of all document sentences into a single score."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=KEY_FAITHFULNESS_ZS,
                display_name="Faithfulness (SummaC ZS)",
                description=(
                    "The faithfulness metric measures how well the summary preserves "
                    "the meaning and factual content of the original text. "
                    "SummaC ZS performs zero-shot aggregation by combining "
                    "sentence-level scores using max and mean operators. This metric "
                    "is more sensitive to outliers than Summac Conv."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
            ),
        ]
    )

    _parameters = [
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        ),
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.Evaluator._get_custom_param_min_test_case(),
    ]
    _modules_needed_by_name = [
        h2o_sonar_config.DEP_UMAP,
        h2o_sonar_config.DEP_NLTK,
        h2o_sonar_config.DEP_SENTENCE_TRANSFORMERS,
        h2o_sonar_config.DEP_TRANSFORMERS,
    ]

    # models used by the evaluator
    e_model_baai_bge = caching.MODEL_BAAI_BGE_SMALL_EN
    e_model_vitamin_c = caching.MODEL_ALBERT_XLARGE_VITAMINC

    _brief_description = """This summarization evaluator, which does
**not require a reference summary**, uses **two faithfulness metrics** based on
SummaC (Conv and ZS) and **one completeness metric**.

- Compatibility: RAG and LLM models."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- The question is the text to be summarized and the actual answer is
  the generated summary.
- Models that calculate the metrics work at the sentence granularity.
- **Completeness** metric (primary) ~ geometric completeness measure:
   - The goal is to measure the completeness of a summary of a context in a geometric
     way as the ratio of the approximate area covered by the summary sentence
     embeddings in reduced dimensionality space to the approximate area covered
     by the context sentence embeddings in reduced dimensionality space.
   - A sentence transformer is used to create an embedding for each sentence
     in the summary and each sentence in the context.
   - Umap, trained on the context points, is used to reduce the dimensionality
     of the sentence embeddings.  Right now, the dimension of the reduced space is 5,
     if there are enough context sentences to use 5D space.
   - For each summary point, the euclidean distance between the summary point
     and the context points are calculated. If the summary point is close enough
     to a context point it is redefined as the closest context point.
     If it is not close to any context point, it is thrown out (the threshold distance
     is set as the 50th percentile of the distance matrix distances fo
     the context points). This prevents the completeness metric from being greater
     than 1 and throws out summary sentences that aren’t grounded in the context.
   - For each set of points the “three segment distance” is calculated by finding
     the longest point to point segment (euclidean distance in reduced space),
     then adding the longest additional segment to the first segment’s endpoints.
   - The completeness measure is the ratio of the three segment distance for
     the summary points to the three segment distance for the context points.
- **SummaC Conv** metric:
   - Trained model consisting of a single learned **convolution layer** compiling
     the distribution of entailment scores of all document sentences into
     a single score.
- **SummaC ZS** metric:
   - The model performs **zero-shot** aggregation by combining sentence-level scores
     using `max` and `mean` operators. This metric is more sensitive to outliers
     than `Summac Conv`.

See also:

- 3rd party SummaC library used: https://github.com/tingofurro/summac
- SummaC paper: https://arxiv.org/abs/2111.09525
- Embedding model used:
  [{e_model_baai_bge}](https://huggingface.co/{e_model_baai_bge}) - "BAAI General
  Embedding" - a suite of open-source text embedding models developed by
  the Beijing Academy of Artificial Intelligence (BAAI).""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.model_conv = None
        self.model_zs = None
        self.args = None
        self.problems = []
        self.log_name = "Summarization evaluator"

        configured_device = h2o_sonar_config.config.resolve_gpu_cpu_device(
            result_format="str"
        )
        self._device = (
            h2o_sonar_config.H2oSonarConfig.VALUE_CPU
            if not configured_device
            else configured_device
        )

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not HAS_REQUIRED_PACKAGES:
            self.logger.warning(
                self._check_compatibility_pckg_err_msg(
                    ["torch", "transformers", "nltk", "sentence-transformers"]
                )
            )
            return False

        if not _CONFIG_MD5_SUCCESSFUL:
            self.logger.error(
                f"{self.log_name}: MD5 hashing for joblib is not available "
                f"(reconfiguration finished with {_CONFIG_MD5_SUCCESSFUL}) "
                f"- NOT COMPATIBLE"
            )
            return False

        if not self.models:
            self.logger.warning(
                f"{self.log_name}: no RAG/LLM models found for evaluation: "
                f"{[m.key for m in self.models]} - NOT COMPATIBLE"
            )
            return False

        if not evaluators.Evaluator._check_llm_dataset_compatibility(
            self, params=params, evaluator_keywords=self._keywords
        ):
            return False

        # check that at least one row has actual answer
        if not self._check_llm_dataset_field_presence(
            params=params,
            require_actual_answer=True,
            require_expected_answer=False,
        ):
            return False

        return True

    def setup(self, model, persistence, **kwargs):
        evaluators.Evaluator.setup(self, model, persistence, **kwargs)

        self._resolve_evaluator_params()

        caching.cache_nltk_punkt()

        self.log_name = f"Summarization evaluator {self.mli_key}/{self.key}"

    _ERR_MSG_SHORT_DOC = "Prompt text to be summarized is too short!"
    _ERR_MSG_SHORT_SUMMARY = "Actual answer summary is too short!"

    def _add_problem_for_short_text(
        self,
        error_message: str,
        row: datasets.LlmDataset.LlmDatasetRow,
        model_name: str,
        evaluator: evaluators.Evaluator,
    ):
        t_html_fragment = e10s.GlobalHtmlFragmentExplanation

        description = (
            (
                f"Evaluator was not able to evaluate the summary generated by "
                f"{model_name} because the reference document is too short: '{row.i}'. "
                f"It has either less than 2 sentences or its embedding does not have "
                f"sufficient size."
            )
            if self._ERR_MSG_SHORT_DOC in error_message
            else (
                f"Evaluator was not able to evaluate the summary of the test case "
                f"{row.key} because this summary created by {model_name} model is too "
                f"short: '{row.actual_output}'. It has less than 2 sentences or its "
                f"embedding does not have sufficient size."
            )
        )

        # IMPROVE: format the description using HTML
        html = airium.Airium()
        html(description)

        actions_description = (
            (
                f"This is not a problem with the model, but with the evaluation / "
                f"input data. Remove the test case from the evaluation, provide "
                f"a longer reference document or do not run "
                f"{SummarizationEvaluator._display_name} evaluator for this input."
            )
            if self._ERR_MSG_SHORT_DOC in error_message
            else (
                f"Please check the actual answer of the model and determine if it is "
                f"a valid summary. If it is, then do not run  "
                f"{SummarizationEvaluator._display_name} evaluator for this input. "
                f"Otherwise experiment with the input data length, content and "
                f"structure to determine why the model produces such summary."
            )
        )

        problem_type = (
            "input data"
            if self._ERR_MSG_SHORT_DOC in error_message
            else "summarization"
        )

        problem_code = (
            problems.AVIDProblemCode.P0100_DATA
            if self._ERR_MSG_SHORT_DOC in error_message
            else problems.AVIDProblemCode.P0200_MODEL
        )

        # make sure to add the problem to indicate the evaluation failure
        problem = problems.ProblemAndAction(
            description=description,
            description_html=html,
            problem_type=problem_type,
            problem_code=problem_code,
            problem_attrs={
                problems.ProblemAndAction.ATTR_EVALUATOR_NAME: evaluator._display_name,
                # input dataset ~ test lab ~ key is the test case key
                problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [row.key],
                problems.ProblemAndAction.ATTR_ROW_KEYS: [(row.key, row.model_key)],
            },
            severity=problems.ProblemSeverity.high,
            actions_description=actions_description,
            evaluator_id=self.evaluator_id(),
            evaluator_name=self._display_name,
            explanation_type=t_html_fragment.explanation_type(),
            explanation_name=t_html_fragment.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
            resources=[],
        )
        self.add_problem(problem)

    def evaluate(self, llm_testset, **kwargs) -> list:
        save_llm_result = self.args.get(
            evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
            evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
        )

        metrics_threshold = self.args.get(
            evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
            evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
        )
        self._metrics_meta.set_threshold(metrics_threshold)

        # models: key -> model
        key_2_evaluated_model = {m.key: m for m in self.models}

        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())
        eval_results = datasets.LlmEvalResults()
        eval_failures = 0

        inputs = [r.i for r in llm_dataset.inputs]
        actual_outputs = [r.actual_output for r in llm_dataset.inputs]
        scores, exceptions = self.calculate_scores(inputs, actual_outputs)
        for i, (r, completeness, faithfulness_conv, faithfulness_zs) in enumerate(
            zip(
                llm_dataset.inputs,
                scores[self.KEY_COMPLETENESS],
                scores[self.KEY_FAITHFULNESS_CONV],
                scores[self.KEY_FAITHFULNESS_ZS],
                strict=False,
            )
        ):
            # handle actual answer retrieval error ~ RAG/LLM client crash
            if evaluators.Evaluator._is_internal_err_answer(r.actual_output):
                # set WORST metrics values
                eval_results.add_result(
                    datasets.LlmEvalResults.LlmEvalResultRow(
                        dataset_row=r,
                        metrics={
                            self.KEY_COMPLETENESS: 0.0,
                            self.KEY_FAITHFULNESS_CONV: 0.0,
                            self.KEY_FAITHFULNESS_ZS: 0.0,
                        },
                    )
                )
            elif i not in exceptions:
                # add to result
                eval_results.add_result(
                    datasets.LlmEvalResults.LlmEvalResultRow(
                        dataset_row=r,
                        metrics={
                            self.KEY_COMPLETENESS: completeness,
                            self.KEY_FAITHFULNESS_CONV: faithfulness_conv,
                            self.KEY_FAITHFULNESS_ZS: faithfulness_zs,
                        },
                    )
                )
            else:
                eval_failures += 1
                evaluated_model = key_2_evaluated_model.get(r.model_key)
                model_name = (
                    evaluated_model.llm_model_name if evaluated_model else r.model_key
                )
                excs = exceptions[i]
                for exc in excs:
                    self.logger.warning(
                        f"{self.log_name}: Evaluation of row {r.key} for model "
                        f"{model_name}"
                        f"failed - {exc}"
                        f"\n{traceback.format_exc()}"
                    )
                    self._add_problem_for_short_text(
                        error_message=str(exc),
                        row=r,
                        model_name=model_name,
                        evaluator=self,
                    )

        if eval_failures == len(llm_dataset.inputs):
            raise ValueError(
                f"{self.log_name}: evaluation of all test cases for all models "
                f"failed - no results."
            )

        #
        # NORMALIZATION of the evaluation RESULTS
        #

        sort_by_metric = self._metrics_meta.get_primary_metric().key

        # EXPLANATIONS
        explanations = []

        # EXPLANATION: all data (per prompt metrics)
        if save_llm_result:
            eval_results_explanation = e10s.LlmEvalResultsExplanation(
                evaluator=self,
                display_name="Summarization evaluation results",
                display_category=e10s.Explanation.DISPLAY_CAT_LLM,
                eval_results=eval_results,
            )
            # FORMATS of the explanation: JSon, CSV, DataTable
            eval_results_explanation.add_json_format()
            eval_results_explanation.add_csv_format()
            eval_results_explanation.add_datatable_format()
            explanations.append(eval_results_explanation)

        # EXPLANATION: heatmap leaderboard
        heatmap_explanation = e10s.LlmHeatmapLeaderboardExplanation.from_eval_results(
            evaluator=self,
            eval_results=eval_results,
            metrics_meta=self._metrics_meta,
            key_2_evaluated_model=key_2_evaluated_model,
            display_name=f"{self._display_name} leaderboard",
            display_category=e10s.GlobalSummaryFeatImpExplanation.DISPLAY_CAT_LLM,
            logger=self.logger,
        )
        heatmap_explanation.add_json_format(
            threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                self._metrics_meta.get_primary_metric().threshold,
            )
        )
        heatmap_explanation.add_markdown_format(sort_by_metric_id=sort_by_metric)
        heatmap_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=sort_by_metric
        )
        explanations.append(heatmap_explanation)

        # PROBLEMS for alerts and actionability
        self._diagnose_problems(
            leaderboard_explanation=heatmap_explanation, eval_results=eval_results
        )

        # INSIGHTS
        self._diagnose_insights(leaderboard_explanation=heatmap_explanation)

        # EXPLANATION: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                html_explanation = e10s.GlobalHtmlFragmentExplanation(
                    evaluator=self,
                    display_name=f"{self._display_name} leaderboard as HTML",
                    display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
                )
                html_explanation.add_html_format(
                    str(
                        heatmap_explanation.as_html(
                            sort_by_metric_id=sort_by_metric,
                        )
                    )
                )
                explanations.append(html_explanation)
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML fragment explanation creation failed: "
                    f"{ex}\n{traceback.format_exc()}"
                )

        return explanations

    def calculate_scores(
        self, inputs: list[str], actual_outputs: list[str]
    ) -> tuple[dict[str, float], dict]:
        SummaCZS, SummaCConv = load_summac()  # noqa
        with resource_mgmt.PytorchModelLifeCycleManager(
            SummaCConv(
                models=["vitc"],
                bins="percentile",
                granularity="sentence",
                nli_labels="e",
                device=self._device,
                start_file="default",
                agg="mean",
            )
        ) as model_conv:
            faithfulness_conv = model_conv.score(inputs, actual_outputs)["scores"]

        with resource_mgmt.PytorchModelLifeCycleManager(
            SummaCZS(granularity="sentence", model_name="vitc", device=self._device)
        ) as model_zs:
            imgs = model_zs.score(inputs, actual_outputs)["images"]
            faithfulness_zs = [np.mean(np.amax(img[0], axis=0)) for img in imgs]

        completenesses, exceptions = self.summary_completeness_batch(
            actual_outputs, inputs
        )
        return {
            self.KEY_COMPLETENESS: completenesses,
            self.KEY_FAITHFULNESS_CONV: faithfulness_conv,
            self.KEY_FAITHFULNESS_ZS: faithfulness_zs,
        }, exceptions

    # Several summac package faith score calculations
    def summac_faith_score1(self, summary: str, refs: str) -> float:
        """Calculate the summac convolution faithfulness score using
        the summac convolution"""
        all_conv = self.model_conv.score([refs], [summary])
        return all_conv["scores"][0]

    def summac_faith_score2(self, summary: str, refs: str) -> float:
        """Max summac/NLI score for individual sentences"""
        all_ind = self.model_zs.score([refs], [summary])
        img = all_ind["images"]
        max_per_sentence = np.amax(img[0][0], axis=0)
        return np.mean(max_per_sentence)

    # Split docs into sentences
    @staticmethod
    def split_sentences(text: str) -> list[str]:
        # """ Split the data into sentences"""
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > 10]
        return sentences

    def summary_completeness_batch(
        self,
        summaries: list[str],
        docs: list[str],
        nearest_neighbors: int = 10,
        umap_dimension: int = 5,
    ) -> tuple[list | None, dict]:
        docs = [self.split_sentences(doc) for doc in docs]
        split_summaries = [self.split_sentences(summary) for summary in summaries]

        exceptions = collections.defaultdict(list)

        for i, doc in enumerate(docs):
            if len(doc) < 2:
                exceptions[i].append(
                    ValueError(
                        f"{SummarizationEvaluator._ERR_MSG_SHORT_DOC} "
                        f"(tokenization to sentences)"
                    )
                )

        for i, split_summary in enumerate(split_summaries):
            if len(split_summary) < 2:
                exceptions[i].append(
                    ValueError(
                        f"{SummarizationEvaluator._ERR_MSG_SHORT_SUMMARY} "
                        f"(tokenization to sentences)"
                    )
                )

        # Pre-calculate embeddings
        embeddings = []
        embeddings_s = []
        with resource_mgmt.PytorchModelLifeCycleManager(
            sentence_transformers.SentenceTransformer(
                SummarizationEvaluator.e_model_baai_bge,
                device=h2o_sonar_config.config.resolve_gpu_cpu_device(
                    result_format="str"
                ),
                revision=caching.REVISIONS_FOR_MODEL[
                    SummarizationEvaluator.e_model_baai_bge
                ],
            )
        ) as embedding_model:
            for i, (doc, summary) in enumerate(
                zip(docs, split_summaries, strict=False)
            ):
                if i in exceptions:
                    embeddings.append(None)
                    embeddings_s.append(None)
                    continue
                try:
                    embeddings.append(
                        embedding_model.encode(doc, show_progress_bar=False)
                    )
                except Exception:
                    embeddings.append(None)
                    exceptions[i].append(
                        ValueError(
                            f"{SummarizationEvaluator._ERR_MSG_SHORT_DOC} "
                            f"(embeddings error)"
                        )
                    )
                else:
                    if len(embeddings[-1]) < 1:
                        exceptions[i].append(
                            ValueError(
                                f"{SummarizationEvaluator._ERR_MSG_SHORT_DOC} "
                                f"(small embeddings)"
                            )
                        )
                if i in exceptions:
                    embeddings_s.append(None)
                    continue
                try:
                    embeddings_s.append(
                        embedding_model.encode(summary, show_progress_bar=False)
                    )
                except Exception:
                    embeddings_s.append(None)
                    exceptions[i].append(
                        ValueError(
                            f"{SummarizationEvaluator._ERR_MSG_SHORT_SUMMARY} "
                            f"(embeddings error)"
                        )
                    )
                else:
                    if len(embeddings_s[-1]) < 1:
                        exceptions[i].append(
                            ValueError(
                                f"{SummarizationEvaluator._ERR_MSG_SHORT_SUMMARY} "
                                f"(small embeddings)"
                            )
                        )

        cov_dist_segment3s = []
        reduced_emb_references = []
        reduced_embeddings_summaries = []

        for i in range(len(embeddings)):
            if i in exceptions:
                cov_dist_segment3s.append(None)
                reduced_emb_references.append(None)
                reduced_embeddings_summaries.append(None)
                continue

            # Reduced embeddings for the references
            # For very small docs, a smaller dimension might be necessary
            # len(docs) - 2 is suggested in https://github.com/lmcinnes/umap/issues/201
            n_comp = max(min(umap_dimension, len(docs[i]) - 2), 2)

            # Dimensionality reduction on the document sentences
            with resource_mgmt.UmapModelLifeCycleManager(
                device=self._device,
                n_neighbors=nearest_neighbors,
                n_components=n_comp,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            ) as umap_model:
                try:
                    (
                        cov_dist_segment3,
                        reduced_emb_reference,
                        reduced_embeddings_summary,
                    ) = self._reduce_embeddings(
                        embeddings[i], embeddings_s[i], umap_model
                    )

                    cov_dist_segment3s.append(cov_dist_segment3)
                    reduced_emb_references.append(reduced_emb_reference)
                    reduced_embeddings_summaries.append(reduced_embeddings_summary)
                except Exception as e:
                    cov_dist_segment3s.append(None)
                    reduced_emb_references.append(None)
                    reduced_embeddings_summaries.append(None)
                    exceptions[i].append(e)

        results = []
        for i in range(len(reduced_emb_references)):
            if reduced_embeddings_summaries[i] is None:
                results.append(None)
                continue
            filtered = []
            # Filtered only includes summary sentences close to at least
            # one point in the reference data.
            # That way the coverage data doesn't include unfaithful points
            for sentence_idx in range(len(reduced_embeddings_summaries[i])):
                row = reduced_embeddings_summaries[i][sentence_idx, :]
                temp = reduced_emb_references[i] - row
                datadist = temp[:, 0] ** 2
                for dint in range(1, temp.shape[1]):
                    datadist += temp[:, dint] ** 2
                # Kim's workaround {
                # if min(datadist) < large_threshold:
                #    filtered.append(list(row))
                filtered.append(reduced_emb_references[i][np.argmin(datadist), :])
                # } end of Kim's workaround

            # Calculate a distance matrix for the current summary based on
            # the filtered data.
            bed = np.array(filtered)
            distance = pairwise_distances_wrapper(bed)
            # Calculate the two coverage distances.
            coverage_distance2 = segment_calc(distance, 1)

            cov_dist_segment3s[i].append(coverage_distance2)
            results.append(
                min(
                    1.0,
                    max(
                        0.0, float(cov_dist_segment3s[i][1] / cov_dist_segment3s[i][0])
                    ),
                )
            )
        return results, exceptions

    def _reduce_embeddings(self, embeddings, embeddings_s, umap_model):
        # Calculate a distance matrix for the full data
        try:
            reduced_emb_references = umap_model.fit_transform(embeddings)
        except Exception as e:
            raise ValueError(
                f"{SummarizationEvaluator._ERR_MSG_SHORT_DOC} (umap error)"
            ) from e
        reduced_emb_references = np.array(reduced_emb_references)
        distance_references = pairwise_distances_wrapper(reduced_emb_references)
        # Find a threshold that defines "Nearby" as the 20th percentile distance for
        # consecutive sentences.
        diff = reduced_emb_references[1:, :] - reduced_emb_references[:(-1), :]
        dist = diff[:, 0] ** 2
        for difind in range(1, diff.shape[1]):
            dist += diff[:, difind] ** 2
        # large_threshold = np.percentile(dist, 50)
        # Calculate two segment based coverage metrics for the full document
        # The first is the largest single segment and the second is the largest
        # 3-segment distance
        coverage_distance2_reference = segment_calc(distance_references, 1)
        cov_dist_segment3 = [coverage_distance2_reference]
        # Calculate the reduced embeddings based on the reference data model
        reduced_embeddings = umap_model.transform(embeddings_s)
        reduced_embeddings = pd.DataFrame(reduced_embeddings)
        reduced_embeddings_summary = np.array(reduced_embeddings)
        return cov_dist_segment3, reduced_emb_references, reduced_embeddings_summary

    def _diagnose_problems(
        self,
        leaderboard_explanation: e10s.LlmHeatmapLeaderboardExplanation,
        eval_results: datasets.LlmEvalResults,
    ):
        # low test case count
        self._diagnose_low_test_case_problem(
            eval_results=eval_results,
            models=self.models,
            test_case_minimum=self.args.get(evaluators.Evaluator.PARAM_MIN_TEST_CASES),
        )
        # threshold failures
        problems.problems_for_heat_leaderboard(
            evaluator=self,
            leaderboard=leaderboard_explanation,
            metric_threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                self._metrics_meta.get_primary_metric().threshold,
            ),
            primary_metric_meta=self._metrics_meta.get_primary_metric(),
            problem_type="summarization",
            problem_code=problems.AVIDProblemCode.P0200_MODEL,
            actions_description=(
                "To improve summarizations, focus on three key areas:  refinement, "
                "evaluation, and training data. The LLM can be equipped with "
                "auto-refinement modules that assess its own summaries and identify "
                "areas for improvement, like missing key points. Additionally, "
                "using metrics that go beyond surface-level similarity to "
                "human-written summaries can guide the training process. Finally, "
                "incorporating diverse and high-quality summaries into the training "
                "data provides the LLM with better examples to learn from, leading "
                "to more comprehensive and informative summaries."
            ),
            explanation_type=e10s.GlobalHtmlFragmentExplanation.explanation_type(),
            explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    def _diagnose_insights(
        self, leaderboard_explanation: e10s.LlmHeatmapLeaderboardExplanation
    ):
        t_html_fragment = e10s.GlobalHtmlFragmentExplanation

        leaderboard_explanation.get_insights(
            metrics_meta=self._metrics_meta,
            extra_description_best=(
                "This model produces responses that most closely resemble the expected "
                "summaries based on the used metrics."
            ),
            insight_type="summarization",
            explanation_type=t_html_fragment.explanation_type(),
            explanation_name=t_html_fragment.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
