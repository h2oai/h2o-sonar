# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import functools
import pickle

import numpy

from h2o_sonar.config import config
from h2o_sonar.lib.api import commons
from h2o_sonar.methods.nlp._tokenizers import TfIdfBasedMliTokenizer


try:
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import scipy

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TextImportance:
    def __init__(
        self,
        text,
        tokens,
        importances,
        y_true,
        y_pred,
        y_pred_label=None,
        text_feature=None,
    ):
        self.text = text
        self.tokens = tokens
        self.importances = importances
        assert all(numpy.array(importances) <= 1) and all(
            numpy.array(importances) >= -1
        ), "Importances need to be in [-1, 1]"
        self.tokens2attentions = dict(zip(self.tokens, self.importances, strict=False))

        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_label = y_pred_label
        self.text_feature = text_feature

    def get_word_weightages_list(self, words=None):
        weightages = []
        for word in words:
            weightage = self.tokens2attentions.get(
                f"{word}",
                0,
            )
            weightages.append(weightage)
        sorted_indices = numpy.argsort(numpy.absolute(weightages))[::-1]
        words = numpy.array(words)[sorted_indices]
        weightages = numpy.round(numpy.array(weightages)[sorted_indices], 5)
        data = list(zip(map(str, words), map(float, weightages), strict=False))
        return dict(data)


class LinearModel:
    def __init__(
        self,
        tokenizer=None,
        max_features=None,
        C=1.0,
        max_iter=100,
        random_state=520,
        target=None,
        text_features_idx=None,
        text_feature=None,
        n_classes=None,
        run_tokenizer=True,
        vocab=None,
        config_overrides=None,
    ):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.target = target
        self.text_features_idx = text_features_idx
        self.text_feature = text_feature
        self.n_classes = n_classes
        self.run_tokenizer = run_tokenizer
        self.tokenizer = tokenizer
        self.config_overrides = config_overrides

        if self.config_overrides:
            config.update(config_overrides=self.config_overrides)
        if not self.tokenizer and self.run_tokenizer:
            c = config
            self.tokenizer = TfIdfBasedMliTokenizer(
                text_features_idx=text_features_idx,
                max_features=max_features,
                token_pattern=r"\b\w+\b",
                min_df=c.mli_nlp_min_df,
                max_df=c.mli_nlp_max_df,
                min_ngram=c.mli_nlp_min_ngram,
                max_ngram=c.mli_nlp_max_ngram,
                stop_words=c.mli_nlp_stop_words,
                append_to_english_stop_words=c.mli_nlp_append_to_english_stop_words,
                use_stop_words=c.mli_nlp_use_stop_words,
            )

        self._vocab = vocab
        self._coefficients = None
        self._class_labels = None
        self.model = None
        self.lb_enc = None

    @functools.cached_property
    def vocab(self):
        if len(self.text_feature) == 1:
            return numpy.array(
                self.tokenizer.get_transformed_token_names(
                    self.tokenizer.per_feature_vectorizers[self.text_feature[0]],
                    self.text_feature[0],
                )
            ).flatten()

        np_arrays = []
        for txt in self.text_feature:
            np_arrays.append(
                numpy.array(
                    self.tokenizer.get_transformed_token_names(
                        self.tokenizer.per_feature_vectorizers[txt],
                        txt,
                    )
                ).flatten()
            )
        return numpy.concatenate(np_arrays, axis=0)

    @property
    def coefficients(self):
        self._coefficients = self.model.coef_
        return self._coefficients

    @property
    def class_labels(self):
        self._class_labels = self.lb_enc.classes_
        return self._class_labels

    def fit(self, frame):
        if not HAS_SKLEARN:
            commons.raise_opt_import_err("scikit-learn")

        if self.n_classes >= 2:
            self.lb_enc = LabelEncoder()
            y = self.lb_enc.fit_transform(frame[:, self.target])
            self.n_classes = len(self.lb_enc.classes_)
        else:
            self.n_classes = 1
            y = frame[:, self.target]

        if self.run_tokenizer:
            x = self.tokenizer.fit(frame).transform(frame).to_pandas().values
        else:
            x = None

        if self.n_classes < 2:
            self.model = LinearRegression(fit_intercept=False)
        elif self.n_classes == 2:
            self.model = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                fit_intercept=False,
                random_state=self.random_state,
            )
        else:
            raise ValueError(
                "Only regression/binomial is supported for "
                "NLP Vectorizer + Linear Model explainer"
            )

        features = list(frame.names)

        if x is None and self.target in features:
            features.remove(self.target)

        self.model.fit(x if x is not None else frame[:, features], y)

        return self

    def predict(self, frame):
        if not HAS_SKLEARN:
            commons.raise_opt_import_err("scikit-learn")

        x = self.tokenizer.transform(frame).to_pandas()
        if isinstance(self.model, LogisticRegression):
            if self.n_classes < 2:
                raise ValueError(
                    "Cannot use Logistic Regression model to predict continous outcome"
                )
            elif self.n_classes == 2:
                y_pred = self.model.predict_proba(x)[:, 1]
            else:
                raise ValueError(
                    "Only regression/binomial is supported for "
                    "NLP Vectorizer + Linear Model explainer"
                )
        elif isinstance(self.model, LinearRegression):
            y_pred = self.model.predict(x)
        else:
            # TODO use inspect() to infer which method to use.
            raise NotImplementedError(
                "Only Linear and Logistic Regression models are currently supported."
            )
        return y_pred

    def get_text_importances(self, frame, y_trues, cut_off=0.5):
        if not HAS_SKLEARN:
            commons.raise_opt_import_err("scikit-learn")
        if not HAS_SCIPY:
            commons.raise_opt_import_err("scipy")

        # TODO normalize attentions by prediction.

        if self.run_tokenizer:
            tokenized_texts = scipy.sparse.csr_matrix(
                self.tokenizer.transform(frame).to_pandas().values
            )
        else:
            tokenized_texts = None
        y_preds = self.predict(frame)

        text_importances = []
        for text, tokenized_text, y_true, y_pred in zip(
            frame[:, self.text_feature],
            tokenized_texts if tokenized_texts is not None else frame,
            y_trues,
            y_preds,
            strict=False,
        ):
            idxs_non_zero = numpy.argwhere(tokenized_text.A[0] != 0).flatten()

            if self.n_classes < 2:
                y_pred_label = None
                y_pred_prob = y_pred
                importance = (
                    self.coefficients[0][idxs_non_zero]
                    * tokenized_text.A[0][idxs_non_zero]
                )
            elif self.n_classes == 2:
                y_pred_label = 1 if y_pred > cut_off else 0
                y_pred_label = str(self.lb_enc.inverse_transform([y_pred_label])[0])
                y_pred_prob = y_pred
                importance = (
                    self.coefficients[0][idxs_non_zero]
                    * tokenized_text.A[0][idxs_non_zero]
                )
            else:
                raise ValueError(
                    "Only regression/binomial is supported for "
                    "NLP Vectorizer + Linear Model explainer"
                )

                # TODO Use below when multinomial support is implemented
                # y_pred_index = numpy.argmax(y_pred)
                # y_pred_prob = y_pred[y_pred_index]
                # y_pred_label = str(
                #     self.lb_enc.inverse_transform([y_pred_index])[0]
                # )
                # importance = (
                #     self.coefficients[y_pred_index][idxs_non_zero]
                #     * tokenized_text.A[0][idxs_non_zero]
                # )
            # raw scores
            importance = numpy.clip(importance, -1, 1)
            tokens = self.vocab[idxs_non_zero]

            text_importances.append(
                TextImportance(
                    text,
                    tokens,
                    importance,
                    y_true,
                    y_pred_prob,
                    y_pred_label,
                    self.text_feature,
                )
            )
        return text_importances

    def get_most_important_words(self):
        if self.n_classes <= 2:
            indices = numpy.argsort(self.coefficients)
            return_dict = numpy.array(self.vocab)[indices]
            weights_dict = (
                self.coefficients[0][indices]
                if len(self.coefficients.shape) > 1
                else self.coefficients[indices]
            )
            results = (
                dict(zip(list(return_dict[0]), weights_dict[0], strict=False))
                if len(self.coefficients.shape) > 1
                else dict(zip(list(return_dict), weights_dict, strict=False))
            )
        else:
            raise ValueError(
                "Only regression/binomial is supported for "
                "NLP Vectorizer + Linear Model explainer"
            )
        return results

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump([self.tokenizer, self.model], f)

    def load(self, path):
        with open(path, "rb") as f:
            self.tokenizer, self.model = pickle.load(f)
        return self
