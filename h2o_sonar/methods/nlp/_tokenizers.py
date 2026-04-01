# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import ast
import re

import datatable as dt
import numpy as np
import pandas as pd

from h2o_sonar.config import config
from h2o_sonar.lib.api import commons
from h2o_sonar.utils import sanitization


try:
    from sklearn.feature_extraction import text
    from sklearn.feature_extraction.text import TfidfVectorizer

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MliTokenizer:
    def fit(self, dataset: dt.Frame):
        raise NotImplementedError

    def transform(self, dataset: dt.Frame):
        raise NotImplementedError

    def transform_tokens(self, dataset: dt.Frame):
        raise NotImplementedError

    def get_n_tokens(self, n=20, mode=config.mli_nlp_min_token_mode):
        raise NotImplementedError

    def get_n_token_frequencies(self, n=20, mode=config.mli_nlp_min_token_mode):
        raise NotImplementedError

    @staticmethod
    def tokenizer_name():
        raise NotImplementedError


class TfIdfBasedMliTokenizer(MliTokenizer):
    def __init__(
        self,
        text_features_idx,
        max_features=100,
        token_pattern=r"\b\w+\b",
        min_df=config.mli_nlp_min_df,
        max_df=config.mli_nlp_max_df,
        min_ngram=config.mli_nlp_min_ngram,
        max_ngram=config.mli_nlp_max_ngram,
        stop_words=config.mli_nlp_stop_words,
        append_to_english_stop_words=config.mli_nlp_append_to_english_stop_words,
        use_stop_words=config.mli_nlp_use_stop_words,
    ):
        if not HAS_SKLEARN:
            commons.raise_opt_import_err("scikit-learn")

        self.max_df = max_df
        self.min_df = min_df
        self.token_pattern = token_pattern
        self.text_features_idx = text_features_idx
        self.max_features = max_features
        self.ngram_range = (min_ngram, max_ngram)

        self.vectorizer = None  # Stores current per-feature vectorizer
        self.per_feature_vectorizers: dict[str, TfidfVectorizer] = {}
        self.global_importances: dict[str, float] = {}
        self.tfidf_matrices: dict[str, dt.Frame] = {}
        self.stop_words = stop_words
        self.append_to_english_stop_words = append_to_english_stop_words
        self.use_stop_words = use_stop_words

        if self.use_stop_words:
            if self.stop_words != "english":
                sw = ast.literal_eval(self.stop_words)
                self.stop_words = [i.strip() for i in sw]

            if self.append_to_english_stop_words and self.stop_words != "english":
                self.stop_words = text.ENGLISH_STOP_WORDS.union(self.stop_words)
        else:
            self.stop_words = None

    def fit(self, dataset: dt.Frame):
        """Fits a separate vectorizer per each text feature (column)

        After all vectorizers are fitted, `self.global_importances` is constructed,
        which is a dict. Keys of the dict are text tokens wrapped with feature name
        they belong to (e.g. `Text_column('abc')`). Values of the dict are values
        of importances of these text tokens relative to their corpus (i.e. features).

        Returns
        -------
        TfIdfBasedMliTokenizer :
            Current instance of TfIdfBasedMliTokenizer

        """
        for id_ in self.text_features_idx:
            colname = self._get_feature_name(dataset, id_)
            data_pd = dataset[:, id_].to_pandas().astype(str).iloc[:, 0]
            try:
                # NOTE: In case user settings fail, try with default <0,1> range
                tf_idf_matrix = self.fit_vectorizer(data_pd, self.min_df, self.max_df)
            except ValueError:
                tf_idf_matrix = self.fit_vectorizer(data_pd, 0, 1)
            self.tfidf_matrices[colname] = dt.Frame(tf_idf_matrix.toarray())
            self.tfidf_matrices[colname].names = self.vectorizer.get_feature_names()

            self.per_feature_vectorizers[colname] = self.vectorizer
            sum_ = tf_idf_matrix.sum(axis=0).getA()[0]
            means = sum_ / tf_idf_matrix.shape[0]

            token_names = self.get_transformed_token_names(self.vectorizer, colname)
            self.global_importances.update(
                dict(zip(token_names, means.tolist(), strict=False))
            )

        return self

    def fit_vectorizer(self, column_data, min_df, max_df):
        if not HAS_SKLEARN:
            commons.raise_opt_import_err("scikit-learn")

        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            max_features=self.get_max_features(),
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
        )
        return self.vectorizer.fit_transform(column_data)

    def get_transformed_token_names(self, vectorizer, column: str) -> list[str]:
        """Transform raw text tokens, wrapping them with column to which they belong

        Parameters
        ----------
        vectorizer: TfidfVectorizer
            Fitted vectorizer
        column: str
            Column name to which tokens belong.

        """
        return [
            self.encode_token(name, column) for name in vectorizer.get_feature_names()
        ]

    @staticmethod
    def encode_token(raw_token: str, column: str) -> str:
        """Encode raw text token, wrapping it with column to which they belong
        This is an inverse operation of `decode_token`

        Example:
        Token `and` in column `description`
        `and -> description('and')`
        """
        return f"{column}('{raw_token}')"

    @staticmethod
    def decode_token(token: str) -> tuple[str, str]:
        """Decode token and get original text token and column
        from transformed token name.
        If token pattern is not matching, return token as column and empty original
        token
        This is an inverse operation of `encode_token`

        Example:
        Token `and` in column `description`
        `description('and') -> description, and`

        Returns
        -------
        Tuple[str, str]
            Decoded Tuple (column, original_token)
        """
        token_pattern = r"(.*?)\s*\('(.*?)'\)"
        match = re.search(token_pattern, token)
        if match:
            return match.group(1), match.group(2)
        else:
            return token, ""

    def get_max_features(self):
        if self.max_features is None or self.max_features <= 0:
            return None
        return self.max_features

    def transform(self, dataset: dt.Frame):
        final_frame = dt.Frame()
        for id_ in self.text_features_idx:
            colname = self._get_feature_name(dataset, id_)
            vectorizer = self.per_feature_vectorizers[colname]
            data_pd = dataset[:, id_].to_pandas().astype(str).iloc[:, 0]
            tf_idf = vectorizer.transform(data_pd)
            token_names = self.get_transformed_token_names(vectorizer, colname)
            importances = dt.Frame(tf_idf.todense().A, names=token_names)
            final_frame.cbind(importances)

        return final_frame

    def transform_tokens(self, dataset: dt.Frame):
        """
        Retrieves list of tokens per each dataset row, where each token (term)
        has non-zero value for TF-IDF for specific row.

        Parameters
        ----------
        dataset: dt.Frame
            frame produced by `self.transform`, containing
            concatenated tf-idf computed for all columns separately

        Returns
        -------
        list[np.array()]
            List of arrays of tokens occuring in each dataset row. Shape M x 1
            where M is number of rows
        """
        per_row_tokens = [np.array([])] * dataset.nrows
        for id_ in self.text_features_idx:
            colname = self._get_feature_name(dataset, id_)
            vectorizer = self.per_feature_vectorizers[colname]
            data_pd = dataset[:, id_].to_pandas().astype(str).iloc[:, 0]
            tf_idf = vectorizer.transform(data_pd)
            tokens = vectorizer.inverse_transform(tf_idf)
            for row_id, row in enumerate(tokens):
                # Wrap raw tokens with column names
                per_row_tokens[row_id] = np.append(
                    per_row_tokens[row_id],
                    [
                        np.array(
                            [self.encode_token(str(token), colname) for token in row]
                        )
                    ],
                )

        return per_row_tokens

    def get_n_tokens(self, n=20, mode=config.mli_nlp_min_token_mode):
        """
        Returns the top N tokens ordered by importance. You can obtain all the tokens
        by passing a non-positive number as the parameter.

        Parameters
        ----------
        n: int
        mode: str

        Returns
        -------

        """
        return self.get_n_token_frequencies(n, mode)[:, "Tokens"].to_list()[0]

    def get_n_token_frequencies(self, n=20, mode=config.mli_nlp_min_token_mode):
        df_tfidf = pd.DataFrame(
            {
                "Tokens": list(self.global_importances.keys()),
                "Importances": list(self.global_importances.values()),
            }
        )
        np.random.seed(1234)
        df_tfidf = df_tfidf.iloc[np.random.permutation(len(df_tfidf))]
        df_tfidf = df_tfidf.sort_values("Importances", ascending=False).reset_index(
            drop=True
        )

        if n > 0:
            if mode == "top":
                return dt.Frame(df_tfidf[0:n])
            elif mode == "bottom":
                return dt.Frame(df_tfidf[-n:])
            elif mode == "linspace":
                idx = np.round(np.linspace(0, len(df_tfidf) - 1, n)).astype(int)
                return dt.Frame(df_tfidf.loc[idx, :])

        return dt.Frame(df_tfidf)

    @staticmethod
    def tokenizer_name():
        return "tfidf"

    def _concat_text_features(self, data):
        return (
            data[:, self.text_features_idx]
            .to_pandas()
            .astype(str)
            .apply(lambda x: " ".join(x), axis=1)
        )

    @staticmethod
    def _get_feature_name(dataset: dt.Frame, feature_id: str | int) -> str:
        if type(feature_id) is str:
            colname = str(feature_id)
        else:
            colname = dataset.names[feature_id]

        return sanitization.sanitize_names(names=colname)


TOKENIZERS = {TfIdfBasedMliTokenizer.tokenizer_name(): TfIdfBasedMliTokenizer}
