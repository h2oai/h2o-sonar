# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar import errors


class CachingMethod:
    """Methods cache."""

    def __init__(self):
        self._explanations_cache = None
        self._extra_explanations_cache = None
        self._json_cache = None

    def explanations(self, kind=None):
        """Return previously computed explanations from cache.

        Parameters
        ----------
        kind: str
            Specify type of the methods to return. If no value is
            specified, then default methods is returned, else use string
            key to get methods specific explanations.

        Returns
        -------
        dict[string or tuple, dict[string, DataFrame]]
            Previously calculated explanations - dictionary with feature name as
            key and value which is dictionary with class (
            regression/binomial/multinomial) as key and Pandas DataFrame value.

        """

        if self._explanations_cache is None:
            raise errors.MliError("No explanations", "Run methods() method first")

        if kind is not None:
            if kind in ["ice"]:
                return self._extra_explanations_cache[kind]
            raise ValueError("Unknown methods kind '" + kind + "'")
        return self._explanations_cache

    def evict_explanations(self):
        """Clear explanations cache."""
        self._explanations_cache = None
        self._extra_explanations_cache = None
