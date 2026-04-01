# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.


class PdCache:
    """PD's cache for ICE computed to compute PD."""

    KEY_ICE_CACHE = "ice"

    _ERROR_MSG_ICE = (
        "ICE cache filter must be dict with keys: feature name, "
        "class name and list of instance (row) identifiers - "
        "check docstring."
    )

    def __init__(self):
        self._ice_cache = None
        self._ice_cache_path = None

    def _check_ice_cache(self, features, ice_cache, ice_cache_path):
        """Initialize disk cache and return initialized memory ICE cache."""
        if ice_cache is not None:
            self._ice_cache = ice_cache
            self._check_ice_cache_filter(features)
            extra_explanations_cache = {self.KEY_ICE_CACHE: {}}
        else:
            extra_explanations_cache = None

        if ice_cache_path is not None:
            if not isinstance(ice_cache_path, str):
                raise ValueError("ICE cache path must be string")
            self._ice_cache_path = ice_cache_path

        return extra_explanations_cache

    def _check_ice_cache_filter(self, features):
        if self._ice_cache is not None:
            if not isinstance(self._ice_cache, dict):
                raise ValueError(
                    "Wrong type - features not dict: " + self._ERROR_MSG_ICE
                )
            if self._ice_cache:
                for feature in self._ice_cache:
                    self._check_ice_cache_filter_feature(features, feature)

    def _check_ice_cache_filter_feature(self, features, feature):
        if feature in features:
            if not isinstance(self._ice_cache[feature], dict):
                raise ValueError(
                    "Wrong type - classes not dict: " + self._ERROR_MSG_ICE
                )
            for clazz in self._ice_cache[feature]:
                if not isinstance(self._ice_cache[feature][clazz], list):
                    raise ValueError(
                        "Wrong type - list of row identifiers: " + self._ERROR_MSG_ICE
                    )
                if not self._ice_cache[feature][clazz]:
                    raise ValueError(
                        "Wrong type - list of row "
                        "identifiers must be non-empty:"
                        " " + self._ERROR_MSG_ICE
                    )
        else:
            raise ValueError(
                "Unknown feature '" + str(feature) + "' " + self._ERROR_MSG_ICE
            )
