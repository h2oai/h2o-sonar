# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import contextlib
import gc

from h2o_sonar.lib.api import commons


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import cuml

    HAS_CUML = True
except ImportError:
    HAS_CUML = False


class GenericModelLifeCycleManager(contextlib.AbstractContextManager):
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        return self._model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model = None
        gc.collect()
        self._finalize()
        return False

    def _finalize(self):
        pass


class PytorchModelLifeCycleManager(GenericModelLifeCycleManager):
    def _finalize(self):
        if not HAS_TORCH:
            commons.raise_opt_import_err("torch")
        torch.cuda.empty_cache()


class UmapModelLifeCycleManager(GenericModelLifeCycleManager):
    def __init__(self, device, **kwargs):
        if device == "cpu":
            if not HAS_UMAP:
                commons.raise_opt_import_err("umap")
            super().__init__(umap.UMAP(**kwargs))
        else:
            if HAS_CUML:
                # currently not tested (dependency conflicts)
                super().__init__(cuml.UMAP(**kwargs))
            elif not HAS_UMAP:
                commons.raise_opt_import_err("umap")
            else:
                super().__init__(cuml.UMAP(**kwargs))

    def _finalize(self):
        pass
