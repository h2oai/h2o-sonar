# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.lib.container import explainer_container


class LocalExplainerContainer(explainer_container.ExplainerContainer):
    """Mock explainer container."""

    TYPE_ID = "MOCK_EXPLAINER_CONTAINER"

    pass
