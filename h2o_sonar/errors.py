# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.


class MliError(Exception):
    """MLI error."""

    # IMPORTANT: suggestion must be optional parameter to ensure pickleability
    def __init__(self, message, suggestion=None):
        """Create new MLI error.

        Parameters
        ----------
        message: str
            Error message.
        suggestion: str
            Suggestion how to solve the problem.

        """
        if suggestion:
            super().__init__(message + " - " + suggestion)
        else:
            super().__init__(message)

        self.suggestion = suggestion


class ExplainerCompatibilityError(MliError):
    """Explainer not compatible error."""

    pass


class UnknownExplainerError(MliError):
    """Explainer not known to explainer container error."""

    def __init__(self, explainer_id: str):
        MliError.__init__(
            self,
            message=(
                f"Unknown explainer - explainer ID '{explainer_id}' is not known to "
                f"the explainer container and therefore it cannot be run."
            ),
            suggestion="Please register to explainer in the explainer container first.",
        )


class MliPredictMethodError(MliError):
    """Predict method failure."""

    pass


class MliTypeError(MliError):
    """Wrong type error."""

    pass


class MliNotFoundError(MliError):
    """Entity not found error."""

    pass


class MliUnsupportedError(MliError):
    pass


class MliUnsupportedOperationError(MliUnsupportedError):
    """Unsupported operation."""

    pass


class MliUnsupportedDataFormatError(MliUnsupportedError):
    """Unsupported data type error."""

    pass


class MliJsonSerializationError(MliError):
    """MLI JSon serialization error."""

    pass


class MliJsonDeserializationError(MliError):
    """MLI JSon deserialization error."""

    pass


class InvalidArgumentError(MliError):
    """Invalid (CLI) argument error."""

    pass


class InvalidArgumentValueError(MliError):
    """Invalid (CLI) argument value error."""

    pass


class InvalidDataError(MliError):
    """Invalid data error."""

    pass


class RenderingError(MliError):
    """Chart or image rendering error."""

    pass


class DatasetTooBigError(MliError):
    """The dataset is too big to be processed by H2O Sonar."""

    pass
