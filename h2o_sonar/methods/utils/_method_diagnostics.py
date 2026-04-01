# Copyright (C) 2022-2026 H2O.ai, Inc. All rights reserved


class MethodDiagnostics:
    """Method diagnostic data for profiling and debugging."""

    SCORER_HISTORY_LIMIT = 50

    @property
    def total_scorer_calls(self):
        """Total number of scorer calls since methods instantiation.

        Returns
        -------
        int
            Total scorer calls.

        """
        return self._total_scorer

    @property
    def scorer_calls_history(self):
        """History of scorer calls count per methods.

        Returns
        -------
        list of int
            Scorer calls history.

        """
        return self._scorer_history

    def __init__(self):
        self._total_scorer = 0
        self._scorer_history = [0]
        self._scorer_history_offset = 0

    def add_scorer_calls_slot(self):
        if self._scorer_history_offset < (self.SCORER_HISTORY_LIMIT - 1):
            if self._scorer_history_offset or self._scorer_history[0]:
                self._scorer_history_offset += 1
                self._scorer_history.append(0)
        else:
            del self._scorer_history[0]
            self._scorer_history.append(0)

    def add_scorer_calls(self, count=1):
        self._total_scorer += count
        self._scorer_history[self._scorer_history_offset] += count
