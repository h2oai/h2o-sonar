# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
"""JSON comparison utilities using DeepDiff."""

import dataclasses
import json

import deepdiff


@dataclasses.dataclass
class DiffSummary:
    """Summary of changes found in a JSON diff.

    Attributes
    ----------
    values_changed : int
        Number of values that changed between dictionaries.
    dictionary_item_added : int
        Number of dictionary items added in current vs baseline.
    dictionary_item_removed : int
        Number of dictionary items removed in current vs baseline.
    iterable_item_added : int
        Number of list/set items added in current vs baseline.
    iterable_item_removed : int
        Number of list/set items removed in current vs baseline.
    type_changes : int
        Number of items where the type changed between dictionaries.

    """

    values_changed: int = 0
    dictionary_item_added: int = 0
    dictionary_item_removed: int = 0
    iterable_item_added: int = 0
    iterable_item_removed: int = 0
    type_changes: int = 0

    def total_changes(self) -> int:
        """Calculate total number of changes.

        Returns
        -------
        int :
            Sum of all change counts.

        """
        return (
            self.values_changed
            + self.dictionary_item_added
            + self.dictionary_item_removed
            + self.iterable_item_added
            + self.iterable_item_removed
            + self.type_changes
        )


class JSONComparator:
    """Compare two JSON-like dictionaries and generate detailed diff reports.

    This class uses DeepDiff to calculate differences between two dictionaries
    and provides methods to generate various report formats (dict, JSON, HTML).

    Parameters
    ----------
    baseline_dict : dict
        The baseline/reference dictionary to compare against.
    current_dict : dict
        The current dictionary to compare with baseline.
    ignore_order : bool
        Whether to ignore list/set order in comparisons. Default: False.
    report_repetition : bool
        Whether to report repeated items. Default: True.
    verbose_level : int
        Verbosity level for DeepDiff (0-2). Default: 2.

    """

    def __init__(
        self,
        baseline_dict: dict,
        current_dict: dict,
        ignore_order: bool = False,
        report_repetition: bool = True,
        verbose_level: int = 2,
    ):
        """Initialize JSON comparator with two dictionaries to compare."""
        self.baseline_dict = baseline_dict
        self.current_dict = current_dict
        self.ignore_order = ignore_order
        self.report_repetition = report_repetition
        self.verbose_level = verbose_level
        self._diff = None

    def calculate_diff(self) -> deepdiff.DeepDiff:
        """Calculate the deep difference between baseline and current dictionaries.

        Returns
        -------
        DeepDiff :
            DeepDiff object containing the calculated differences.

        """
        if self._diff is None:
            self._diff = deepdiff.DeepDiff(
                self.baseline_dict,
                self.current_dict,
                ignore_order=self.ignore_order,
                report_repetition=self.report_repetition,
                verbose_level=self.verbose_level,
            )
        return self._diff

    def to_dict(self) -> dict:
        """Convert the diff to a dictionary structure.

        Returns
        -------
        dict :
            Dictionary representation of the diff.

        """
        diff = self.calculate_diff()
        return diff.to_dict()

    def to_json(self, indent: int = 2) -> str:
        """Convert the diff to a JSON string.

        Parameters
        ----------
        indent : int
            Number of spaces for JSON indentation. Default: 2.

        Returns
        -------
        str :
            JSON string representation of the diff.

        """
        return json.dumps(self.to_dict(), indent=indent)

    def get_diff_summary(self) -> DiffSummary:
        """Get a summary of changes found in the diff.

        Returns
        -------
        DiffSummary :
            Dataclass containing counts of different types of changes.

        """
        diff = self.calculate_diff()
        diff_dict = diff.to_dict()

        return DiffSummary(
            values_changed=len(diff_dict.get("values_changed", {})),
            dictionary_item_added=len(diff_dict.get("dictionary_item_added", {})),
            dictionary_item_removed=len(diff_dict.get("dictionary_item_removed", {})),
            iterable_item_added=len(diff_dict.get("iterable_item_added", {})),
            iterable_item_removed=len(diff_dict.get("iterable_item_removed", {})),
            type_changes=len(diff_dict.get("type_changes", {})),
        )

    def has_differences(self) -> bool:
        """Check if there are any differences between the dictionaries.

        Returns
        -------
        bool :
            True if differences exist, False otherwise.

        """
        diff = self.calculate_diff()
        return bool(diff)

    def get_changed_paths(self) -> list[str]:
        """Get list of all paths where values changed.

        Returns
        -------
        list[str] :
            List of path strings where changes occurred.

        """
        diff = self.calculate_diff()
        diff_dict = diff.to_dict()

        paths = []

        # extract paths from values_changed
        if "values_changed" in diff_dict:
            paths.extend(diff_dict["values_changed"].keys())

        # extract paths from dictionary_item_added
        if "dictionary_item_added" in diff_dict:
            paths.extend(diff_dict["dictionary_item_added"].keys())

        # extract paths from dictionary_item_removed
        if "dictionary_item_removed" in diff_dict:
            paths.extend(diff_dict["dictionary_item_removed"].keys())

        return sorted(paths)
