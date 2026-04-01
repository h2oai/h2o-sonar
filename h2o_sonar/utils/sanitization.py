# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import re

import numpy


_SANITIZATION_SPECIAL_CHARS = "|,=[]<\t\r\n:.~"


class SanitizationMap:
    """Map of original (raw) dataset column names/features to sanitized names and
    vice versa.

    """

    def __init__(
        self,
        # TODO raw names
        raw_names: list[str],
        sanitized_names: list[str],
    ):
        """Map is initialized with two arrays - raw feature names and sanitized
        feature names. Columns must be of the same length. Be sure that you sanitize
        *all* the features at once before initializing this map - multiple raw
        features can be mapped to the same sanitized string (N:1) and therefore names
        must be make unique by the sanitization function.

        Parameters
        ----------
        raw_names : list[str]
          List of original (raw) names.
        sanitized_names : list[str]
          List of sanitized (raw) names.

        """
        if (
            not raw_names
            or not sanitized_names
            or len(raw_names) != len(sanitized_names)
        ):
            raise ValueError(
                f"Sanitization map cannot be initialized with raw or sanitized list "
                f"which is empty OR isn't of the same size (raw/sanitized):\n"
                f"{raw_names}\n{sanitized_names}"
            )

        self.raw = raw_names
        self.sanitized = sanitized_names
        self.to_raw_dict = {s: raw_names[i] for i, s in enumerate(sanitized_names)}
        self.to_sanitized_dict = {
            r: sanitized_names[i] for i, r in enumerate(raw_names)
        }

    def __str__(self):
        return (
            f"SanitizationMap:\n"
            f"  raw             : {self.raw}\n"
            f"  sanitized       : {self.sanitized}\n"
        )

    def to_raw(self, names: str | list[str]):
        """Sanitized name(s) to original (raw) name(s)."""
        if names is None:
            return None
        if isinstance(names, (list, tuple)):
            return [self.to_raw_dict.get(feature, feature) for feature in names]
        return self.to_raw_dict.get(names, names)

    def to_sanitized(self, names: str | list[str]):
        """Original (raw) name(s) to sanitized name(s)."""
        if names is None:
            return None
        if isinstance(names, (list, tuple)):
            return [self.to_sanitized_dict.get(feature, feature) for feature in names]
        return self.to_sanitized_dict.get(names, names)

    @staticmethod
    def ensure(cols, col) -> list[str]:
        if cols:
            if isinstance(cols, tuple):
                result = list(cols)
            else:
                result = cols.copy()
        else:
            result = list()
        if col is not None and col not in result:
            result.append(col)
        return result

    @staticmethod
    def sanitize_value(
        values: str | list[str],
        special_chars: str = _SANITIZATION_SPECIAL_CHARS,
    ) -> str | list[str]:
        """Method for feature **values** (labels, classes) sanitization. Note that
        column/feature name sanitization (handled by map) typically has different
        requirements than value sanitization. Also note that value sanitization is
        one way (original to sanitized only) and potentially may have collisions
        if sanitized in multiple calls to this method (collisions within one call
        of this function are resolved).

        """
        if values not in [None, ""]:
            if isinstance(values, list):
                values = [str(v) for v in values]
            else:
                values = str(values)
            return sanitize_strings(strings=values, special_chars=special_chars)

        return values


class DriverlessAiSanitizationMap(SanitizationMap):
    """Driverless AI model sanitization map.

    Driverless AI (auto ML) model provides its own sanitization map. The purpose of
    this class is to make Driverless AI sanitization available vis standard
    ``SanitizationMap`` interface.

    """

    pass


def sanitize_names(
    names: str | list[str],
    sanitization_map: SanitizationMap | None = None,
) -> SanitizationMap | None:
    """Sanitize column/feature **name(s)** either using (model's) sanitization map
    (if available) or using universal sanitization method.

    Parameters
    ----------
    names : str | list[str]
      Name(s) to be sanitized.
    sanitization_map : SanitizationMap | None
      Optional sanitization map.

    """

    if names is not None:
        if isinstance(names, list):
            if sanitization_map is None:
                return sanitize_strings(names)
            else:
                return sanitization_map.to_sanitized(names)
        else:
            if sanitization_map:
                return sanitize_strings([names])[0]
            else:
                assert isinstance(names, (str, numpy.str_))
                return sanitization_map.to_sanitized(names)

    return None


def sanitize_strings(
    strings: str | list[str],
    replace_with: str = "_",
    special_chars: str = _SANITIZATION_SPECIAL_CHARS,
):
    """Sanitize a string or a list of strings.

    Parameters
    ----------
    strings : str | list[str]
      Strings to be sanitized.
    replace_with : str
      Character to be used for replacement for characters to be forbidden.
    special_chars : str
      Optional special characters to be sanitized.

    Returns
    -------
    str | list[str]
      Sanitized strings.

    """

    if not strings:
        return strings

    strings_ = [strings] if isinstance(strings, str) else strings
    escaped_strings = [None] * len(strings_)
    escaped_set = set()
    idxs = list(range(len(strings_)))
    if len(strings_) > 0:
        idxs, strings_sorted = zip(
            *sorted(zip(idxs, strings_, strict=False), key=lambda x: x[1]), strict=False
        )
    else:
        idxs, strings_sorted = idxs, strings_

    for idx, s in zip(idxs, strings_sorted, strict=False):
        assert s is not None, "None cannot be sanitized"
        s = str(s)
        for special_char in special_chars:
            s = s.replace(special_char, replace_with)
        while s in escaped_set:
            s += replace_with
        escaped_strings[idx] = s
        escaped_set.add(s)

    return escaped_strings[0] if isinstance(strings, str) else escaped_strings


def sanitize_frame(
    frame, sanitization_map: SanitizationMap | None = None
) -> SanitizationMap | None:
    # TODO frame sanitization to be ported from former code base OR implemented from
    #  scratch
    return frame


def _markdown_sanitize_link(match):
    """Sanitize a Markdown link by surrounding it with backticks."""
    display_text = match.group(1)
    url = match.group(2)
    return f"`[{display_text}]({url})`"


def sanitize_markdown(md_fragment: str) -> str:
    """The purpose of this function is to sanitize a Markdown fragment string.
    It is NOT meant to sanitize whole Markdown documents, but its fragments where
    string (to be stored in Markdown) would interact with other Markdown elements.

    Parameters
    ----------
    md_fragment : str
      A Markdown fragment string.

    Returns
    -------
    str :
      Sanitized Markdown string fragment without dangerous characters and links.

    """
    # remove special characters which may interfere with Markdown
    output = (
        md_fragment.replace("|", " ")  # avoid table parsing
        .replace("\n", " ")  # avoid line breaks
        .replace("\r", " ")  # avoid line breaks
        .replace("\t", " ")  # avoid tab characters
        .replace("`", "")  # avoid backticks
    )
    # remove <html> and <script> opening tags (closing tags are harmless)
    output = re.sub(
        pattern=r"(?i)<\s*script\b[^>]*>|<\s*html\b[^>]*>",
        repl=" ",
        string=output,
    )
    # surround links with backticks to neutralize them
    output = re.sub(
        pattern=r"\[(.*?)\]\((.*?)\)", repl=_markdown_sanitize_link, string=output
    )
    return output
