# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import collections
import dataclasses
from typing import Any


TOKENIZATION_TYPE_S = "sentence_level"
TOKENIZATION_TYPE_S_PUNKT = "sentence_level_punkt"
TOKENIZATION_TYPE_F = "fragment_level"

META_ERR_MSG = "error_message"
META_ERR_MSG_HTML = "error_message_html"


@dataclasses.dataclass(kw_only=True)
class TextFragment:
    text: str
    metrics: dict[str, float]
    meta: dict[Any, Any]  # = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, a_dict):
        # handle missing 'meta' field for backwards compatibility
        if "meta" not in a_dict:
            a_dict = {**a_dict, "meta": {}}
        return cls(**a_dict)

    def to_dict(self):
        self.meta = self.meta or {}
        return dataclasses.asdict(self)


FragmentMeta = collections.namedtuple(
    typename="Failure",
    field_names=[
        "start",
        "end",
        "metrics",
    ],
)


@dataclasses.dataclass(kw_only=True)
class Tokenization:
    tokenization: str
    data: list[TextFragment] | None

    @classmethod
    def from_dict(cls, a_dict):
        if "data" not in a_dict:
            a_dict = {**a_dict, "data": None}
        elif a_dict.get("data"):
            a_dict = {
                **a_dict,
                "data": [TextFragment.from_dict(d) for d in a_dict["data"]],
            }
        return cls(**a_dict)

    @classmethod
    def from_text_fragments(
        cls,
        tokenization_type: str,
        text: str,
        fragments: list[FragmentMeta],
        err_msg: str = None,
    ):
        """Create the tokenization from fragments, where the fragment is a tuple of
        start position, position, metrics and the error message.

        Parameters
        ----------
        tokenization_type : str
            The type of tokenization.
        text : str
            The original text.
        fragments : list[FragmentMeta]
            List of tuples with: start position, end position, and metrics for
            the text interval.
        err_msg : str | None
            The error message for the whole text.

        Returns
        -------
        tokenization.Tokenization | None :
            Return tokenization with fragments, None if no fragments.

        """
        t = Tokenization(tokenization=tokenization_type, data=[])

        meta = {META_ERR_MSG: err_msg} if err_msg else {}

        # fragments are tuples with start and end positions:
        # - the order is not guaranteed
        # - may overlap
        # - may be included in each other
        if not fragments:
            t.data.append(TextFragment(text=text, metrics={}, meta=meta))
            return t

        # sort fragments by start position
        fragments = sorted(fragments, key=lambda x: x[0])
        # complete the list of fragments with missing intervals to cover the whole text
        last_end = -1
        for f in fragments:
            (start, end, m) = f

            if start == last_end + 1:
                t.data.append(TextFragment(text=text[start:end], metrics=m, meta=meta))
                last_end = end
                # store meta to the first fragment only
                meta = {}
            elif start > last_end + 1:
                # add missing interval
                t.data.append(
                    TextFragment(text=text[last_end + 1 : start], metrics={}, meta=meta)
                )
                # store meta to the first fragment only
                meta = {}
                # add the fragment
                t.data.append(TextFragment(text=text[start:end], metrics=m, meta=meta))
                last_end = end
            else:
                # overlap or included
                raise NotImplementedError(
                    f"Overlap or included fragments are not supported: {f}"
                )

        # add the last missing interval
        if last_end < len(text) - 1:
            t.data.append(
                TextFragment(text=text[last_end + 1 :], metrics={}, meta=meta)
            )

        return t

    def to_dict(self):
        result = dataclasses.asdict(self)
        if result.get("data") is None:
            del result["data"]
        return result
