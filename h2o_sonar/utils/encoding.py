# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import base64
import binascii
import enum
import re


class EncodingType(enum.Enum):
    """Specifies the type of encoding/decoding to perform."""

    BASE16 = "base16"
    BASE16_SPACES = "base16_spaces"
    BASE64 = "base64"

    @staticmethod
    def str_to_enum(value: str) -> "EncodingType | None":
        """Converts a string to the corresponding EncodingType enum."""
        member_name = value.split(".")[-1]
        enum_member = None
        try:
            enum_member = EncodingType[member_name]
        except KeyError:
            pass

        return enum_member


def encode(data: str, encoding_type: EncodingType) -> str:
    """Encodes a UTF-8 string using the specified encoding type.

    Parameters
    ----------
    data : str
        The UTF-8 string to encode.
    encoding_type : EncodingType
        The target encoding (BASE16, BASE16_SPACES, BASE64).

    Returns
    -------
    str
        The encoded string.

    Raises
    ------
    ValueError
        If encoding_type is unsupported.
    UnicodeEncodeError
        If data cannot be UTF-8 encoded.

    """
    if encoding_type == EncodingType.BASE16:
        return base64.b16encode(data.encode("utf-8")).decode("utf-8")
    elif encoding_type == EncodingType.BASE16_SPACES:
        encoded = base64.b16encode(data.encode("utf-8")).decode("utf-8")
        return " ".join(encoded[i : i + 2] for i in range(0, len(encoded), 2))
    elif encoding_type == EncodingType.BASE64:
        return base64.b64encode(data.encode("utf-8")).decode("utf-8")
    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")


def decode(data: str, encoding_type: EncodingType) -> str:
    """Decodes a string based on the specified encoding type.

    Parameters
    ----------
    data : str
        The encoded string to decode.
    encoding_type : EncodingType
        The encoding used for the input data.

    Returns
    -------
    str
        The decoded UTF-8 string.

    Raises
    ------
    ValueError:
        If encoding_type is unsupported.
    binascii.Error:
        If the input data is malformed for the encoding type.

    """
    if encoding_type == EncodingType.BASE16:
        return base64.b16decode(data.encode("utf-8")).decode("utf-8")
    elif encoding_type == EncodingType.BASE16_SPACES:
        sanitized = data.replace(" ", "")
        return base64.b16decode(sanitized.encode("utf-8")).decode("utf-8")
    elif encoding_type == EncodingType.BASE64:
        return base64.b64decode(data.encode("utf-8")).decode("utf-8")
    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")


def validate_string(s: str, enc: EncodingType) -> bool:
    """Validates if a string conforms to the specified encoding rules.

    Parameters
    ----------
    s : str
        The input string to validate.
    enc: EncodingType
        The EncodingType to validate against.

    Returns
    -------
    bool
        ``True`` if the string is valid for the encoding, ``False`` otherwise.

    """
    if not isinstance(s, str):
        return False

    if enc == EncodingType.BASE64:
        if not s:
            return False
        # check Base64 structure (characters and padding)
        if not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", s):
            return False
        # attempt decoding to ensure validity
        try:
            _ = decode(data=s, encoding_type=EncodingType.BASE64)
            return True
        except (ValueError, binascii.Error):
            return False

    elif enc == EncodingType.BASE16_SPACES:
        if not s:
            return False
        hex_without_spaces = "".join(s.split())
        if not hex_without_spaces:
            return False
        # check for even length after removing spaces
        if len(hex_without_spaces) % 2 != 0:
            return False
        # check if all non-space characters are valid hex digits
        base_hex_chars = "0123456789abcdefABCDEF"
        return all(c in base_hex_chars for c in hex_without_spaces)

    elif enc == EncodingType.BASE16:
        if not s:  # Empty string is not valid
            return False
        # check for even length
        if len(s) % 2 != 0:
            return False
        # check if all characters are valid hex digits
        base_hex_chars = "0123456789abcdefABCDEF"
        return all(c in base_hex_chars for c in s)

    else:
        return False
