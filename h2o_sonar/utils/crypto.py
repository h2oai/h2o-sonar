# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import base64
import os

from h2o_sonar.lib.api import commons


try:
    from cryptography import fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf import pbkdf2

    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

# shell environment variables
ENV_VAR_H2O_SONAR_ENCRYPTION_KEY: str = "H2O_SONAR_ENCRYPTION_KEY"


__SLT = b"H20S0N3RS31T"
__UTF_8 = "utf-8"


def resolve_encryption_key(encryption_key: str = "") -> str:
    if not encryption_key:
        encryption_key = os.getenv(ENV_VAR_H2O_SONAR_ENCRYPTION_KEY, "")

    if not encryption_key:
        raise ValueError(
            f"Encryption key is neither provided nor set in the "
            f"environment variable {ENV_VAR_H2O_SONAR_ENCRYPTION_KEY}"
        )

    return encryption_key


def __get_fernet(encryption_key: str):
    if not HAS_CRYPTOGRAPHY:
        commons.raise_opt_import_err("cryptography")

    # derivation function
    kdf = pbkdf2.PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=__SLT,
        iterations=480000,
    )
    encoded_encryption_key = base64.urlsafe_b64encode(
        kdf.derive(encryption_key.encode(__UTF_8))
    )

    return fernet.Fernet(encoded_encryption_key)


def encrypt(encryption_key: str, data: str) -> str:
    if not encryption_key:
        raise ValueError("Encryption key is required")

    if data:
        return (
            __get_fernet(encryption_key).encrypt(data.encode(__UTF_8)).decode(__UTF_8)
        )

    return data


def decrypt(encryption_key: str, data: str) -> str:
    if not encryption_key:
        raise ValueError("Encryption key is required")

    if data:
        return (
            __get_fernet(encryption_key).decrypt(data.encode(__UTF_8)).decode(__UTF_8)
        )

    return data
