"""``provider_configs`` 表：租户自定义 Provider 凭据（对称加密存储）。

加密使用 Fernet（AES-128-CBC + HMAC-SHA256），密钥由平台 secret 通过
PBKDF2-HMAC-SHA256 派生。secret 轮换后旧密文不可解，需要重新录入凭据。
"""
from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from sqlalchemy import Boolean, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.engine import setting
from omnistream.db.mixins import TenantMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import EncryptedBytes, JSONB

_KDF_SALT = b"omnistream.provider_config.v1"
_KDF_ROUNDS = 200_000


def _fernet_key() -> bytes:
    """从平台 secret 派生 32 字节 Fernet key（urlsafe base64 编码）。"""
    secret = str(
        setting("security.secret_key", "")
        or setting("security.jwt_secret", "")
        or "omnistream-dev-insecure-secret-change-me"
    )
    raw = hashlib.pbkdf2_hmac("sha256", secret.encode("utf-8"), _KDF_SALT, _KDF_ROUNDS, dklen=32)
    return base64.urlsafe_b64encode(raw)


def encrypt_credentials(payload: dict[str, Any]) -> bytes:
    """加密凭据字典。"""
    from cryptography.fernet import Fernet

    plaintext = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return Fernet(_fernet_key()).encrypt(plaintext)


def decrypt_credentials(blob: bytes | None) -> dict[str, Any]:
    """解密凭据字典；密文损坏或密钥已轮换时返回空字典。"""
    if not blob:
        return {}
    from cryptography.fernet import Fernet, InvalidToken

    try:
        plaintext = Fernet(_fernet_key()).decrypt(bytes(blob))
    except (InvalidToken, ValueError):
        return {}
    try:
        data = json.loads(plaintext.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def mask_credentials(payload: dict[str, Any]) -> dict[str, str]:
    """对外展示用的掩码形态，只保留末 4 位。"""
    masked: dict[str, str] = {}
    for key, value in payload.items():
        text = str(value)
        masked[key] = ("*" * max(0, len(text) - 4)) + text[-4:] if len(text) > 4 else "****"
    return masked


class ProviderConfig(ULIDPkMixin, TenantMixin, TimestampMixin, Base):
    __tablename__ = "provider_configs"
    __id_prefix__ = "pcf_"

    provider: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    credentials_enc: Mapped[bytes | None] = mapped_column(EncryptedBytes, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    priority: Mapped[int] = mapped_column(nullable=False, default=100)
    #: 非敏感的 provider 选项：模型名、region、超时等
    options: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    #: unknown | healthy | unhealthy
    health: Mapped[str] = mapped_column(String(16), nullable=False, default="unknown")
    last_error: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    __table_args__ = (
        UniqueConstraint("tenant_id", "provider", name="uq_provider_configs_tenant_id_provider"),
        Index("ix_provider_configs_enabled_priority", "enabled", "priority"),
    )

    def credentials(self) -> dict[str, Any]:
        return decrypt_credentials(self.credentials_enc)

    def set_credentials(self, payload: dict[str, Any]) -> None:
        self.credentials_enc = encrypt_credentials(payload) if payload else None

    def masked_credentials(self) -> dict[str, str]:
        return mask_credentials(self.credentials())


__all__ = [
    "ProviderConfig",
    "decrypt_credentials",
    "encrypt_credentials",
    "mask_credentials",
]
