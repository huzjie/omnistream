"""密钥、令牌与凭证加密。

设计取舍：

* **口令哈希** 用标准库 ``hashlib.scrypt``（内存硬，抗 GPU 爆破），
  不引入 argon2/bcrypt 依赖，也不自己发明算法；
* **Provider 密钥落库加密** 用 AES-256-GCM，依赖可选包 ``cryptography``。
  未安装时明确抛 :class:`ConfigError` 并给出安装提示，**绝不**降级成明文存储；
* **令牌生成** 一律走 ``secrets``，绝不用 ``random``。

存储格式全部是自描述的、带版本号的紧凑字符串，便于将来无损轮换算法。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass

from omnistream.core.constants import API_KEY_PLAINTEXT_PREFIX
from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import ConfigError

__all__ = [
    "ENCRYPTION_PREFIX",
    "PASSWORD_HASH_PREFIX",
    "ApiKeyPair",
    "b64decode",
    "b64encode",
    "decrypt_secret",
    "derive_key",
    "encrypt_secret",
    "generate_api_key",
    "generate_token",
    "hash_password",
    "mask_secret",
    "verify_password",
]

#: scrypt 参数：N=2^15、r=8、p=1 ≈ 32MB 内存、单次约 60ms，兼顾安全与登录延迟。
_SCRYPT_N = 1 << 15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
#: OpenSSL 默认 maxmem 是 32MiB，恰好卡在 N=2^15,r=8 的 128*N*r 门槛上，必须显式抬高。
_SCRYPT_MAXMEM = 128 * _SCRYPT_N * _SCRYPT_R * 2
_SALT_BYTES = 16
_GCM_NONCE_BYTES = 12

PASSWORD_HASH_PREFIX = "scrypt$1$"
ENCRYPTION_PREFIX = "oms.enc.v1:"


# --------------------------------------------------------------------------- base64
def b64encode(data: bytes) -> str:
    """URL-safe base64（无填充），用于所有紧凑二进制字段。"""
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64decode(text: str) -> bytes:
    """:func:`b64encode` 的逆操作，自动补回填充。"""
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


# --------------------------------------------------------------------------- 令牌
def generate_token(n_bytes: int = 32, *, prefix: str = "") -> str:
    """生成密码学安全的随机令牌（URL-safe）。"""
    if n_bytes < 16:
        raise ValueError("token entropy must be >= 16 bytes")
    return f"{prefix}{b64encode(secrets.token_bytes(n_bytes))}"


@dataclass(frozen=True, slots=True)
class ApiKeyPair:
    """新签发的 API Key：明文只在签发瞬间存在一次，服务端只留 ``digest``。"""

    plaintext: str
    digest: str
    prefix: str

    def masked(self) -> str:
        """展示用掩码，如 ``oms_live_ab12…7f9c``。"""
        return mask_secret(self.plaintext)


def generate_api_key(salt: str, *, key_prefix: str = API_KEY_PLAINTEXT_PREFIX) -> ApiKeyPair:
    """签发一枚 API Key。

    Args:
        salt: 服务端全局 salt（``settings.security.api_key_salt``）。
        key_prefix: 明文前缀，用于人眼与日志快速识别环境。

    Returns:
        :class:`ApiKeyPair`，其中 ``digest`` 是入库值，``plaintext`` 只返回给用户一次。
    """
    plaintext = f"{key_prefix}{b64encode(secrets.token_bytes(30))}"
    digest = hashlib.blake2b(plaintext.encode("utf-8"),
                             key=salt.encode("utf-8")[:64],
                             digest_size=32).hexdigest()
    return ApiKeyPair(plaintext=plaintext, digest=digest, prefix=plaintext[:12])


def mask_secret(secret: str, *, head: int = 12, tail: int = 4) -> str:
    """把密钥掩成 ``头…尾`` 形式，可安全写进日志与前端。"""
    if len(secret) <= head + tail:
        return "*" * len(secret)
    return f"{secret[:head]}…{secret[-tail:]}"


# --------------------------------------------------------------------------- 口令
def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """scrypt 口令哈希，输出 ``scrypt$1$<salt_b64>$<dk_b64>``。"""
    if not password:
        raise ValueError("password must not be empty")
    real_salt = salt if salt is not None else secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=real_salt,
        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN,
        maxmem=_SCRYPT_MAXMEM,
    )
    return f"{PASSWORD_HASH_PREFIX}{b64encode(real_salt)}${b64encode(derived)}"


def verify_password(password: str, encoded: str) -> bool:
    """恒定时间校验口令。格式非法或算法未知一律返回 ``False``。"""
    if not encoded.startswith(PASSWORD_HASH_PREFIX):
        return False
    try:
        salt_b64, dk_b64 = encoded[len(PASSWORD_HASH_PREFIX):].split("$", 1)
        salt = b64decode(salt_b64)
        expected = b64decode(dk_b64)
    except Exception:
        return False
    actual = hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=len(expected) or _SCRYPT_DKLEN,
        maxmem=_SCRYPT_MAXMEM,
    )
    return hmac.compare_digest(actual, expected)


def derive_key(master: str, *, context: str = "omnistream", length: int = 32) -> bytes:
    """从主密钥派生子密钥（HKDF-SHA256，Extract + Expand）。

    Args:
        master: 主密钥字符串（``settings.security.credential_encryption_key``）。
        context: 用途标签，不同用途派生出互不相关的子密钥。
        length: 输出字节数，1..255*32。
    """
    if not master:
        raise ConfigError("master key is empty", code=ErrorCode.CONFIG_MISSING)
    if not 1 <= length <= 8160:
        raise ValueError("length out of range")
    prk = hmac.new(b"omnistream-hkdf-salt", master.encode("utf-8"), hashlib.sha256).digest()
    info = context.encode("utf-8")
    okm = b""
    block = b""
    counter = 1
    while len(okm) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        okm += block
        counter += 1
    return okm[:length]


# --------------------------------------------------------------------------- 对称加密
def _load_aesgcm() -> type:
    """按需加载 ``cryptography``；缺失时给出可执行的修复建议。"""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - 取决于是否安装可选依赖
        raise ConfigError(
            "AES-GCM requires the `cryptography` package",
            code=ErrorCode.CONFIG_MISSING,
            detail={"install": "pip install 'cryptography>=42,<45'"},
            cause=exc,
        ) from exc
    return AESGCM


def encrypt_secret(plaintext: str, master_key: str, *, context: str = "provider-credential") -> str:
    """AES-256-GCM 加密 Provider 密钥，返回 ``oms.enc.v1:<nonce_b64>.<ct_b64>``。

    Args:
        plaintext: 明文密钥。
        master_key: 主密钥，来自配置且**不得**入库。
        context: HKDF 上下文，同时作为 GCM 的 AAD 绑定用途，防止密文跨用途重放。
    """
    aesgcm_cls = _load_aesgcm()
    key = derive_key(master_key, context=context, length=32)
    nonce = os.urandom(_GCM_NONCE_BYTES)
    ciphertext = aesgcm_cls(key).encrypt(nonce, plaintext.encode("utf-8"),
                                         context.encode("utf-8"))
    return f"{ENCRYPTION_PREFIX}{b64encode(nonce)}.{b64encode(ciphertext)}"


def decrypt_secret(token: str, master_key: str, *, context: str = "provider-credential") -> str:
    """解密 :func:`encrypt_secret` 产物。格式或认证失败抛 :class:`ConfigError`。"""
    if not token.startswith(ENCRYPTION_PREFIX):
        raise ConfigError("unrecognized ciphertext format",
                          code=ErrorCode.CONFIG_INVALID,
                          detail={"expected_prefix": ENCRYPTION_PREFIX})
    aesgcm_cls = _load_aesgcm()
    body = token[len(ENCRYPTION_PREFIX):]
    try:
        nonce_b64, ct_b64 = body.split(".", 1)
        nonce = b64decode(nonce_b64)
        ciphertext = b64decode(ct_b64)
    except Exception as exc:
        raise ConfigError("malformed ciphertext", code=ErrorCode.CONFIG_INVALID,
                          cause=exc) from exc
    key = derive_key(master_key, context=context, length=32)
    try:
        plain = aesgcm_cls(key).decrypt(nonce, ciphertext, context.encode("utf-8"))
    except Exception as exc:
        raise ConfigError("credential decryption failed (wrong key or tampered data)",
                          code=ErrorCode.CONFIG_INVALID, cause=exc) from exc
    return plain.decode("utf-8")
