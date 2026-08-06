"""密码哈希与强度校验。

优先使用 argon2id（``argon2-cffi``）；未安装时自动降级为标准库
``hashlib.pbkdf2_hmac('sha256')``，参数与格式均写入哈希串，保证：

* 无第三方依赖也能真实跑通注册/登录；
* 后续安装 argon2 后，旧 PBKDF2 密码仍可校验，并通过 :func:`needs_rehash`
  在下次登录时自动升级。

哈希串格式（PBKDF2 分支）::

    pbkdf2-sha256$<iterations>$<salt_b64url>$<dk_b64url>
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import string
from dataclasses import dataclass, field

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import ValidationError

__all__ = [
    "MIN_PASSWORD_LENGTH",
    "PasswordStrength",
    "check_strength",
    "generate_password",
    "hash_password",
    "needs_rehash",
    "verify_password",
]

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 256
_PBKDF2_ITERATIONS = 240_000
_PBKDF2_SALT_BYTES = 16
_PBKDF2_DK_BYTES = 32
_PBKDF2_TAG = "pbkdf2-sha256"

#: 常见弱密码黑名单（登录场景足够，生产可在租户设置里扩展）
_COMMON_PASSWORDS: frozenset[str] = frozenset(
    {
        "password", "123456", "12345678", "123456789", "qwerty", "abc123",
        "111111", "iloveyou", "admin", "welcome", "monkey", "dragon",
        "letmein", "password1", "qwerty123", "sunshine", "master", "888888",
        "666666", "passw0rd", "admin123", "root", "toor", "changeme",
    }
)

try:  # pragma: no cover - 取决于运行环境是否安装 argon2-cffi
    from argon2 import PasswordHasher as _Argon2Hasher
    from argon2.exceptions import InvalidHash as _Argon2InvalidHash
    from argon2.exceptions import VerificationError as _Argon2VerificationError
    from argon2.exceptions import VerifyMismatchError as _Argon2Mismatch

    _HASHER = _Argon2Hasher(time_cost=3, memory_cost=65536, parallelism=2, hash_len=32, salt_len=16)
    ARGON2_AVAILABLE = True
except Exception:  # noqa: BLE001
    _HASHER = None
    _Argon2InvalidHash = Exception  # type: ignore[assignment,misc]
    _Argon2VerificationError = Exception  # type: ignore[assignment,misc]
    _Argon2Mismatch = Exception  # type: ignore[assignment,misc]
    ARGON2_AVAILABLE = False


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _pbkdf2_hash(password: str, *, salt: bytes | None = None, iterations: int = _PBKDF2_ITERATIONS) -> str:
    real_salt = salt or secrets.token_bytes(_PBKDF2_SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), real_salt, iterations, dklen=_PBKDF2_DK_BYTES
    )
    return f"{_PBKDF2_TAG}${iterations}${_b64e(real_salt)}${_b64e(derived)}"


def _pbkdf2_verify(password: str, encoded: str) -> bool:
    try:
        tag, iter_text, salt_text, dk_text = encoded.split("$")
    except ValueError:
        return False
    if tag != _PBKDF2_TAG:
        return False
    try:
        iterations = int(iter_text)
        salt = _b64d(salt_text)
        expected = _b64d(dk_text)
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations, dklen=len(expected)
    )
    return hmac.compare_digest(actual, expected)


def hash_password(password: str) -> str:
    """生成密码哈希。空密码或超长密码直接拒绝。"""
    if not password:
        raise ValidationError("密码不能为空", code=ErrorCode.INVALID_REQUEST)
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValidationError(
            f"密码长度不能超过 {MAX_PASSWORD_LENGTH}", code=ErrorCode.INVALID_REQUEST
        )
    if ARGON2_AVAILABLE and _HASHER is not None:
        return str(_HASHER.hash(password))
    return _pbkdf2_hash(password)


def verify_password(password: str, encoded: str | None) -> bool:
    """校验密码。``encoded`` 为空时恒定返回 False，但仍消耗等量时间以缓解时序侧信道。"""
    if not encoded:
        # 走一次真实 KDF，避免"用户不存在"被时间差探测出来
        _pbkdf2_hash(password or "", salt=b"\x00" * _PBKDF2_SALT_BYTES, iterations=1000)
        return False
    if encoded.startswith("$argon2"):
        if not (ARGON2_AVAILABLE and _HASHER is not None):
            return False
        try:
            return bool(_HASHER.verify(encoded, password))
        except (_Argon2Mismatch, _Argon2VerificationError, _Argon2InvalidHash):
            return False
        except Exception:  # noqa: BLE001
            return False
    return _pbkdf2_verify(password, encoded)


def needs_rehash(encoded: str | None) -> bool:
    """判断是否需要在下次成功登录时重算哈希（算法升级或参数过时）。"""
    if not encoded:
        return True
    if ARGON2_AVAILABLE and _HASHER is not None:
        if not encoded.startswith("$argon2"):
            return True
        try:
            return bool(_HASHER.check_needs_rehash(encoded))
        except Exception:  # noqa: BLE001
            return True
    if encoded.startswith("$argon2"):
        # argon2 已卸载，无法校验，必须重设
        return True
    parts = encoded.split("$")
    if len(parts) != 4 or parts[0] != _PBKDF2_TAG:
        return True
    try:
        return int(parts[1]) < _PBKDF2_ITERATIONS
    except ValueError:
        return True


@dataclass(slots=True)
class PasswordStrength:
    """密码强度评估结果。"""

    score: int = 0
    ok: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"score": self.score, "ok": self.ok, "reasons": list(self.reasons)}


def check_strength(password: str, *, min_length: int = MIN_PASSWORD_LENGTH) -> PasswordStrength:
    """0-100 打分 + 不通过原因。score >= 50 且无硬性缺陷才算 ok。"""
    reasons: list[str] = []
    if not password:
        return PasswordStrength(score=0, ok=False, reasons=["密码不能为空"])

    length = len(password)
    has_lower = bool(re.search(r"[a-z]", password))
    has_upper = bool(re.search(r"[A-Z]", password))
    has_digit = bool(re.search(r"\d", password))
    has_symbol = bool(re.search(r"[^\w\s]", password))
    classes = sum([has_lower, has_upper, has_digit, has_symbol])

    score = 0
    score += min(length, 20) * 3            # 长度最多 60 分
    score += (classes - 1) * 10 if classes else 0  # 字符类最多 30 分
    if length >= 16:
        score += 10
    score = max(0, min(score, 100))

    if length < min_length:
        reasons.append(f"长度至少 {min_length} 位")
    if length > MAX_PASSWORD_LENGTH:
        reasons.append(f"长度不能超过 {MAX_PASSWORD_LENGTH} 位")
    if classes < 2:
        reasons.append("至少包含大小写字母、数字、符号中的两类")
    if password.lower() in _COMMON_PASSWORDS:
        reasons.append("命中常见弱密码黑名单")
        score = min(score, 20)
    if re.fullmatch(r"(.)\1+", password):
        reasons.append("不能全部为重复字符")
        score = min(score, 10)
    if re.search(r"(0123|1234|2345|3456|4567|5678|6789|abcd|qwer)", password.lower()):
        reasons.append("包含连续字符序列")
        score = max(0, score - 15)

    return PasswordStrength(score=score, ok=not reasons and score >= 50, reasons=reasons)


def generate_password(length: int = 20) -> str:
    """生成满足强度要求的随机密码（用于初始化管理员账号）。"""
    size = max(MIN_PASSWORD_LENGTH, min(int(length), 64))
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*-_=+"
    while True:
        candidate = "".join(secrets.choice(alphabet) for _ in range(size))
        if check_strength(candidate).ok:
            return candidate
