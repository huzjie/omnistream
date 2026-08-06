"""安全配置：JWT、API Key 哈希、凭据加密、CORS 与 PII 开关。

生产环境**必须**通过环境变量覆盖 ``jwt_secret`` / ``api_key_salt`` /
``credential_encryption_key``，默认值仅供本地开发。
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: 仅用于本地开发的占位密钥。生产使用会在启动时打印严重告警。
DEV_PLACEHOLDER_SECRET = "dev-only-change-me-in-production"


class SecuritySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMS_SECURITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------ JWT
    jwt_secret: str = Field(default=DEV_PLACEHOLDER_SECRET, repr=False)
    jwt_algorithm: str = Field(default="HS256", description="HS256|HS512|RS256")
    jwt_issuer: str = "omnistream"
    jwt_audience: str = "omnistream-console"
    access_token_ttl_s: int = Field(default=3600, ge=60)
    refresh_token_ttl_s: int = Field(default=604_800, ge=3600)
    #: 会话专用 token 的有效期，短到足以防重放
    session_token_ttl_s: int = Field(default=300, ge=30)

    # ------------------------------------------------------------ API Key
    api_key_salt: str = Field(default=DEV_PLACEHOLDER_SECRET, repr=False)
    api_key_prefix: str = "oms_live_"
    api_key_default_ttl_days: int = Field(default=365, ge=1)
    #: 明文哈希算法：blake2b（快）或 argon2（更抗暴力，需额外依赖）
    api_key_hash: str = Field(default="blake2b", description="blake2b|argon2")

    # ------------------------------------------------------------ 凭据加密
    #: Fernet 密钥（44 字符 urlsafe base64）。留空则从 jwt_secret 派生。
    credential_encryption_key: str | None = Field(default=None, repr=False)

    # ------------------------------------------------------------ 密码策略
    password_min_length: int = Field(default=10, ge=6)
    password_require_mixed: bool = True
    login_max_attempts: int = Field(default=8, ge=1)
    login_lockout_s: int = Field(default=900, ge=30)

    # ------------------------------------------------------------ 网络
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    cors_allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_headers: list[str] = Field(default_factory=lambda: ["*"])
    trusted_hosts: list[str] = Field(default_factory=lambda: ["*"])
    #: 允许匿名访问的路径前缀（健康检查、文档、静态资源）
    public_paths: list[str] = Field(default_factory=lambda: [
        "/healthz", "/readyz", "/metrics", "/docs", "/redoc", "/openapi.json"])
    require_https: bool = False
    hsts_max_age_s: int = Field(default=31_536_000, ge=0)

    # ------------------------------------------------------------ 隐私合规
    redact_pii: bool = True
    pii_kinds: list[str] = Field(default_factory=lambda: [
        "phone", "email", "id_card", "bank_card"])
    #: 日志中是否允许出现转写全文（生产建议 False）
    log_transcripts: bool = False
    audit_enabled: bool = True
    audit_hash_chain: bool = True

    @property
    def using_dev_secrets(self) -> bool:
        """是否仍在使用开发占位密钥。启动检查据此告警。"""
        return DEV_PLACEHOLDER_SECRET in (self.jwt_secret, self.api_key_salt)

    def is_public_path(self, path: str) -> bool:
        return any(path.startswith(p) for p in self.public_paths)
