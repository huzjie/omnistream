"""平台治理层公共 API。

汇总租户、鉴权、RBAC、配额/限流、计费、Webhook 的稳定出口。下游（API 路由、worker、
WS 编排器）只从这里导入，不要直接依赖子模块内部实现。
"""
from __future__ import annotations

from omnistream.platform.auth import (
    ApiKeyService,
    authenticate_request,
    authenticate_user,
    describe_auth_backends,
    generate_api_key,
    tokens_for_principal,
    verify_api_key,
)
from omnistream.platform.auth.jwt import (
    decode_access_token,
    issue_token_pair,
    jwt_config,
)
from omnistream.platform.auth.principal import (
    principal_from_api_key,
    principal_from_claims,
    principal_from_user,
    system_principal,
)
from omnistream.platform.billing import (
    UsageMeter,
    aggregate,
    aggregate_rollup,
    load_price_book,
    price_for,
)
from omnistream.platform.quota import (
    QuotaEnforcer,
    QuotaPolicy,
    enforce_rate_limit,
    resolve_policy,
)
from omnistream.platform.rbac import enforce, enforce_tenant, require, require_any, require_role
from omnistream.platform.rbac.permissions import (
    ALL_PERMISSIONS,
    PERMISSION_GROUPS,
    PERMISSION_LABELS,
    Permission,
    WILDCARD,
)
from omnistream.platform.rbac.roles import (
    DEFAULT_ROLE,
    ROLE_DESCRIPTIONS,
    ROLE_ORDER,
    Role,
    permissions_for,
    role_at_least,
    role_rank,
)
from omnistream.platform.tenancy import (
    TenantContext,
    TenantResolver,
    current_tenant_id,
    ensure_default_tenant,
    slug_from_host,
    validate_slug,
)
from omnistream.platform.webhooks import (
    PlatformEvent,
    WebhookDispatcher,
    WebhookEvent,
    default_dispatcher,
    load_subscriptions,
    make_event,
)

__all__ = [
    "ALL_PERMISSIONS",
    "ApiKeyService",
    "DEFAULT_ROLE",
    "Permission",
    "PlatformEvent",
    "QuotaEnforcer",
    "QuotaPolicy",
    "ROLE_DESCRIPTIONS",
    "ROLE_ORDER",
    "Role",
    "TenantContext",
    "TenantResolver",
    "WebhookDispatcher",
    "WebhookEvent",
    "WILDCARD",
    "authenticate_request",
    "authenticate_user",
    "current_tenant_id",
    "default_dispatcher",
    "decode_access_token",
    "describe_auth_backends",
    "enforce",
    "enforce_rate_limit",
    "enforce_tenant",
    "ensure_default_tenant",
    "generate_api_key",
    "issue_token_pair",
    "jwt_config",
    "load_price_book",
    "load_subscriptions",
    "make_event",
    "permissions_for",
    "PERMISSION_GROUPS",
    "PERMISSION_LABELS",
    "principal_from_api_key",
    "principal_from_claims",
    "principal_from_user",
    "price_for",
    "QuotaEnforcer",
    "require",
    "require_any",
    "require_role",
    "resolve_policy",
    "role_at_least",
    "role_rank",
    "slug_from_host",
    "system_principal",
    "tokens_for_principal",
    "UsageMeter",
    "aggregate",
    "aggregate_rollup",
    "validate_slug",
    "verify_api_key",
]
