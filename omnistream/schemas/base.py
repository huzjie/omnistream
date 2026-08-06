"""所有领域模型基类。统一 orjson、别名、ORM 转换行为。"""
from __future__ import annotations

from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict


class OmniModel(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=False,
        validate_assignment=False,
        arbitrary_types_allowed=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    def to_json(self) -> bytes:
        return orjson.dumps(self.model_dump(mode="json", exclude_none=True))

    def merged(self, patch: dict[str, Any]) -> Any:
        return self.model_copy(update=patch, deep=True)
