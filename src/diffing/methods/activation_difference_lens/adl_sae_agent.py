from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from .agents import ADLAgent


class ADLSAEAgent(ADLAgent):
    """ADL agent variant that appends SAE summary evidence."""

    @property
    def name(self) -> str:
        return "ADL_SAE"

    def _load_sae_overview(self) -> Dict[str, Any]:
        agent_cfg = self.cfg.diffing.method.agent
        sae_cfg = getattr(agent_cfg, "sae_overview", None)
        if sae_cfg is None or not bool(getattr(sae_cfg, "enabled", False)):
            return {"disabled": True, "reason": "sae_overview.enabled=false"}

        path_str = str(getattr(sae_cfg, "path", "")).strip()
        if not path_str:
            return {"disabled": True, "reason": "sae_overview.path is empty"}

        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            logger.warning(f"SAE overview file not found: {path}")
            return {"disabled": True, "reason": f"missing file: {path}"}

        raw = path.read_text(encoding="utf-8")
        max_chars = int(getattr(sae_cfg, "max_chars", 12000))
        if max_chars > 0 and len(raw) > max_chars:
            raw = raw[:max_chars]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(f"Invalid SAE overview JSON at {path}: {exc}")
            return {"disabled": True, "reason": f"invalid json: {path}"}

    def build_first_user_message(self, method: Any) -> str:
        adl_message = super().build_first_user_message(method)
        sae_payload = self._load_sae_overview()
        sae_str = json.dumps(sae_payload)
        return (
            adl_message
            + "\n\nSAE_OVERVIEW:\n"
            + sae_str
            + "\n\nUse SAE_OVERVIEW as additional evidence and combine it with ADL signals."
        )

