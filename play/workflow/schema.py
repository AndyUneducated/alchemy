"""Minimal workflow.yaml schema validation.

Per plan §12 fail-fast philosophy: detect programmer errors at first failure
point, no friendly hints, no migration helpers, no smart inference. The
validator only catches "missing required field" / "wrong shape" mistakes.
Anything else (missing referenced stage, unknown function, runtime KeyError
in template) is allowed to surface as a normal Python exception when the
runner reaches it.
"""

from __future__ import annotations

import sys
from typing import Any


VALID_STAGE_TYPES = {"deterministic", "agent"}
VALID_VAR_TYPES = {"str", "int", "float", "bool"}


def _err(msg: str) -> None:
    sys.exit(f"Error: workflow.yaml: {msg}")


def validate(meta: dict[str, Any]) -> None:
    """Validate top-level ``workflow.yaml`` shape; fail-fast on missing fields."""
    if not isinstance(meta, dict):
        _err("top-level YAML must be a mapping.")

    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        _err("missing required string 'name'.")

    stages = meta.get("stages")
    if not isinstance(stages, list) or not stages:
        _err("'stages' must be a non-empty list.")

    seen_names: set[str] = set()
    for i, stage in enumerate(stages, 1):
        where = f"stages[{i}]"
        if not isinstance(stage, dict):
            _err(f"{where} must be a mapping.")
        sname = stage.get("name")
        if not isinstance(sname, str) or not sname.strip():
            _err(f"{where} missing required string 'name'.")
        if sname in seen_names:
            _err(f"{where} duplicate stage name {sname!r}.")
        seen_names.add(sname)
        stype = stage.get("type")
        if stype not in VALID_STAGE_TYPES:
            _err(
                f"{where} '{sname}' has type={stype!r}; "
                f"must be one of: {sorted(VALID_STAGE_TYPES)}."
            )
        if stype == "deterministic":
            fn = stage.get("fn")
            if not isinstance(fn, str) or not fn.strip():
                _err(f"{where} '{sname}' missing required string 'fn' for deterministic stage.")
        elif stype == "agent":
            scn = stage.get("scenario")
            if not isinstance(scn, str) or not scn.strip():
                _err(f"{where} '{sname}' missing required string 'scenario' for agent stage.")

    vars_cfg = meta.get("vars")
    if vars_cfg is not None:
        if not isinstance(vars_cfg, dict):
            _err("'vars' must be a mapping.")
        for vname, spec in vars_cfg.items():
            if not isinstance(spec, dict):
                _err(f"vars['{vname}'] must be a mapping (with 'required' or 'default').")
            if "required" in spec and not isinstance(spec["required"], bool):
                _err(f"vars['{vname}'].required must be a boolean.")
            if "type" in spec and spec["type"] not in VALID_VAR_TYPES:
                _err(
                    f"vars['{vname}'].type={spec['type']!r}; "
                    f"must be one of: {sorted(VALID_VAR_TYPES)}."
                )
            if "required" not in spec and "default" not in spec:
                _err(
                    f"vars['{vname}'] needs either 'required: true' or a 'default'."
                )
