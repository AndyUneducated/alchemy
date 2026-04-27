from __future__ import annotations

import importlib
import sys
from typing import Any


def _resolve_fn(fn_str: str, *, hooks_module: str | None) -> Any:
    if ":" in fn_str:
        mod_path, _, func_name = fn_str.partition(":")
    else:
        if not hooks_module:
            sys.exit(
                f"Error: stage fn={fn_str!r} is a bare name but workflow has "
                f"no top-level 'hooks_module'. Either set hooks_module or use "
                f"'pkg.sub:func' colon syntax."
            )
        mod_path = hooks_module
        func_name = fn_str
    module = importlib.import_module(mod_path)
    return getattr(module, func_name)


def run(stage: dict, args: dict, *, hooks_module: str | None) -> Any:
    fn = _resolve_fn(stage["fn"], hooks_module=hooks_module)
    return fn(**args)
