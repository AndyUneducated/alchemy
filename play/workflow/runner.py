"""``Workflow``: parse + run a workflow.yaml.

A Workflow is the outer deterministic pipeline (plan §2.1). It runs a list
of stages in declaration order, threading state through interpolation:

- ``vars.<name>`` — workflow inputs (CLI ``--vars k=v`` or YAML default)
- ``stages.<name>.output`` — previous stage's return value
- ``pkg_dir`` — workflow.yaml's parent dir (for resolving template paths)

No conditional / parallel / loop / retry / DAG (plan §9). Stage failures
propagate as exceptions; runner does not catch.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import yaml

from . import schema
from .executors import agent as agent_exec
from .executors import deterministic as det_exec
from .state import interpolate


_VAR_CASTS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": lambda v: str(v).lower() in {"1", "true", "yes", "on"},
}


@dataclass
class Workflow:
    """Parsed workflow.yaml ready to ``run(vars_dict)``."""

    path: str
    name: str
    description: str | None
    vars_spec: dict[str, dict]
    hooks_module: str | None
    stages: list[dict]
    workflow_dir: str

    @classmethod
    def from_yaml(cls, path: str) -> "Workflow":
        with open(path, encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        if not isinstance(meta, dict):
            sys.exit(f"Error: {path} is not a valid YAML mapping.")
        schema.validate(meta)
        return cls(
            path=os.path.abspath(path),
            name=meta["name"],
            description=meta.get("description"),
            vars_spec=meta.get("vars") or {},
            hooks_module=meta.get("hooks_module"),
            stages=meta["stages"],
            workflow_dir=os.path.dirname(os.path.abspath(path)),
        )

    def run(self, vars_input: dict[str, str]) -> dict[str, Any]:
        """Execute every stage in order; return the final state dict."""
        resolved_vars = self._resolve_vars(vars_input)
        state: dict[str, Any] = {
            "vars": resolved_vars,
            "stages": {},
            "pkg_dir": self.workflow_dir,
        }

        print(
            f"\n{'=' * 60}\n  workflow: {self.name}  "
            f"({len(self.stages)} stages)\n{'=' * 60}",
            flush=True,
        )

        for stage in self.stages:
            self._run_stage(stage, state)

        print(f"\n{'=' * 60}\n  workflow: {self.name} done\n{'=' * 60}\n",
              flush=True)
        return state

    # -- internals ----------------------------------------------------------

    def _resolve_vars(self, vars_input: dict[str, str]) -> dict[str, Any]:
        """Apply required / default / type cast per ``vars`` spec."""
        out: dict[str, Any] = {}
        for vname, spec in self.vars_spec.items():
            if vname in vars_input:
                raw = vars_input[vname]
            elif spec.get("required"):
                sys.exit(
                    f"Error: workflow {self.name!r} requires --vars {vname}=... "
                    f"(declared 'required: true')."
                )
            else:
                raw = spec.get("default", "")
            cast = _VAR_CASTS[spec.get("type", "str")]
            out[vname] = cast(raw)
        # Allow extra keys the schema didn't declare — pass through as str.
        for vname in vars_input:
            if vname not in out:
                out[vname] = vars_input[vname]
        return out

    def _run_stage(self, stage: dict, state: dict[str, Any]) -> None:
        sname = stage["name"]
        stype = stage["type"]
        output_key = stage.get("output", sname)

        t0 = time.monotonic()
        print(f"\n▶ stage '{sname}' (type={stype})", flush=True)

        if stype == "deterministic":
            args = interpolate(stage.get("args", {}) or {}, state)
            value = det_exec.run(stage, args, hooks_module=self.hooks_module)
        elif stype == "agent":
            config = interpolate(stage.get("config", {}) or {}, state)
            value = agent_exec.run(stage, config, workflow_dir=self.workflow_dir)
        else:
            # Unreachable: schema validator rejects other types.
            sys.exit(f"Error: stage '{sname}': unknown type {stype!r}.")

        dt_ms = int((time.monotonic() - t0) * 1000)
        print(f"  ✓ stage '{sname}' done ({dt_ms} ms)", flush=True)
        state["stages"][output_key] = {"output": value}
