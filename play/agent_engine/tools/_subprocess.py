from __future__ import annotations

import json
import subprocess


def run_json_subprocess(cmd: list[str]) -> tuple[int, dict | None]:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return result.returncode, None
    try:
        return 0, json.loads(result.stdout)
    except (ValueError, TypeError):
        return 0, None
