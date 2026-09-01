"""Synthetic triple builder: mine true failure attempt + synthesis from agent_engine envelope
Correctly respond to triples - bypassing the thin signal path of "waiting for the model to recover itself".

Approach B (difference from [`extractor.py`](extractor.py)):

| pairing strategy | extractor.py | synthesize.py |
|--------------------|----------------------------------|--------------------------------------------------|
| Trigger condition | First attempt fails + subsequent attempts succeed | First attempt fails (i.e. any nudge fired) |
| corrected_response | speaker.content of real successful attempt | programmatically synthesized by step.instruction + tool name |
| yield | ~3-25% (depends on model recovery rate) | 100% (all fires) |
| supervision semantics | "real samples corrected by the model itself" | "failure demonstrations are real, standard answers are templates" |

Why use: 7B nudge recovery rate is only ~3% in most scenarios, resulting in extractor path
The yield is extremely low. The synthesize path uses the literal `tool(args)` template in instruction (fallback: universal
"I now call X" packaging) creates the correct response, and each fire can produce training samples.

Supports the same Triple schema as [`extractor.py`](extractor.py), downstream [`split.py`](split.py) /
[`formatter.py`](formatter.py) is not aware of the data source."""
[`formatter.py`](formatter.py) is not aware of the data source."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Shared extractor's PLAY_DIR sys.path injection path - agent_engine + evals are the same as monorepo
# Shared extractor's PLAY_DIR sys.path injection path - agent_engine + evals are the same as monorepo
# Shared extractor's PLAY_DIR sys.path injection path - agent_engine + evals are the same as monorepo
# Shared extractor's PLAY_DIR sys.path injection path - agent_engine + evals are the same as monorepo
# Shared extractor's PLAY_DIR sys.path injection path - agent_engine + evals are the same as monorepo
# Shared extractor's PLAY_DIR sys.path injection path - agent_engine + evals are the same as monorepo
# Shared extractor's PLAY_DIR sys.path injection path - agent_engine + evals are the same as monorepo
#Sister package, directly connected to agent_engine.Result/Scenario after DECISIONS §13, only evals.metrics.nudge is retained
# The classify_failure_mode public face import
REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"
if str(PLAY_DIR) not in sys.path:
    sys.path.insert(0, str(PLAY_DIR))

from agent_engine import (  # noqa: E402  pylint: disable=wrong-import-position
    ExpandedTurn,
    Result,
    Scenario,
    SpeakerEntry,
    TurnView,
)
from evals.metrics.nudge import (  # noqa: E402  pylint: disable=wrong-import-position
    classify_failure_mode,
)

# Reuse extractor's Triple + helpers + scenario path analysis to avoid data schema drift
# Reuse extractor's Triple + helpers + scenario path analysis to avoid data schema drift
# Reuse extractor's Triple + helpers + scenario path analysis to avoid data schema drift
# Reuse extractor's Triple + helpers + scenario path analysis to avoid data schema drift
# Reuse extractor's Triple + helpers + scenario path analysis to avoid data schema drift
# Reuse extractor's Triple + helpers + scenario path analysis to avoid data schema drift
# Reuse extractor's Triple + helpers + scenario path analysis to avoid data schema drift
from extractor import (  # noqa: E402  pylint: disable=wrong-import-position
    NUDGE_TEMPLATE,
    Triple,
    _attempt_called_required,
    _parse_envelope_name,
    resolve_scenario_path,
    write_triples_jsonl,
)


def synthesize_corrected_response(instruction: str, required_tool: str) -> str:
    """Create a 'corrected' response string based on the step.instruction text."""

    Priority: Grab the literal call template (such as `append_section("review_a", "...")`) in the instruction as the main body.
    Fallback: Use the universal wrapper "I now call {tool} to complete this step:\n{instruction}".

    Template grabbing support:
      - Single layer paren, cross-line args (most scenario.instructions are in this form)
      - Chinese quotation mark / string literal mixing (regex does not parse the inner part, only the first unbalanced `)`)

    The return value is deterministic plain text - the same (instruction, tool) always produces the same output."""
    The return value is deterministic plain text - the same (instruction, tool) always produces the same output."""
    instruction = (instruction or "").strip()
    template = _extract_call_template(instruction, required_tool)
    if template:
return f"Okay, I now call `{required_tool}`:\n\n{template}"
    return (
f"Okay, I now call `{required_tool}` to complete this step:\n\n{instruction}"
        if instruction
else f"Okay, I'll call `{required_tool}` now."
    )


def _extract_call_template(instruction: str, tool: str) -> str | None:
    """Looks for the literal `{tool}(...)` fragment in instruction; returns None if not found."""

    Matching strategy:
      - `\\b{tool}\\s*\\(` Start
      - args gets the first unbalanced `)` (a single layer of brackets, enough to cover the actual writing of scenario)
      - Do not parse internal string literals; only look at paren balancing"""
      - Do not parse internal string literals; only look at paren balancing"""
    needle = re.escape(tool)
    pattern = re.compile(rf"\b{needle}\s*\(", re.MULTILINE)
    m = pattern.search(instruction)
    if not m:
        return None
    start = m.start()
    open_paren = m.end() - 1
    depth = 0
    for i in range(open_paren, len(instruction)):
        ch = instruction[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return instruction[start:i + 1]
    return None  # Unbalanced paren — no forced matching


def envelope_to_synthetic_triples(
    envelope: dict,
    scenario_path: str | Path,
    *,
    run_id: int,
    scenario_name: str | None = None,
) -> list[Triple]:
    """envelope dict + scenario YAML → list[Triple], pairing strategy = "each fire → 1 triple"."""
    scenario_path = Path(scenario_path)
    if scenario_name is None:
        scenario_name = scenario_path.stem

    result = Result.from_dict(envelope)
    transcript = result.transcript
    expanded = Scenario.from_yaml(str(scenario_path)).expanded_turns()
    turns: list[TurnView] = result.turns()
    expanded_by_turn: dict[int, ExpandedTurn] = {e.turn_idx: e for e in expanded}

    out: list[Triple] = []
    for exp in expanded:
        if not exp.require_tool:
            continue
        turn_idx = exp.turn_idx
        agent = exp.agent
        required_tool = exp.require_tool

        seg_idx = turn_idx - 1
        if seg_idx >= len(turns):
            continue  # subprocess crashes midway
        tv = turns[seg_idx]

        attempts = tv.attempts(agent)
        speaker_entries = [
            (i, e) for i, e in enumerate(tv.entries)
            if isinstance(e, SpeakerEntry) and e.speaker == agent
        ]
        if not attempts or not speaker_entries:
            continue
        if _attempt_called_required(attempts[0], agent, required_tool):
            continue  # Success the first time — no nudge fire, no triple

        failure_mode = classify_failure_mode(attempts[0], agent, required_tool)
        if failure_mode == "wrong_args":
            continue  # deferred to Phase 5; defensive skip

        first_speaker_local_idx, first_speaker_entry = speaker_entries[0]
        failed_content = first_speaker_entry.content
        first_speaker_global_idx = tv.start_offset + first_speaker_local_idx
        context = list(transcript[:first_speaker_global_idx])

        instruction = expanded_by_turn[turn_idx].instruction.strip()
        corrected_content = synthesize_corrected_response(instruction, required_tool)

        out.append(Triple(
            run_id=run_id,
            scenario=scenario_name,
            turn_idx=turn_idx,
            step_id=exp.step_id,
            agent=agent,
            required_tool=required_tool,
            failure_mode=failure_mode,
            context=context,
            instruction=instruction,
            failed_response=failed_content,
            nudge=NUDGE_TEMPLATE.format(tool=required_tool),
            corrected_response=corrected_content,
        ))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
epilog="Mutually exclusive relationship with extractor.py: just take one data strategy and produce triples.jsonl.",
    )
    parser.add_argument(
        "--in", dest="in_dir", required=True,
        help="directory of envelope JSONs named '<scenario>-r<N>.json'",
    )
    parser.add_argument(
        "--out", dest="out_path", required=True,
        help="output triples.jsonl path",
    )
    parser.add_argument(
        "--upstream", action="store_true",
help="Use upstream agent_engine/scenarios/<name>.md to parse; default to fast copy "
"data/scenarios/<name>_fast.md, must match the version used by mine_triples",
    )
    parser.add_argument(
        "--scenarios-root", default=None,
help="Explicitly overwrite the scenarios directory, use less - prefer --upstream / default fast copy",
    )
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        print(f"ERROR: --in {in_dir} is not a directory", file=sys.stderr)
        return 2

    explicit_root = Path(args.scenarios_root) if args.scenarios_root else None

    envelopes = sorted(in_dir.glob("*.json"))
    if not envelopes:
        print(f"ERROR: no envelope JSONs under {in_dir}", file=sys.stderr)
        return 2

    all_triples: list[Triple] = []
    per_file_summary: list[tuple[str, int]] = []
    for env_path in envelopes:
        scen_name, run_id = _parse_envelope_name(env_path.stem)
        if explicit_root is not None:
            scen_path = explicit_root / f"{scen_name}.md"
        else:
            scen_path = resolve_scenario_path(scen_name, upstream=args.upstream)
        with env_path.open("r", encoding="utf-8") as f:
            envelope = json.load(f)
        triples = envelope_to_synthetic_triples(
            envelope, scen_path, run_id=run_id, scenario_name=scen_name
        )
        per_file_summary.append((env_path.name, len(triples)))
        all_triples.extend(triples)

    write_triples_jsonl(all_triples, args.out_path)
    print(f"\n=== Synthetic extraction summary (Approach B) ===")
    for name, count in per_file_summary:
        print(f"  {name}: {count} triples")
    print(f"  TOTAL: {len(all_triples)} triples → {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
