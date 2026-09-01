"""Triple extractor: agent_engine envelope + scenario → list of (failed, nudge, corrected) triples.

DECISIONS §13 Post-direct connection `agent_engine.Scenario / Result / TurnView`: transcript segmentation
(`Result.turns()`, `TurnView.start_offset` provide global offset) + intra-segment attempt segmentation
(`TurnView.attempts(agent)`) + static step expansion (`Scenario.expanded_turns()`, including
`instruction` transparent transmission) are all provided by agent_engine. §16 Upgrade to `TranscriptEntry` typed
union (`SpeakerEntry / ToolCallEntry / ArtifactEventEntry / ...`), use isinstance on the consumer side
dispatch takes fields, no more `entry.get("...")` defense. This module only retains:
  - "first attempt fails → subsequent attempts succeed" pair selection
  - failure_mode classification (still `from evals.metrics.nudge import classify_failure_mode`,
    This is the public side of evals, cross-project import is legal)
  - SFT triple schema form

Triple schema (aligned with plan §Schemas):
  - run_id, scenario, turn_idx, step_id, agent, required_tool, failure_mode
  - context: transcript prefix until first attempt's speaker entry (list[TranscriptEntry],
    Convert dataclasses.asdict to list[dict] during JSON serialization)
  - instruction: step.instruction (raw scenario YAML)
  - SpeakerEntry.content of failed_response: first attempt (for diagnostic purposes, does not enter F1 input)
  - nudge: engine hardcoded nudge text (restored by required_tool; do not enter F1 input)
  - corrected_response: SpeakerEntry.content (F1 target) of the final successful attempt

The case where triple is not generated:
  - The first attempt is successful → no failure signal
  - All attempts failed → no positive samples, discarded
  - segment number < expected turn_idx (subprocess crashes midway) → no attempt data
  - failure_mode == 'wrong_args' (deferred to Phase 5 in metrics/nudge.py) → defensive skip"""
  - failure_mode == 'wrong_args' (deferred to Phase 5 in metrics/nudge.py) → defensive skip"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# agent_engine and evals.metrics.nudge.classify_failure_mode are both sister packages of the same monorepo.
# One-way sys.path injection allows import to resolve; the same idea as evals/_ae_bridge.py.
# One-way sys.path injection allows import to resolve; the same idea as evals/_ae_bridge.py.
# One-way sys.path injection allows import to resolve; the same idea as evals/_ae_bridge.py.
# One-way sys.path injection allows import to resolve; the same idea as evals/_ae_bridge.py.
# One-way sys.path injection allows import to resolve; the same idea as evals/_ae_bridge.py.
# One-way sys.path injection allows import to resolve; the same idea as evals/_ae_bridge.py.
# One-way sys.path injection allows import to resolve; the same idea as evals/_ae_bridge.py.
# One-way sys.path injection allows import to resolve; the same idea as evals/_ae_bridge.py.
# One-way sys.path injection allows import to resolve; the same idea as evals/_ae_bridge.py.
# One-way sys.path injection allows import to resolve; the same idea as evals/_ae_bridge.py.
# One-way sys.path injection allows import to resolve; the same idea as evals/_ae_bridge.py.
# One-way sys.path injection allows import to resolve; the same idea as evals/_ae_bridge.py.
REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"
if str(PLAY_DIR) not in sys.path:
    sys.path.insert(0, str(PLAY_DIR))

from agent_engine import (  # noqa: E402  pylint: disable=wrong-import-position
    ArtifactEventEntry,
    ExpandedTurn,
    Result,
    Scenario,
    SpeakerEntry,
    ToolCallEntry,
    TranscriptEntry,
    TurnView,
)
from evals.metrics.nudge import (  # noqa: E402  pylint: disable=wrong-import-position
    classify_failure_mode,
)

# Engine nudge text format (hardcoded in discussion.py); press required_tool to restore
NUDGE_TEMPLATE = "You did not call the `{tool}` tool just now. Please make the call now to complete this round of tasks."

# scenarios_root / filename analysis: default to the same fast copy of mine_triples (max_retries=0
# / delete open+finalize / short max_tokens), --upstream switch back to agent_engine/scenarios/<name>.md.
# Must be consistent with the scenario YAML used when generating the envelope - Scenario.expanded_turns press step
# Expand turn_idx sequentially. After the fast copy deletes open / finalize, the difference between turn_idx and the upstream is 1. Mixing will cause
# Cause expected agent / required_tool / step.instruction to be all misplaced (synthesize is still 0
# triple, extractor will miss all attempts across segments).
FAST_SCENARIOS_DIR = REPO_ROOT / "play" / "agent_sft" / "data" / "scenarios"
UPSTREAM_SCENARIOS_DIR = REPO_ROOT / "play" / "agent_engine" / "scenarios"


def resolve_scenario_path(scenario_name: str, *, upstream: bool) -> Path:
    """Fast / upstream path selection strategy for Mirror mine_triples.py."""
    if upstream:
        return UPSTREAM_SCENARIOS_DIR / f"{scenario_name}.md"
    return FAST_SCENARIOS_DIR / f"{scenario_name}_fast.md"


@dataclass
class Triple:
    """A (failed, nudge, corrected) supervision triple, ready to be fed to the formatter."""

    run_id: int
    scenario: str
    turn_idx: int
    step_id: str | None
    agent: str
    required_tool: str
    failure_mode: str  # "missed" | "wrong_tool"（wrong_args deferred）
    context: list[TranscriptEntry] = field(default_factory=list)
    instruction: str = ""
    failed_response: str = ""
    nudge: str = ""
    corrected_response: str = ""


def _attempt_called_required(
    events: list[TranscriptEntry], agent: str, tool: str,
) -> bool:
    """Is there a `(caller=agent, tool=required_tool)` tool event in attempt - the same as agent_engine"""
`discussion._called_tool` inspection surface."""
    for e in events:
        if isinstance(e, (ToolCallEntry, ArtifactEventEntry)):
            if e.caller == agent and e.tool == tool:
                return True
    return False


def extract_triples(
    envelope: dict,
    scenario_path: str | Path,
    *,
    run_id: int,
    scenario_name: str | None = None,
) -> list[Triple]:
    """envelope dict (per agent_engine.result.Result asdict) + scenario YAML → list[Triple]."""
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
            continue  # subprocess crashes/scenario truncation — no attempt to mine
        tv = turns[seg_idx]

        attempts = tv.attempts(agent)
        speaker_entries = [
            (i, e) for i, e in enumerate(tv.entries)
            if isinstance(e, SpeakerEntry) and e.speaker == agent
        ]
        if not attempts or not speaker_entries:
            continue  #The agent is completely silent in this segment
        if _attempt_called_required(attempts[0], agent, required_tool):
            continue  # The first attempt succeeds — no supervision signal

        success_idx = next(
            (
                i for i, att in enumerate(attempts)
                if _attempt_called_required(att, agent, required_tool)
            ),
            None,
        )
        if success_idx is None:
            continue  # All failed - no positive samples
        if success_idx >= len(speaker_entries):
            continue  # Defense: speaker entry and attempt should correspond 1:1

        failure_mode = classify_failure_mode(attempts[0], agent, required_tool)
        if failure_mode == "wrong_args":
            continue  # deferred; defensive skip

        first_speaker_local_idx, first_speaker_entry = speaker_entries[0]
        _, success_speaker_entry = speaker_entries[success_idx]
        failed_content = first_speaker_entry.content
        corrected_content = success_speaker_entry.content

        first_speaker_global_idx = tv.start_offset + first_speaker_local_idx
        context = list(transcript[:first_speaker_global_idx])

        instruction = expanded_by_turn[turn_idx].instruction.strip()

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


# --- file I/O -------------------------------------------------------------

def write_triples_jsonl(triples: list[Triple], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


def _parse_envelope_name(stem: str) -> tuple[str, int]:
    """'tool_chain-r3' → ('tool_chain', 3)."""
    """'tool_chain-r3' → ('tool_chain', 3)."""
    if "-r" not in stem:
        raise ValueError(f"envelope filename must match '<scenario>-r<N>': {stem!r}")
    scen, _, run = stem.rpartition("-r")
    try:
        run_id = int(run)
    except ValueError as exc:
        raise ValueError(f"envelope filename run_id not int: {stem!r}") from exc
    return scen, run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
help="Use upstream agent_engine/scenarios/<name>.md to parse (consistent with baseline eval);"
"The default is to use fast copy data/scenarios/<name>_fast.md, which must match the version used by mine_triples",
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
        triples = extract_triples(
            envelope, scen_path, run_id=run_id, scenario_name=scen_name
        )
        per_file_summary.append((env_path.name, len(triples)))
        all_triples.extend(triples)

    write_triples_jsonl(all_triples, args.out_path)
    print(f"\n=== Extraction summary ===")
    for name, count in per_file_summary:
        print(f"  {name}: {count} triples")
    print(f"  TOTAL: {len(all_triples)} triples → {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
