"""Read and compare saved scientific contracts without executing YAML object tags."""

from dataclasses import asdict
from pathlib import Path

import yaml


def action_contract(action: dict) -> dict:
    """Exclude descriptive provenance, retaining every executable action setting."""
    return {name: value for name, value in action.items() if name != "parameter_source"}


def validate_environment_contract(saved: dict, cfg, *, for_resume: bool = False) -> None:
    """Compare policy, timing, plant and non-bank event semantics.

    Evaluation also requires the same rewards, terminations and simulation
    allocation. Resume retains its existing allowance for reward/termination and
    allocation changes; neither mode treats this as exact simulation continuation.
    Reset-bank selection, world count and descriptive parameter provenance are
    not execution compatibility checks. Explicit task-owned scientific contracts
    are always strict, including resume. The original saved mapping is not changed.
    """
    current = yaml.load(yaml.dump(asdict(cfg)), Loader=yaml.BaseLoader)

    def task_contracts(record):
        # A task may explicitly declare objective/geometry semantics in an
        # existing termination parameter. These are not tunable reward weights
        # and must survive the ordinary resume allowance for termination edits.
        result = {}
        terms = record.get("terminations", {})
        if not isinstance(terms, dict):
            raise ValueError("Invalid saved/current terminations mapping")
        for name, term in terms.items():
            if not isinstance(term, dict):
                raise ValueError(f"Invalid saved/current termination for {name}")
            params = term.get("params", {})
            if not isinstance(params, dict):
                raise ValueError(f"Invalid saved/current termination parameters for {name}")
            if "task_contract" in params:
                if not isinstance(params["task_contract"], dict):
                    raise ValueError(f"Invalid saved/current task contract for {name}")
                result[name] = params["task_contract"]
        return result

    def observations(record):
        # Checking each term or the concatenated group detects the same NaN/Inf
        # values; only the error message's context differs.
        return {
            name: {key: value for key, value in group.items() if key != "nan_check_per_term"}
            if isinstance(group, dict)
            else group
            for name, group in record["observations"].items()
        }

    def contract(record):
        fields = ("decimation", "episode_length_s", "is_finite_horizon")
        if not for_resume:
            fields += ("rewards", "terminations")
        result = {
            "actions": {
                name: action_contract(action) for name, action in record["actions"].items()
            },
            "task-owned scientific contracts": task_contracts(record),
            "observations": observations(record),
            **{field: record[field] for field in fields},
            "sim": record["sim"]["mujoco"] if for_resume else record["sim"],
            "scene entities": record["scene"]["entities"],
            "startup/reset events": {
                name: event for name, event in record["events"].items() if name != "reset_from_bank"
            },
        }
        return result

    try:
        recorded = contract(saved)
    except KeyError as error:
        raise ValueError(f"Saved environment lacks required field: {error.args[0]}") from error
    expected = contract(current)
    for field, value in expected.items():
        if recorded[field] != value:
            raise ValueError(f"Checkpoint/current {field} mismatch")


def checkpoint_environment(checkpoint: Path, config_path: Path | None = None) -> tuple[dict, Path]:
    if config_path is None:
        run_dir = checkpoint.parent.parent if checkpoint.is_dir() else checkpoint.parent
        config_path = run_dir / "params" / "env.yaml"
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Saved checkpoint environment not found: {config_path}. "
            "Provide --training-env-config; observation shape alone does not establish compatibility."
        )
    saved = yaml.load(config_path.read_text(), Loader=yaml.BaseLoader)
    if not isinstance(saved, dict):
        raise ValueError(f"Expected a saved environment mapping in {config_path}")
    return saved, config_path
