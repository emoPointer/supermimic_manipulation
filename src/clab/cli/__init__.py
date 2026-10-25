"""The ``clab`` command line: ``clab <group> <command> [options]``.

Commands are imported only when invoked, so ``clab --help`` stays fast.
"""

from __future__ import annotations

import sys
from importlib import import_module

# name -> (module, function, summary) or a nested group of the same.
COMMANDS = {
    "tasks": ("clab.cli.tasks", "main", "List the registered tasks"),
    "train": {
        "ppo": ("clab.cli.train", "ppo", "Train a state policy with PPO (mjlab / RSL-RL)"),
        "supermimic": ("clab.cli.train", "supermimic", "Train with SuperMimic world-model BPTT"),
    },
    "eval": {
        "policy": ("clab.cli.evaluate", "policy", "Evaluate a learned checkpoint"),
    },
    "resets": {
        "collect": ("clab.cli.resets", "collect", "Collect a reset bank for a task"),
    },
}


def _usage(prog: str, table: dict) -> str:
    width = max(map(len, table))
    lines = [f"usage: {prog} <command> [options]", "", "commands:"]
    for name, entry in table.items():
        summary = entry[2] if isinstance(entry, tuple) else " | ".join(entry)
        lines.append(f"  {name:<{width}}  {summary}")
    lines.append(f"\nRun '{prog} <command> --help' for the options of a command.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    table, prog = COMMANDS, "clab"
    while isinstance(table, dict):
        if not args or args[0] in ("-h", "--help"):
            print(_usage(prog, table))
            return 0 if args else 2
        name = args.pop(0)
        if name not in table:
            print(f"{prog}: unknown command {name!r}\n\n{_usage(prog, table)}", file=sys.stderr)
            return 2
        table, prog = table[name], f"{prog} {name}"
    module, function, _ = table
    status = getattr(import_module(module), function)(args, prog=prog)
    return status if isinstance(status, int) else 0
