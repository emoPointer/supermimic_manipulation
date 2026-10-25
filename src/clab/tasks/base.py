"""Task composition contract shared by training, evaluation and reset collection."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from clab.contracts import validate_environment_contract


class ManipulationTask:
    """Bind one task package's configuration, reset bank and success readers.

    A task package provides ``config.py`` (``make_env_cfg``), ``reset_data.py``
    (``validate_bank``/``bank_metadata``), ``resets.py``, ``collect.py`` (reset-bank
    collection CLI) and ``supermimic.py`` (the SuperMimic adapter).
    """

    name: str
    label: str
    robot: object
    reset_types: tuple[str, ...]
    reset_class: type
    # PPO training allocation (nconmax, njmax); None keeps the environment config.
    ppo_capacity: tuple[int, int] | None = None
    # SuperMimic adapter class in the task's ``supermimic`` module; None: unsupported.
    supermimic_adapter: str | None = None

    @staticmethod
    def make_env_cfg(*, play: bool = False, reset_bank: str | Path | None = None):
        raise NotImplementedError

    @property
    def task_id(self) -> str:
        return self.robot.task_id

    def module(self, name: str):
        return import_module(f"clab.tasks.{self.name}.{name}")

    def make_ppo_cfg(self):
        from clab.algorithms.ppo import make_state_ppo_cfg

        return make_state_ppo_cfg(
            experiment_name=self.robot.experiment_name, wandb_project=self.robot.wandb_project
        )

    # Reset banks.

    def validate_bank(self, bank, model=None):
        return self.module("reset_data").validate_bank(bank, model)

    def bank_metadata(self, bank) -> dict:
        return self.module("reset_data").bank_metadata(bank)

    def reset_term(self, env):
        term = env.event_manager.get_term_cfg("reset_from_bank").func
        if not isinstance(term, self.reset_class):
            raise TypeError(f"{self.name}: unexpected reset implementation {type(term)}")
        return term

    def reset_groups(self, env):
        """Reset category of each world, as sampled by the bank reset term."""
        return self.reset_term(env).task_id

    def episode_success(self, env, extras):
        """Read the explicit outcome; termination alone never implies success."""
        del env
        if "episode_success" not in extras:
            raise RuntimeError(f"Completed {self.name} episode has no explicit success outcome")
        return extras["episode_success"]

    # Learned-policy evaluation.

    def make_eval_cfg(self, saved: dict, reset_bank: Path):
        """Evaluation config that must match a checkpoint's saved training environment."""
        cfg = self.make_env_cfg(play=True, reset_bank=reset_bank)
        cfg.events["reset_from_bank"].params["reaching_only"] = False
        corruption = saved["observations"]["actor"]["enable_corruption"]
        if corruption not in ("true", "false"):
            raise ValueError("Saved actor corruption setting must be explicit")
        # Compare against the training noise setting; evaluation disables it afterwards.
        cfg.observations["actor"].enable_corruption = corruption == "true"
        self.restore_eval_allocation(saved, cfg)
        validate_environment_contract(saved, cfg)
        return cfg

    @staticmethod
    def restore_eval_allocation(saved: dict, cfg) -> None:
        """Reuse the training run's simulation capacities (allocation, not physics)."""
        for name in ("nconmax", "njmax"):
            value = saved["sim"][name]
            capacity = None if value == "null" else int(value)
            if capacity is not None and capacity <= 0:
                raise ValueError(f"Saved simulation capacity {name} must be positive")
            setattr(cfg.sim, name, capacity)

    # SuperMimic.

    def configure_supermimic(self, cfg) -> None:
        """Prepare an environment config for SuperMimic transition capture."""
        from clab.tasks.supermimic import install_capture

        if self.supermimic_adapter is None:
            raise ValueError(f"{self.name} has no SuperMimic adapter")
        install_capture(cfg)

    def make_supermimic_adapter(self, env, *, capture: bool = True, **options):
        if self.supermimic_adapter is None:
            raise ValueError(f"{self.name} has no SuperMimic adapter")
        adapter = getattr(self.module("supermimic"), self.supermimic_adapter)
        return adapter(env, capture=capture, **options)

    def supermimic_contract(self, env) -> dict:
        """Read the state-model contract without capture or simulation writes."""
        return self.make_supermimic_adapter(env, capture=False).mdp.contract()
