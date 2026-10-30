"""Reusable state-policy PPO defaults."""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class ClabPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL configuration using the compatible W&B writer."""

    logging_backend: Literal["both", "tensorboard"] = "both"

    cuda_graph_mode: Literal["auto", "plain", "substeps"] = "auto"

    upload_model: bool = False

    # "native": the task's reward manager. "imagined": the task's SuperMimic MDP
    # reward on each real transition, the function SuperMimic optimizes in
    # imagination (with its imagination-only shaping), for a like-for-like baseline.
    reward_source: Literal["native", "imagined"] = "native"

    task_name: str = ""

    logger: dict[str, str] = field(
        default_factory=lambda: {
            "class_name": "clab.logging.wandb_writer:ClabWandbLogWriter",
            "project_name": "mjlab",
        }
    )


def make_state_ppo_cfg(*, experiment_name: str, wandb_project: str = "mjlab") -> ClabPpoRunnerCfg:
    """PPO hyperparameters with state-dependent Gaussian exploration."""
    return ClabPpoRunnerCfg(
        wandb_project=wandb_project,
        logger={
            "class_name": "clab.logging.wandb_writer:ClabWandbLogWriter",
            "project_name": wandb_project,
        },
        actor=RslRlModelCfg(
            class_name="clab.algorithms.ppo.gsde:GsdeMLPModel",
            hidden_dims=(512, 256, 128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GsdeDistribution",
                "init_std": 1.0,
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128, 64), activation="elu", obs_normalization=True
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            normalize_advantage_per_mini_batch=False,
            clip_param=0.2,
            entropy_coef=0.006,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=experiment_name,
        num_steps_per_env=32,
        max_iterations=40000,
        save_interval=100,
    )
