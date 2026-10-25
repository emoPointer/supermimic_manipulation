"""MugHang observation contract on the ARX5 binding."""

from clab.robots.arx5.binding import Arx5Binding


class Binding(Arx5Binding):
    task_id = "Mjlab-Manipulation-MugHang-ARX5"
    actor_dim = 210
    critic_dim = 240
    experiment_name = "clab_arx5_mug_hang_state"
    wandb_project = "clab_mug_hang_state"


BINDING = Binding()
