"""Drawer observation contract on the ARX5 binding, including its sequence history."""

from clab.robots.arx5.binding import Arx5Binding

from ..process import PhysicalSequence


class Binding(Arx5Binding):
    task_id = "Mjlab-Manipulation-Drawer-ARX5"
    # Only Drawer opts into the corrected non-convex finger geometry.
    finger_collision_geometry = "convex_decomposition"
    # The simplified pre-opened task owns the process-history width.
    actor_dim = (39 + 2 + PhysicalSequence.observation_dim) * 5
    # Current actor + time + arm/finger velocity + two body velocities.
    critic_dim = 39 + 2 + PhysicalSequence.observation_dim + 1 + 8 + 12
    experiment_name = "clab_arx5_drawer_state"
    wandb_project = "clab_drawer_state"


BINDING = Binding()
