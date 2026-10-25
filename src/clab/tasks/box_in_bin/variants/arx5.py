"""BoxInBin observation contract on the ARX5 binding."""

from clab.robots.arx5.binding import Arx5Binding


class Binding(Arx5Binding):
    task_id = "Mjlab-Manipulation-BoxInBin-ARX5"
    actor_dim = 195
    critic_dim = 164
    experiment_name = "clab_arx5_box_in_bin_state"
    wandb_project = "arx5_box_in_bin_state"
    # Reset-proposal joint bands for collection, distinct from the robot's joint limits.
    arm_proposal_low = (-0.8, 1.2, 0.7, -1.60, -0.65, -1.5)
    arm_proposal_high = (0.8, 2.4, 2.1, -0.2, 0.65, 1.5)


BINDING = Binding()
