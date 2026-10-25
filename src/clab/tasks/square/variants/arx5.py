"""Square observation contract on the ARX5 binding."""

from clab.robots.arx5.binding import Arx5Binding


class Binding(Arx5Binding):
    task_id = "Mjlab-Manipulation-Square-ARX5"
    actor_dim = 195
    critic_dim = 153
    experiment_name = "clab_arx5_square_state"
    wandb_project = "clab_square_state"


BINDING = Binding()
