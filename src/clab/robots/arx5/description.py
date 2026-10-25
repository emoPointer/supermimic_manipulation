"""Ordered names and frames of the packaged robot model."""

JOINTS = tuple(f"joint{i}" for i in range(1, 9))
ARM_JOINTS = JOINTS[:6]
FINGER_JOINTS = JOINTS[6:]
ROOT_BODY = "arx5"
EE_BODY = "link6"  # Sole current EEF: body origin, no tool offset.
FINGER_BODIES = ("link7", "link8")
RIGID_BODIES = (ROOT_BODY, *(f"link{i}" for i in range(1, 9)))
COLLISION_GEOMS = ("base_link_collision", *(f"link{i}_collision" for i in range(1, 9)))
FINGER_ACTUATORS = tuple(f"{joint}_position" for joint in FINGER_JOINTS)
GRASP_OFFSET = (0.145, 0.0, 0.0)  # Grasp geometry only; NOT the EEF frame.
GRIPPER_CLOSING_AXIS = (0.0, 1.0, 0.0)

HOME = (0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.02, 0.02)
GRASP_SITE = "grasp_point"
