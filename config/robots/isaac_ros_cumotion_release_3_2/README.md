# NVIDIA Isaac ROS cuMotion UR10e + Robotiq baseline

These robot-description files are copied unchanged from
`NVIDIA-ISAAC-ROS/isaac_ros_cumotion` branch `release-3.2` at commit
`dbaa7e8264f6314f8baca516511414186ad1105d`:

- `ur10e_robotiq_2f_140.urdf`
- `ur10e_robotiq_2f_140.xrdf`

That release provides the UR10e + Robotiq 2F-140 pair, including its
`grasp_frame`, c-space configuration, collision spheres, self-collision
exclusions, and `attached_object` planning frame. The files are immutable
input to cuRobo's XRDF converter. `scripts/prepare_ur10e_robot_profile.py`
checks their SHA-256 digests before conversion.

The copied repository license is in `LICENSE` (Apache-2.0). Runtime simulation
continues to use Isaac Sim 5.1's bundled `ur_gripper.usd`; the mandatory FK
alignment gate must pass before a trajectory is accepted.
