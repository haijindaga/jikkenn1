"""Reviewed tabletop scene geometry for the Franka integration tests.

The coordinate convention follows NVIDIA's Isaac Lab Franka lift task: the
robot mounting plane and tabletop are at z=0, while the room floor is below
them.  The target remains inside the task's documented x/y sampling region.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


ISAAC_LAB_LIFT_CONFIG = (
    "https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/"
    "isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py"
)
ISAAC_LAB_FRANKA_CONFIG = (
    "https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/"
    "isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py"
)


@dataclass(frozen=True)
class TabletopSceneLayout:
    """Metric world poses for the reproducible tabletop smoke scene."""

    robot_base_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ground_z_m: float = -1.05
    table_center_m: tuple[float, float, float] = (0.50, 0.0, -0.025)
    table_size_m: tuple[float, float, float] = (0.80, 1.00, 0.05)
    target_center_m: tuple[float, float, float] = (0.50, 0.20, 0.025)
    target_size_m: tuple[float, float, float] = (0.20, 0.05, 0.05)
    obstacle_center_m: tuple[float, float, float] = (0.50, -0.20, 0.05)
    obstacle_size_m: tuple[float, float, float] = (0.10, 0.10, 0.10)
    camera_position_m: tuple[float, float, float] = (1.25, 0.0, 1.35)
    camera_target_m: tuple[float, float, float] = (0.48, 0.0, 0.08)

    @property
    def table_top_z_m(self) -> float:
        return self.table_center_m[2] + 0.5 * self.table_size_m[2]

    @property
    def table_bottom_z_m(self) -> float:
        return self.table_center_m[2] - 0.5 * self.table_size_m[2]

    def validation_report(self) -> dict:
        """Return inspectable checks that prevent the old floor/table mismatch."""

        tolerance = 1e-9
        table_min_x = self.table_center_m[0] - 0.5 * self.table_size_m[0]
        table_max_x = self.table_center_m[0] + 0.5 * self.table_size_m[0]
        table_min_y = self.table_center_m[1] - 0.5 * self.table_size_m[1]
        table_max_y = self.table_center_m[1] + 0.5 * self.table_size_m[1]
        target_bottom = self.target_center_m[2] - 0.5 * self.target_size_m[2]
        obstacle_bottom = self.obstacle_center_m[2] - 0.5 * self.obstacle_size_m[2]
        checks = {
            "robot_mount_matches_tabletop": abs(
                self.robot_base_position_m[2] - self.table_top_z_m
            )
            <= tolerance,
            "floor_is_below_table": self.ground_z_m < self.table_bottom_z_m,
            "target_rests_on_table": abs(target_bottom - self.table_top_z_m) <= tolerance,
            "obstacle_rests_on_table": abs(obstacle_bottom - self.table_top_z_m) <= tolerance,
            "target_footprint_is_on_table": (
                table_min_x <= self.target_center_m[0] - 0.5 * self.target_size_m[0]
                and self.target_center_m[0] + 0.5 * self.target_size_m[0] <= table_max_x
                and table_min_y <= self.target_center_m[1] - 0.5 * self.target_size_m[1]
                and self.target_center_m[1] + 0.5 * self.target_size_m[1] <= table_max_y
            ),
            "target_is_in_isaac_lab_xy_region": (
                0.4 <= self.target_center_m[0] <= 0.6
                and -0.25 <= self.target_center_m[1] <= 0.25
            ),
        }
        return {
            "status": "success" if all(checks.values()) else "failure",
            "reference": {
                "coordinate_convention": "robot mounting plane and tabletop at world z=0",
                "isaac_lab_lift_config": ISAAC_LAB_LIFT_CONFIG,
                "isaac_lab_franka_config": ISAAC_LAB_FRANKA_CONFIG,
            },
            "layout": asdict(self),
            "derived": {
                "table_top_z_m": self.table_top_z_m,
                "table_bottom_z_m": self.table_bottom_z_m,
                "target_bottom_z_m": target_bottom,
                "obstacle_bottom_z_m": obstacle_bottom,
            },
            "automatic_checks": checks,
        }


DEFAULT_TABLETOP_LAYOUT = TabletopSceneLayout()

