import unittest

from panda_handover.handover_overlay import handover_region_overlay_spec


class HandoverRegionOverlayTests(unittest.TestCase):
    def test_reviewed_box_becomes_translucent_visual_spec(self):
        spec = handover_region_overlay_spec(
            {
                "frame": "robot_base",
                "units": "metres",
                "up_axis": "Z",
                "humanDirection": [-1.0, 0.0, 0.0],
                "entities": {
                    "handover": {
                        "shape": "box",
                        "pos": [-0.29, 0.06, 0.15],
                        "size": [0.25, 0.25, 0.30],
                        "yaw": 15.0,
                        "color": "#df57dc",
                    }
                },
            },
            source="reviewed.json",
            opacity=0.18,
        )
        for actual, expected in zip(
            spec.center_robot_base_m, (-0.29, 0.06, 0.15), strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(spec.size_m, (0.25, 0.25, 0.30), strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(spec.yaw_degrees, 15.0)
        self.assertAlmostEqual(spec.color_rgb[0], 223 / 255)
        self.assertEqual(spec.opacity, 0.18)
        self.assertFalse(spec.to_report()["affects_physics"])

    def test_point_is_not_silently_invented_as_a_box(self):
        with self.assertRaisesRegex(ValueError, "requires a box"):
            handover_region_overlay_spec(
                {
                    "frame": "robot_base",
                    "units": "metres",
                    "up_axis": "Z",
                    "humanDirection": [-1.0, 0.0, 0.0],
                    "entities": {
                        "handover": {
                            "shape": "point",
                            "pos": [-0.29, 0.06, 0.15],
                        }
                    },
                },
                source="point.json",
                opacity=0.18,
            )

    def test_invalid_opacity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "opacity"):
            handover_region_overlay_spec(
                {
                    "frame": "robot_base",
                    "units": "metres",
                    "up_axis": "Z",
                    "humanDirection": [-1.0, 0.0, 0.0],
                    "entities": {
                        "handover": {
                            "shape": "box",
                            "pos": [0.0, 0.0, 0.0],
                            "size": [0.1, 0.1, 0.1],
                        }
                    },
                },
                source="box.json",
                opacity=0.0,
            )
