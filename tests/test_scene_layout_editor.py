import unittest
from pathlib import Path


class SceneLayoutEditorTests(unittest.TestCase):
    def test_windows_editor_exports_reviewed_support_layout(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "scene_layout_editor_windows.html"
        ).read_text(encoding="utf-8")
        self.assertIn("jikkenn1-support-layout-v1", source)
        self.assertIn("downloadSupportJson", source)
        self.assertIn("applyAuthoredSupports", source)
        self.assertIn("support_centers_world_m", source)
        self.assertIn("support_sizes_m", source)
        self.assertIn("台JSON保存", source)
        self.assertIn("ハンマー1台", source)
        self.assertIn("ハサミ2台", source)
        self.assertIn("frame:'world'", source)
        self.assertIn("units:'metres'", source)
        self.assertIn("up_axis:'Z'", source)


if __name__ == "__main__":
    unittest.main()
