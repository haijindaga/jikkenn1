import json
from pathlib import Path
import unittest


class RoboLabAssetManifestTests(unittest.TestCase):
    def test_baseline_assets_are_pinned_with_licenses(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "config" / "robolab_baseline_assets.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(manifest["source_commit"]), 40)
        self.assertEqual(set(manifest["assets"]), {"hammer", "scissors"})
        for record in manifest["assets"].values():
            self.assertTrue(record["path"].endswith(".usd"))
            self.assertEqual(len(record["sha256"]), 64)
            self.assertTrue(record["license_path"].endswith("LICENSE"))
            self.assertGreater(record["mass_kg"], 0.0)
            self.assertGreaterEqual(record["static_friction"], 0.0)


if __name__ == "__main__":
    unittest.main()
