from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "platforms" / "hf-spaces"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from demolib import cached_illustration, frame_cache, provenance
import gradio_app


class FrameCacheTests(unittest.TestCase):
    def test_cache_round_trip_persists_each_step_image_and_outputs(self):
        image = b"input-image-bytes"
        key = frame_cache.make_key(image, "checkpoint-abc", 5)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = {
                "steps": 1,
                "nfe": 2,
                "seconds": 1.25,
                "map_values": {"mean": 0.1, "p99": 0.2, "max": 0.3},
                "map_png": b"map-png",
                "recon_png": b"recon-png",
                "timings": [0.5, 0.6],
                "frames": [{
                    "phase": "invert", "i": 0, "t": 0, "sec": 0.5,
                    "eps_rms": 0.7, "map_mean": 0.1,
                    "x_t_png": b"x-t-png", "x0_hat_png": b"x0-hat-png",
                    "x0_png": b"compatibility-alias-must-not-be-serialized",
                }],
            }
            frame_cache.store(key, run, root=root)
            loaded = frame_cache.load(key, root=root)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["map_png"], b"map-png")
            self.assertEqual(loaded["recon_png"], b"recon-png")
            self.assertEqual(loaded["frames"][0]["x_t_png"], b"x-t-png")
            self.assertEqual(loaded["frames"][0]["x0_hat_png"], b"x0-hat-png")
            self.assertTrue((root / key / "x_t-000.png").is_file())
            self.assertEqual(frame_cache.clear(root=root), 1)

    def test_cache_key_changes_with_image_model_or_steps(self):
        key = frame_cache.make_key(b"a", "checkpoint", 5)
        self.assertNotEqual(key, frame_cache.make_key(b"b", "checkpoint", 5))
        self.assertNotEqual(key, frame_cache.make_key(b"a", "other-checkpoint", 5))
        self.assertNotEqual(key, frame_cache.make_key(b"a", "checkpoint", 10))

    def test_cached_illustration_has_every_step_and_is_labeled_synthetic(self):
        manifest = cached_illustration.manifest()
        self.assertEqual(set(manifest["steps"]), {"5", "10", "20"})
        self.assertFalse(manifest["is_adm_inference"])
        self.assertFalse(manifest["is_measurement"])
        for steps, expected in ((5, 10), (10, 20), (20, 40)):
            self.assertEqual(len(manifest["steps"][str(steps)]), expected)
            panels, description = cached_illustration.frame(steps, expected - 1)
            self.assertEqual(len(panels), 2)
            self.assertTrue(all(Path(path).is_file() for path, _ in panels))
            self.assertIn("không phải DIRE map", description)


class GalleryOverviewTests(unittest.TestCase):
    def test_overview_shows_generator_maps_and_separates_real_map(self):
        samples, problems = provenance.discover(ROOT / "modal-runs" / "gallery")
        self.assertEqual(problems, [])
        chart, fake_maps, real_maps = gradio_app.gallery_map_overview(samples)
        self.assertIn("<svg", chart)
        self.assertEqual(len(fake_maps), 56)
        self.assertEqual(len(real_maps), 8)
        self.assertTrue(all("Map" not in cap or Path(path).is_file() for path, cap in fake_maps))
        self.assertNotIn("ACC", chart)
        self.assertNotIn("AP", chart)


if __name__ == "__main__":
    unittest.main()
