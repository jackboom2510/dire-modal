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
                "map_array": __import__("numpy").arange(3 * 224 * 224, dtype="float32").reshape(3, 224, 224) / 100000,
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
            self.assertEqual(loaded["map_array"].shape, (3, 224, 224))
            self.assertTrue((loaded["map_array"] >= 0).all())
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
        chart, fake_items, real_items = gradio_app.gallery_map_overview(samples)
        self.assertIn("<svg", chart)
        self.assertEqual(len(fake_items), 112)  # original + map for 56 generated samples
        self.assertEqual(len(real_items), 16)    # original + map for 8 real samples
        self.assertTrue(all(Path(path).is_file() for path, _ in fake_items + real_items))
        self.assertTrue(any("Ảnh gốc" in cap and "Ảnh sinh" in cap for _, cap in fake_items))
        self.assertTrue(any("DIRE map" in cap for _, cap in fake_items))
        self.assertIn("Ảnh gốc", fake_items[0][1])
        self.assertIn("DIRE map", fake_items[1][1])

        # Khi ảnh tái tạo ADM+DDIM thật đã được lưu, overview chèn đúng panel đó
        # giữa original và map; không tự tạo recon từ map tuyệt đối.
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as td:
            recon = Path(td) / "sample_recon.png"
            recon.write_bytes(b"PNG fixture")
            with_recon = [replace(samples[0], recon=recon), *samples[1:]]
            _, fake_with_recon, _ = gradio_app.gallery_map_overview(with_recon)
            self.assertEqual(len(fake_with_recon), 113)
            self.assertIn("Ảnh tái tạo", fake_with_recon[1][1])
        self.assertNotIn("ACC", chart)
        self.assertNotIn("AP", chart)


if __name__ == "__main__":
    unittest.main()
