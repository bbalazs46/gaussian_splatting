import unittest
from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pygame

from gaussian_viewer import Camera, Gaussian3D, HEIGHT, MOUSE_SENS, WIDTH, project_gaussians, render


class GaussianViewerTests(unittest.TestCase):
    @patch("pygame.key.get_mods", return_value=0)
    def test_camera_update_inverts_vertical_mouse_rotation(self, _get_mods):
        cam = Camera(pitch=10.0)
        keys = defaultdict(bool)

        cam.update(keys, dt=0.0, dmx=0.0, dmy=4.0)
        self.assertAlmostEqual(cam.pitch, 10.0 - 4.0 * MOUSE_SENS)

        cam.update(keys, dt=0.0, dmx=0.0, dmy=-10.0)
        self.assertAlmostEqual(cam.pitch, 10.0 + 6.0 * MOUSE_SENS)

    @patch("pygame.key.get_mods", return_value=0)
    def test_camera_update_orbits_around_origin_while_key_3_is_pressed(self, _get_mods):
        cam = Camera(pos=(0.0, 0.0, 4.0), yaw=0.0, pitch=0.0)
        keys = defaultdict(bool, {pygame.K_3: True})

        initial_radius = np.linalg.norm(cam.pos)
        cam.update(keys, dt=1.0, dmx=20.0, dmy=8.0)

        self.assertAlmostEqual(np.linalg.norm(cam.pos), initial_radius)
        np.testing.assert_allclose(cam.pos, -cam.forward * initial_radius, atol=1e-6)

    def test_project_gaussians_sorts_front_to_back(self):
        cam = Camera(pos=(0.0, 0.0, 4.5))
        gaussians = [
            Gaussian3D(
                position=np.array([0.0, 0.0, -2.0]),
                scale=np.array([0.3, 0.3, 0.3]),
                rotation=np.array([1.0, 0.0, 0.0, 0.0]),
                color=np.array([0.0, 0.0, 1.0]),
                opacity=0.8,
            ),
            Gaussian3D(
                position=np.array([0.0, 0.0, 1.0]),
                scale=np.array([0.3, 0.3, 0.3]),
                rotation=np.array([1.0, 0.0, 0.0, 0.0]),
                color=np.array([1.0, 0.0, 0.0]),
                opacity=0.8,
            ),
        ]

        projected = project_gaussians(gaussians, cam)

        self.assertEqual(len(projected), 2)
        self.assertLess(projected[0]["depth"], projected[1]["depth"])

    def test_render_front_splat_occludes_back_splat(self):
        fb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
        projected = [
            {
                "depth": 3.0,
                "px": WIDTH / 2,
                "py": HEIGHT / 2,
                "sig_inv": np.eye(2, dtype=np.float32) * 0.5,
                "max_std": 2.0,
                "color": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "opacity": 0.5,
            },
            {
                "depth": 6.0,
                "px": WIDTH / 2,
                "py": HEIGHT / 2,
                "sig_inv": np.eye(2, dtype=np.float32) * 0.5,
                "max_std": 2.0,
                "color": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                "opacity": 0.8,
            },
        ]

        render(projected, fb)

        center = fb[HEIGHT // 2, WIDTH // 2]
        np.testing.assert_allclose(center, np.array([0.5, 0.0, 0.4]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
