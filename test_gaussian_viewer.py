import unittest

import numpy as np

from gaussian_viewer import Camera, Gaussian3D, HEIGHT, WIDTH, project_gaussians, render


class GaussianViewerTests(unittest.TestCase):
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
