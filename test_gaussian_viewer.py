import json
import unittest
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pygame

from gaussian_viewer import (
    Camera,
    DEFAULT_SCENE_FILENAME,
    Gaussian3D,
    HEIGHT,
    MOUSE_SENS,
    WIDTH,
    create_gaussian_scene_file,
    evaluate_gaussian_scene_consistency,
    improve_gaussian_scene_consistency,
    open_image_folder,
    project_gaussians,
    render,
    select_working_folder,
)


class GaussianViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def _create_test_image(self, path: Path, color: tuple[int, int, int], size: tuple[int, int] = (4, 2)) -> None:
        surface = pygame.Surface(size)
        surface.fill(color)
        pygame.image.save(surface, str(path))

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

    def test_open_image_folder_lists_supported_images(self):
        with TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            self._create_test_image(folder / "b.bmp", (255, 0, 0))
            self._create_test_image(folder / "a.bmp", (0, 255, 0))
            (folder / "notes.txt").write_text("ignore", encoding="utf-8")

            image_paths = open_image_folder(folder)
            loaded_surface = pygame.image.load(str(image_paths[0]))

            self.assertEqual([path.name for path in image_paths], ["a.bmp", "b.bmp"])
            self.assertEqual(loaded_surface.get_size(), (4, 2))

    def test_create_gaussian_scene_file_writes_camera_angles_and_gaussians(self):
        with TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            self._create_test_image(folder / "front.bmp", (255, 0, 0), size=(4, 2))
            self._create_test_image(folder / "back.bmp", (0, 0, 255), size=(2, 4))

            scene_path = create_gaussian_scene_file(folder)

            self.assertEqual(scene_path, folder / DEFAULT_SCENE_FILENAME)
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            self.assertEqual([image["file"] for image in scene["images"]], ["back.bmp", "front.bmp"])
            self.assertEqual(scene["images"][0]["camera_angles"]["yaw_deg"], 0.0)
            self.assertEqual(scene["images"][1]["camera_angles"]["yaw_deg"], 180.0)
            self.assertEqual(len(scene["gaussians"]), 2)
            self.assertEqual(scene["gaussians"][0]["source_image"], "back.bmp")
            self.assertAlmostEqual(scene["gaussians"][1]["color"][0], 1.0, places=4)

    def test_select_working_folder_returns_selected_path(self):
        with TemporaryDirectory() as selected_dir:
            initial_path = Path("non-existent-initial-dir")
            selected_path = Path(selected_dir)
            seen_initial_dirs = []

            def dialog_opener(passed_initial_dir):
                seen_initial_dirs.append(passed_initial_dir)
                return str(selected_path)

            result = select_working_folder(initial_dir=initial_path, dialog_opener=dialog_opener)

            self.assertEqual(result, selected_path.resolve())
            self.assertEqual(seen_initial_dirs, [None])

    def test_select_working_folder_returns_none_when_cancelled(self):
        result = select_working_folder(dialog_opener=lambda _initial_dir: "")

        self.assertIsNone(result)

    def test_select_working_folder_passes_existing_initial_dir_to_dialog(self):
        with TemporaryDirectory() as initial_dir:
            initial_path = Path(initial_dir)
            seen_initial_dirs = []

            def dialog_opener(passed_initial_dir):
                seen_initial_dirs.append(passed_initial_dir)
                return ""

            select_working_folder(initial_dir=initial_path, dialog_opener=dialog_opener)

            self.assertEqual(seen_initial_dirs, [initial_path.resolve()])

    def test_evaluate_and_improve_gaussian_scene_consistency(self):
        with TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            self._create_test_image(folder / "left.bmp", (255, 64, 64))
            self._create_test_image(folder / "right.bmp", (64, 64, 255))

            scene_path = create_gaussian_scene_file(folder)
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            good_score = evaluate_gaussian_scene_consistency(scene, folder)

            degraded_scene = deepcopy(scene)
            degraded_scene["images"][0]["camera_angles"]["yaw_deg"] = 135.0
            degraded_scene["images"][1]["camera_angles"]["pitch_deg"] = 30.0
            degraded_scene["gaussians"][0]["color"] = [0.0, 1.0, 0.0]
            degraded_scene["gaussians"][0]["opacity"] = 0.1
            degraded_scene["gaussians"][1]["scale"] = [1.5, 1.5, 1.5]
            degraded_score = evaluate_gaussian_scene_consistency(degraded_scene, folder)

            self.assertLess(degraded_score, good_score)

            improved_scene = improve_gaussian_scene_consistency(degraded_scene, folder, step_size=1.0)
            improved_score = evaluate_gaussian_scene_consistency(improved_scene, folder)

            self.assertGreater(improved_score, degraded_score)
            self.assertAlmostEqual(improved_score, good_score, places=4)
            self.assertEqual(improved_scene["images"][0]["camera_angles"]["yaw_deg"], 0.0)
            np.testing.assert_allclose(improved_scene["gaussians"][0]["color"], scene["gaussians"][0]["color"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
