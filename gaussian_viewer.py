#!/usr/bin/env python3
"""
gaussian_viewer.py – Interaktív 3D Gauss-buborék megjelenítő
(Gaussian Splatting prototípus)

Matematikai alap:
  - Minden Gaussian3D objektum tartalmaz 3D pozíciót, skálát (sx,sy,sz),
    kvaterniós elforgatást, RGB színt és opacitást.
  - A 3D kovariancia-mátrix: Σ3D = R S² Rᵀ  (ahol R a kvaternióból kapott
    forgatási mátrix, S = diag(sx,sy,sz))
  - Kamera-vetítés: perspektív projekció a Jacobi-mátrix segítségével
    2D kovariancia-elipszisekbe: Σ2D = J W Σ3D Wᵀ Jᵀ
  - Alpha compositing (Porter-Duff) hátra → előre sorrendben

Vezérlők:
  WASD / Nyilak  – előre/hátra/balra/jobbra mozgás
  Q / E          – lejjebb / feljebb mozgás
  Egér           – nézési irány
  O              – használandó mappa kiválasztása
  C              – jelenet konzisztencia kiértékelése
  I              – jelenet javítása
  +/-            – javítás sebességének állítása
  F              – folyamatos javítás be/ki
  N              – új random Gaussian-folt hozzáadása
  R              – meglévő Gaussian-foltok randomizálása és mentése
  Shift          – gyors mozgás (3×)
  ESC            – kilépés
"""

import copy
import json
import sys
import math
import numpy as np
import pygame
from dataclasses import dataclass
from pathlib import Path

# ── Ablak és renderelési paraméterek ──────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
FOV_Y_DEG    = 60.0       # függőleges látószög (fok)
NEAR         = 0.1        # közeli vágósík
MOVE_SPEED   = 3.0        # egység / másodperc
MOUSE_SENS   = 0.15       # fok / pixel
BG_COLOR     = (10, 10, 20)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_SCENE_FILENAME = "gaussian_scene.json"
GAUSSIAN_CAMERA_RADIUS = 1.5
GAUSSIAN_VERTICAL_POSITION_FACTOR = 0.5
COLOR_ERROR_WEIGHT = 1.8
SCALE_ERROR_WEIGHT = 1.0
OPACITY_ERROR_WEIGHT = 0.8
YAW_ERROR_WEIGHT = 0.4
PITCH_ERROR_WEIGHT = 0.2
ROLL_ERROR_WEIGHT = 0.1
POSITION_ERROR_WEIGHT = 0.8
ROTATION_ERROR_WEIGHT = 0.3
MAX_YAW_DEVIATION_DEG = 180.0
MAX_PITCH_DEVIATION_DEG = 90.0
MAX_ROLL_DEVIATION_DEG = 180.0
INITIAL_POSITION_FACTOR = 0.8
INITIAL_TARGET_SCALE_WEIGHT = 0.35
INITIAL_TARGET_COLOR_WEIGHT = 0.75
INITIAL_TARGET_OPACITY_WEIGHT = 0.5
NEUTRAL_GAUSSIAN_COLOR = np.array([0.5, 0.5, 0.5], dtype=np.float64)
NEUTRAL_GAUSSIAN_SCALE = np.array([0.45, 0.45, 0.25], dtype=np.float64)
NEUTRAL_GAUSSIAN_OPACITY = 0.65
RANDOMIZED_POSITION_RANGE_FACTOR = 2.0
RANDOMIZED_SCALE_MIN = 0.05
RANDOMIZED_SCALE_MAX = 1.25
RANDOMIZED_OPACITY_MIN = 0.1
RANDOMIZED_OPACITY_MAX = 1.0
DEFAULT_IMPROVE_STEP_SIZE = 1.0
IMPROVE_STEP_SIZE_DELTA = 0.1
CONTINUOUS_IMPROVE_INTERVAL_SECONDS = 0.2
MAX_RANDOM_GAUSSIAN_SUFFIX = 10_000


# ── Gauss adatstruktúra ────────────────────────────────────────────────────────
@dataclass
class Gaussian3D:
    position: np.ndarray   # (3,)  világ-koordináta  [x, y, z]
    scale:    np.ndarray   # (3,)  féltengelyek (pozitív)  [sx, sy, sz]
    rotation: np.ndarray   # (4,)  kvaternió  [w, x, y, z]
    color:    np.ndarray   # (3,)  RGB  [0 .. 1]
    opacity:  float        # [0 .. 1]


def open_image_folder(folder_path: str | Path) -> list[Path]:
    """Kinyit egy képmappát, és visszaadja a támogatott képfájlokat."""
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"A mappa nem található: {folder}")

    images = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"A mappában nincs támogatott kép: {folder}")
    return images


def _resolve_existing_directory(folder_path: str | Path | None) -> Path | None:
    """Felold egy opcionális mappaútvonalat, és csak létező könyvtár esetén adja vissza."""
    if not folder_path:
        return None

    resolved = Path(folder_path).expanduser().resolve()
    return resolved if resolved.is_dir() else None


def select_working_folder(
    initial_dir: str | Path | None = None,
    dialog_opener=None,
) -> Path | None:
    """
    Megnyit egy mappaválasztó ablakot, és visszaadja a kijelölt mappát.

    A `dialog_opener` paraméter teszteléshez injektálható.
    """
    resolved_initial_dir = _resolve_existing_directory(initial_dir)
    if dialog_opener is not None:
        selected = dialog_opener(resolved_initial_dir)
        return Path(selected).expanduser().resolve() if selected else None

    try:
        import tkinter as tk
        from tkinter import filedialog
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("A mappaválasztó párbeszédablak nem érhető el.") from exc

    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        selected = filedialog.askdirectory(
            initialdir=str(resolved_initial_dir) if resolved_initial_dir else None,
            title="Válaszd ki a használandó mappát",
        )
    finally:
        try:
            root.destroy()
        except (tk.TclError, AttributeError):
            pass

    return Path(selected).expanduser().resolve() if selected else None


def _load_image_statistics(image_path: str | Path) -> dict:
    """Betölti a képet, és visszaadja a méretét, átlagos színét, fényerejét és kontrasztját."""
    surface = pygame.image.load(str(image_path))
    width, height = surface.get_size()
    pixels = pygame.surfarray.array3d(surface).astype(np.float32)
    mean_color = pixels.mean(axis=(0, 1)) / 255.0
    brightness = float(mean_color.mean())
    contrast = float(pixels.std() / 255.0)
    return {
        "width": int(width),
        "height": int(height),
        "mean_color": mean_color,
        "brightness": brightness,
        "contrast": contrast,
    }


def _expected_camera_yaw(index: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(index * 360.0 / total)


def _target_scale_from_stats(stats: dict) -> np.ndarray:
    max_dim = max(stats["width"], stats["height"], 1)
    width_ratio = stats["width"] / max_dim
    height_ratio = stats["height"] / max_dim
    return np.array([
        0.25 + width_ratio * 0.45,
        0.25 + height_ratio * 0.45,
        0.15 + min(stats["contrast"], 1.0) * 0.35,
    ], dtype=np.float64)


def _target_opacity_from_stats(stats: dict) -> float:
    return float(np.clip(0.35 + stats["brightness"] * 0.65, 0.1, 1.0))


def _wrap_degrees(angle: float) -> float:
    """Egy szöget a [-180, 180) tartományba normalizál."""
    return float(((angle + 180.0) % 360.0) - 180.0)


def _clamp_improve_step_size(step_size: float) -> float:
    """A javítás lépésközét a [0, 1] tartományban tartja."""
    return float(np.clip(step_size, 0.0, 1.0))


def _build_random_gaussian_entry(random_generator: np.random.Generator, source_image: str) -> dict:
    """Létrehoz egy új random Gaussian-bejegyzést."""
    rotation = random_generator.normal(size=4).astype(np.float64)
    rotation_norm = float(np.linalg.norm(rotation))
    if rotation_norm < 1e-10:
        rotation_values = [1.0, 0.0, 0.0, 0.0]
    else:
        rotation_values = (rotation / rotation_norm).tolist()

    return {
        "source_image": source_image,
        "position": random_generator.uniform(
            -GAUSSIAN_CAMERA_RADIUS * RANDOMIZED_POSITION_RANGE_FACTOR,
            GAUSSIAN_CAMERA_RADIUS * RANDOMIZED_POSITION_RANGE_FACTOR,
            size=3,
        ).astype(np.float64).tolist(),
        "scale": random_generator.uniform(
            RANDOMIZED_SCALE_MIN,
            RANDOMIZED_SCALE_MAX,
            size=3,
        ).astype(np.float64).tolist(),
        "rotation": rotation_values,
        "color": random_generator.uniform(0.0, 1.0, size=3).astype(np.float64).tolist(),
        "opacity": float(random_generator.uniform(RANDOMIZED_OPACITY_MIN, RANDOMIZED_OPACITY_MAX)),
    }


def _next_random_gaussian_source_name(scene_data: dict, prefix: str = "random_added") -> str:
    """Egyedi source_image nevet készít az új random Gaussianhoz."""
    image_extension = ".bmp"
    images = scene_data.get("images", [])
    if images:
        image_name = str(images[0].get("file", ""))
        if image_name:
            image_extension = Path(image_name).suffix or image_extension

    used_names = {
        gaussian.get("source_image")
        for gaussian in scene_data.get("gaussians", [])
        if gaussian.get("source_image")
    }
    for suffix in range(1, MAX_RANDOM_GAUSSIAN_SUFFIX):
        candidate = f"{prefix}_{suffix:03d}{image_extension}"
        if candidate not in used_names:
            return candidate
    raise ValueError("Nem sikerült egyedi nevet készíteni az új random Gaussianhoz.")


def _build_gaussian_entry(image_name: str, camera_angles: dict, stats: dict) -> dict:
    """Létrehoz egy Gaussian-spot leírást a képnévhez, kameraálláshoz és képi statisztikákhoz."""
    yaw_rad = math.radians(camera_angles["yaw_deg"])
    pitch_rad = math.radians(camera_angles["pitch_deg"])
    position = np.array([
        math.sin(yaw_rad) * GAUSSIAN_CAMERA_RADIUS,
        math.sin(pitch_rad) * GAUSSIAN_CAMERA_RADIUS * GAUSSIAN_VERTICAL_POSITION_FACTOR,
        math.cos(yaw_rad) * GAUSSIAN_CAMERA_RADIUS,
    ], dtype=np.float64)
    return {
        "source_image": image_name,
        "position": position.tolist(),
        "scale": _target_scale_from_stats(stats).tolist(),
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "color": stats["mean_color"].astype(np.float64).tolist(),
        "opacity": _target_opacity_from_stats(stats),
    }


def _build_initial_gaussian_entry(image_name: str, camera_angles: dict, stats: dict) -> dict:
    """Durva kezdeti Gaussian-becslést készít, amit a javítás közelít a célértékekhez."""
    target_gaussian = _build_gaussian_entry(image_name, camera_angles, stats)
    target_position = np.asarray(target_gaussian["position"], dtype=np.float64)
    target_scale = np.asarray(target_gaussian["scale"], dtype=np.float64)
    target_color = np.asarray(target_gaussian["color"], dtype=np.float64)

    return {
        "source_image": image_name,
        "position": (target_position * INITIAL_POSITION_FACTOR).tolist(),
        "scale": (
            (1.0 - INITIAL_TARGET_SCALE_WEIGHT) * NEUTRAL_GAUSSIAN_SCALE +
            INITIAL_TARGET_SCALE_WEIGHT * target_scale
        ).tolist(),
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "color": (
            INITIAL_TARGET_COLOR_WEIGHT * target_color +
            (1.0 - INITIAL_TARGET_COLOR_WEIGHT) * NEUTRAL_GAUSSIAN_COLOR
        ).tolist(),
        "opacity": float(
            INITIAL_TARGET_OPACITY_WEIGHT * float(target_gaussian["opacity"]) +
            (1.0 - INITIAL_TARGET_OPACITY_WEIGHT) * NEUTRAL_GAUSSIAN_OPACITY
        ),
    }


def _quaternion_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Előjelfüggetlen távolság két kvaternió között."""
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    lhs_norm = np.linalg.norm(lhs)
    rhs_norm = np.linalg.norm(rhs)
    if lhs_norm < 1e-10 or rhs_norm < 1e-10:
        return 1.0

    lhs = lhs / lhs_norm
    rhs = rhs / rhs_norm
    # A q és -q ugyanazt a forgatást reprezentálja, ezért a kisebbik eltérést vesszük.
    return float(min(np.linalg.norm(lhs - rhs), np.linalg.norm(lhs + rhs)) / 2.0)


def build_gaussian_scene_data(folder_path: str | Path) -> dict:
    """Egyszerű jelenetleírást készít a mappa képeihez becsült kameraállásokkal."""
    image_paths = open_image_folder(folder_path)
    scene = {
        "folder": str(Path(folder_path).expanduser().resolve()),
        "images": [],
        "gaussians": [],
    }

    total = len(image_paths)
    for index, image_path in enumerate(image_paths):
        stats = _load_image_statistics(image_path)
        camera_angles = {
            "yaw_deg": _expected_camera_yaw(index, total),
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
        }
        scene["images"].append({
            "file": image_path.name,
            "camera_angles": camera_angles,
        })
        scene["gaussians"].append(_build_initial_gaussian_entry(image_path.name, camera_angles, stats))

    return scene


def create_gaussian_scene_file(folder_path: str | Path, output_filename: str = DEFAULT_SCENE_FILENAME) -> Path:
    """Létrehozza a jelenetfájlt a kiválasztott képmappában."""
    scene = build_gaussian_scene_data(folder_path)
    scene_path = Path(folder_path).expanduser().resolve() / output_filename
    scene_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    return scene_path


def load_gaussian_scene_file(
    folder_path: str | Path | None,
    scene_filename: str = DEFAULT_SCENE_FILENAME,
) -> tuple[Path, dict]:
    """Betölti a kiválasztott mappához tartozó jelenetfájlt."""
    if folder_path is None:
        raise ValueError("Előbb válassz mappát az O billentyűvel.")

    folder = Path(folder_path).expanduser().resolve()
    scene_path = folder / scene_filename
    if not scene_path.exists():
        raise FileNotFoundError(
            f"Nem található jelenetfájl: {scene_path.name}. "
            f"Hozd létre előbb a {scene_filename} fájlt."
        )

    scene_data = json.loads(scene_path.read_text(encoding="utf-8"))
    return scene_path, scene_data


def evaluate_gaussian_scene_consistency(scene_data: dict, folder_path: str | Path | None = None) -> float:
    """
    0..1 közötti pontszámmal becsüli, mennyire vannak összhangban
    a kameraállások és a gauss-foltok a bemeneti képekkel.
    """
    images = list(scene_data.get("images", []))
    if not images:
        return 0.0

    folder = Path(folder_path or scene_data.get("folder", ".")).expanduser().resolve()
    gaussians_by_image = {
        gaussian.get("source_image"): gaussian
        for gaussian in scene_data.get("gaussians", [])
        if gaussian.get("source_image")
    }

    total_score = 0.0
    for index, image_entry in enumerate(images):
        image_name = image_entry["file"]
        gaussian = gaussians_by_image.get(image_name)
        if gaussian is None:
            continue

        stats = _load_image_statistics(folder / image_name)
        target_yaw = _expected_camera_yaw(index, len(images))
        target_scale = _target_scale_from_stats(stats)
        target_opacity = _target_opacity_from_stats(stats)

        camera_angles = image_entry.get("camera_angles", {})
        target_gaussian = _build_gaussian_entry(image_name, camera_angles, stats)
        yaw_error = abs(_wrap_degrees(float(camera_angles.get("yaw_deg", 0.0)) - target_yaw)) / MAX_YAW_DEVIATION_DEG
        pitch_error = abs(float(camera_angles.get("pitch_deg", 0.0))) / MAX_PITCH_DEVIATION_DEG
        roll_error = abs(float(camera_angles.get("roll_deg", 0.0))) / MAX_ROLL_DEVIATION_DEG
        position_error = float(
            np.mean(
                np.abs(
                    np.asarray(gaussian["position"], dtype=np.float64) -
                    np.asarray(target_gaussian["position"], dtype=np.float64)
                )
            ) / max(GAUSSIAN_CAMERA_RADIUS, 1e-6)
        )
        rotation_error = _quaternion_distance(gaussian.get("rotation", [1.0, 0.0, 0.0, 0.0]), target_gaussian["rotation"])
        color_error = float(np.mean(np.abs(np.asarray(gaussian["color"], dtype=np.float64) - stats["mean_color"])))
        scale_error = float(np.mean(np.abs(np.asarray(gaussian["scale"], dtype=np.float64) - target_scale)))
        opacity_error = abs(float(gaussian["opacity"]) - target_opacity)

        penalty = (
            position_error * POSITION_ERROR_WEIGHT +
            rotation_error * ROTATION_ERROR_WEIGHT +
            color_error * COLOR_ERROR_WEIGHT +
            scale_error * SCALE_ERROR_WEIGHT +
            opacity_error * OPACITY_ERROR_WEIGHT +
            yaw_error * YAW_ERROR_WEIGHT +
            pitch_error * PITCH_ERROR_WEIGHT +
            roll_error * ROLL_ERROR_WEIGHT
        )
        total_score += max(0.0, 1.0 - min(penalty, 1.0))

    return total_score / max(len(images), len(scene_data.get("gaussians", [])), 1)


def improve_gaussian_scene_consistency(
    scene_data: dict,
    folder_path: str | Path | None = None,
    step_size: float = 0.5,
) -> dict:
    """A jelenetleírást a képek statisztikái felé tolja, hogy javuljon a pontszám."""
    folder = Path(folder_path or scene_data.get("folder", ".")).expanduser().resolve()
    improved_scene = copy.deepcopy(scene_data)
    images = improved_scene.setdefault("images", [])
    gaussians = improved_scene.setdefault("gaussians", [])
    gaussians_by_image = {
        gaussian.get("source_image"): gaussian
        for gaussian in gaussians
        if gaussian.get("source_image")
    }

    step_size = _clamp_improve_step_size(step_size)
    new_gaussians = []

    for index, image_entry in enumerate(images):
        image_name = image_entry["file"]
        target_yaw = _expected_camera_yaw(index, len(images))
        stats = _load_image_statistics(folder / image_name)
        target_camera = {
            "yaw_deg": target_yaw,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
        }
        camera_angles = image_entry.setdefault("camera_angles", target_camera.copy())
        current_yaw = float(camera_angles.get("yaw_deg", target_yaw))
        camera_angles["yaw_deg"] = float(current_yaw + (target_yaw - current_yaw) * step_size)
        camera_angles["pitch_deg"] = float(camera_angles.get("pitch_deg", 0.0) * (1.0 - step_size))
        camera_angles["roll_deg"] = float(camera_angles.get("roll_deg", 0.0) * (1.0 - step_size))

        target_gaussian = _build_gaussian_entry(image_name, camera_angles, stats)
        gaussian = gaussians_by_image.get(image_name)
        if gaussian is None:
            new_gaussians.append(target_gaussian)
            continue

        gaussian["source_image"] = image_name
        gaussian["position"] = (
            (1.0 - step_size) * np.asarray(gaussian.get("position", target_gaussian["position"]), dtype=np.float64) +
            step_size * np.asarray(target_gaussian["position"], dtype=np.float64)
        ).tolist()
        gaussian["scale"] = (
            (1.0 - step_size) * np.asarray(gaussian.get("scale", target_gaussian["scale"]), dtype=np.float64) +
            step_size * np.asarray(target_gaussian["scale"], dtype=np.float64)
        ).tolist()
        gaussian["color"] = (
            (1.0 - step_size) * np.asarray(gaussian.get("color", target_gaussian["color"]), dtype=np.float64) +
            step_size * np.asarray(target_gaussian["color"], dtype=np.float64)
        ).tolist()
        gaussian["rotation"] = target_gaussian["rotation"]
        gaussian["opacity"] = float(
            (1.0 - step_size) * float(gaussian.get("opacity", target_gaussian["opacity"])) +
            step_size * float(target_gaussian["opacity"])
        )
        new_gaussians.append(gaussian)

    improved_scene["folder"] = str(folder)
    improved_scene["gaussians"] = new_gaussians
    improved_scene["consistency_score"] = evaluate_gaussian_scene_consistency(improved_scene, folder)
    return improved_scene


def randomize_gaussian_scene(
    scene_data: dict,
    folder_path: str | Path | None = None,
    seed: int | None = None,
) -> dict:
    """A meglévő Gaussian-bejegyzéseket random értékekre állítja és újrapontozza a jelenetet."""
    folder = Path(folder_path or scene_data.get("folder", ".")).expanduser().resolve()
    randomized_scene = copy.deepcopy(scene_data)
    gaussians = randomized_scene.setdefault("gaussians", [])
    random_generator = np.random.default_rng(seed)

    for gaussian in gaussians:
        random_gaussian = _build_random_gaussian_entry(
            random_generator,
            gaussian.get("source_image", _next_random_gaussian_source_name(randomized_scene)),
        )
        gaussian["position"] = random_gaussian["position"]
        gaussian["scale"] = random_gaussian["scale"]
        gaussian["rotation"] = random_gaussian["rotation"]
        gaussian["color"] = random_gaussian["color"]
        gaussian["opacity"] = random_gaussian["opacity"]

    randomized_scene["folder"] = str(folder)
    randomized_scene["consistency_score"] = evaluate_gaussian_scene_consistency(randomized_scene, folder)
    return randomized_scene


def append_random_gaussian_scene(
    scene_data: dict,
    folder_path: str | Path | None = None,
    seed: int | None = None,
) -> dict:
    """Hozzáad egy új random Gaussian-bejegyzést, majd újrapontozza a jelenetet."""
    folder = Path(folder_path or scene_data.get("folder", ".")).expanduser().resolve()
    expanded_scene = copy.deepcopy(scene_data)
    gaussians = expanded_scene.setdefault("gaussians", [])
    random_generator = np.random.default_rng(seed)
    gaussians.append(_build_random_gaussian_entry(
        random_generator,
        _next_random_gaussian_source_name(expanded_scene),
    ))
    expanded_scene["folder"] = str(folder)
    expanded_scene["consistency_score"] = evaluate_gaussian_scene_consistency(expanded_scene, folder)
    return expanded_scene


def evaluate_selected_gaussian_scene(
    folder_path: str | Path | None,
    scene_filename: str = DEFAULT_SCENE_FILENAME,
) -> tuple[Path, float]:
    """Betölti a kiválasztott mappához tartozó jelenetet és visszaadja a pontszámát."""
    scene_path, scene_data = load_gaussian_scene_file(folder_path, scene_filename)
    score = evaluate_gaussian_scene_consistency(scene_data, folder_path)
    return scene_path, score


def improve_selected_gaussian_scene(
    folder_path: str | Path | None,
    scene_filename: str = DEFAULT_SCENE_FILENAME,
    step_size: float = DEFAULT_IMPROVE_STEP_SIZE,
) -> tuple[Path, float, float]:
    """Javítja a kiválasztott jelenetet, elmenti, és visszaadja az előtte/utána pontszámot."""
    scene_path, scene_data = load_gaussian_scene_file(folder_path, scene_filename)
    step_size = _clamp_improve_step_size(step_size)
    previous_score = evaluate_gaussian_scene_consistency(scene_data, folder_path)
    improved_scene = improve_gaussian_scene_consistency(scene_data, folder_path, step_size=step_size)
    scene_path.write_text(json.dumps(improved_scene, indent=2), encoding="utf-8")
    improved_score = float(improved_scene.get("consistency_score", previous_score))
    return scene_path, previous_score, improved_score


def randomize_selected_gaussian_scene(
    folder_path: str | Path | None,
    scene_filename: str = DEFAULT_SCENE_FILENAME,
    seed: int | None = None,
) -> tuple[Path, float, float]:
    """Randomizálja a kiválasztott jelenetet, elmenti, és visszaadja az előtte/utána pontszámot."""
    scene_path, scene_data = load_gaussian_scene_file(folder_path, scene_filename)
    previous_score = evaluate_gaussian_scene_consistency(scene_data, folder_path)
    randomized_scene = randomize_gaussian_scene(scene_data, folder_path, seed=seed)
    scene_path.write_text(json.dumps(randomized_scene, indent=2), encoding="utf-8")
    randomized_score = float(randomized_scene.get("consistency_score", previous_score))
    return scene_path, previous_score, randomized_score


def append_random_gaussian_selected_scene(
    folder_path: str | Path | None,
    scene_filename: str = DEFAULT_SCENE_FILENAME,
    seed: int | None = None,
) -> tuple[Path, float, float]:
    """Hozzáad egy új random Gaussian-bejegyzést, elmenti, és visszaadja az előtte/utána pontszámot."""
    scene_path, scene_data = load_gaussian_scene_file(folder_path, scene_filename)
    previous_score = evaluate_gaussian_scene_consistency(scene_data, folder_path)
    expanded_scene = append_random_gaussian_scene(scene_data, folder_path, seed=seed)
    scene_path.write_text(json.dumps(expanded_scene, indent=2), encoding="utf-8")
    expanded_score = float(expanded_scene.get("consistency_score", previous_score))
    return scene_path, previous_score, expanded_score


def _scene_data_to_gaussians(scene_data: dict) -> list[Gaussian3D]:
    """A jelenetfájl Gaussian-bejegyzéseit renderelhető Gaussian3D objektumokká alakítja."""
    gaussians = []
    for gaussian in scene_data.get("gaussians", []):
        position = np.asarray(gaussian["position"], dtype=np.float64)
        scale = np.asarray(gaussian["scale"], dtype=np.float64)
        rotation = np.asarray(gaussian["rotation"], dtype=np.float64)
        color = np.asarray(gaussian["color"], dtype=np.float64)

        if position.shape != (3,) or scale.shape != (3,) or rotation.shape != (4,) or color.shape != (3,):
            raise ValueError("A gaussian_scene.json egyik Gaussian-bejegyzése hibás alakú.")

        gaussians.append(Gaussian3D(
            position=position,
            scale=np.clip(scale, 1e-3, None),
            rotation=rotation,
            color=np.clip(color, 0.0, 1.0),
            opacity=float(np.clip(float(gaussian["opacity"]), 0.0, 1.0)),
        ))

    return gaussians


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Egységkvaternióból 3×3 forgatási mátrix."""
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def build_sigma3d(scale: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """3D kovariancia-mátrix: Σ = R S² Rᵀ"""
    R = quat_to_rotmat(rotation)
    S2 = np.diag(scale ** 2)
    return R @ S2 @ R.T


# ── Kamera ────────────────────────────────────────────────────────────────────
class Camera:
    def __init__(self, pos=(0.0, 0.0, 4.0), yaw=0.0, pitch=0.0):
        self.pos   = np.array(pos, dtype=np.float64)
        self.yaw   = float(yaw)    # vízszintes elforgatás (fok)
        self.pitch = float(pitch)  # függőleges elforgatás (fok)

    @property
    def forward(self) -> np.ndarray:
        """Egységvektor arra, amerre a kamera néz (világ-koordináta)."""
        yr = math.radians(self.yaw)
        pr = math.radians(self.pitch)
        return np.array([
            math.sin(yr) * math.cos(pr),
             math.sin(pr),
            -math.cos(yr) * math.cos(pr),
        ])

    @property
    def right(self) -> np.ndarray:
        """Jobbra mutató egységvektor (vízszintes, nincs pitch-hatás)."""
        yr = math.radians(self.yaw)
        return np.array([math.cos(yr), 0.0, math.sin(yr)])

    def rot_matrix(self) -> np.ndarray:
        """3×3 kamera-forgatási mátrix (world → cam). Sorok: right, up, forward."""
        f = self.forward
        r = self.right
        u = np.cross(r, f)
        u /= np.linalg.norm(u) + 1e-12
        return np.array([r, u, f])   # W: p_cam = W @ (p_world - pos)

    def update(self, keys, dt: float, dmx: float, dmy: float):
        self.yaw   += dmx * MOUSE_SENS
        self.pitch  = float(np.clip(self.pitch - dmy * MOUSE_SENS, -89.0, 89.0))

        if keys[pygame.K_3] and (dmx != 0.0 or dmy != 0.0):
            radius = np.linalg.norm(self.pos)
            if radius > 0.0:
                self.pos = -self.forward * radius

        speed = MOVE_SPEED * (3.0 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 1.0)
        f, r = self.forward, self.right

        if keys[pygame.K_w] or keys[pygame.K_UP]:    self.pos += f * speed * dt
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:  self.pos -= f * speed * dt
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  self.pos -= r * speed * dt
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: self.pos += r * speed * dt
        if keys[pygame.K_e]:  self.pos[1] += speed * dt
        if keys[pygame.K_q]:  self.pos[1] -= speed * dt


# ── Projekció ─────────────────────────────────────────────────────────────────
def project_gaussians(gaussians: list[Gaussian3D], cam: Camera) -> list[dict]:
    """
    Minden Gaussian3D-t levetít 2D képernyő-koordinátákra.
    Visszatér mélység szerint növekvő sorrendben rendezett listával
    (front-to-back alpha compositing-hoz).
    """
    # Fókusztávolság pixelben  (négyzetpixel-feltétel: fx == fy)
    f_px = (HEIGHT / 2.0) / math.tan(math.radians(FOV_Y_DEG / 2.0))
    cx, cy = WIDTH / 2.0, HEIGHT / 2.0

    W  = cam.rot_matrix()   # 3×3: sorok = right, up, forward
    t  = -W @ cam.pos       # transzláció kamera-térben

    result = []
    for g in gaussians:
        # Kamera-koordináta:
        #   p_cam[0] = dot(right,   p - pos)  → jobbra pozitív
        #   p_cam[1] = dot(up,      p - pos)  → felfelé pozitív
        #   p_cam[2] = dot(forward, p - pos)  → mélység (pozitív = előttünk)
        p_cam = W @ g.position + t
        z = p_cam[2]
        if z <= NEAR:
            continue

        x_cam, y_cam = p_cam[0], p_cam[1]

        # Perspektív vetítés
        px =  f_px * x_cam / z + cx
        py = -f_px * y_cam / z + cy    # Y negáció: világ-up → képernyő-fent

        # 3D → 2D kovariancia:  Σ2D = J W Σ3D Wᵀ Jᵀ
        sigma3d = build_sigma3d(g.scale, g.rotation)

        # Jacobi-mátrix: ∂(px,py)/∂(x_cam, y_cam, z_cam)
        J = np.array([
            [ f_px / z,       0.0,  -f_px * x_cam / (z * z)],
            [      0.0,  -f_px / z,  f_px * y_cam / (z * z)],
        ])
        sigma2d = J @ W @ sigma3d @ W.T @ J.T
        sigma2d += np.eye(2) * 0.3     # regularizáció a szinguláris eset ellen

        try:
            sigma2d_inv = np.linalg.inv(sigma2d)
        except np.linalg.LinAlgError:
            continue

        # Bounding box a Gauss-buborék körül (3σ)
        eigvals = np.linalg.eigvalsh(sigma2d)
        max_std = math.sqrt(max(float(eigvals[-1]), 0.0)) * 3.0

        result.append({
            "depth":    z,
            "px":       px,
            "py":       py,
            "sig_inv":  sigma2d_inv,
            "max_std":  max_std,
            "color":    g.color,
            "opacity":  g.opacity,
        })

    # Előre → hátra rendezés (transzmisszió-alapú compositing)
    result.sort(key=lambda s: s["depth"])
    return result


# ── Renderelő ─────────────────────────────────────────────────────────────────
def render(projected: list[dict], fb: np.ndarray) -> None:
    """
    Alpha compositing (Porter-Duff, front-to-back) a float32 framebuffer-re.
      fb: (H, W, 3) float32 – függvénybe lépéskor nulla (fekete háttér)
      T:  (H, W)   float32 – fennmaradó átlátszatlanság (1 = teljesen átlátszó)
    """
    T    = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    xs_a = np.arange(WIDTH,  dtype=np.float32)
    ys_a = np.arange(HEIGHT, dtype=np.float32)

    for s in projected:
        px, py   = s["px"], s["py"]
        ms       = s["max_std"]
        inv      = s["sig_inv"]
        color    = s["color"]
        opacity  = s["opacity"]

        # AABB a buborék körül (vágva a képernyő határaira)
        x0 = max(0,      int(px - ms))
        x1 = min(WIDTH,  int(px + ms) + 1)
        y0 = max(0,      int(py - ms))
        y1 = min(HEIGHT, int(py + ms) + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        # Eltolásrács a patch-en belül
        dx = xs_a[x0:x1][np.newaxis, :] - px   # (1, w)
        dy = ys_a[y0:y1][:, np.newaxis] - py   # (h, 1)

        # Mahalanobis-távolság négyzete:  dᵀ Σ⁻¹ d
        a00, a01, a11 = inv[0, 0], inv[0, 1], inv[1, 1]
        maha2 = a00 * dx*dx + 2.0*a01 * dx*dy + a11 * dy*dy

        gauss  = np.exp(-0.5 * maha2)          # Gauss-súly  (h, w)
        alpha  = opacity * gauss               # effektív alpha  (h, w)

        # C_out += T * alpha * color  (Porter-Duff over, front-to-back)
        T_crop = T[y0:y1, x0:x1]
        contrib = T_crop * alpha
        fb[y0:y1, x0:x1, 0] += contrib * color[0]
        fb[y0:y1, x0:x1, 1] += contrib * color[1]
        fb[y0:y1, x0:x1, 2] += contrib * color[2]

        T[y0:y1, x0:x1] *= (1.0 - alpha)      # átlátszatlanság csökkentése


# ── Hard-coded jelenet ─────────────────────────────────────────────────────────
SCENE: list[Gaussian3D] = [
    Gaussian3D(                          # piros gömb (középen)
        position = np.array([ 0.0,  0.0,  0.0]),
        scale    = np.array([0.50, 0.50, 0.50]),
        rotation = np.array([1.0, 0.0, 0.0, 0.0]),
        color    = np.array([1.0, 0.2, 0.2]),
        opacity  = 0.90,
    ),
    Gaussian3D(                          # zöld, megnyúlt Y-ban
        position = np.array([ 1.5,  0.0,  0.0]),
        scale    = np.array([0.28, 0.75, 0.28]),
        rotation = np.array([1.0, 0.0, 0.0, 0.0]),
        color    = np.array([0.2, 1.0, 0.2]),
        opacity  = 0.85,
    ),
    Gaussian3D(                          # kék, elforgatott (~45° X-tengely körül)
        position = np.array([-1.5,  0.0,  0.0]),
        scale    = np.array([0.35, 0.18, 0.72]),
        rotation = np.array([0.924, 0.383, 0.0, 0.0]),
        color    = np.array([0.2, 0.45, 1.0]),
        opacity  = 0.90,
    ),
    Gaussian3D(                          # sárga, lapos korong (felül)
        position = np.array([ 0.0,  1.3,  0.0]),
        scale    = np.array([0.85, 0.12, 0.45]),
        rotation = np.array([1.0, 0.0, 0.0, 0.0]),
        color    = np.array([1.0, 0.90, 0.1]),
        opacity  = 0.85,
    ),
    Gaussian3D(                          # lila, megnyúlt Z-ban (alul)
        position = np.array([ 0.0, -1.3,  0.0]),
        scale    = np.array([0.22, 0.22, 0.80]),
        rotation = np.array([0.707, 0.0, 0.707, 0.0]),
        color    = np.array([0.90, 0.3, 0.90]),
        opacity  = 0.75,
    ),
    Gaussian3D(                          # cián kis gömb (jobb-felső)
        position = np.array([ 0.9,  0.9, -0.4]),
        scale    = np.array([0.18, 0.18, 0.18]),
        rotation = np.array([1.0, 0.0, 0.0, 0.0]),
        color    = np.array([0.1, 0.95, 0.95]),
        opacity  = 0.95,
    ),
    Gaussian3D(                          # narancs, elforgatott (bal-alsó, kicsit hátul)
        position = np.array([-0.8, -0.6,  0.9]),
        scale    = np.array([0.38, 0.52, 0.18]),
        rotation = np.array([0.866, 0.0, 0.5, 0.0]),
        color    = np.array([1.0, 0.52, 0.1]),
        opacity  = 0.88,
    ),
    Gaussian3D(                          # szürke nagy lapos Gauss (háttér-sík)
        position = np.array([ 0.0,  0.0, -2.2]),
        scale    = np.array([1.60, 1.60, 0.08]),
        rotation = np.array([1.0, 0.0, 0.0, 0.0]),
        color    = np.array([0.75, 0.75, 0.78]),
        opacity  = 0.28,
    ),
]


# ── Főprogram ──────────────────────────────────────────────────────────────────
def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Gaussian Splatting – prototípus")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 17, bold=True)
    font_s = pygame.font.SysFont("monospace", 14)

    # Egér befogása (FPS-stílusú nézés)
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    cam = Camera(pos=[0.0, 0.0, 4.5])
    selected_folder: Path | None = None
    active_scene = list(SCENE)
    folder_status = "Nincs kijelölt mappa"
    improve_step_size = DEFAULT_IMPROVE_STEP_SIZE
    continuous_improve_enabled = False
    continuous_improve_elapsed = 0.0

    while True:
        dt   = clock.tick(60) / 1000.0
        dmx  = 0.0
        dmy  = 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_o:
                try:
                    chosen_folder = select_working_folder(selected_folder)
                    if chosen_folder is None:
                        folder_status = "Mappaválasztás megszakítva"
                    else:
                        image_count = len(open_image_folder(chosen_folder))
                        continuous_improve_enabled = False
                        continuous_improve_elapsed = 0.0
                        selected_folder = chosen_folder
                        try:
                            _, scene_data = load_gaussian_scene_file(selected_folder)
                            active_scene = _scene_data_to_gaussians(scene_data)
                            folder_status = (
                                f"Kijelölt mappa: {selected_folder.name} "
                                f"({image_count} kép, {len(active_scene)} Gaussian)"
                            )
                        except FileNotFoundError:
                            active_scene = list(SCENE)
                            folder_status = (
                                f"Kijelölt mappa: {selected_folder.name} "
                                f"({image_count} kép, nincs {DEFAULT_SCENE_FILENAME})"
                            )
                except (RuntimeError, FileNotFoundError, ValueError) as exc:
                    folder_status = str(exc)
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                improve_step_size = _clamp_improve_step_size(improve_step_size - IMPROVE_STEP_SIZE_DELTA)
                folder_status = f"Javítási sebesség: {improve_step_size:.2f}"
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                improve_step_size = _clamp_improve_step_size(improve_step_size + IMPROVE_STEP_SIZE_DELTA)
                folder_status = f"Javítási sebesség: {improve_step_size:.2f}"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                continuous_improve_enabled = not continuous_improve_enabled
                continuous_improve_elapsed = 0.0
                folder_status = (
                    f"Folyamatos javítás: {'BE' if continuous_improve_enabled else 'KI'} "
                    f"(sebesség: {improve_step_size:.2f})"
                )
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                try:
                    scene_path, scene_data = load_gaussian_scene_file(selected_folder)
                    active_scene = _scene_data_to_gaussians(scene_data)
                    score = evaluate_gaussian_scene_consistency(scene_data, selected_folder)
                    folder_status = f"Konzisztencia: {score:.3f} ({scene_path.name})"
                except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                    folder_status = str(exc)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                try:
                    scene_path, previous_score, improved_score = improve_selected_gaussian_scene(
                        selected_folder,
                        step_size=improve_step_size,
                    )
                    _, improved_scene_data = load_gaussian_scene_file(selected_folder)
                    active_scene = _scene_data_to_gaussians(improved_scene_data)
                    folder_status = (
                        f"Javítás ({improve_step_size:.2f}): {previous_score:.3f} → {improved_score:.3f} "
                        f"({scene_path.name})"
                    )
                except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                    folder_status = str(exc)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                try:
                    scene_path, previous_score, randomized_score = randomize_selected_gaussian_scene(selected_folder)
                    _, randomized_scene_data = load_gaussian_scene_file(selected_folder)
                    active_scene = _scene_data_to_gaussians(randomized_scene_data)
                    folder_status = (
                        f"Randomizálás: {previous_score:.3f} → {randomized_score:.3f} "
                        f"({scene_path.name})"
                    )
                except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                    folder_status = str(exc)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                try:
                    scene_path, previous_score, expanded_score = append_random_gaussian_selected_scene(selected_folder)
                    _, expanded_scene_data = load_gaussian_scene_file(selected_folder)
                    active_scene = _scene_data_to_gaussians(expanded_scene_data)
                    folder_status = (
                        f"Új random Gaussian: {previous_score:.3f} → {expanded_score:.3f} "
                        f"({scene_path.name}, {len(expanded_scene_data.get('gaussians', []))} Gaussian)"
                    )
                except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                    folder_status = str(exc)
            if event.type == pygame.MOUSEMOTION:
                dmx, dmy = event.rel

        cam.update(pygame.key.get_pressed(), dt, dmx, dmy)
        if continuous_improve_enabled:
            continuous_improve_elapsed += dt
            if continuous_improve_elapsed >= CONTINUOUS_IMPROVE_INTERVAL_SECONDS:
                continuous_improve_elapsed = 0.0
                try:
                    scene_path, previous_score, improved_score = improve_selected_gaussian_scene(
                        selected_folder,
                        step_size=improve_step_size,
                    )
                    _, improved_scene_data = load_gaussian_scene_file(selected_folder)
                    active_scene = _scene_data_to_gaussians(improved_scene_data)
                    folder_status = (
                        f"Folyamatos javítás ({improve_step_size:.2f}): "
                        f"{previous_score:.3f} → {improved_score:.3f} ({scene_path.name})"
                    )
                except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                    folder_status = str(exc)
                    continuous_improve_enabled = False

        # ── Renderelés ─────────────────────────────────────────────
        fb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
        projected = project_gaussians(active_scene, cam)
        render(projected, fb)

        # Float32 → uint8 → pygame Surface
        img  = (np.clip(fb, 0.0, 1.0) * 255.0).astype(np.uint8)
        surf = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
        screen.blit(surf, (0, 0))

        # ── HUD ────────────────────────────────────────────────────
        fps = clock.get_fps()
        hud_top = font.render(
            f"FPS: {fps:5.1f}   "
            f"X:{cam.pos[0]:+.2f}  Y:{cam.pos[1]:+.2f}  Z:{cam.pos[2]:+.2f}   "
            f"Yaw:{cam.yaw % 360:.1f}°  Pitch:{cam.pitch:.1f}°",
            True, (255, 240, 80),
        )
        hud_bot = font_s.render(
            (
                "WASD/Nyilak  |  Q/E: le/fel  |  Egér  |  Shift  |  O: mappa  |  "
                f"C: konsziszt.  |  I: javít  |  +/-: seb. {improve_step_size:.2f}  |  "
                f"F: {'BE' if continuous_improve_enabled else 'KI'}  |  R: random  |  N: új random"
            ),
            True, (170, 170, 170),
        )
        hud_folder = font_s.render(folder_status, True, (150, 220, 255))

        # Félig átlátszó sáv a felső szöveg mögé
        bar = pygame.Surface((hud_top.get_width() + 20, hud_top.get_height() + 10), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 140))
        screen.blit(bar, (5, 5))
        screen.blit(hud_top, (15, 10))
        screen.blit(hud_bot, (10, HEIGHT - 42))
        screen.blit(hud_folder, (10, HEIGHT - 22))

        pygame.display.flip()


if __name__ == "__main__":
    main()
