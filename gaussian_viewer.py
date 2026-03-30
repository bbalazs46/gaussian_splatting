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
  Shift          – gyors mozgás (3×)
  ESC            – kilépés
"""

import sys
import math
import numpy as np
import pygame
from dataclasses import dataclass

# ── Ablak és renderelési paraméterek ──────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
FOV_Y_DEG    = 60.0       # függőleges látószög (fok)
NEAR         = 0.1        # közeli vágósík
MOVE_SPEED   = 3.0        # egység / másodperc
MOUSE_SENS   = 0.15       # fok / pixel
BG_COLOR     = (10, 10, 20)


# ── Gauss adatstruktúra ────────────────────────────────────────────────────────
@dataclass
class Gaussian3D:
    position: np.ndarray   # (3,)  világ-koordináta  [x, y, z]
    scale:    np.ndarray   # (3,)  féltengelyek (pozitív)  [sx, sy, sz]
    rotation: np.ndarray   # (4,)  kvaternió  [w, x, y, z]
    color:    np.ndarray   # (3,)  RGB  [0 .. 1]
    opacity:  float        # [0 .. 1]


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
            if event.type == pygame.MOUSEMOTION:
                dmx, dmy = event.rel

        cam.update(pygame.key.get_pressed(), dt, dmx, dmy)

        # ── Renderelés ─────────────────────────────────────────────
        fb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
        projected = project_gaussians(SCENE, cam)
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
            "WASD/Nyilak: mozgás  |  Q/E: le/fel  |  Egér: nézés  |  Shift: gyors  |  ESC: kilépés",
            True, (170, 170, 170),
        )

        # Félig átlátszó sáv a felső szöveg mögé
        bar = pygame.Surface((hud_top.get_width() + 20, hud_top.get_height() + 10), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 140))
        screen.blit(bar, (5, 5))
        screen.blit(hud_top, (15, 10))
        screen.blit(hud_bot, (10, HEIGHT - 22))

        pygame.display.flip()


if __name__ == "__main__":
    main()
