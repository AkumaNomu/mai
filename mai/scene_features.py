"""Computer-vision scene analysis: an image's *feel* as a mood vector.

Given a still frame (a movie scene, a photo, cover art) this module reads the
colour and light of the image and projects it into the same compact mood space
the audio side already uses — ``valence`` / ``arousal`` / ``tension`` /
``warmth`` — so a scene and a song can be compared directly.

The always-on path needs only Pillow + numpy + scikit-learn (a dominant-colour
``KMeans`` palette plus global colour statistics). It is deterministic and
deps-light. A learned vision-language seam (CLIP) is added in
``scene_context``; this module is the colour-grounded baseline.

Colour → mood mapping (perceptual heuristics, documented per axis):

* **warmth**  — share of warm hue (reds/oranges/yellows) and saturation; cool,
  desaturated frames read cold.
* **valence** — brightness + saturation + warm hue; bright, vivid, warm frames
  feel positive, dark/desaturated ones negative.
* **arousal** — saturation + contrast + colourfulness + brightness; vivid,
  high-contrast frames feel energetic.
* **tension** — contrast + darkness + coolness; dark, harsh, cold frames feel
  tense, calm pastel frames do not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

# Mood axes shared with the audio side (mai.sentiment.SENTIMENT_DIMS, minus the
# ``sentiment_`` prefix). Kept local to avoid a hard import cycle.
SCENE_MOOD_DIMS = ('valence', 'arousal', 'tension', 'warmth')

# Hue bands in degrees [0, 360). Warm = reds/oranges/yellows; cool = cyans/blues.
_WARM_HUE_BANDS = ((0.0, 60.0), (330.0, 360.0))
_COOL_HUE_BANDS = ((180.0, 270.0),)

# Pixels are downsampled to this long edge before analysis; plenty for colour
# statistics and keeps KMeans cheap on 4K frames.
_ANALYSIS_LONG_EDGE = 160
_PALETTE_SIZE = 5


@dataclass(slots=True)
class SceneFeatures:
    """Colour/light descriptors of an image plus its derived mood vector."""

    mood: dict[str, float]                      # valence/arousal/tension/warmth in [0, 1]
    palette: list[tuple[int, int, int]]         # dominant RGB colours, most-common first
    palette_weights: list[float]                # fraction of pixels per palette colour
    brightness: float = 0.0
    saturation: float = 0.0
    contrast: float = 0.0
    colorfulness: float = 0.0
    warm_share: float = 0.0
    cool_share: float = 0.0
    dark_share: float = 0.0
    source: str = 'color'
    extra: dict[str, Any] = field(default_factory=dict)

    def mood_vector(self) -> np.ndarray:
        return np.array([float(self.mood.get(dim, 0.5)) for dim in SCENE_MOOD_DIMS], dtype=np.float64)


def _import_pillow():
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            'Scene image analysis requires Pillow. Install it with `pip install pillow`.'
        ) from exc
    return Image


def _load_rgb_array(image_path: str) -> np.ndarray:
    Image = _import_pillow()
    with Image.open(image_path) as handle:
        rgb = handle.convert('RGB')
        long_edge = max(rgb.size)
        if long_edge > _ANALYSIS_LONG_EDGE:
            scale = _ANALYSIS_LONG_EDGE / float(long_edge)
            new_size = (max(1, round(rgb.size[0] * scale)), max(1, round(rgb.size[1] * scale)))
            rgb = rgb.resize(new_size, Image.BILINEAR)
        return np.asarray(rgb, dtype=np.float64) / 255.0


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB->HSV. ``rgb`` is (..., 3) in [0, 1]. Hue returned in degrees."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    hue = np.zeros_like(max_c)
    safe = delta > 1e-9
    # Channel-wise hue, guarding the achromatic (delta==0) pixels.
    rc = np.where(safe & (max_c == r), ((g - b) / np.where(safe, delta, 1.0)) % 6.0, 0.0)
    gc = np.where(safe & (max_c == g), ((b - r) / np.where(safe, delta, 1.0)) + 2.0, 0.0)
    bc = np.where(safe & (max_c == b), ((r - g) / np.where(safe, delta, 1.0)) + 4.0, 0.0)
    hue = (rc + gc + bc) * 60.0
    hue = np.where(hue < 0.0, hue + 360.0, hue)

    saturation = np.where(max_c > 1e-9, delta / np.where(max_c > 1e-9, max_c, 1.0), 0.0)
    value = max_c
    return hue, saturation, value


def _hue_band_share(hue: np.ndarray, weights: np.ndarray, bands) -> float:
    mask = np.zeros(hue.shape, dtype=bool)
    for lo, hi in bands:
        mask |= (hue >= lo) & (hue < hi)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(weights[mask].sum() / total)


def _colorfulness(rgb_flat: np.ndarray) -> float:
    """Hasler-Süsstrunk colourfulness metric, normalised to roughly [0, 1]."""
    r, g, b = rgb_flat[:, 0], rgb_flat[:, 1], rgb_flat[:, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.sqrt(rg.var() + yb.var())
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    metric = std + 0.3 * mean  # rg/yb are in [-1, 1]; metric ~ [0, ~0.8]
    return float(np.clip(metric / 0.6, 0.0, 1.0))


def _palette(rgb_flat: np.ndarray, k: int) -> tuple[list[tuple[int, int, int]], list[float]]:
    from sklearn.cluster import KMeans

    sample = rgb_flat
    if len(sample) > 4000:
        # Deterministic stride sample keeps KMeans fast without an RNG dependency.
        sample = rgb_flat[:: max(1, len(rgb_flat) // 4000)]
    k = int(max(1, min(k, len(sample))))
    model = KMeans(n_clusters=k, n_init=4, random_state=0)
    labels = model.fit_predict(sample)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    order = np.argsort(-counts)
    weights = counts[order] / max(counts.sum(), 1.0)
    centers = np.clip(model.cluster_centers_[order] * 255.0, 0, 255).astype(int)
    palette = [tuple(int(c) for c in center) for center in centers]
    return palette, [float(w) for w in weights]


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _mood_from_stats(
    *, brightness: float, saturation: float, contrast: float, colorfulness: float,
    warm_share: float, cool_share: float, dark_share: float,
) -> dict[str, float]:
    warmth = _clip01(0.60 * warm_share + 0.25 * saturation + 0.15 * (1.0 - cool_share))
    valence = _clip01(
        0.40 * brightness + 0.22 * saturation + 0.20 * warm_share + 0.18 * (1.0 - dark_share)
    )
    arousal = _clip01(
        0.38 * saturation + 0.26 * contrast + 0.20 * colorfulness + 0.16 * brightness
    )
    tension = _clip01(
        0.40 * contrast + 0.26 * dark_share + 0.20 * cool_share + 0.14 * (1.0 - brightness)
    )
    return {'valence': valence, 'arousal': arousal, 'tension': tension, 'warmth': warmth}


def analyze_scene_image(image_path: str) -> SceneFeatures:
    """Read an image file and return its colour descriptors and mood vector."""
    rgb = _load_rgb_array(image_path)
    rgb_flat = rgb.reshape(-1, 3)
    if len(rgb_flat) == 0:
        raise ValueError(f'scene image has no pixels: {image_path!r}')

    hue, saturation_map, value_map = _rgb_to_hsv(rgb)
    hue_flat = hue.reshape(-1)
    # Weight hue shares by saturation*value so washed-out / black pixels do not
    # vote for a colour temperature they barely express.
    hue_weights = (saturation_map.reshape(-1) * value_map.reshape(-1)) + 1e-6

    luminance = (0.2126 * rgb_flat[:, 0] + 0.7152 * rgb_flat[:, 1] + 0.0722 * rgb_flat[:, 2])
    brightness = float(np.mean(value_map))
    saturation = float(np.mean(saturation_map))
    contrast = float(np.clip(np.std(luminance) / 0.30, 0.0, 1.0))  # 0.30 ~ very high-contrast frame
    colorfulness = _colorfulness(rgb_flat)
    warm_share = _hue_band_share(hue_flat, hue_weights, _WARM_HUE_BANDS)
    cool_share = _hue_band_share(hue_flat, hue_weights, _COOL_HUE_BANDS)
    dark_share = float(np.mean(luminance < 0.25))

    mood = _mood_from_stats(
        brightness=brightness, saturation=saturation, contrast=contrast,
        colorfulness=colorfulness, warm_share=warm_share, cool_share=cool_share,
        dark_share=dark_share,
    )

    try:
        palette, palette_weights = _palette(rgb_flat, _PALETTE_SIZE)
    except Exception:
        logger.exception('Palette extraction failed; reporting mean colour only.')
        mean_rgb = tuple(int(c) for c in np.clip(rgb_flat.mean(axis=0) * 255.0, 0, 255))
        palette, palette_weights = [mean_rgb], [1.0]

    return SceneFeatures(
        mood=mood, palette=palette, palette_weights=palette_weights,
        brightness=brightness, saturation=saturation, contrast=contrast,
        colorfulness=colorfulness, warm_share=warm_share, cool_share=cool_share,
        dark_share=dark_share, source='color',
    )
