"""NLP scene context: a scene description's *feel* as a mood vector + genre hints.

Colour tells you the light of a frame; words tell you what is happening in it. A
red frame could be a sunset romance or a burning battlefield — the caption
disambiguates. This module turns a free-text scene description (a caption, a
synopsis line, hand tags) into the same ``valence / arousal / tension / warmth``
mood space the image and audio sides use, plus soft genre hints.

The always-on path is a small, curated emotion lexicon: matched cue words pull
the scene toward documented mood anchors and vote for fitting genres. It needs no
model and no network. When ``transformers`` + ``torch`` are installed and
``MAI_CLIP_MODEL`` names a checkpoint, :func:`clip_scene_alignment` is the seam
for grounding the *image itself* against mood/genre prompts in a learned joint
space; the lexicon stays the baseline.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

import numpy as np


logger = logging.getLogger(__name__)

SCENE_MOOD_DIMS = ('valence', 'arousal', 'tension', 'warmth')

# Each cue maps to a mood anchor (valence, arousal, tension, warmth) in [0, 1]
# and optional genre votes. Multiple matched cues are averaged by weight. Anchors
# are intentionally broad emotional archetypes, not per-film labels.
_LEXICON: dict[str, dict] = {
    # high-energy conflict
    'battle':   {'mood': (0.45, 0.95, 0.85, 0.40), 'genres': ('epic', 'orchestral', 'metal', 'electronic')},
    'fight':    {'mood': (0.45, 0.92, 0.82, 0.40), 'genres': ('metal', 'rock', 'electronic')},
    'chase':    {'mood': (0.50, 0.95, 0.78, 0.45), 'genres': ('electronic', 'drum and bass', 'rock')},
    'war':      {'mood': (0.30, 0.88, 0.88, 0.35), 'genres': ('epic', 'orchestral')},
    'explosion':{'mood': (0.45, 0.97, 0.80, 0.50), 'genres': ('electronic', 'metal')},
    'action':   {'mood': (0.55, 0.90, 0.65, 0.45), 'genres': ('electronic', 'rock')},
    # triumph / uplift
    'victory':  {'mood': (0.92, 0.85, 0.30, 0.70), 'genres': ('epic', 'orchestral', 'pop')},
    'triumph':  {'mood': (0.92, 0.85, 0.30, 0.70), 'genres': ('epic', 'orchestral')},
    'celebrat': {'mood': (0.90, 0.80, 0.20, 0.75), 'genres': ('pop', 'funk', 'dance')},
    'hope':     {'mood': (0.80, 0.55, 0.25, 0.70), 'genres': ('ambient', 'orchestral', 'indie')},
    # calm / serene
    'calm':     {'mood': (0.70, 0.20, 0.15, 0.65), 'genres': ('ambient', 'acoustic', 'lo-fi')},
    'peaceful': {'mood': (0.75, 0.18, 0.12, 0.70), 'genres': ('ambient', 'acoustic')},
    'serene':   {'mood': (0.75, 0.20, 0.12, 0.68), 'genres': ('ambient', 'classical')},
    'nature':   {'mood': (0.72, 0.30, 0.18, 0.62), 'genres': ('ambient', 'folk', 'acoustic')},
    'dawn':     {'mood': (0.72, 0.35, 0.20, 0.68), 'genres': ('ambient', 'indie')},
    'sunset':   {'mood': (0.70, 0.35, 0.22, 0.78), 'genres': ('lo-fi', 'indie', 'soul')},
    # romance / warmth
    'romanc':   {'mood': (0.82, 0.45, 0.25, 0.85), 'genres': ('soul', 'r&b', 'acoustic', 'pop')},
    'love':     {'mood': (0.82, 0.45, 0.25, 0.85), 'genres': ('soul', 'r&b', 'pop')},
    'intimate': {'mood': (0.70, 0.35, 0.30, 0.80), 'genres': ('soul', 'jazz', 'acoustic')},
    'tender':   {'mood': (0.78, 0.30, 0.20, 0.82), 'genres': ('acoustic', 'soul')},
    # sorrow
    'sad':      {'mood': (0.20, 0.30, 0.45, 0.40), 'genres': ('acoustic', 'classical', 'indie')},
    'grief':    {'mood': (0.12, 0.30, 0.55, 0.35), 'genres': ('classical', 'ambient')},
    'funeral':  {'mood': (0.12, 0.25, 0.55, 0.30), 'genres': ('classical', 'ambient')},
    'lonely':   {'mood': (0.25, 0.28, 0.45, 0.38), 'genres': ('lo-fi', 'indie', 'ambient')},
    'melanchol':{'mood': (0.28, 0.30, 0.42, 0.45), 'genres': ('indie', 'classical', 'lo-fi')},
    'rain':     {'mood': (0.35, 0.30, 0.40, 0.40), 'genres': ('lo-fi', 'jazz', 'ambient')},
    # fear / dread
    'horror':   {'mood': (0.12, 0.70, 0.95, 0.20), 'genres': ('industrial', 'ambient', 'electronic')},
    'fear':     {'mood': (0.18, 0.72, 0.90, 0.25), 'genres': ('industrial', 'ambient')},
    'dark':     {'mood': (0.25, 0.55, 0.75, 0.30), 'genres': ('industrial', 'electronic', 'ambient')},
    'eerie':    {'mood': (0.22, 0.50, 0.85, 0.25), 'genres': ('ambient', 'industrial')},
    'tense':    {'mood': (0.30, 0.65, 0.85, 0.35), 'genres': ('electronic', 'orchestral')},
    'suspense': {'mood': (0.32, 0.62, 0.85, 0.35), 'genres': ('orchestral', 'electronic')},
    'mystery':  {'mood': (0.35, 0.45, 0.65, 0.40), 'genres': ('ambient', 'jazz', 'electronic')},
    # urban / nocturnal
    'city':     {'mood': (0.55, 0.60, 0.45, 0.45), 'genres': ('electronic', 'hip hop', 'synthwave')},
    'night':    {'mood': (0.45, 0.45, 0.50, 0.40), 'genres': ('synthwave', 'lo-fi', 'electronic')},
    'neon':     {'mood': (0.58, 0.70, 0.45, 0.50), 'genres': ('synthwave', 'electronic')},
    'party':    {'mood': (0.88, 0.85, 0.25, 0.65), 'genres': ('dance', 'pop', 'house')},
    'epic':     {'mood': (0.70, 0.85, 0.50, 0.55), 'genres': ('epic', 'orchestral')},
}

_TOKEN_RE = re.compile(r"[a-z]+")

# Cues that are truncated stems matched by prefix (catch inflections a fixed
# suffix set would miss: romance/romantic, celebrate/celebration, ...).
_STEM_CUES = frozenset({'romanc', 'celebrat', 'melanchol'})
# Common inflections appended to a full-word cue, so 'battles'/'exploded' match
# 'battle'/'explosion'-style words without a stemmer — and without the substring
# trap where 'war' would match 'warm' / 'toward'.
_INFLECTIONS = ('', 's', 'es', 'ed', 'd', 'ing')


def _cue_matches(cue: str, tokens: list[str]) -> bool:
    if cue in _STEM_CUES:
        return any(token.startswith(cue) for token in tokens)
    variants = {cue + suffix for suffix in _INFLECTIONS}
    return any(token in variants for token in tokens)


@dataclass(slots=True)
class SceneContext:
    """Mood vector and genre hints derived from a scene description."""

    mood: dict[str, float] | None             # None when no cue matched
    genre_weights: dict[str, float] = field(default_factory=dict)
    matched_cues: list[str] = field(default_factory=list)
    source: str = 'lexicon'

    def mood_vector(self) -> np.ndarray | None:
        if self.mood is None:
            return None
        return np.array([float(self.mood.get(dim, 0.5)) for dim in SCENE_MOOD_DIMS], dtype=np.float64)


def analyze_scene_text(text: str | None) -> SceneContext:
    """Map a free-text scene description to a mood vector and genre hints."""
    if not text or not str(text).strip():
        return SceneContext(mood=None)

    tokens = _TOKEN_RE.findall(str(text).lower())

    anchors: list[np.ndarray] = []
    weights: list[float] = []
    genre_weights: dict[str, float] = {}
    matched: list[str] = []

    for cue, entry in _LEXICON.items():
        if _cue_matches(cue, tokens):
            anchors.append(np.asarray(entry['mood'], dtype=np.float64))
            weights.append(1.0)
            matched.append(cue)
            for genre in entry.get('genres', ()):  # accumulate soft genre votes
                genre_weights[genre] = genre_weights.get(genre, 0.0) + 1.0

    if not anchors:
        return SceneContext(mood=None, source='lexicon')

    stacked = np.vstack(anchors)
    weight_array = np.asarray(weights, dtype=np.float64)
    mood_vector = (stacked * weight_array[:, None]).sum(axis=0) / weight_array.sum()
    mood = {dim: float(np.clip(mood_vector[i], 0.0, 1.0)) for i, dim in enumerate(SCENE_MOOD_DIMS)}

    total_votes = sum(genre_weights.values()) or 1.0
    genre_weights = {genre: round(votes / total_votes, 4) for genre, votes in genre_weights.items()}

    return SceneContext(mood=mood, genre_weights=genre_weights, matched_cues=matched, source='lexicon')


def clip_scene_alignment(image_path: str, prompts: list[str]):  # pragma: no cover - optional heavy path
    """Optional CLIP grounding of an image against text prompts.

    Returns a dict ``{prompt: probability}`` over ``prompts`` when
    ``transformers`` + ``torch`` are installed and ``MAI_CLIP_MODEL`` names a
    checkpoint; otherwise ``None`` so callers fall back to the lexicon + colour
    mood. This is the learned upgrade seam mirroring the audio side's CLAP hook.
    """
    model_name = str(os.getenv('MAI_CLIP_MODEL') or '').strip()
    if not model_name or not prompts:
        return None
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:
        logger.info('CLIP scene grounding unavailable (%r); using lexicon + colour.', exc)
        return None

    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        with Image.open(image_path) as handle:
            image = handle.convert('RGB')
        inputs = processor(text=list(prompts), images=image, return_tensors='pt', padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image.softmax(dim=1).reshape(-1)
        return {prompt: float(score) for prompt, score in zip(prompts, logits.tolist())}
    except Exception:
        logger.exception('CLIP scene grounding failed; using lexicon + colour.')
        return None
