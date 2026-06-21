"""Retrieval ⊕ generation: when no track fits a scene, generate one.

Retrieval picks the best *existing* song for a scene. Sometimes nothing in the
library fits — a niche affect, an empty catalogue, a bespoke cue. The dual move
is to *generate*: translate the scene's affect + genre hints into a text prompt
and synthesise audio with a music generation model (MusicGen). Same affect space,
two outputs — pick the closest track, or render a new one.

This is a torch-guarded seam (audiocraft or transformers MusicGen). It imports
without those heavy deps; :func:`generation_available` reports whether it can run,
and :func:`scene_to_prompt` — the affect→text translation — is pure-python and
always available (useful on its own for text-conditioned APIs).
"""

from __future__ import annotations

import logging

from .scene_match import SCENE_MOOD_DIMS, SceneMoodTarget


logger = logging.getLogger(__name__)

# Affect-axis wording, low→high, for turning a mood vector into natural language.
_AXIS_WORDS = {
    'valence': ('dark and bleak', 'bittersweet', 'bright and uplifting'),
    'arousal': ('calm and still', 'steady', 'energetic and driving'),
    'tension': ('relaxed and resolved', 'unsettled', 'tense and ominous'),
    'warmth': ('cold and clinical', 'neutral', 'warm and intimate'),
}


def _bucket(value: float) -> int:
    return 0 if value < 0.34 else (1 if value < 0.67 else 2)


def scene_to_prompt(target: SceneMoodTarget, *, max_genres: int = 2) -> str:
    """Translate a scene's affect + genre hints into a MusicGen text prompt."""
    descriptors = [_AXIS_WORDS[axis][_bucket(target.mood.get(axis, 0.5))] for axis in SCENE_MOOD_DIMS]
    genres = []
    if target.context is not None and target.context.genre_weights:
        genres = [g for g, _ in sorted(target.context.genre_weights.items(), key=lambda kv: -kv[1])[:max_genres]]
    genre_text = (', '.join(genres) + ' ') if genres else ''
    return f'{genre_text}instrumental music that is ' + ', '.join(descriptors)


def generation_available() -> bool:
    """True when a MusicGen backend (audiocraft or transformers) + torch is present."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    try:
        import audiocraft  # noqa: F401
        return True
    except Exception:
        pass
    try:
        from transformers import MusicgenForConditionalGeneration  # noqa: F401
        return True
    except Exception:
        return False


def generate_music_for_scene(
    target: SceneMoodTarget,
    *,
    out_path: str,
    duration_s: float = 10.0,
    model_name: str = 'facebook/musicgen-small',
) -> str:  # pragma: no cover - optional heavy path
    """Synthesise a scene-conditioned music clip to ``out_path``; returns the path.

    Uses transformers MusicGen. Raises a clear error when the backend is absent so
    callers fall back to retrieval (:mod:`mai.scene_match` / :mod:`mai.scene_index`).
    """
    prompt = scene_to_prompt(target)
    try:
        import scipy.io.wavfile
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            'Scene music generation requires torch + transformers (+ scipy for WAV output). '
            'Retrieval works without them.'
        ) from exc

    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    inputs = processor(text=[prompt], padding=True, return_tensors='pt')
    sample_rate = model.config.audio_encoder.sampling_rate
    max_new_tokens = int(duration_s * model.config.audio_encoder.frame_rate)
    with torch.no_grad():
        audio = model.generate(**inputs, max_new_tokens=max_new_tokens)
    waveform = audio[0, 0].cpu().numpy()
    scipy.io.wavfile.write(out_path, rate=sample_rate, data=waveform)
    logger.info('Generated scene music (%.1fs) for prompt %r -> %s', duration_s, prompt, out_path)
    return out_path
