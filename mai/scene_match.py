"""Match songs to an image scene.

Ties the colour-grounded image mood (``scene_features``) and the text-grounded
scene context (``scene_context``) into a single target mood, then ranks a track
library by how well each song's mood fits that target — so a frame of a film can
pick the songs that score it.

Both sides land in the shared ``valence / arousal / tension / warmth`` space, so
matching is a weighted distance in four numbers. Genre hints from the scene
description give a soft re-rank on top. Optionally the matched top-k is handed to
the existing transition engine so the picks come back as a flowing sequence, not
just a ranked list.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .scene_context import SCENE_MOOD_DIMS, SceneContext, analyze_scene_text
from .scene_features import SceneFeatures, analyze_scene_image
from .sentiment import add_sentiment_features


logger = logging.getLogger(__name__)

# Per-axis weight when measuring scene<->track mood distance. Tension and valence
# carry the most narrative intent (a tense scene must not get a happy song), so
# they weigh a little heavier than arousal/warmth.
_AXIS_WEIGHTS = np.array([1.1, 1.0, 1.2, 0.9], dtype=np.float64)  # valence, arousal, tension, warmth
# Distance -> score falloff. ~0.32 means a per-axis mood gap of ~1/3 roughly
# halves the score; tight enough to discriminate, loose enough to rank a library.
_MOOD_SIGMA = 0.32
_GENRE_BOOST = 0.15  # max multiplicative lift for a track whose genre the scene wants


@dataclass(slots=True)
class SceneMoodTarget:
    """The fused mood a scene asks for, with provenance for reporting."""

    mood: dict[str, float]
    image_features: SceneFeatures | None = None
    context: SceneContext | None = None
    text_weight: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)

    def mood_vector(self) -> np.ndarray:
        return np.array([float(self.mood.get(dim, 0.5)) for dim in SCENE_MOOD_DIMS], dtype=np.float64)


def build_scene_target(
    image_path: str | None = None,
    scene_text: str | None = None,
    *,
    text_weight: float = 0.5,
) -> SceneMoodTarget:
    """Fuse the image mood and the text mood into one target mood vector.

    ``text_weight`` in [0, 1] sets how much the caption pulls the colour mood when
    both are present. With only one source available, that source is used as-is.
    """
    image_features = analyze_scene_image(image_path) if image_path else None
    context = analyze_scene_text(scene_text)

    image_vector = image_features.mood_vector() if image_features is not None else None
    text_vector = context.mood_vector()

    if image_vector is not None and text_vector is not None:
        blend = float(np.clip(text_weight, 0.0, 1.0))
        fused = (1.0 - blend) * image_vector + blend * text_vector
    elif image_vector is not None:
        fused = image_vector
    elif text_vector is not None:
        fused = text_vector
    else:
        raise ValueError('a scene needs at least an --image or --scene-text to match against')

    mood = {dim: float(np.clip(fused[i], 0.0, 1.0)) for i, dim in enumerate(SCENE_MOOD_DIMS)}
    return SceneMoodTarget(
        mood=mood, image_features=image_features, context=context, text_weight=text_weight,
    )


def _track_mood_matrix(library: pd.DataFrame) -> np.ndarray:
    enriched = add_sentiment_features(library)
    columns = [f'sentiment_{dim}' for dim in SCENE_MOOD_DIMS]
    arrays = []
    for column in columns:
        if column in enriched.columns:
            arrays.append(pd.to_numeric(enriched[column], errors='coerce').fillna(0.5).to_numpy(dtype=np.float64))
        else:
            arrays.append(np.full(len(enriched), 0.5, dtype=np.float64))
    return np.column_stack(arrays)


def _genre_boost(library: pd.DataFrame, genre_weights: dict[str, float]) -> np.ndarray:
    """Per-track multiplicative lift in [1, 1+_GENRE_BOOST] from scene genre hints."""
    n = len(library)
    if not genre_weights:
        return np.ones(n, dtype=np.float64)
    column = next((c for c in ('genre_primary', 'mix_group', 'style_cluster') if c in library.columns), None)
    if column is None:
        return np.ones(n, dtype=np.float64)
    labels = library[column].fillna('').astype(str).str.strip().str.lower()
    max_weight = max(genre_weights.values()) or 1.0
    # Resolve the best matching hint per *distinct* label (cardinality << N rows),
    # then map back — keeps the hot path off a per-row python loop. Substring match
    # both ways so 'hip hop' hints lift a 'boom bap hip hop' track and vice versa.
    weight_for_label = {
        label: (max((w for g, w in genre_weights.items() if g in label or label in g), default=0.0)
                if label else 0.0)
        for label in labels.unique()
    }
    best = labels.map(weight_for_label).to_numpy(dtype=np.float64)
    return 1.0 + _GENRE_BOOST * (best / max_weight)


def score_library_against_scene(
    library: pd.DataFrame,
    target: SceneMoodTarget,
    *,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Return ``library`` with a ``scene_fit`` score in [0, 1], best first.

    Score is a Gaussian on the axis-weighted mood distance to the scene target,
    re-ranked by the scene's soft genre hints. Adds ``mood_distance`` and the
    per-axis ``scene_*`` target columns for transparency.
    """
    if library.empty:
        result = library.copy()
        result['scene_fit'] = pd.Series(dtype=float)
        return result

    track_moods = _track_mood_matrix(library)
    target_vector = target.mood_vector()

    deltas = (track_moods - target_vector[None, :]) * _AXIS_WEIGHTS[None, :]
    distances = np.sqrt(np.sum(deltas ** 2, axis=1) / float(np.sum(_AXIS_WEIGHTS)))
    base_score = np.exp(-(distances ** 2) / (2.0 * _MOOD_SIGMA ** 2))

    genre_weights = target.context.genre_weights if target.context is not None else {}
    boosted = np.clip(base_score * _genre_boost(library, genre_weights), 0.0, 1.0)

    result = library.copy()
    result['mood_distance'] = distances
    result['scene_fit'] = boosted
    for dim, value in target.mood.items():
        result[f'scene_{dim}'] = float(value)

    result = result.sort_values('scene_fit', ascending=False, kind='stable').reset_index(drop=True)
    if top_k is not None and top_k > 0:
        result = result.head(int(top_k)).reset_index(drop=True)
    return result


def order_scene_playlist(matched: pd.DataFrame, target: SceneMoodTarget) -> pd.DataFrame:
    """Reorder the matched tracks into a flowing sequence via the transition engine.

    Falls back to the scene-fit ranking if the transition engine cannot run (too
    few tracks). The arc is oriented so the set opens near the scene's energy.
    """
    if len(matched) < 3:
        return matched
    from .playlist_generation import compute_transition_scores, generate_playlist_paths

    transitions, scored = compute_transition_scores(matched)
    paths, _, _ = generate_playlist_paths(
        scored, transitions, playlist_size=len(matched), num_playlists=1, allow_reuse=False,
    )
    if not paths or not paths[0]:
        return matched
    order = paths[0]
    ordered = matched.iloc[order].reset_index(drop=True)
    ordered['scene_position'] = np.arange(1, len(ordered) + 1)
    return ordered


def _format_target(target: SceneMoodTarget) -> str:
    mood_text = ', '.join(f'{dim}={target.mood[dim]:.2f}' for dim in SCENE_MOOD_DIMS)
    parts = [f'scene mood: {mood_text}']
    features = target.image_features
    if features is not None:
        palette = ' '.join('#%02x%02x%02x' % colour for colour in features.palette[:5])
        parts.append(
            f'colour: brightness={features.brightness:.2f} saturation={features.saturation:.2f} '
            f'contrast={features.contrast:.2f} warm={features.warm_share:.2f} cool={features.cool_share:.2f}'
        )
        parts.append(f'palette: {palette}')
    context = target.context
    if context is not None and context.matched_cues:
        parts.append('cues: ' + ', '.join(context.matched_cues))
        if context.genre_weights:
            top_genres = sorted(context.genre_weights.items(), key=lambda kv: -kv[1])[:5]
            parts.append('genre hints: ' + ', '.join(f'{g} {w:.2f}' for g, w in top_genres))
    return '\n'.join(parts)


def _display_label(row: pd.Series) -> str:
    for column in ('title', 'track_name', 'name', 'video_title'):
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            title = str(row[column]).strip()
            for artist_column in ('artist', 'artists', 'channel_title', 'uploader'):
                if artist_column in row.index and pd.notna(row[artist_column]) and str(row[artist_column]).strip():
                    return f'{str(row[artist_column]).strip()} - {title}'
            return title
    return '<unknown track>'


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Match songs to an image scene by colour + context mood.')
    parser.add_argument('--csv', required=True, help='track library CSV (with audio/sentiment descriptors).')
    parser.add_argument('--image', help='scene image file (movie still, photo, cover art).')
    parser.add_argument('--scene-text', help='free-text scene description (caption, synopsis, tags).')
    parser.add_argument('--text-weight', type=float, default=0.5,
                        help='blend [0,1] of caption mood vs colour mood when both given (default 0.5).')
    parser.add_argument('--top-k', type=int, default=15, help='how many songs to return (default 15).')
    parser.add_argument('--order', action='store_true', help='reorder the matches into a flowing sequence.')
    parser.add_argument('--out', help='write the ranked matches to this CSV.')
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = build_arg_parser().parse_args(argv)
    if not args.image and not args.scene_text:
        build_arg_parser().error('provide --image and/or --scene-text to describe the scene.')

    library = pd.read_csv(args.csv)
    library.columns = [c.strip() for c in library.columns]

    target = build_scene_target(args.image, args.scene_text, text_weight=args.text_weight)
    matched = score_library_against_scene(library, target, top_k=args.top_k)
    if args.order:
        matched = order_scene_playlist(matched, target)

    print(_format_target(target))
    print(f'\nTop {len(matched)} songs for this scene:')
    for rank, (_, row) in enumerate(matched.iterrows(), start=1):
        print(f'  {rank:02d}. {_display_label(row)}  [fit {float(row["scene_fit"]):.3f}]')

    if args.out:
        matched.to_csv(args.out, index=False)
        print(f'\nWrote {len(matched)} matches to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
