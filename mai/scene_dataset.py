"""MAI-Bench: the scene→music benchmark schema, loaders, and ground-truth adapters.

A citable benchmark is the highest-leverage research artifact: it converts the
matcher from a demo into something the field can measure against. This module
defines the dataset format and the tooling around it — but never invents ground
truth. Real relevance comes from an external source you supply:

* **Film/TV cue sheets** — scenes already paired with the licensed tracks that
  scored them. :func:`ingest_cue_sheets` turns a cue-sheet table (film, scene,
  cued track) into labelled examples. You bring the dump; this maps it.
* **Hand curation / human study** — authored ``relevant_ids`` per scene.

:func:`make_synthetic_benchmark` fabricates *clearly-labelled* synthetic scenes
for smoke-testing the harness only; it is not a substitute for real labels.

On-disk format (JSONL, one scene per line)::

    {"scene_id": "godfather_baptism", "image_path": null,
     "scene_text": "intercut baptism and assassinations, sacred and brutal",
     "relevant_ids": ["t_4412"], "graded": {"t_4412": 1.0},
     "source": "cue_sheet:the_godfather"}
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .scene_eval import BenchmarkExample


logger = logging.getLogger(__name__)

MAI_BENCH_VERSION = '0.1'


# --------------------------------------------------------------------------- #
# Load / save                                                                 #
# --------------------------------------------------------------------------- #

def _example_from_record(record: dict[str, Any]) -> BenchmarkExample:
    if 'scene_id' not in record:
        raise ValueError(f'benchmark record missing scene_id: {record!r}')
    relevant = record.get('relevant_ids') or []
    graded = {k: float(v) for k, v in (record.get('graded') or {}).items()}
    return BenchmarkExample(
        scene_id=str(record['scene_id']),
        relevant_ids=set(relevant),
        image_path=record.get('image_path') or None,
        scene_text=record.get('scene_text') or None,
        graded=graded,
    )


def load_benchmark(path: str) -> list[BenchmarkExample]:
    """Load a MAI-Bench file. ``.jsonl`` (preferred) or ``.csv`` are supported."""
    if str(path).lower().endswith('.csv'):
        frame = pd.read_csv(path)
        examples = []
        for _, row in frame.iterrows():
            raw_ids = str(row.get('relevant_ids', '') or '')
            relevant = [piece.strip() for piece in raw_ids.split('|') if piece.strip()]
            examples.append(_example_from_record({
                'scene_id': row.get('scene_id'),
                'image_path': row.get('image_path'),
                'scene_text': row.get('scene_text'),
                'relevant_ids': relevant,
            }))
        return examples

    examples = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(_example_from_record(json.loads(line)))
    return examples


def save_benchmark(examples: Iterable[BenchmarkExample], path: str, *, source: str | None = None) -> None:
    """Write examples to a JSONL MAI-Bench file."""
    with open(path, 'w', encoding='utf-8') as handle:
        for example in examples:
            record = asdict(example)
            record['relevant_ids'] = sorted(record['relevant_ids'], key=str)
            if source and 'source' not in record:
                record['source'] = source
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #

def validate_benchmark(
    examples: list[BenchmarkExample],
    library_ids: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Sanity-check a benchmark; returns a report of problems found."""
    library = set(library_ids) if library_ids is not None else None
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    empty_relevant: list[str] = []
    no_scene_signal: list[str] = []
    missing_in_library: dict[str, list[Any]] = {}

    for example in examples:
        if example.scene_id in seen:
            duplicate_ids.append(example.scene_id)
        seen.add(example.scene_id)
        if not example.relevant_ids:
            empty_relevant.append(example.scene_id)
        if not example.image_path and not example.scene_text:
            no_scene_signal.append(example.scene_id)
        if library is not None:
            missing = [item for item in example.relevant_ids if item not in library]
            if missing:
                missing_in_library[example.scene_id] = missing

    report = {
        'n_examples': len(examples),
        'duplicate_scene_ids': duplicate_ids,
        'empty_relevant': empty_relevant,
        'no_scene_signal': no_scene_signal,
        'missing_in_library': missing_in_library,
        'ok': not (duplicate_ids or empty_relevant or no_scene_signal or missing_in_library),
    }
    if not report['ok']:
        logger.warning('MAI-Bench validation found issues: %s',
                       {k: v for k, v in report.items() if k not in ('n_examples', 'ok') and v})
    return report


# --------------------------------------------------------------------------- #
# Ground-truth adapters                                                       #
# --------------------------------------------------------------------------- #

def ingest_cue_sheets(
    cue_sheet: pd.DataFrame,
    *,
    film_column: str,
    track_id_column: str,
    scene_column: str | None = None,
    scene_text_column: str | None = None,
    image_path_column: str | None = None,
) -> list[BenchmarkExample]:
    """Turn a cue-sheet table into labelled scenes.

    A cue sheet pairs a film (and optionally a scene/cue within it) with the
    track(s) that scored it — real scene→music ground truth. Rows are grouped by
    ``(film, scene)``; the cued tracks become ``relevant_ids``.
    """
    if film_column not in cue_sheet.columns or track_id_column not in cue_sheet.columns:
        raise ValueError('cue_sheet must contain the film and track id columns')

    group_keys = [film_column] + ([scene_column] if scene_column else [])
    examples: list[BenchmarkExample] = []
    for keys, group in cue_sheet.groupby(group_keys, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        scene_id = ':'.join(str(part) for part in keys)
        relevant = {tid for tid in group[track_id_column].tolist() if pd.notna(tid)}
        if not relevant:
            continue
        scene_text = None
        if scene_text_column and scene_text_column in group.columns:
            texts = [str(t) for t in group[scene_text_column].dropna().tolist()]
            scene_text = next((t for t in texts if t.strip()), None)
        image_path = None
        if image_path_column and image_path_column in group.columns:
            paths = [str(p) for p in group[image_path_column].dropna().tolist()]
            image_path = next((p for p in paths if p.strip()), None)
        examples.append(BenchmarkExample(
            scene_id=scene_id, relevant_ids=relevant,
            image_path=image_path, scene_text=scene_text,
        ))
    logger.info('Ingested %d cue-sheet rows into %d labelled scenes.', len(cue_sheet), len(examples))
    return examples


# Templates pairing a description with the genre that should win it, used only to
# fabricate synthetic ground truth for testing the harness.
_SYNTH_TEMPLATES = (
    ('a tense horror scene, dark and eerie', 'industrial'),
    ('a tender romantic moment at sunset', 'soul'),
    ('a joyful celebration, bright and hopeful', 'pop'),
    ('a peaceful calm morning in nature', 'ambient'),
    ('an explosive battle, chaotic and loud', 'metal'),
    ('a neon city chase at night', 'synthwave'),
)


def make_synthetic_benchmark(
    library: pd.DataFrame,
    *,
    relevant_per_scene: int = 3,
    seed: int = 0,
) -> list[BenchmarkExample]:
    """Fabricate synthetic labelled scenes from the library (SMOKE TEST ONLY).

    Relevance is defined as the library's own best affect matches for each
    template, so a working matcher should score highly — this validates the
    harness end to end, not model quality. Real evaluation needs real labels.
    """
    from .scene_match import build_scene_target, score_library_against_scene

    id_column = next((c for c in ('track_id', 'video_id', 'id') if c in library.columns), None)
    examples: list[BenchmarkExample] = []
    rng = np.random.default_rng(seed)
    for offset, (text, _genre) in enumerate(_SYNTH_TEMPLATES):
        target = build_scene_target(scene_text=text)
        matched = score_library_against_scene(library, target, top_k=relevant_per_scene)
        if id_column:
            relevant = set(matched[id_column].tolist())
        else:
            relevant = set(matched.index.tolist())
        examples.append(BenchmarkExample(
            scene_id=f'synthetic_{offset}_{int(rng.integers(0, 1_000_000))}',
            relevant_ids=relevant, scene_text=text,
        ))
    return examples
