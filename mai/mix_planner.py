"""Turn an ordered playlist into a DJ mix plan and export it.

``mai.overlay`` scores and selects overlays between :class:`RegionDescriptor`
sections, but something has to *build* those sections from real track rows and
emit the result. This module is that bridge:

    dataframe rows ──▶ per-track regions ──▶ plan_mix ──▶ cue-sheet export

Most catalogue rows carry only aggregate + intro/outro edge features (not a full
time series), so each track is split into a coarse ``intro / body / outro`` set of
regions derived from the columns Mai already computes; when beat-synchronous
features are available later, the same plan path produces finer sections. The
output is a mix plan (one overlay per consecutive pair) plus a flat cue-sheet
DataFrame/CSV that a DJ tool or the existing exporters can consume.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from .overlay import RegionDescriptor, plan_mix
from .sentiment import add_sentiment_features


logger = logging.getLogger(__name__)

_ID_CANDIDATES = ('track_id', 'video_id', 'id', 'resolved_video_id')
_TITLE_CANDIDATES = ('title', 'track_name', 'name', 'video_title')


def _first(row: pd.Series, columns, default=None):
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            return row[column]
    return default


def _num(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row.get(column, default) if column in row.index else default
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _band_profile(brightness: float) -> np.ndarray:
    """Coarse bass/mid/high energy share from spectral brightness in [0, 1].

    A proxy until per-band energies are threaded through: dark tracks weight bass,
    bright tracks weight highs, with a steady mid floor. Replace with real band
    energies when the analysis time series is available.
    """
    b = float(np.clip(brightness, 0.0, 1.0))
    profile = np.array([1.0 - b, 0.5, b], dtype=np.float64) + 1e-3
    return profile / profile.sum()


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def regions_from_row(row: pd.Series, track_id, *, frames: int = 32) -> list[RegionDescriptor]:
    """Build intro / body / outro mixable regions for one track row."""
    tempo = _num(row, 'tempo', 120.0)
    key = int(_num(row, 'key', -1))
    energy = float(np.clip(_num(row, 'energy', _num(row, 'rms', 0.5)), 0.0, 1.0))
    speechiness = _num(row, 'speechiness', 0.0)
    harmonic = _num(row, 'harmonic_ratio', 0.5)
    vocal = float(np.clip(0.6 * speechiness + 0.4 * harmonic, 0.0, 1.0))

    def region(edge: str, position: float, bars: int) -> RegionDescriptor:
        brightness = _norm(_num(row, f'{edge}_spectral_centroid', _num(row, 'spectral_centroid', 2500.0)), 1000.0, 5000.0)
        local_tempo = _num(row, f'{edge}_tempo', tempo) if edge else tempo
        # A flat onset envelope (no time series): aligns at lag 0 with low
        # confidence; real beat-grids upgrade this to a true cross-correlation.
        onset = np.full(frames, _norm(_num(row, f'{edge}_onset_strength', _num(row, 'onset_strength', 1.0)), 0.0, 5.0) + 0.1)
        return RegionDescriptor(
            track_id=track_id, start_s=0.0, end_s=0.0, position=position, tempo=local_tempo,
            key=key, energy=energy, vocal_activity=vocal, band_profile=_band_profile(brightness),
            onset_envelope=onset, bars=bars, is_drop=energy > 0.8,
        )

    return [region('intro', 0.05, 16), region('', 0.5, 32), region('outro', 0.95, 16)]


def build_regions(df: pd.DataFrame) -> dict:
    """Per-track region lists keyed by track id (falls back to row position)."""
    enriched = add_sentiment_features(df).reset_index(drop=True)
    id_column = next((c for c in _ID_CANDIDATES if c in enriched.columns), None)
    regions: dict = {}
    for position, (_, row) in enumerate(enriched.iterrows()):
        track_id = row[id_column] if id_column else position
        regions[track_id] = regions_from_row(row, track_id)
    return regions


def _order_from_dataframe(df: pd.DataFrame) -> list:
    id_column = next((c for c in _ID_CANDIDATES if c in df.columns), None)
    if 'position' in df.columns:
        df = df.sort_values('position', kind='stable')
    return df[id_column].tolist() if id_column else list(range(len(df)))


def plan_mix_from_dataframe(df: pd.DataFrame, order: list | None = None) -> list:
    """Build per-track regions then the overlay plan for ``order`` (or row order)."""
    regions = build_regions(df)
    if order is None:
        order = _order_from_dataframe(df)
    return plan_mix(regions, order)


def mix_plan_to_dataframe(plan: list, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Flatten a mix plan into a cue-sheet table (one row per transition)."""
    titles = {}
    if df is not None:
        id_column = next((c for c in _ID_CANDIDATES if c in df.columns), None)
        if id_column:
            for _, row in df.iterrows():
                titles[row[id_column]] = _first(row, _TITLE_CANDIDATES, default=row[id_column])

    rows = []
    for step, match in enumerate(plan, start=1):
        exit_region, entry_region = match.exit_region, match.entry_region
        rows.append({
            'step': step,
            'from_track': titles.get(exit_region.track_id, exit_region.track_id),
            'to_track': titles.get(entry_region.track_id, entry_region.track_id),
            'blend_type': match.blend_type,
            'beat_offset': match.beat_offset,
            'score': round(float(match.score), 4),
            'exit_position': round(float(exit_region.position), 3),
            'entry_position': round(float(entry_region.position), 3),
            'tempo_lock': round(float(match.components.get('tempo_lock', 0.0)), 3),
            'complementarity': round(float(match.components.get('complementarity', 0.0)), 3),
            'vocal_compat': round(float(match.components.get('vocal_compat', 0.0)), 3),
        })
    return pd.DataFrame(rows)


def export_cue_sheet(plan: list, path: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Write the mix plan as a cue-sheet CSV and return the table."""
    table = mix_plan_to_dataframe(plan, df)
    table.to_csv(path, index=False)
    logger.info('Wrote %d-transition mix plan to %s', len(table), path)
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Build a DJ mix plan from a playlist CSV.')
    parser.add_argument('--csv', required=True, help='ordered playlist CSV (audio features).')
    parser.add_argument('--out', help='write the cue sheet to this CSV.')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    plan = plan_mix_from_dataframe(df)
    table = mix_plan_to_dataframe(plan, df)

    print(f'Mix plan: {len(table)} transitions')
    for _, row in table.iterrows():
        print(f"  {row['step']:02d}. {row['from_track']} → {row['to_track']}  "
              f"[{row['blend_type']} off={row['beat_offset']} score={row['score']:.2f}]")
    if args.out:
        export_cue_sheet(plan, args.out, df)
        print(f'Wrote cue sheet to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
