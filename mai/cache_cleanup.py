from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Sequence

from .audio_analysis import _load_feature_cache_table
from .config import DEFAULT_CONFIG_PATH, get_config_value, load_project_config


logger = logging.getLogger(__name__)
TEMP_CACHE_SUFFIXES = {'.part', '.ytdl', '.tmp', '.temp'}


def _bootstrap_config(argv: Sequence[str] | None = None) -> tuple[dict, str, bool]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config', default=DEFAULT_CONFIG_PATH)
    bootstrap.add_argument('--no-config', action='store_true')
    known, _ = bootstrap.parse_known_args(argv)
    config = load_project_config(known.config, use_config=not known.no_config)
    return config, str(known.config), bool(known.no_config)


def _safe_unlink(path: Path, *, dry_run: bool) -> int:
    try:
        size = int(path.stat().st_size)
    except OSError:
        size = 0
    if not dry_run:
        try:
            path.unlink()
        except FileNotFoundError:
            return 0
    return size


def _remove_empty_dirs(root: Path, *, dry_run: bool) -> int:
    if not root.exists():
        return 0
    removed = 0
    for path in sorted((candidate for candidate in root.rglob('*') if candidate.is_dir()), key=lambda item: len(item.parts), reverse=True):
        try:
            if any(path.iterdir()):
                continue
        except OSError:
            continue
        if not dry_run:
            try:
                path.rmdir()
            except OSError:
                continue
        removed += 1
    return removed


def _is_temp_cache_file(path: Path) -> bool:
    suffixes = {suffix.lower() for suffix in path.suffixes}
    return bool(suffixes & TEMP_CACHE_SUFFIXES)


def _audio_cache_video_id(path: Path) -> str:
    name = path.name
    if not name:
        return ''
    return name.split('.', 1)[0].strip()


def format_bytes(num_bytes: int) -> str:
    value = float(max(int(num_bytes), 0))
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f'{value:.1f} {unit}'
        value /= 1024.0
    return f'{value:.1f} TB'


def clean_useless_cache(
    *,
    cache_dir: str = 'data/cache',
    audio_cache_dir: str = 'data/audio_cache',
    feature_cache_dir: str | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    cache_root = Path(str(cache_dir)).resolve()
    audio_root = Path(str(audio_cache_dir)).resolve()
    feature_cache_df, feature_cache_csv_path = _load_feature_cache_table(feature_cache_dir or str(cache_root / 'audio_features.csv'))
    feature_video_ids = {
        str(video_id).strip()
        for video_id in feature_cache_df.get('video_id', [])
        if str(video_id).strip()
    }
    feature_csv_path = Path(feature_cache_csv_path).resolve()
    legacy_feature_cache_dir = feature_csv_path.with_suffix('')

    yt_dlp_candidates = {cache_root / 'yt_dlp'}

    summary = {
        'feature_rows': int(len(feature_video_ids)),
        'audio_files_deleted': 0,
        'audio_temp_files_deleted': 0,
        'audio_files_kept': 0,
        'legacy_feature_json_deleted': 0,
        'yt_dlp_cache_files_deleted': 0,
        'directories_removed': 0,
        'bytes_freed': 0,
    }

    if audio_root.exists():
        for path in sorted(candidate for candidate in audio_root.iterdir() if candidate.is_file()):
            if _is_temp_cache_file(path):
                summary['audio_temp_files_deleted'] += 1
                summary['bytes_freed'] += _safe_unlink(path, dry_run=dry_run)
                continue
            if _audio_cache_video_id(path) in feature_video_ids:
                summary['audio_files_deleted'] += 1
                summary['bytes_freed'] += _safe_unlink(path, dry_run=dry_run)
            else:
                summary['audio_files_kept'] += 1

    if legacy_feature_cache_dir.exists():
        for path in sorted(legacy_feature_cache_dir.glob('*.json')):
            if path.stem in feature_video_ids:
                summary['legacy_feature_json_deleted'] += 1
                summary['bytes_freed'] += _safe_unlink(path, dry_run=dry_run)

    for yt_dlp_dir in sorted(yt_dlp_candidates):
        if not yt_dlp_dir.exists():
            continue
        for path in sorted(candidate for candidate in yt_dlp_dir.iterdir() if candidate.is_file()):
            if path.name.endswith('_sanitized.txt') or _is_temp_cache_file(path):
                summary['yt_dlp_cache_files_deleted'] += 1
                summary['bytes_freed'] += _safe_unlink(path, dry_run=dry_run)

    for root in {audio_root, legacy_feature_cache_dir, *yt_dlp_candidates}:
        summary['directories_removed'] += _remove_empty_dirs(root, dry_run=dry_run)
    return summary


def build_parser(config: dict[str, Any], config_path: str, no_config: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='clean redundant Mai cache files')
    parser.add_argument('--config', default=config_path, help='Path to project TOML config')
    parser.add_argument('--no-config', action='store_true', default=no_config, help='Ignore the TOML config and use CLI/default values only')
    parser.add_argument('--cache-dir', default=get_config_value(config, 'cache.root_dir', 'data/cache'), help='Directory for reusable metadata caches')
    parser.add_argument('--audio-cache', default=get_config_value(config, 'cache.audio_dir', 'data/audio_cache'), help='Directory for downloaded audio files')
    parser.add_argument('--feature-cache', default=os.path.join(get_config_value(config, 'cache.root_dir', 'data/cache'), 'audio_features.csv'), help='Global audio feature cache CSV')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without deleting files')
    parser.add_argument('--log-level', default=get_config_value(config, 'logging.level', 'INFO'), choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'], help='Verbosity')
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    config, config_path, no_config = _bootstrap_config(argv)
    parser = build_parser(config, config_path=config_path, no_config=no_config)
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format='%(levelname)s: %(message)s', force=True)
    summary = clean_useless_cache(
        cache_dir=args.cache_dir,
        audio_cache_dir=args.audio_cache,
        feature_cache_dir=args.feature_cache,
        dry_run=bool(args.dry_run),
    )
    mode = 'Dry run' if args.dry_run else 'Cache cleanup'
    print(
        f"{mode}: feature_rows={summary['feature_rows']} "
        f"audio_files_deleted={summary['audio_files_deleted']} "
        f"audio_temp_files_deleted={summary['audio_temp_files_deleted']} "
        f"legacy_feature_json_deleted={summary['legacy_feature_json_deleted']} "
        f"yt_dlp_cache_files_deleted={summary['yt_dlp_cache_files_deleted']} "
        f"audio_files_kept={summary['audio_files_kept']} "
        f"directories_removed={summary['directories_removed']} "
        f"bytes_freed={summary['bytes_freed']} ({format_bytes(summary['bytes_freed'])})"
    )


if __name__ == '__main__':
    main()
