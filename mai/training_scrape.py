from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime
import hashlib
import json
import logging
import os
import re
import tempfile
from threading import Lock
import time
import unicodedata
from typing import Any, Callable, Optional, Sequence

import pandas as pd
from yt_dlp import YoutubeDL

from .audio_analysis import analyze_youtube_playlist_audio
from .cli_progress import CliProgressRenderer, configure_cli_logging
from .config import DEFAULT_CONFIG_PATH, get_config_value, load_project_config
from .tabular_cache import read_sqlite_table, resolve_sqlite_cache_path, write_sqlite_table
from .yt_dlp_auth import apply_yt_dlp_auth_options


logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, int, int, str], None]

DEFAULT_CHANNEL_URL = 'https://www.youtube.com/@mai_dq/videos'
DEFAULT_OUTPUT_PATH = os.path.join('data', 'training', 'positive_transitions.csv')
DEFAULT_LABEL = 'excellent'
DEFAULT_LABEL_SOURCE = 'youtube_mix_curation'
MAX_SEARCH_RESULTS = 5
DEFAULT_METADATA_WORKERS = 4
DEFAULT_SEARCH_WORKERS = 4
YTDLP_SOCKET_TIMEOUT_SECONDS = 20.0
YTDLP_EXTRACTOR_RETRIES = 2
YTDLP_RATE_LIMIT_MAX_RETRIES = 3
YTDLP_RATE_LIMIT_BACKOFF_SECONDS = 2.0
YTDLP_RATE_LIMIT_MAX_BACKOFF_SECONDS = 30.0
YTDLP_MIN_REQUEST_INTERVAL_SECONDS = 0.0
FUTURE_WAIT_HEARTBEAT_SECONDS = 10.0
SOURCE_TRACK_CACHE_VERSION = 4
RESOLUTION_CACHE_VERSION = 4
MAX_SEARCH_DURATION_SECONDS = 15 * 60
TIMESTAMP_LINE_RE = re.compile(r'^\s*((?:\d{1,2}:)?\d{1,2}:\d{2})\s*(?:[-\u2013\u2014|]\s*)?(.+?)\s*$')
SEPARATOR_RE = re.compile(r'\s(?:-|\u2013|\u2014)\s')
LEADING_TIMESTAMP_TEXT_RE = re.compile(r'^\s*(?:\d{1,2}:)?\d{1,2}:\d{2}\s*(?:[-\u2013\u2014|]\s*)?')
URL_CHUNK_RE = re.compile(r'(?i)(?:https?://\S+|www\.\S+|(?:open\.)?spotify\.com/\S+|soundcloud\.com/\S+|youtube\.com/\S+|youtu\.be/\S+)')
EMPTY_BRACKETS_RE = re.compile(r'\(\s*\)|\[\s*\]|\{\s*\}')
SEARCH_NOISE_BRACKET_RE = re.compile(
    r'\s*[\(\[]\s*(?:\d{4}(?:-\d{2,4})?|official(?:\s+(?:video|audio))?|lyrics?|lyric video|visualizer|4k special|slowed(?:\s+by\s+me)?|sped up|nightcore|full version|ver\.?.*?)\s*[\)\]]',
    re.IGNORECASE,
)
URL_TEXT_RE = re.compile(r'(?:https?://|www\.|youtube\.com/|youtu\.be/|soundcloud\.com/|spotify\.com/)', re.IGNORECASE)
NON_TRACK_CANDIDATE_RE = re.compile(
    r'\b(?:tutorial|walkthrough|gameplay|gaming test|benchmark|review|reaction|compilation|playlist|mix(?:\s+\d+)?|1 hour|2 hour|3 hour|night drive|edit|amv)\b',
    re.IGNORECASE,
)
TOKEN_SPLIT_RE = re.compile(r'[^\w]+', re.UNICODE)
MATCH_STOPWORDS = {
    'official', 'audio', 'video', 'music', 'topic', 'lyrics', 'lyric', 'visualizer', 'feat', 'featuring',
    'ft', 'prod', 'full', 'version', 'ver', 'by', 'the', 'and',
}
UNAVAILABLE_TITLES = {'[deleted video]', '[private video]'}
WATCH_METADATA_LIST_KEYS = ('music_tracks', 'tracks', 'tracklist', 'music_sections')

_YTDLP_REQUEST_LOCK = Lock()
_YTDLP_NEXT_REQUEST_AT = 0.0
_YTDLP_GLOBAL_RATE_LIMIT_LOCK = Lock()
_YTDLP_GLOBAL_RATE_LIMIT_UNTIL = 0.0


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name) or '').strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name) or '').strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


YTDLP_RATE_LIMIT_MAX_RETRIES = _env_int('MAI_YTDLP_RATE_LIMIT_RETRIES', YTDLP_RATE_LIMIT_MAX_RETRIES)
YTDLP_RATE_LIMIT_BACKOFF_SECONDS = _env_float('MAI_YTDLP_RATE_LIMIT_BACKOFF', YTDLP_RATE_LIMIT_BACKOFF_SECONDS)
YTDLP_RATE_LIMIT_MAX_BACKOFF_SECONDS = _env_float('MAI_YTDLP_RATE_LIMIT_MAX_BACKOFF', YTDLP_RATE_LIMIT_MAX_BACKOFF_SECONDS)
YTDLP_MIN_REQUEST_INTERVAL_SECONDS = _env_float('MAI_YTDLP_MIN_REQUEST_INTERVAL', YTDLP_MIN_REQUEST_INTERVAL_SECONDS)
YTDLP_GLOBAL_PAUSE_ON_429 = _env_int('MAI_YTDLP_GLOBAL_PAUSE_ON_429', 1) != 0

SOURCE_TRACK_COLUMNS = [
    'video_id',
    'channel_url',
    'channel_handle',
    'label',
    'label_source',
    'source_cache_version',
    'video_title',
    'video_url',
    'description_length',
    'track_source',
    'chapter_title',
    'chapter_timestamp_s',
    'position',
    'timestamp',
    'timestamp_s',
    'track_raw',
    'artist_guess',
    'title_guess',
]
SOURCE_TRACK_CACHE_COLUMNS = [
    'video_id',
    'position',
    'source_signature',
    'channel_handle',
    'description_length',
    'track_source',
    'chapter_title',
    'chapter_timestamp_s',
    'timestamp_s',
    'track_raw',
    'artist_guess',
    'title_guess',
]

RESOLUTION_COLUMNS = [
    'video_id',
    'position',
    'search_query',
    'normalized_search_query',
    'search_max_results',
    'resolution_cache_version',
    'resolution_status',
    'resolved_video_id',
    'resolved_title',
    'resolved_artist',
    'resolved_url',
    'resolved_duration_seconds',
]
RESOLUTION_CACHE_COLUMNS = [
    'video_id',
    'position',
    'resolution_signature',
    'resolution_status',
    'resolved_video_id',
    'resolved_title',
    'resolved_artist',
    'resolved_duration_seconds',
]

TRACK_COLUMNS = SOURCE_TRACK_COLUMNS + [column for column in RESOLUTION_COLUMNS if column not in SOURCE_TRACK_COLUMNS]

BASE_TRANSITION_COLUMNS = [
    'video_id',
    'label',
    'label_source',
    'channel_handle',
    'channel_url',
    'video_title',
    'video_url',
    'description_length',
    'from_position',
    'to_position',
    'from_timestamp',
    'to_timestamp',
    'from_timestamp_s',
    'to_timestamp_s',
    'transition_duration_s',
    'from_track_raw',
    'to_track_raw',
    'from_artist_guess',
    'from_title_guess',
    'to_artist_guess',
    'to_title_guess',
    'from_track_source',
    'to_track_source',
    'from_video_id',
    'to_video_id',
    'from_resolved_title',
    'from_resolved_artist',
    'from_resolved_url',
    'from_resolved_duration_seconds',
    'to_resolved_title',
    'to_resolved_artist',
    'to_resolved_url',
    'to_resolved_duration_seconds',
]

# Compact export used for the training transitions CSV.
# Keep high-signal transition metadata and textual context while dropping
# bulky timing and dense numeric feature columns.
COMPACT_TRANSITION_COLUMNS = [
    'video_id',
    'label',
    'label_source',
    'channel_handle',
    'channel_url',
    'video_title',
    'video_url',
    'from_position',
    'to_position',
    'transition_duration_s',
    'from_track_raw',
    'to_track_raw',
    'from_artist_guess',
    'from_title_guess',
    'to_artist_guess',
    'to_title_guess',
    'from_track_source',
    'to_track_source',
    'from_video_id',
    'to_video_id',
    'from_resolved_title',
    'from_resolved_artist',
    'from_resolved_url',
    'from_resolved_duration_seconds',
    'to_resolved_title',
    'to_resolved_artist',
    'to_resolved_url',
    'to_resolved_duration_seconds',
    'from_uploader',
    'to_uploader',
    'from_channel',
    'to_channel',
    'from_category',
    'to_category',
    'from_tags',
    'to_tags',
    'from_description',
    'to_description',
]

ANALYSIS_REQUIRED_COLUMNS = [
    'tempo',
    'key',
    'mode',
    'intro_tempo',
    'outro_tempo',
    'intro_seconds_used',
    'outro_seconds_used',
]

ANALYSIS_REQUIRED_DEEP_COLUMNS = [
    'intro_attack_time_s',
    'outro_abruptness',
    'outro_release_time_s',
    'intro_flux_peak',
]


class SessionErrorCaptureHandler(logging.Handler):
    def __init__(self, level: int = logging.WARNING):
        super().__init__(level=level)
        self._entries: list[dict[str, str]] = []
        self._lock = Lock()

    @property
    def entries(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._entries)

    @property
    def warning_count(self) -> int:
        with self._lock:
            return sum(1 for entry in self._entries if entry.get('level') == 'WARNING')

    @property
    def error_count(self) -> int:
        with self._lock:
            return sum(1 for entry in self._entries if entry.get('level') in {'ERROR', 'CRITICAL'})

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < self.level:
            return
        try:
            timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
            entry = {
                'timestamp': timestamp,
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
            }
            with self._lock:
                self._entries.append(entry)
        except Exception:  # pragma: no cover - avoid logging recursion
            return


def _emit_progress(
    progress_callback: ProgressCallback | None,
    label: str,
    current: int,
    total: int,
    detail: str = '',
) -> None:
    if progress_callback is not None:
        progress_callback(label, int(current), int(total), str(detail or ''))


class _CapturingYtDlpLogger:
    def __init__(self, base_logger: logging.Logger) -> None:
        self._base_logger = base_logger
        self.messages: list[str] = []

    def debug(self, message: str) -> None:
        self._base_logger.debug(str(message))

    def info(self, message: str) -> None:
        self._base_logger.info(str(message))

    def warning(self, message: str) -> None:
        payload = str(message)
        self.messages.append(payload)
        self._base_logger.warning(payload)

    def error(self, message: str) -> None:
        payload = str(message)
        self.messages.append(payload)
        self._base_logger.debug(payload)


def _yt_dlp_wait_for_slot() -> None:
    while True:
        now = time.monotonic()
        with _YTDLP_GLOBAL_RATE_LIMIT_LOCK:
            global_until = _YTDLP_GLOBAL_RATE_LIMIT_UNTIL
        if now < global_until:
            time.sleep(max(0.0, global_until - now))
            continue
        break

    interval = float(YTDLP_MIN_REQUEST_INTERVAL_SECONDS or 0.0)
    if interval <= 0:
        return
    now = time.monotonic()
    global _YTDLP_NEXT_REQUEST_AT
    with _YTDLP_REQUEST_LOCK:
        scheduled_at = max(_YTDLP_NEXT_REQUEST_AT, now)
        with _YTDLP_GLOBAL_RATE_LIMIT_LOCK:
            if _YTDLP_GLOBAL_RATE_LIMIT_UNTIL > scheduled_at:
                scheduled_at = _YTDLP_GLOBAL_RATE_LIMIT_UNTIL
        _YTDLP_NEXT_REQUEST_AT = scheduled_at + interval
    sleep_for = scheduled_at - now
    if sleep_for > 0:
        time.sleep(sleep_for)


def _is_rate_limit_text(text: str) -> bool:
    lowered = str(text or '').lower()
    return 'http error 429' in lowered or 'too many requests' in lowered or 'rate limit' in lowered


def _yt_dlp_log_indicates_rate_limit(messages: Sequence[str]) -> bool:
    return any(_is_rate_limit_text(message) for message in messages)


def _yt_dlp_exception_indicates_rate_limit(exc: Exception) -> bool:
    return _is_rate_limit_text(str(exc))


def _yt_dlp_backoff_delay(attempt: int) -> float:
    base = max(float(YTDLP_RATE_LIMIT_BACKOFF_SECONDS or 0.0), 0.0)
    if base <= 0:
        return 0.0
    delay = base * (2 ** max(int(attempt), 0))
    return min(delay, float(YTDLP_RATE_LIMIT_MAX_BACKOFF_SECONDS or delay))


def _sleep_for_rate_limit(attempt: int, *, context: str, total_attempts: int) -> None:
    delay = _yt_dlp_backoff_delay(attempt)
    if delay <= 0:
        return
    if YTDLP_GLOBAL_PAUSE_ON_429:
        now = time.monotonic()
        global_until = now + delay
        with _YTDLP_GLOBAL_RATE_LIMIT_LOCK:
            global _YTDLP_GLOBAL_RATE_LIMIT_UNTIL
            if global_until > _YTDLP_GLOBAL_RATE_LIMIT_UNTIL:
                _YTDLP_GLOBAL_RATE_LIMIT_UNTIL = global_until
        with _YTDLP_REQUEST_LOCK:
            global _YTDLP_NEXT_REQUEST_AT
            if global_until > _YTDLP_NEXT_REQUEST_AT:
                _YTDLP_NEXT_REQUEST_AT = global_until
    suffix = f' ({context})' if context else ''
    logger.warning(
        'yt-dlp rate limit detected%s; backing off %.1fs before retry %d/%d',
        suffix,
        delay,
        attempt + 1,
        total_attempts,
    )
    time.sleep(delay)


def _yt_dlp_extract_info(
    url: str,
    ydl_opts: dict[str, Any],
    *,
    allow_empty: bool,
    context: str,
) -> dict[str, Any]:
    attempts = max(int(YTDLP_RATE_LIMIT_MAX_RETRIES or 1), 1)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        _yt_dlp_wait_for_slot()
        capture_logger = _CapturingYtDlpLogger(logger)
        active_opts = dict(ydl_opts or {})
        active_opts['logger'] = capture_logger
        try:
            with YoutubeDL(apply_yt_dlp_auth_options(active_opts)) as ydl:
                payload = ydl.extract_info(url, download=False) or {}
        except Exception as exc:  # pragma: no cover - depends on live network behavior
            last_exc = exc
            if _yt_dlp_exception_indicates_rate_limit(exc) and attempt < attempts - 1:
                _sleep_for_rate_limit(attempt, context=context, total_attempts=attempts)
                continue
            raise
        if payload:
            return payload
        if _yt_dlp_log_indicates_rate_limit(capture_logger.messages) and attempt < attempts - 1:
            _sleep_for_rate_limit(attempt, context=context, total_attempts=attempts)
            continue
        if allow_empty:
            return payload
        last_exc = RuntimeError(f'yt-dlp returned no metadata for {context or url}')
    if last_exc is not None:
        raise last_exc
    return {}


def _worker_count(requested_workers: int, total_items: int) -> int:
    requested = max(int(requested_workers or 1), 1)
    return max(1, min(requested, max(int(total_items), 1)))


def _bootstrap_config(argv: Sequence[str] | None = None) -> tuple[dict, str, bool]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config', default=DEFAULT_CONFIG_PATH)
    bootstrap.add_argument('--no-config', action='store_true')
    known, _ = bootstrap.parse_known_args(argv)
    config = load_project_config(known.config, use_config=not known.no_config)
    return config, str(known.config), bool(known.no_config)


def _add_bool_override(
    parser: argparse.ArgumentParser,
    *,
    true_flag: str,
    false_flag: str,
    dest: str,
    default: bool,
    true_help: str,
    false_help: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(true_flag, dest=dest, action='store_true', default=bool(default), help=true_help)
    group.add_argument(false_flag, dest=dest, action='store_false', help=false_help)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if int(denominator or 0) <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _pct_text(numerator: int, denominator: int) -> str:
    if int(denominator or 0) <= 0:
        return 'n/a'
    return f'{100.0 * _safe_ratio(numerator, denominator):.1f}%'


def _default_errors_report_path(output_csv_path: str, run_started_at: datetime) -> str:
    root, _ = os.path.splitext(output_csv_path)
    timestamp = run_started_at.strftime('%Y%m%d_%H%M%S')
    return f'{root}_errors_{timestamp}.log'


def _write_session_errors_report(
    path: str,
    *,
    entries: list[dict[str, str]],
    run_started_at: datetime,
    run_finished_at: datetime,
    run_failed: bool,
) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    lines = [
        'Mai Training Scrape Session Errors',
        f'run_started={run_started_at.strftime("%Y-%m-%d %H:%M:%S")}',
        f'run_finished={run_finished_at.strftime("%Y-%m-%d %H:%M:%S")}',
        f'run_status={"failed" if run_failed else "success"}',
        f'captured_entries={len(entries)}',
        '',
    ]
    if entries:
        lines.append('timestamp | level | logger | message')
        for entry in entries:
            lines.append(
                f"{entry.get('timestamp', '')} | {entry.get('level', '')} | {entry.get('logger', '')} | {entry.get('message', '')}"
            )
    else:
        lines.append('No warnings or errors were captured for this session.')
    lines.append('')

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def format_scrape_summary_report(
    summary: dict[str, int],
    *,
    output_path: str,
    errors_path: str,
    warning_count: int,
    error_count: int,
) -> str:
    channels_scanned = int(summary.get('channels_scanned', 0))
    channels_failed = int(summary.get('channels_failed', 0))
    videos_scanned = int(summary.get('videos_scanned', 0))
    videos_with_tracklist = int(summary.get('videos_with_tracklist', 0))
    videos_skipped = int(summary.get('videos_skipped', 0))
    tracks_parsed = int(summary.get('tracks_parsed', 0))
    tracks_resolved = int(summary.get('tracks_resolved', 0))
    tracks_unresolved = int(summary.get('tracks_unresolved', 0))
    tracks_unavailable = int(summary.get('tracks_unavailable', 0))
    tracks_analyzed = int(summary.get('tracks_analyzed', 0))
    tracks_with_features = int(summary.get('tracks_with_features', 0))
    tracks_analysis_failed = int(summary.get('tracks_analysis_failed', 0))
    positive_pairs = int(summary.get('positive_pairs', 0))
    pairs_skipped = int(summary.get('pairs_skipped', 0))
    total_pairs_considered = positive_pairs + pairs_skipped
    analysis_denominator = tracks_resolved if tracks_resolved > 0 else tracks_analyzed

    return '\n'.join([
        'Training Scrape Report',
        f'- Output CSV: {output_path}',
        f'- Session Errors: {errors_path}',
        (
            f'- Sources: channels_scanned={channels_scanned} '
            f'channels_failed={channels_failed} ({_pct_text(channels_failed, channels_scanned)})'
        ),
        (
            f'- Videos: scanned={videos_scanned} with_tracklist={videos_with_tracklist} ({_pct_text(videos_with_tracklist, videos_scanned)}) '
            f'skipped={videos_skipped} ({_pct_text(videos_skipped, videos_scanned)})'
        ),
        (
            f'- Tracks: parsed={tracks_parsed} resolved={tracks_resolved} ({_pct_text(tracks_resolved, tracks_parsed)}) '
            f'unresolved={tracks_unresolved} ({_pct_text(tracks_unresolved, tracks_parsed)}) '
            f'unavailable={tracks_unavailable} ({_pct_text(tracks_unavailable, tracks_parsed)})'
        ),
        (
            f'- Analysis: analyzed_unique={tracks_analyzed} with_features={tracks_with_features} ({_pct_text(tracks_with_features, analysis_denominator)}) '
            f'analysis_failed={tracks_analysis_failed} ({_pct_text(tracks_analysis_failed, analysis_denominator)})'
        ),
        (
            f'- Pairs: positive={positive_pairs} skipped={pairs_skipped} '
            f'kept_rate={_pct_text(positive_pairs, total_pairs_considered)}'
        ),
        f'- Logged issues: warnings={warning_count} errors={error_count}',
    ])


def _seconds_to_timestamp(total_seconds: int) -> str:
    seconds = max(int(total_seconds), 0)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remainder = seconds % 60
    if hours > 0:
        return f'{hours}:{minutes:02d}:{remainder:02d}'
    return f'{minutes:02d}:{remainder:02d}'


def timestamp_to_seconds(timestamp: str) -> int:
    parts = [int(part) for part in str(timestamp).strip().split(':')]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f'invalid timestamp: {timestamp}')


def guess_artist_title(raw_track: str) -> tuple[str, str]:
    text = normalize_track_text(raw_track)
    if not text:
        return '', ''
    matches = list(SEPARATOR_RE.finditer(text))
    if len(matches) != 1:
        return '', ''
    match = matches[0]
    title_guess = text[:match.start()].strip()
    artist_guess = text[match.end():].strip()
    if not title_guess or not artist_guess:
        return '', ''
    return artist_guess, title_guess


def normalize_track_text(text: str) -> str:
    normalized = str(text or '').strip()
    if not normalized:
        return ''
    normalized = LEADING_TIMESTAMP_TEXT_RE.sub('', normalized, count=1)
    normalized = URL_CHUNK_RE.sub('', normalized)
    normalized = EMPTY_BRACKETS_RE.sub('', normalized)
    normalized = re.sub(r'[\(\[\{]\s*(?=[-\u2013\u2014|:;,.])', '', normalized)
    normalized = re.sub(r'(?<=[-\u2013\u2014|:;,.])\s*[\)\]\}]', '', normalized)
    normalized = re.sub(r'\s{2,}', ' ', normalized)
    normalized = normalized.strip(' -\u2013\u2014|:;,.')
    normalized = re.sub(r'\s{2,}', ' ', normalized)
    return normalized.strip()


def simplify_track_search_text(text: str) -> str:
    simplified = normalize_track_text(text)
    if not simplified:
        return ''
    simplified = SEARCH_NOISE_BRACKET_RE.sub('', simplified)
    simplified = EMPTY_BRACKETS_RE.sub('', simplified)
    simplified = re.sub(r'\s{2,}', ' ', simplified)
    return simplified.strip(' -\u2013\u2014|:;,.').strip()


def _normalized_match_text(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', normalize_track_text(text)).casefold()
    normalized = TOKEN_SPLIT_RE.sub(' ', normalized)
    return ' '.join(normalized.split())


def _match_tokens(text: str) -> set[str]:
    tokens = []
    for token in _normalized_match_text(text).split():
        if token in MATCH_STOPWORDS:
            continue
        if len(token) >= 2 or not token.isascii():
            tokens.append(token)
    return set(tokens)


def build_track_search_queries(track_raw: str, artist_guess: str = '', title_guess: str = '') -> list[str]:
    del artist_guess, title_guess
    queries: list[str] = []
    for candidate in [normalize_track_text(track_raw), simplify_track_search_text(track_raw)]:
        if not candidate:
            continue
        if candidate not in queries:
            queries.append(candidate)
    return queries


def _expected_track_parts(track_raw: str) -> tuple[str, str]:
    cleaned = normalize_track_text(track_raw)
    matches = list(SEPARATOR_RE.finditer(cleaned))
    if len(matches) != 1:
        return '', ''
    match = matches[0]
    left = cleaned[:match.start()].strip()
    right = cleaned[match.end():].strip()
    if not left or not right:
        return '', ''
    return left, right


def _token_overlap(expected: set[str], actual: set[str]) -> float:
    if not expected:
        return 0.0
    return float(len(expected & actual)) / float(len(expected))


def _candidate_match_score(
    candidate: dict[str, Any],
    *,
    track_raw: str,
    query_variants: Optional[Sequence[str]] = None,
) -> float:
    candidate_title = str(candidate.get('title') or '').strip()
    candidate_artist = str(candidate.get('uploader') or candidate.get('channel') or '').strip()
    title_tokens = _match_tokens(candidate_title)
    artist_tokens = _match_tokens(candidate_artist)
    combined_tokens = title_tokens | artist_tokens

    cleaned_track = normalize_track_text(track_raw)
    query_variants = list(query_variants or [])
    if cleaned_track and cleaned_track not in query_variants:
        query_variants.insert(0, cleaned_track)
    expected_tokens = set()
    for query in query_variants:
        expected_tokens |= _match_tokens(query)

    combined_overlap = _token_overlap(expected_tokens, combined_tokens)
    title_overlap = _token_overlap(expected_tokens, title_tokens)
    left, right = _expected_track_parts(cleaned_track)
    orientation_score = 0.0
    if left and right:
        left_tokens = _match_tokens(left)
        right_tokens = _match_tokens(right)
        orientation_score = max(
            0.65 * _token_overlap(left_tokens, artist_tokens) + 0.35 * _token_overlap(right_tokens, title_tokens),
            0.65 * _token_overlap(right_tokens, artist_tokens) + 0.35 * _token_overlap(left_tokens, title_tokens),
            0.5 * _token_overlap(left_tokens, combined_tokens) + 0.5 * _token_overlap(right_tokens, combined_tokens),
        )

    candidate_text = f'{candidate_artist} {candidate_title}'.strip()
    normalized_candidate_text = _normalized_match_text(candidate_text)
    exact_bonus = 0.0
    for query in query_variants:
        normalized_query = _normalized_match_text(query)
        if normalized_query and normalized_query in normalized_candidate_text:
            exact_bonus = max(exact_bonus, 0.6)
        elif normalized_query and all(token in normalized_candidate_text for token in _match_tokens(query)):
            exact_bonus = max(exact_bonus, 0.3)

    duration_seconds = _candidate_duration_seconds(candidate)
    short_bonus = 0.15 if duration_seconds is not None and duration_seconds <= 10 * 60 else 0.0
    penalty = 0.0
    if NON_TRACK_CANDIDATE_RE.search(candidate_title) and combined_overlap < 0.85:
        penalty += 0.7
    if duration_seconds is None:
        penalty += 0.05

    return 1.1 * combined_overlap + 0.9 * title_overlap + 1.2 * orientation_score + exact_bonus + short_bonus - penalty


def looks_like_non_track_text(text: str) -> bool:
    normalized = normalize_track_text(text)
    if not normalized:
        return True
    return len(normalized) < 2


def parse_tracklist_description(description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in str(description or '').splitlines():
        match = TIMESTAMP_LINE_RE.match(line)
        if not match:
            continue
        timestamp = match.group(1).strip()
        track_raw = normalize_track_text(match.group(2))
        if not track_raw:
            continue
        if looks_like_non_track_text(track_raw):
            logger.info('Skipping malformed description line: %s', match.group(2).strip())
            continue
        artist_guess, title_guess = guess_artist_title(track_raw)
        rows.append({
            'position': len(rows) + 1,
            'timestamp': timestamp,
            'timestamp_s': timestamp_to_seconds(timestamp),
            'track_raw': track_raw,
            'artist_guess': artist_guess,
            'title_guess': title_guess,
            'track_source': 'description',
            'chapter_title': '',
            'chapter_timestamp_s': pd.NA,
        })
    return rows


def parse_tracklist_chapters(chapters: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chapter in chapters or []:
        if not isinstance(chapter, dict):
            continue
        title = normalize_track_text(chapter.get('title') or '')
        start_time = chapter.get('start_time')
        if not title or start_time is None:
            continue
        timestamp_s = int(round(float(start_time)))
        artist_guess, title_guess = guess_artist_title(title)
        rows.append({
            'position': len(rows) + 1,
            'timestamp': _seconds_to_timestamp(timestamp_s),
            'timestamp_s': timestamp_s,
            'track_raw': title,
            'artist_guess': artist_guess,
            'title_guess': title_guess,
            'track_source': 'chapters',
            'chapter_title': title,
            'chapter_timestamp_s': timestamp_s,
        })
    return rows


def _metadata_timestamp_to_seconds(value: Any) -> Optional[int]:
    if value is None or value == '':
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if ':' in text:
            try:
                return timestamp_to_seconds(text)
            except ValueError:
                return None
        try:
            return int(round(float(text)))
        except ValueError:
            return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _item_timestamp_seconds(item: dict[str, Any]) -> Optional[int]:
    if 'start_time' in item:
        return _metadata_timestamp_to_seconds(item.get('start_time'))
    if 'start_time_ms' in item:
        try:
            return int(round(float(item.get('start_time_ms')) / 1000.0))
        except (TypeError, ValueError):
            return None
    if 'timestamp' in item:
        return _metadata_timestamp_to_seconds(item.get('timestamp'))
    return _metadata_timestamp_to_seconds(item.get('time_text'))


def _metadata_track_label(item: dict[str, Any]) -> str:
    title = str(item.get('title') or item.get('track') or item.get('name') or '').strip()
    artists = item.get('artists')
    if isinstance(artists, list):
        artist_names = [str(artist.get('name') if isinstance(artist, dict) else artist).strip() for artist in artists]
        artist_names = [name for name in artist_names if name]
    else:
        artist_names = []
    artist_text = ', '.join(artist_names) if artist_names else str(item.get('artist') or '').strip()
    if title:
        if artist_text and artist_text.casefold() not in title.casefold():
            return f'{title} - {artist_text}'
        return title
    track_name = str(item.get('song') or '').strip()
    if track_name and artist_text:
        return f'{track_name} - {artist_text}'
    if track_name:
        return track_name
    return artist_text


def parse_tracklist_watch_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    chapter_rows = parse_tracklist_chapters(metadata.get('chapters'))
    if chapter_rows:
        return chapter_rows

    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, str]] = set()
    for key in WATCH_METADATA_LIST_KEYS:
        items = metadata.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            timestamp_s = _item_timestamp_seconds(item)
            track_raw = normalize_track_text(_metadata_track_label(item))
            if timestamp_s is None or not track_raw:
                continue
            dedupe_key = (timestamp_s, track_raw.casefold())
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            artist_guess, title_guess = guess_artist_title(track_raw)
            rows.append({
                'position': len(rows) + 1,
                'timestamp': _seconds_to_timestamp(timestamp_s),
                'timestamp_s': timestamp_s,
                'track_raw': track_raw,
                'artist_guess': artist_guess,
                'title_guess': title_guess,
                'track_source': 'chapters',
                'chapter_title': track_raw,
                'chapter_timestamp_s': timestamp_s,
            })
    return rows


def channel_handle_from_url(channel_url: str) -> str:
    match = re.search(r'youtube\.com/(@[^/?#]+)', str(channel_url or ''))
    return match.group(1) if match else ''


def normalize_video_url(video_id: str, video_url: str = '') -> str:
    normalized = str(video_url or '').strip()
    if normalized.startswith('http://') or normalized.startswith('https://'):
        return normalized
    return f'https://www.youtube.com/watch?v={video_id}'


def extract_youtube_video_id(value: str) -> str:
    normalized = str(value or '').strip()
    if _looks_like_youtube_video_id(normalized):
        return normalized
    patterns = [
        r'[?&]v=([^&#?/]+)',
        r'youtu\.be/([^&#?/]+)',
        r'/shorts/([^&#?/]+)',
        r'/live/([^&#?/]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)
    return ''


def _looks_like_youtube_video_id(video_id: str) -> bool:
    return bool(re.fullmatch(r'[A-Za-z0-9_-]{11}', str(video_id or '').strip()))


def build_track_search_query(track_raw: str, artist_guess: str = '', title_guess: str = '') -> str:
    queries = build_track_search_queries(track_raw, artist_guess=artist_guess, title_guess=title_guess)
    return queries[0] if queries else ''


def _training_cache_root(cache_dir: str) -> str:
    return os.path.join(str(cache_dir), 'training')


def _slugify(value: str) -> str:
    text = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value or '').strip())
    return text.strip('_') or 'cache'


def _hash_text(value: str) -> str:
    return hashlib.sha1(str(value).encode('utf-8')).hexdigest()


def _source_track_signature(
    channel_url: str,
    label: str,
    label_source: str,
    *,
    cache_version: int = SOURCE_TRACK_CACHE_VERSION,
) -> str:
    payload = '||'.join([
        str(cache_version),
        str(channel_url),
        str(label),
        str(label_source),
    ])
    return f'v{int(cache_version)}|{_hash_text(payload)[:16]}'


def _source_track_signature_from_row(row: pd.Series | dict[str, Any]) -> str:
    signature = str(row.get('source_signature') or '').strip()
    if signature:
        return signature
    try:
        cache_version = int(float(row.get('source_cache_version')))
    except (TypeError, ValueError):
        cache_version = int(SOURCE_TRACK_CACHE_VERSION)
    return _source_track_signature(
        str(row.get('channel_url') or ''),
        str(row.get('label') or ''),
        str(row.get('label_source') or ''),
        cache_version=cache_version,
    )


def _resolution_signature(
    normalized_query: str,
    max_results: int,
    *,
    cache_version: int = RESOLUTION_CACHE_VERSION,
) -> str:
    payload = '||'.join([
        str(cache_version),
        str(max(1, int(max_results))),
        str(normalized_query or ''),
    ])
    return f'v{int(cache_version)}|{_hash_text(payload)[:16]}'


def _resolution_signature_from_row(row: pd.Series | dict[str, Any]) -> str:
    signature = str(row.get('resolution_signature') or '').strip()
    if signature:
        return signature
    normalized_query = str(row.get('normalized_search_query') or '').strip()
    if not normalized_query:
        normalized_query = _normalize_search_query(str(row.get('search_query') or ''))
    try:
        max_results = int(float(row.get('search_max_results')))
    except (TypeError, ValueError):
        max_results = int(MAX_SEARCH_RESULTS)
    try:
        cache_version = int(float(row.get('resolution_cache_version')))
    except (TypeError, ValueError):
        cache_version = int(RESOLUTION_CACHE_VERSION)
    return _resolution_signature(
        normalized_query=normalized_query,
        max_results=max_results,
        cache_version=cache_version,
    )


def _compact_channel_video_entries(entries: Sequence[Any]) -> list[dict[str, str]]:
    compacted: list[dict[str, str]] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        video_id = str(entry.get('id') or '').strip()
        if not video_id:
            continue
        compacted.append({
            'id': video_id,
            'title': str(entry.get('title') or '').strip(),
        })
    return compacted


def _compact_search_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': str(candidate.get('id') or '').strip(),
        'title': str(candidate.get('title') or '').strip(),
        'uploader': str(candidate.get('uploader') or '').strip(),
        'channel': str(candidate.get('channel') or '').strip(),
        'duration': candidate.get('duration'),
        'availability': str(candidate.get('availability') or '').strip(),
    }


def _compact_search_entries(entries: Sequence[Any]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        compacted.append(_compact_search_candidate(entry))
    return compacted


def _compact_metadata_artist_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get('name') or item.get('title') or item.get('label') or '').strip()
        else:
            text = str(item or '').strip()
        if text:
            names.append(text)
    return names


def _compact_metadata_track_items(items: Any) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        compacted_item = {
            'title': str(item.get('title') or '').strip(),
            'track': str(item.get('track') or '').strip(),
            'name': str(item.get('name') or '').strip(),
            'song': str(item.get('song') or '').strip(),
            'artist': str(item.get('artist') or '').strip(),
            'artists': _compact_metadata_artist_list(item.get('artists')),
            'start_time': item.get('start_time'),
            'start_time_ms': item.get('start_time_ms'),
            'timestamp': item.get('timestamp'),
            'time_text': item.get('time_text'),
        }
        compacted.append({key: value for key, value in compacted_item.items() if value not in (None, '', [])})
    return compacted


def _compact_chapters(chapters: Any) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for chapter in chapters or []:
        if not isinstance(chapter, dict):
            continue
        title = str(chapter.get('title') or '').strip()
        start_time = chapter.get('start_time')
        if start_time in (None, '') and not title:
            continue
        compacted.append({
            key: value
            for key, value in {
                'title': title,
                'start_time': start_time,
            }.items()
            if value not in (None, '')
        })
    return compacted


def _compact_video_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    compacted = {
        'title': str(metadata.get('title') or '').strip(),
        'description': str(metadata.get('description') or '').strip(),
        'uploader_id': str(metadata.get('uploader_id') or '').strip(),
        'channel_id': str(metadata.get('channel_id') or '').strip(),
        'artist': str(metadata.get('artist') or '').strip(),
        'uploader': str(metadata.get('uploader') or '').strip(),
        'channel': str(metadata.get('channel') or '').strip(),
        'tags': [str(tag).strip() for tag in (metadata.get('tags') or []) if str(tag).strip()],
        'categories': [str(category).strip() for category in (metadata.get('categories') or []) if str(category).strip()],
        'chapters': _compact_chapters(metadata.get('chapters')),
        'music_tracks': _compact_metadata_track_items(metadata.get('music_tracks')),
        'tracks': _compact_metadata_track_items(metadata.get('tracks')),
        'tracklist': _compact_metadata_track_items(metadata.get('tracklist')),
        'music_sections': _compact_metadata_track_items(metadata.get('music_sections')),
    }
    category_text = _metadata_value_to_text(metadata.get('category'))
    if category_text and not compacted['categories']:
        compacted['categories'] = [category_text]
    return {key: value for key, value in compacted.items() if value not in (None, '', [], {})}


def _compact_source_track_cache_rows(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame(columns=SOURCE_TRACK_CACHE_COLUMNS)
    prepared = rows_df.copy()
    if 'source_signature' not in prepared.columns:
        prepared['source_signature'] = ''
    missing_signature = prepared['source_signature'].fillna('').astype(str).str.strip().eq('')
    if missing_signature.any():
        prepared.loc[missing_signature, 'source_signature'] = prepared.loc[missing_signature].apply(
            _source_track_signature_from_row,
            axis=1,
        )
    if 'timestamp_s' not in prepared.columns and 'timestamp' in prepared.columns:
        prepared['timestamp_s'] = prepared['timestamp'].map(timestamp_to_seconds)
    if 'channel_handle' not in prepared.columns:
        prepared['channel_handle'] = ''
    if 'description_length' not in prepared.columns:
        prepared['description_length'] = pd.NA
    if 'track_source' not in prepared.columns:
        prepared['track_source'] = ''
    if 'chapter_title' not in prepared.columns:
        prepared['chapter_title'] = ''
    if 'chapter_timestamp_s' not in prepared.columns:
        prepared['chapter_timestamp_s'] = pd.NA
    if 'artist_guess' not in prepared.columns:
        prepared['artist_guess'] = ''
    if 'title_guess' not in prepared.columns:
        prepared['title_guess'] = ''
    compacted = prepared.reindex(columns=SOURCE_TRACK_CACHE_COLUMNS)
    return compacted.sort_values(['video_id', 'position'], kind='stable').reset_index(drop=True)


def _restore_source_track_cache_rows(
    cached_rows_df: pd.DataFrame,
    *,
    channel_config: dict[str, str],
    video: dict[str, str],
) -> pd.DataFrame:
    if cached_rows_df.empty:
        return pd.DataFrame(columns=SOURCE_TRACK_COLUMNS)
    compacted = _compact_source_track_cache_rows(cached_rows_df)
    video_id = str(video.get('video_id') or '').strip()
    video_title = str(video.get('video_title') or '').strip()
    video_url = normalize_video_url(video_id, str(video.get('video_url') or '').strip())
    restored = compacted.copy()
    restored['channel_url'] = str(channel_config['url'])
    restored['label'] = str(channel_config['label'])
    restored['label_source'] = str(channel_config['label_source'])
    restored['source_cache_version'] = int(SOURCE_TRACK_CACHE_VERSION)
    restored['video_title'] = video_title
    restored['video_url'] = video_url
    restored['timestamp'] = restored['timestamp_s'].map(
        lambda value: '' if pd.isna(value) or value == '' else _seconds_to_timestamp(int(value))
    )
    return restored.reindex(columns=SOURCE_TRACK_COLUMNS).sort_values(['video_id', 'position'], kind='stable').reset_index(drop=True)


def _compact_resolution_cache_rows(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame(columns=RESOLUTION_CACHE_COLUMNS)
    prepared = rows_df.copy()
    if 'resolution_signature' not in prepared.columns:
        prepared['resolution_signature'] = ''
    missing_signature = prepared['resolution_signature'].fillna('').astype(str).str.strip().eq('')
    if missing_signature.any():
        prepared.loc[missing_signature, 'resolution_signature'] = prepared.loc[missing_signature].apply(
            _resolution_signature_from_row,
            axis=1,
        )
    if 'resolution_status' not in prepared.columns:
        prepared['resolution_status'] = ''
    if 'resolved_video_id' not in prepared.columns:
        prepared['resolved_video_id'] = ''
    if 'resolved_title' not in prepared.columns:
        prepared['resolved_title'] = ''
    if 'resolved_artist' not in prepared.columns:
        prepared['resolved_artist'] = ''
    if 'resolved_duration_seconds' not in prepared.columns:
        prepared['resolved_duration_seconds'] = pd.NA
    compacted = prepared.reindex(columns=RESOLUTION_CACHE_COLUMNS)
    return compacted.sort_values(['video_id', 'position'], kind='stable').reset_index(drop=True)


def _restore_resolution_cache_row(
    resolution_row: dict[str, Any],
    *,
    query: str,
    normalized_query: str,
    max_results: int,
) -> dict[str, Any]:
    resolved_video_id = str(resolution_row.get('resolved_video_id') or '').strip()
    return {
        'video_id': str(resolution_row.get('video_id') or '').strip(),
        'position': resolution_row.get('position', pd.NA),
        'search_query': str(query).strip(),
        'normalized_search_query': str(normalized_query).strip(),
        'search_max_results': int(max(1, int(max_results))),
        'resolution_cache_version': int(RESOLUTION_CACHE_VERSION),
        'resolution_status': str(resolution_row.get('resolution_status') or '').strip(),
        'resolved_video_id': resolved_video_id,
        'resolved_title': str(resolution_row.get('resolved_title') or '').strip(),
        'resolved_artist': str(resolution_row.get('resolved_artist') or '').strip(),
        'resolved_url': normalize_video_url(resolved_video_id) if resolved_video_id else '',
        'resolved_duration_seconds': resolution_row.get('resolved_duration_seconds', pd.NA),
    }


def _channel_cache_path(cache_dir: str, channel_url: str) -> str:
    slug = channel_handle_from_url(channel_url) or _slugify(channel_url)
    return os.path.join(_training_cache_root(cache_dir), 'channel_videos', f'{slug}_{_hash_text(channel_url)[:10]}.json')


def _video_metadata_cache_path(cache_dir: str, video_id: str) -> str:
    return os.path.join(_training_cache_root(cache_dir), 'video_metadata', f'{video_id}.json')


def _search_cache_path(cache_dir: str, normalized_query: str, max_results: int) -> str:
    cache_key = f'{max(1, int(max_results))}||{normalized_query}'
    return os.path.join(_training_cache_root(cache_dir), 'search_results', f'{_hash_text(cache_key)}.json')


def _source_track_cache_path(cache_dir: str) -> str:
    return os.path.join(_training_cache_root(cache_dir), 'source_tracks.sqlite')


def _resolution_cache_path(cache_dir: str) -> str:
    return os.path.join(_training_cache_root(cache_dir), 'track_resolutions.sqlite')


def _ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    directory = os.path.dirname(path) or '.'
    fd, temp_path = tempfile.mkstemp(prefix='mai_cache_', suffix='.tmp', dir=directory)
    os.close(fd)
    try:
        with open(temp_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _read_json(path: str) -> Optional[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _read_cache_table(path: str, columns: list[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=columns)
    if str(path).lower().endswith(('.sqlite', '.db')):
        sqlite_path, legacy_csv_path = resolve_sqlite_cache_path(path, default_path=path)
        if os.path.exists(sqlite_path):
            df = read_sqlite_table(sqlite_path, columns=columns)
            if df.empty:
                return pd.DataFrame(columns=columns)
            return df
        if legacy_csv_path and os.path.exists(legacy_csv_path):
            return _read_cache_table(legacy_csv_path, columns)
        return pd.DataFrame(columns=columns)
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    if df.empty:
        return pd.DataFrame(columns=columns)
    ordered_columns = [column for column in columns if column in df.columns]
    ordered_columns.extend(sorted(column for column in df.columns if column not in ordered_columns))
    return df.reindex(columns=ordered_columns)


def _write_cache_table(path: str, df: pd.DataFrame, columns: list[str], *, key_columns: Optional[list[str]] = None) -> None:
    ordered_columns = [column for column in columns if column in df.columns]
    ordered_columns.extend(sorted(column for column in df.columns if column not in ordered_columns))
    prepared = df.reindex(columns=ordered_columns) if ordered_columns else df.copy()
    if str(path).lower().endswith(('.sqlite', '.db')):
        write_sqlite_table(
            path,
            prepared,
            columns=prepared.columns.tolist() or columns,
            key_columns=key_columns or [],
        )
        return
    _ensure_parent_dir(path)
    directory = os.path.dirname(path) or '.'
    fd, temp_path = tempfile.mkstemp(prefix='mai_cache_', suffix='.tmp', dir=directory)
    os.close(fd)
    try:
        prepared.to_csv(temp_path, index=False, encoding='utf-8')
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def compact_training_cache(cache_dir: str = 'data/cache') -> dict[str, int]:
    training_root = _training_cache_root(cache_dir)
    summary = {
        'channel_video_files': 0,
        'search_result_files': 0,
        'video_metadata_files': 0,
        'source_track_rows': 0,
        'resolution_rows': 0,
        'bytes_before': 0,
        'bytes_after': 0,
    }
    if not os.path.isdir(training_root):
        return summary

    source_track_cache_path = _source_track_cache_path(cache_dir)
    source_track_sqlite_path, source_track_legacy_csv_path = resolve_sqlite_cache_path(source_track_cache_path, default_path=source_track_cache_path)
    if os.path.exists(source_track_sqlite_path) or os.path.exists(source_track_legacy_csv_path):
        bytes_before_path = source_track_sqlite_path if os.path.exists(source_track_sqlite_path) else source_track_legacy_csv_path
        summary['bytes_before'] += int(os.path.getsize(bytes_before_path))
        source_track_cache_df = _compact_source_track_cache_rows(
            _read_cache_table(source_track_cache_path, SOURCE_TRACK_CACHE_COLUMNS)
        )
        _write_cache_table(
            source_track_cache_path,
            source_track_cache_df,
            SOURCE_TRACK_CACHE_COLUMNS,
            key_columns=['video_id', 'position'],
        )
        summary['bytes_after'] += int(os.path.getsize(source_track_cache_path))
        summary['source_track_rows'] = int(len(source_track_cache_df))

    resolution_cache_path = _resolution_cache_path(cache_dir)
    resolution_sqlite_path, resolution_legacy_csv_path = resolve_sqlite_cache_path(resolution_cache_path, default_path=resolution_cache_path)
    if os.path.exists(resolution_sqlite_path) or os.path.exists(resolution_legacy_csv_path):
        bytes_before_path = resolution_sqlite_path if os.path.exists(resolution_sqlite_path) else resolution_legacy_csv_path
        summary['bytes_before'] += int(os.path.getsize(bytes_before_path))
        resolution_cache_df = _compact_resolution_cache_rows(
            _read_cache_table(resolution_cache_path, RESOLUTION_CACHE_COLUMNS)
        )
        _write_cache_table(
            resolution_cache_path,
            resolution_cache_df,
            RESOLUTION_CACHE_COLUMNS,
            key_columns=['video_id', 'position'],
        )
        summary['bytes_after'] += int(os.path.getsize(resolution_cache_path))
        summary['resolution_rows'] = int(len(resolution_cache_df))

    for directory_name, payload_builder, counter_key in (
        ('channel_videos', lambda payload: {'entries': _compact_channel_video_entries(payload.get('entries') or [])}, 'channel_video_files'),
        ('search_results', lambda payload: {'entries': _compact_search_entries(payload.get('entries') or [])}, 'search_result_files'),
        ('video_metadata', _compact_video_metadata, 'video_metadata_files'),
    ):
        directory = os.path.join(training_root, directory_name)
        if not os.path.isdir(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith('.json'):
                continue
            path = os.path.join(directory, filename)
            payload = _read_json(path)
            if payload is None:
                continue
            summary['bytes_before'] += int(os.path.getsize(path))
            compacted_payload = payload_builder(payload)
            _write_json_atomic(path, compacted_payload)
            summary['bytes_after'] += int(os.path.getsize(path))
            summary[counter_key] += 1

    return summary


def _source_track_rows_are_valid(cached_rows_df: pd.DataFrame, channel_config: dict[str, str]) -> bool:
    if cached_rows_df.empty:
        return False
    if 'source_signature' in cached_rows_df.columns:
        signature_values = cached_rows_df['source_signature'].fillna('').astype(str).str.strip()
        expected_signature = _source_track_signature(
            str(channel_config['url']),
            str(channel_config['label']),
            str(channel_config['label_source']),
        )
        return bool(not signature_values.empty and signature_values.eq(expected_signature).all())
    if 'source_cache_version' not in cached_rows_df.columns:
        return False

    version_values = cached_rows_df['source_cache_version'].dropna().astype(str).str.strip()
    if version_values.empty or any(value != str(SOURCE_TRACK_CACHE_VERSION) for value in version_values):
        return False

    for column, expected in (
        ('channel_url', str(channel_config['url'])),
        ('label', str(channel_config['label'])),
        ('label_source', str(channel_config['label_source'])),
    ):
        if column not in cached_rows_df.columns:
            return False
        actual_values = cached_rows_df[column].fillna('').astype(str).str.strip()
        if any(value != expected for value in actual_values):
            return False
    return True


def _resolution_row_is_valid(
    resolution_row: dict[str, Any],
    *,
    query: str,
    normalized_query: str,
    max_results: int,
) -> bool:
    cached_signature = str(resolution_row.get('resolution_signature') or '').strip()
    if cached_signature:
        return cached_signature == _resolution_signature(
            normalized_query=normalized_query,
            max_results=max_results,
        )

    try:
        cache_version = int(resolution_row.get('resolution_cache_version'))
        cached_max_results = int(resolution_row.get('search_max_results'))
    except (TypeError, ValueError):
        return False

    if cache_version != RESOLUTION_CACHE_VERSION:
        return False
    if cached_max_results != max(1, int(max_results)):
        return False
    if str(resolution_row.get('search_query') or '').strip() != str(query).strip():
        return False
    if str(resolution_row.get('normalized_search_query') or '').strip() != str(normalized_query).strip():
        return False
    return True


def _metadata_value_to_text(value: Any) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = str(item.get('name') or item.get('title') or item.get('label') or '').strip()
            else:
                text = str(item).strip()
            if text:
                parts.append(text)
        return ', '.join(parts)
    if isinstance(value, dict):
        return str(value.get('name') or value.get('title') or value.get('label') or '').strip()
    if value is None or pd.isna(value):
        return ''
    return str(value).strip()


def _replace_source_track_rows(cache_df: pd.DataFrame, video_id: str, rows_df: pd.DataFrame) -> pd.DataFrame:
    prepared = _compact_source_track_cache_rows(cache_df)
    if not prepared.empty and 'video_id' in prepared.columns:
        prepared = prepared.loc[prepared['video_id'].fillna('').astype(str) != str(video_id)].copy()
    if rows_df.empty:
        return prepared.reset_index(drop=True)
    combined = pd.concat([prepared, _compact_source_track_cache_rows(rows_df)], ignore_index=True, sort=False)
    return combined.sort_values(['video_id', 'position'], kind='stable').reset_index(drop=True)


def _upsert_resolution_rows(cache_df: pd.DataFrame, rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return _compact_resolution_cache_rows(cache_df)
    prepared = _compact_resolution_cache_rows(cache_df)
    new_rows_df = _compact_resolution_cache_rows(rows_df)
    if not prepared.empty:
        existing_keys = (
            prepared[['video_id', 'position']]
            .astype({'video_id': 'string', 'position': 'string'})
            .fillna('')
            .agg('||'.join, axis=1)
        )
        new_keys = set(
            new_rows_df[['video_id', 'position']]
            .astype({'video_id': 'string', 'position': 'string'})
            .fillna('')
            .agg('||'.join, axis=1)
            .tolist()
        )
        prepared = prepared.loc[~existing_keys.isin(new_keys)].copy()
    combined = pd.concat([prepared, new_rows_df.reindex(columns=RESOLUTION_CACHE_COLUMNS)], ignore_index=True, sort=False)
    return combined.sort_values(['video_id', 'position'], kind='stable').reset_index(drop=True)


def _normalize_search_query(query: str) -> str:
    return ' '.join(str(query or '').strip().lower().split())


def _extract_search_entries(info: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for entry in info.get('entries') or []:
        if isinstance(entry, dict):
            entries.append(_compact_search_candidate(entry))
    return entries


def _search_youtube_track_candidates_once(
    query: str,
    max_results: int,
    extract_flat: str | None = None,
) -> list[dict[str, Any]]:
    search_query = f'ytsearch{max(1, int(max_results))}:{query}'
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'socket_timeout': YTDLP_SOCKET_TIMEOUT_SECONDS,
        'extractor_retries': YTDLP_EXTRACTOR_RETRIES,
        'extractor_args': {'youtube': {'player_skip': ['js']}},
    }
    if extract_flat is not None:
        ydl_opts['extract_flat'] = extract_flat
    info = _yt_dlp_extract_info(
        search_query,
        ydl_opts,
        allow_empty=True,
        context=f'search "{query}"',
    )
    return _extract_search_entries(info)


def search_youtube_track_candidates(
    query: str,
    max_results: int = MAX_SEARCH_RESULTS,
    cache_dir: Optional[str] = None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    normalized_query = _normalize_search_query(query)
    normalized_max_results = max(1, int(max_results))
    cache_path = _search_cache_path(cache_dir, normalized_query, normalized_max_results) if cache_dir else None
    if cache_path and not refresh:
        cached = _read_json(cache_path)
        if cached is not None:
            entries = _extract_search_entries(cached)
            logger.info('YouTube search cache hit [%d]: %s -> %d candidates', normalized_max_results, query, len(entries))
            return entries

    entries: list[dict[str, Any]] = []
    error_message = ''
    logger.info('YouTube search live [%d]: %s', normalized_max_results, query)
    try:
        entries = _search_youtube_track_candidates_once(query, max_results=max_results)
    except Exception as exc:  # pragma: no cover - depends on live yt-dlp behavior
        logger.warning('Track search failed for %r; retrying with flat search results: %s', query, exc)
        error_message = str(exc)
        entries = []
    if not entries:
        try:
            entries = _search_youtube_track_candidates_once(
                query,
                max_results=max_results,
                extract_flat='in_playlist',
            )
        except Exception as exc:  # pragma: no cover - depends on live yt-dlp behavior
            logger.warning('Track search failed for %r even with flat fallback: %s', query, exc)
            if not error_message:
                error_message = str(exc)
            entries = []

    if cache_path:
        _write_json_atomic(
            cache_path,
            {
                'entries': _compact_search_entries(entries),
            },
        )
    logger.info('YouTube search result [%d]: %s -> %d candidates', normalized_max_results, query, len(entries))
    return entries


def _merge_candidate_results(
    normalized_query_variants: Sequence[str],
    search_results: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for normalized_query in normalized_query_variants:
        for candidate in search_results.get(str(normalized_query), []):
            candidate_id = str(candidate.get('id') or '').strip()
            fallback_key = '||'.join(
                [
                    candidate_id,
                    str(candidate.get('title') or '').strip(),
                    str(candidate.get('uploader') or candidate.get('channel') or '').strip(),
                    str(candidate.get('webpage_url') or candidate.get('url') or '').strip(),
                ]
            )
            candidate_key = candidate_id or fallback_key
            if candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)
            merged.append(candidate)
    return merged


def _candidate_duration_seconds(candidate: dict[str, Any]) -> Optional[int]:
    duration = candidate.get('duration')
    if duration is None:
        return None
    try:
        return int(duration)
    except (TypeError, ValueError):
        return None


def _select_track_candidate_with_status(
    candidates: list[dict[str, Any]],
    excluded_video_ids: Optional[set[str]] = None,
    *,
    track_raw: str = '',
    query_variants: Optional[Sequence[str]] = None,
) -> tuple[str, Optional[dict[str, Any]]]:
    excluded = {str(video_id) for video_id in (excluded_video_ids or set()) if str(video_id)}
    filtered: list[dict[str, Any]] = []
    saw_unavailable = False
    for candidate in candidates:
        candidate_id = str(candidate.get('id') or '').strip()
        candidate_title = str(candidate.get('title') or '').strip().lower()
        candidate_availability = str(candidate.get('availability') or '').strip().lower()
        if not candidate_id or candidate_id in excluded:
            continue
        if candidate_title in UNAVAILABLE_TITLES or candidate_availability in {'private', 'unavailable'}:
            saw_unavailable = True
            continue
        filtered.append(candidate)
    if not filtered:
        return ('unavailable' if saw_unavailable else 'no_match', None)

    cleaned_track = normalize_track_text(track_raw)
    query_variants = [variant for variant in (query_variants or []) if str(variant).strip()]
    if cleaned_track and cleaned_track not in query_variants:
        query_variants = [cleaned_track] + query_variants
    expected_tokens = set()
    for variant in query_variants:
        expected_tokens |= _match_tokens(variant)

    ranked_candidates: list[tuple[float, float, int, dict[str, Any]]] = []
    if expected_tokens:
        for candidate in filtered:
            duration_seconds = _candidate_duration_seconds(candidate)
            duration_rank = duration_seconds if duration_seconds is not None else MAX_SEARCH_DURATION_SECONDS + 1
            score = _candidate_match_score(
                candidate,
                track_raw=cleaned_track,
                query_variants=query_variants,
            )
            candidate_title = str(candidate.get('title') or '').strip()
            candidate_artist = str(candidate.get('uploader') or candidate.get('channel') or '').strip()
            overlap = _token_overlap(expected_tokens, _match_tokens(f'{candidate_artist} {candidate_title}'))
            ranked_candidates.append((score, overlap, duration_rank, candidate))
        ranked_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        best_score, best_overlap, _, selected = ranked_candidates[0]
        if best_score < 1.05 or (best_overlap < 0.5 and best_score < 1.45):
            return ('no_match', None)
    else:
        short_form = []
        for candidate in filtered:
            duration_seconds = _candidate_duration_seconds(candidate)
            if duration_seconds is not None and duration_seconds <= MAX_SEARCH_DURATION_SECONDS:
                short_form.append(candidate)
        selected = short_form[0] if short_form else filtered[0]
    resolved_video_id = str(selected.get('id') or '').strip()
    return 'resolved', {
        'resolved_video_id': resolved_video_id,
        'resolved_title': str(selected.get('title') or '').strip(),
        'resolved_artist': str(selected.get('uploader') or selected.get('channel') or '').strip(),
        'resolved_url': normalize_video_url(
            resolved_video_id,
            str(selected.get('webpage_url') or selected.get('url') or '').strip(),
        ),
        'resolved_duration_seconds': _candidate_duration_seconds(selected),
    }


def select_track_candidate(
    candidates: list[dict[str, Any]],
    excluded_video_ids: Optional[set[str]] = None,
    *,
    track_raw: str = '',
    query_variants: Optional[Sequence[str]] = None,
) -> Optional[dict[str, Any]]:
    _, selected = _select_track_candidate_with_status(
        candidates,
        excluded_video_ids=excluded_video_ids,
        track_raw=track_raw,
        query_variants=query_variants,
    )
    return selected


def source_tracks_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=SOURCE_TRACK_COLUMNS)
    df = pd.DataFrame(rows)
    return df.reindex(columns=SOURCE_TRACK_COLUMNS).sort_values(['video_id', 'position'], kind='stable').reset_index(drop=True)


def scraped_tracks_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=TRACK_COLUMNS)
    df = pd.DataFrame(rows)
    return df.reindex(columns=TRACK_COLUMNS).sort_values(['video_id', 'position'], kind='stable').reset_index(drop=True)


def resolution_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=RESOLUTION_COLUMNS)
    df = pd.DataFrame(rows)
    return df.reindex(columns=RESOLUTION_COLUMNS).sort_values(['video_id', 'position'], kind='stable').reset_index(drop=True)


def _normalize_channel_config(channel: dict[str, Any], default_label: str, default_label_source: str) -> dict[str, str]:
    url = str(channel.get('url') or '').strip()
    if not url:
        raise ValueError('training channel config is missing `url`')
    return {
        'source_type': 'channel',
        'name': str(channel.get('name') or channel_handle_from_url(url) or url).strip(),
        'url': url,
        'label': str(channel.get('label') or default_label).strip() or DEFAULT_LABEL,
        'label_source': str(channel.get('label_source') or default_label_source).strip() or DEFAULT_LABEL_SOURCE,
    }


def _normalize_training_video_config(video: dict[str, Any], default_label: str, default_label_source: str) -> dict[str, str]:
    url = str(video.get('url') or '').strip()
    if not url:
        raise ValueError('training video config is missing `url`')
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError(f'could not determine video id from training video url: {url}')
    return {
        'source_type': 'video',
        'name': str(video.get('name') or video_id).strip() or video_id,
        'url': normalize_video_url(video_id, url),
        'label': str(video.get('label') or default_label).strip() or DEFAULT_LABEL,
        'label_source': str(video.get('label_source') or default_label_source).strip() or DEFAULT_LABEL_SOURCE,
    }


def _normalize_training_source_config(source: dict[str, Any], default_label: str, default_label_source: str) -> dict[str, str]:
    url = str(source.get('url') or '').strip()
    if not url:
        raise ValueError('training source config is missing `url`')
    if extract_youtube_video_id(url):
        return _normalize_training_video_config(source, default_label=default_label, default_label_source=default_label_source)
    return _normalize_channel_config(source, default_label=default_label, default_label_source=default_label_source)


def resolve_training_sources(config: dict[str, Any], channel_url: Optional[str] = None) -> list[dict[str, str]]:
    default_label = str(get_config_value(config, 'training.label', DEFAULT_LABEL) or DEFAULT_LABEL)
    default_label_source = str(get_config_value(config, 'training.label_source', DEFAULT_LABEL_SOURCE) or DEFAULT_LABEL_SOURCE)
    if channel_url:
        return [_normalize_training_source_config(
            {
                'name': channel_handle_from_url(channel_url) or extract_youtube_video_id(channel_url) or channel_url,
                'url': str(channel_url),
                'label': default_label,
                'label_source': default_label_source,
            },
            default_label=default_label,
            default_label_source=default_label_source,
        )]

    configured_channels = get_config_value(config, 'training.channels', []) or []
    configured_videos = get_config_value(config, 'training.videos', []) or []
    configured_sources = list(configured_channels) + list(configured_videos)
    if not configured_sources:
        configured_sources = [{
            'name': 'mai_dq',
            'url': DEFAULT_CHANNEL_URL,
            'label': default_label,
            'label_source': 'mai_dq_mix_curation',
        }]
    return [
        _normalize_training_source_config(source, default_label=default_label, default_label_source=default_label_source)
        for source in configured_sources
    ]


def resolve_training_channels(config: dict[str, Any], channel_url: Optional[str] = None) -> list[dict[str, str]]:
    return resolve_training_sources(config, channel_url=channel_url)


def _build_source_track_rows(
    video_id: str,
    video_url: str,
    metadata: dict[str, Any],
    channel_config: dict[str, str],
) -> list[dict[str, Any]]:
    description = str(metadata.get('description') or '')
    description_rows = parse_tracklist_description(description)
    chapter_rows = parse_tracklist_watch_metadata(metadata)

    selected_rows: list[dict[str, Any]] = []
    if len(description_rows) >= 2:
        selected_rows = []
        unmatched_chapters = list(chapter_rows)
        for index, row in enumerate(description_rows, start=1):
            matched_chapter = None
            best_distance = None
            for chapter_row in unmatched_chapters:
                distance = abs(int(chapter_row['timestamp_s']) - int(row['timestamp_s']))
                if distance <= 5 and (best_distance is None or distance < best_distance):
                    matched_chapter = chapter_row
                    best_distance = distance
            if matched_chapter is None and len(chapter_rows) == len(description_rows) and index - 1 < len(chapter_rows):
                matched_chapter = chapter_rows[index - 1]
            if matched_chapter is not None and matched_chapter in unmatched_chapters:
                unmatched_chapters.remove(matched_chapter)
            track_source = 'description+chapters' if matched_chapter is not None else 'description'
            selected_rows.append({
                **row,
                'track_source': track_source,
                'chapter_title': str((matched_chapter or {}).get('track_raw') or ''),
                'chapter_timestamp_s': (matched_chapter or {}).get('timestamp_s', pd.NA),
            })
    elif len(chapter_rows) >= 2:
        selected_rows = chapter_rows

    source_video_title = str(metadata.get('title') or '')
    source_channel_handle = str(metadata.get('uploader_id') or channel_handle_from_url(channel_config['url']) or metadata.get('channel_id') or '')
    rows: list[dict[str, Any]] = []
    for row in selected_rows:
        rows.append({
            'video_id': str(video_id),
            'channel_url': str(channel_config['url']),
            'channel_handle': source_channel_handle,
            'label': str(channel_config['label']),
            'label_source': str(channel_config['label_source']),
            'source_cache_version': int(SOURCE_TRACK_CACHE_VERSION),
            'video_title': source_video_title,
            'video_url': normalize_video_url(video_id, video_url),
            'description_length': len(description),
            'track_source': str(row.get('track_source') or ''),
            'chapter_title': str(row.get('chapter_title') or ''),
            'chapter_timestamp_s': row.get('chapter_timestamp_s', pd.NA),
            'position': int(row['position']),
            'timestamp': str(row['timestamp']),
            'timestamp_s': int(row['timestamp_s']),
            'track_raw': str(row['track_raw']),
            'artist_guess': str(row.get('artist_guess') or ''),
            'title_guess': str(row.get('title_guess') or ''),
        })
    return rows


def fetch_channel_video_entries(
    channel_url: str,
    cache_dir: Optional[str] = None,
    max_videos: Optional[int] = None,
    refresh: bool = False,
) -> list[dict[str, str]]:
    cache_path = _channel_cache_path(cache_dir, channel_url) if cache_dir else None
    payload = _read_json(cache_path) if cache_path and not refresh else None
    if payload is None:
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'skip_download': True,
            'no_warnings': True,
            'ignoreerrors': True,
            'socket_timeout': YTDLP_SOCKET_TIMEOUT_SECONDS,
            'extractor_retries': YTDLP_EXTRACTOR_RETRIES,
            'extractor_args': {'youtube': {'player_skip': ['js']}},
        }
        info = _yt_dlp_extract_info(
            channel_url,
            ydl_opts,
            allow_empty=False,
            context=f'channel {channel_url}',
        )
        if not info:
            raise RuntimeError(f'yt-dlp returned no channel metadata for {channel_url}')
        payload = {
            'entries': _compact_channel_video_entries(info.get('entries') or []),
        }
        if cache_path:
            _write_json_atomic(cache_path, payload)

    entries = _compact_channel_video_entries(payload.get('entries') or [])
    seen_video_ids: set[str] = set()
    rows: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        video_id = str(entry.get('id') or '').strip()
        if not video_id or video_id in seen_video_ids:
            continue
        seen_video_ids.add(video_id)
        rows.append({
            'video_id': video_id,
            'video_title': str(entry.get('title') or '').strip(),
            'video_url': normalize_video_url(video_id, str(entry.get('url') or '').strip()),
        })
        if max_videos is not None and len(rows) >= int(max_videos):
            break
    return rows


def fetch_video_metadata(
    video_url: str,
    video_id: str,
    cache_dir: Optional[str] = None,
    refresh: bool = False,
) -> dict[str, Any]:
    cache_path = _video_metadata_cache_path(cache_dir, video_id) if cache_dir else None
    cached = _read_json(cache_path) if cache_path and not refresh else None
    if cached is not None:
        return _compact_video_metadata(cached)

    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'socket_timeout': YTDLP_SOCKET_TIMEOUT_SECONDS,
        'extractor_retries': YTDLP_EXTRACTOR_RETRIES,
        'extractor_args': {'youtube': {'player_skip': ['js']}},
    }
    metadata = _yt_dlp_extract_info(
        video_url,
        ydl_opts,
        allow_empty=False,
        context=f'video {video_id}',
    )
    if not metadata:
        raise RuntimeError(f'yt-dlp returned no video metadata for {video_id}')
    compacted_metadata = _compact_video_metadata(metadata)
    if cache_path:
        _write_json_atomic(cache_path, compacted_metadata)
    return compacted_metadata


def _fetch_resolved_track_metadata_row(
    row: dict[str, Any],
    cache_dir: Optional[str],
    refresh: bool,
) -> tuple[str, dict[str, str]]:
    video_id = str(row.get('video_id') or '').strip()
    if not _looks_like_youtube_video_id(video_id):
        return video_id, {
            'title': str(row.get('title') or '').strip(),
            'artist': str(row.get('artist') or '').strip(),
            'uploader': '',
            'channel': '',
            'description': '',
            'tags': '',
            'category': '',
            'url': str(row.get('url') or normalize_video_url(video_id)).strip(),
        }
    metadata = fetch_video_metadata(
        video_url=str(row.get('url') or normalize_video_url(video_id)),
        video_id=video_id,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    categories = metadata.get('categories')
    category = _metadata_value_to_text(categories if categories is not None else metadata.get('category'))
    payload = {
        'title': str(metadata.get('title') or row.get('title') or '').strip(),
        'artist': str(row.get('artist') or metadata.get('artist') or metadata.get('uploader') or metadata.get('channel') or '').strip(),
        'uploader': str(metadata.get('uploader') or row.get('artist') or '').strip(),
        'channel': str(metadata.get('channel') or metadata.get('uploader') or row.get('artist') or '').strip(),
        'description': str(metadata.get('description') or '').strip(),
        'tags': _metadata_value_to_text(metadata.get('tags')),
        'category': category,
        'url': normalize_video_url(
            video_id,
            str(metadata.get('webpage_url') or metadata.get('url') or row.get('url') or '').strip(),
        ),
    }
    return video_id, payload


def _enrich_resolved_tracks_with_metadata(
    unique_tracks: pd.DataFrame,
    *,
    cache_dir: Optional[str],
    refresh: bool,
    metadata_workers: int,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    if unique_tracks.empty:
        return unique_tracks

    rows = unique_tracks.to_dict(orient='records')
    total_tracks = len(rows)
    enriched_rows: dict[str, dict[str, str]] = {}
    max_workers = _worker_count(metadata_workers, total_tracks)
    _emit_progress(
        progress_callback,
        'Fetching resolved track metadata',
        0,
        max(total_tracks, 1),
        f'starting {total_tracks} metadata lookups',
    )

    def _store_result(video_id: str, payload: dict[str, str], completed: int) -> None:
        enriched_rows[video_id] = payload
        detail = payload.get('title') or video_id
        _emit_progress(progress_callback, 'Fetching resolved track metadata', completed, total_tracks, detail)

    if max_workers == 1:
        for index, row in enumerate(rows, start=1):
            fallback_video_id = str(row.get('video_id') or '').strip()
            try:
                video_id, payload = _fetch_resolved_track_metadata_row(row, cache_dir=cache_dir, refresh=refresh)
            except Exception as exc:  # pragma: no cover - depends on live network behavior
                logger.warning('Failed to fetch resolved track metadata for %s: %s', fallback_video_id, exc)
                video_id = fallback_video_id
                payload = {
                    'title': str(row.get('title') or '').strip(),
                    'artist': str(row.get('artist') or '').strip(),
                    'uploader': str(row.get('artist') or '').strip(),
                    'channel': str(row.get('artist') or '').strip(),
                    'description': '',
                    'tags': '',
                    'category': '',
                    'url': str(row.get('url') or normalize_video_url(fallback_video_id)).strip(),
                }
            _store_result(video_id, payload, index)
    else:
        future_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for row in rows:
                future = executor.submit(
                    _fetch_resolved_track_metadata_row,
                    row,
                    cache_dir,
                    refresh,
                )
                future_map[future] = row
            completed = 0
            pending_futures = set(future_map)
            while pending_futures:
                done_futures, pending_futures = wait(
                    pending_futures,
                    timeout=FUTURE_WAIT_HEARTBEAT_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
                if not done_futures:
                    waiting_ids = sorted(
                        str((future_map[future] or {}).get('video_id') or '').strip()
                        for future in pending_futures
                    )
                    preview = ', '.join(video_id for video_id in waiting_ids[:3] if video_id)
                    if len(waiting_ids) > 3:
                        preview = f'{preview}, +{len(waiting_ids) - 3} more' if preview else f'+{len(waiting_ids) - 3} more'
                    _emit_progress(
                        progress_callback,
                        'Fetching resolved track metadata',
                        completed,
                        max(total_tracks, 1),
                        f'waiting on {len(pending_futures)} metadata lookups' + (f' ({preview})' if preview else ''),
                    )
                    continue

                for future in done_futures:
                    completed += 1
                    row = future_map[future]
                    video_id = str(row.get('video_id') or '').strip()
                    try:
                        resolved_video_id, payload = future.result()
                    except Exception as exc:  # pragma: no cover - depends on live network behavior
                        logger.warning('Failed to fetch resolved track metadata for %s: %s', video_id, exc)
                        payload = {
                            'title': str(row.get('title') or '').strip(),
                            'artist': str(row.get('artist') or '').strip(),
                            'uploader': str(row.get('artist') or '').strip(),
                            'channel': str(row.get('artist') or '').strip(),
                            'description': '',
                            'tags': '',
                            'category': '',
                            'url': str(row.get('url') or normalize_video_url(video_id)).strip(),
                        }
                        resolved_video_id = video_id
                    _store_result(resolved_video_id, payload, completed)

    enriched_df = unique_tracks.copy()
    for column in ['title', 'artist', 'uploader', 'channel', 'description', 'tags', 'category', 'url']:
        values = enriched_df['video_id'].map(lambda video_id: enriched_rows.get(str(video_id), {}).get(column, ''))
        if column in enriched_df.columns:
            incoming_mask = values.fillna('').astype(str).str.strip().ne('')
            enriched_df.loc[incoming_mask, column] = values.loc[incoming_mask]
        else:
            enriched_df[column] = values
    return enriched_df


def _fetch_source_video_rows(
    video: dict[str, str],
    channel_config: dict[str, str],
    cache_dir: Optional[str],
    refresh: bool,
) -> tuple[str, pd.DataFrame]:
    source_video_id = str(video['video_id'])
    metadata = fetch_video_metadata(
        video_url=video['video_url'],
        video_id=source_video_id,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    parsed_rows_df = source_tracks_dataframe(
        _build_source_track_rows(
            video_id=source_video_id,
            video_url=video['video_url'],
            metadata=metadata,
            channel_config=channel_config,
        )
    )
    return source_video_id, parsed_rows_df


def scrape_channel_track_rows(
    channel_url: str = DEFAULT_CHANNEL_URL,
    max_videos: Optional[int] = None,
    cache_dir: Optional[str] = 'data/cache',
    refresh: bool = False,
    label: str = DEFAULT_LABEL,
    label_source: str = DEFAULT_LABEL_SOURCE,
    metadata_workers: int = DEFAULT_METADATA_WORKERS,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    channel_config = _normalize_channel_config(
        {'url': channel_url, 'label': label, 'label_source': label_source},
        default_label=DEFAULT_LABEL,
        default_label_source=DEFAULT_LABEL_SOURCE,
    )
    summary = {
        'channels_failed': 0,
        'videos_scanned': 0,
        'videos_with_tracklist': 0,
        'videos_skipped': 0,
        'tracks_parsed': 0,
    }
    channel_label = channel_handle_from_url(channel_url) or channel_url
    _emit_progress(progress_callback, 'Loading channel index', 0, 1, channel_label)
    try:
        video_entries = fetch_channel_video_entries(
            channel_url=channel_url,
            cache_dir=cache_dir,
            max_videos=max_videos,
            refresh=refresh,
        )
    except Exception as exc:  # pragma: no cover - depends on live network behavior
        logger.warning('Skipping channel %s due to channel metadata fetch error: %s', channel_url, exc)
        summary['channels_failed'] = 1
        _emit_progress(progress_callback, 'Loading channel index', 1, 1, f'{channel_label} skipped')
        _emit_progress(progress_callback, 'Scraping source videos', 1, 1, f'{channel_handle_from_url(channel_url) or channel_url} skipped')
        return source_tracks_dataframe([]), summary
    _emit_progress(progress_callback, 'Loading channel index', 1, 1, f'{channel_label} found {len(video_entries)} videos')

    source_track_cache_path = _source_track_cache_path(cache_dir) if cache_dir else ''
    source_track_cache_df = (
        _compact_source_track_cache_rows(_read_cache_table(source_track_cache_path, SOURCE_TRACK_CACHE_COLUMNS))
        if cache_dir else pd.DataFrame(columns=SOURCE_TRACK_CACHE_COLUMNS)
    )
    cached_source_track_groups = {
        str(video_id): group.reindex(columns=SOURCE_TRACK_CACHE_COLUMNS).copy()
        for video_id, group in source_track_cache_df.groupby('video_id', sort=False)
    } if not source_track_cache_df.empty else {}

    track_rows: list[dict[str, Any]] = []
    total_videos = len(video_entries)
    pending_videos: list[tuple[int, dict[str, str]]] = []
    cache_updates: list[tuple[str, pd.DataFrame]] = []
    for video_number, video in enumerate(video_entries, start=1):
        summary['videos_scanned'] += 1
        source_video_id = str(video['video_id'])
        cached_rows_df = cached_source_track_groups.get(source_video_id, pd.DataFrame(columns=SOURCE_TRACK_CACHE_COLUMNS)) if not refresh else pd.DataFrame(columns=SOURCE_TRACK_CACHE_COLUMNS)
        if not _source_track_rows_are_valid(cached_rows_df, channel_config):
            cached_rows_df = pd.DataFrame(columns=SOURCE_TRACK_CACHE_COLUMNS)

        if cached_rows_df.empty:
            pending_videos.append((video_number, video))
            continue
        active_rows_df = _restore_source_track_cache_rows(
            cached_rows_df,
            channel_config=channel_config,
            video=video,
        )

        parsed_count = int(len(active_rows_df))
        if parsed_count < 2:
            summary['videos_skipped'] += 1
            _emit_progress(progress_callback, 'Scraping source videos', video_number, total_videos, f'{source_video_id} skipped')
            continue

        summary['videos_with_tracklist'] += 1
        summary['tracks_parsed'] += parsed_count
        track_rows.extend(active_rows_df.to_dict(orient='records'))
        _emit_progress(
            progress_callback,
            'Scraping source videos',
            video_number,
            total_videos,
            f'{source_video_id} parsed {parsed_count} tracks',
        )

    if pending_videos:
        _emit_progress(
            progress_callback,
            'Scraping source videos',
            total_videos - len(pending_videos),
            max(total_videos, 1),
            f'fetching metadata for {len(pending_videos)} uncached videos',
        )
        max_workers = _worker_count(metadata_workers, len(pending_videos))
        if max_workers == 1:
            iterator = []
            for video_number, video in pending_videos:
                try:
                    result = _fetch_source_video_rows(video, channel_config=channel_config, cache_dir=cache_dir, refresh=refresh)
                    iterator.append((video_number, video, result, None))
                except Exception as exc:  # pragma: no cover - depends on live network behavior
                    iterator.append((video_number, video, None, exc))
        else:
            future_map = {}
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                for video_number, video in pending_videos:
                    future = executor.submit(
                        _fetch_source_video_rows,
                        video,
                        channel_config,
                        cache_dir,
                        refresh,
                    )
                    future_map[future] = (video_number, video)
                iterator = []
                pending_futures = set(future_map)
                while pending_futures:
                    done_futures, pending_futures = wait(
                        pending_futures,
                        timeout=FUTURE_WAIT_HEARTBEAT_SECONDS,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done_futures:
                        waiting_ids = sorted(
                            str((future_map[future][1] or {}).get('video_id') or '').strip()
                            for future in pending_futures
                        )
                        preview = ', '.join(video_id for video_id in waiting_ids[:3] if video_id)
                        if len(waiting_ids) > 3:
                            preview = f'{preview}, +{len(waiting_ids) - 3} more' if preview else f'+{len(waiting_ids) - 3} more'
                        _emit_progress(
                            progress_callback,
                            'Scraping source videos',
                            total_videos - len(pending_videos),
                            max(total_videos, 1),
                            (
                                f'waiting on {len(pending_futures)} metadata fetches'
                                + (f' ({preview})' if preview else '')
                            ),
                        )
                        continue

                    for future in done_futures:
                        video_number, video = future_map[future]
                        try:
                            iterator.append((video_number, video, future.result(), None))
                        except Exception as exc:  # pragma: no cover - depends on live network behavior
                            iterator.append((video_number, video, None, exc))
            finally:
                executor.shutdown(wait=True)

        for video_number, video, result, error in sorted(iterator, key=lambda item: item[0]):
            source_video_id = str(video['video_id'])
            if error is not None:
                logger.warning('Skipping %s due to metadata fetch error: %s', source_video_id, error)
                summary['videos_skipped'] += 1
                _emit_progress(progress_callback, 'Scraping source videos', video_number, total_videos, f'{source_video_id} skipped')
                continue

            _, active_rows_df = result
            active_rows_df = active_rows_df.reindex(columns=SOURCE_TRACK_COLUMNS)
            cache_updates.append((source_video_id, _compact_source_track_cache_rows(active_rows_df)))

            parsed_count = int(len(active_rows_df))
            if parsed_count < 2:
                summary['videos_skipped'] += 1
                _emit_progress(progress_callback, 'Scraping source videos', video_number, total_videos, f'{source_video_id} skipped')
                continue

            summary['videos_with_tracklist'] += 1
            summary['tracks_parsed'] += parsed_count
            track_rows.extend(active_rows_df.to_dict(orient='records'))
            _emit_progress(
                progress_callback,
                'Scraping source videos',
                video_number,
                total_videos,
                f'{source_video_id} parsed {parsed_count} tracks',
            )

    if cache_dir and cache_updates:
        for source_video_id, rows_df in cache_updates:
            source_track_cache_df = _replace_source_track_rows(source_track_cache_df, source_video_id, rows_df)
        _write_cache_table(
            source_track_cache_path,
            source_track_cache_df,
            SOURCE_TRACK_CACHE_COLUMNS,
            key_columns=['video_id', 'position'],
        )

    return source_tracks_dataframe(track_rows), summary


def scrape_video_track_rows(
    video_url: str,
    cache_dir: Optional[str] = 'data/cache',
    refresh: bool = False,
    label: str = DEFAULT_LABEL,
    label_source: str = DEFAULT_LABEL_SOURCE,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    source_config = _normalize_training_video_config(
        {'url': video_url, 'label': label, 'label_source': label_source},
        default_label=DEFAULT_LABEL,
        default_label_source=DEFAULT_LABEL_SOURCE,
    )
    source_video_id = extract_youtube_video_id(source_config['url'])
    summary = {
        'channels_failed': 0,
        'videos_scanned': 1,
        'videos_with_tracklist': 0,
        'videos_skipped': 0,
        'tracks_parsed': 0,
    }
    display_label = source_video_id or source_config['url']
    _emit_progress(progress_callback, 'Loading source video', 0, 1, display_label)
    try:
        _, active_rows_df = _fetch_source_video_rows(
            {
                'video_id': source_video_id,
                'video_title': source_config.get('name') or source_video_id,
                'video_url': source_config['url'],
            },
            channel_config=source_config,
            cache_dir=cache_dir,
            refresh=refresh,
        )
    except Exception as exc:  # pragma: no cover - depends on live network behavior
        logger.warning('Skipping source video %s due to metadata fetch error: %s', source_config['url'], exc)
        summary['videos_skipped'] = 1
        _emit_progress(progress_callback, 'Loading source video', 1, 1, f'{display_label} skipped')
        _emit_progress(progress_callback, 'Scraping source videos', 1, 1, f'{display_label} skipped')
        return source_tracks_dataframe([]), summary
    _emit_progress(progress_callback, 'Loading source video', 1, 1, display_label)

    active_rows_df = active_rows_df.reindex(columns=SOURCE_TRACK_COLUMNS)
    parsed_count = int(len(active_rows_df))
    if parsed_count < 2:
        summary['videos_skipped'] = 1
        _emit_progress(progress_callback, 'Scraping source videos', 1, 1, f'{display_label} skipped')
        return source_tracks_dataframe([]), summary

    summary['videos_with_tracklist'] = 1
    summary['tracks_parsed'] = parsed_count
    _emit_progress(progress_callback, 'Scraping source videos', 1, 1, f'{display_label} parsed {parsed_count} tracks')
    return source_tracks_dataframe(active_rows_df.to_dict(orient='records')), summary


def resolve_scraped_tracks(
    track_df: pd.DataFrame,
    cache_dir: Optional[str] = 'data/cache',
    max_results: int = MAX_SEARCH_RESULTS,
    refresh: bool = False,
    search_workers: int = DEFAULT_SEARCH_WORKERS,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if track_df.empty:
        return scraped_tracks_dataframe([]), {
            'tracks_resolved': 0,
            'tracks_unresolved': 0,
            'tracks_unavailable': 0,
        }

    resolution_cache_path = _resolution_cache_path(cache_dir) if cache_dir else ''
    resolution_cache_df = (
        _compact_resolution_cache_rows(_read_cache_table(resolution_cache_path, RESOLUTION_CACHE_COLUMNS))
        if cache_dir else pd.DataFrame(columns=RESOLUTION_CACHE_COLUMNS)
    )
    resolution_lookup = {
        (str(row.get('video_id') or ''), int(row.get('position'))): row
        for row in resolution_cache_df.to_dict(orient='records')
        if str(row.get('video_id') or '').strip() and pd.notna(row.get('position'))
    }

    rows: list[dict[str, Any]] = []
    summary = {'tracks_resolved': 0, 'tracks_unresolved': 0, 'tracks_unavailable': 0}
    total_tracks = int(len(track_df))
    pending_rows: list[dict[str, Any]] = []
    new_resolution_rows: list[dict[str, Any]] = []
    query_plan: dict[str, str] = {}
    _emit_progress(progress_callback, 'Resolving tracks', 0, max(total_tracks, 1), f'preparing {total_tracks} scraped tracks')
    for _, row in track_df.iterrows():
        source_video_id = str(row.get('video_id') or '')
        position = int(row.get('position') or 0)
        cache_key = (source_video_id, position)
        query_variants = build_track_search_queries(
            track_raw=str(row.get('track_raw') or ''),
            artist_guess=str(row.get('artist_guess') or ''),
            title_guess=str(row.get('title_guess') or ''),
        )
        normalized_query_variants: list[str] = []
        for variant in query_variants:
            normalized_variant = _normalize_search_query(variant)
            if normalized_variant and normalized_variant not in normalized_query_variants:
                normalized_query_variants.append(normalized_variant)
        query = query_variants[0] if query_variants else ''
        normalized_query = normalized_query_variants[0] if normalized_query_variants else ''
        cached_resolution_row = resolution_lookup.get(cache_key)
        if not refresh and cached_resolution_row and _resolution_row_is_valid(
            cached_resolution_row,
            query=query,
            normalized_query=normalized_query,
            max_results=max_results,
        ):
            pending_rows.append({
                'row': row.to_dict(),
                'resolution_row': _restore_resolution_cache_row(
                    cached_resolution_row.copy(),
                    query=query,
                    normalized_query=normalized_query,
                    max_results=max_results,
                ),
                'resolution_source': 'cache',
            })
            continue
        if not query:
            resolution_row = {
                'video_id': source_video_id,
                'position': position,
                'search_query': '',
                'normalized_search_query': '',
                'search_max_results': int(max(1, int(max_results))),
                'resolution_cache_version': int(RESOLUTION_CACHE_VERSION),
                'resolution_status': 'ignored',
                'resolved_video_id': '',
                'resolved_title': '',
                'resolved_artist': '',
                'resolved_url': '',
                'resolved_duration_seconds': pd.NA,
            }
            pending_rows.append({
                'row': row.to_dict(),
                'resolution_row': resolution_row.copy(),
                'resolution_source': 'ignored',
            })
            new_resolution_rows.append(resolution_row)
            resolution_lookup[cache_key] = resolution_row.copy()
            logger.info('Skipping suspicious track search for %s #%d: %s', source_video_id, position, row.get('track_raw') or '')
            continue
        for normalized_variant, query_variant in zip(normalized_query_variants, query_variants):
            query_plan.setdefault(normalized_variant, query_variant)
        pending_rows.append({
            'row': row.to_dict(),
            'cache_key': cache_key,
            'query': query,
            'normalized_query': normalized_query,
            'query_variants': query_variants,
            'normalized_query_variants': normalized_query_variants,
            'resolution_source': 'search',
        })

    search_results: dict[str, list[dict[str, Any]]] = {}
    if query_plan:
        _emit_progress(
            progress_callback,
            'Resolving tracks',
            0,
            max(total_tracks, 1),
            f'queued {len(query_plan)} unique YouTube searches',
        )
        unique_queries = sorted(query_plan.items())
        max_workers = _worker_count(search_workers, len(unique_queries))
        total_queries = len(unique_queries)
        _emit_progress(
            progress_callback,
            'Searching YouTube',
            0,
            max(total_queries, 1),
            f'running {total_queries} searches with {max_workers} workers',
        )
        if max_workers == 1:
            for query_number, (normalized_query, query) in enumerate(unique_queries, start=1):
                logger.info('YouTube search %d/%d: %s', query_number, total_queries, query)
                _emit_progress(progress_callback, 'Searching YouTube', query_number - 1, total_queries, f'searching {query}')
                results = search_youtube_track_candidates(
                    query,
                    max_results=max_results,
                    cache_dir=cache_dir,
                    refresh=refresh,
                )
                search_results[normalized_query] = results
                _emit_progress(progress_callback, 'Searching YouTube', query_number, total_queries, f'{query} -> {len(results)} candidates')
        else:
            future_map = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for query_number, (normalized_query, query) in enumerate(unique_queries, start=1):
                    logger.info('YouTube search %d/%d: %s', query_number, total_queries, query)
                    future = executor.submit(
                        search_youtube_track_candidates,
                        query,
                        max_results,
                        cache_dir,
                        refresh,
                    )
                    future_map[future] = (normalized_query, query)
                completed_searches = 0
                for future in as_completed(future_map):
                    completed_searches += 1
                    normalized_query, query = future_map[future]
                    results = future.result()
                    search_results[normalized_query] = results
                    _emit_progress(progress_callback, 'Searching YouTube', completed_searches, total_queries, f'{query} -> {len(results)} candidates')
    else:
        _emit_progress(progress_callback, 'Searching YouTube', 1, 1, 'no new searches needed')

    for track_number, pending in enumerate(pending_rows, start=1):
        row_data = dict(pending['row'])
        source_video_id = str(row_data.get('video_id') or '')
        resolution_source = str(pending.get('resolution_source') or '')
        if 'resolution_row' in pending:
            resolution_row = dict(pending['resolution_row'])
        else:
            query = str(pending['query'])
            normalized_query = str(pending['normalized_query'])
            query_variants = [str(variant) for variant in pending.get('query_variants') or [] if str(variant).strip()]
            normalized_query_variants = [str(variant) for variant in pending.get('normalized_query_variants') or [] if str(variant).strip()]
            candidates = _merge_candidate_results(normalized_query_variants, search_results)
            status, selected = _select_track_candidate_with_status(
                candidates,
                excluded_video_ids={source_video_id},
                track_raw=str(row_data.get('track_raw') or ''),
                query_variants=query_variants,
            )
            resolution_row = {
                'video_id': source_video_id,
                'position': int(row_data.get('position') or 0),
                'search_query': query,
                'normalized_search_query': normalized_query,
                'search_max_results': int(max(1, int(max_results))),
                'resolution_cache_version': int(RESOLUTION_CACHE_VERSION),
                'resolution_status': status,
                'resolved_video_id': '',
                'resolved_title': '',
                'resolved_artist': '',
                'resolved_url': '',
                'resolved_duration_seconds': pd.NA,
            }
            if selected:
                resolution_row.update(selected)
            new_resolution_rows.append(resolution_row)
            resolution_lookup[pending['cache_key']] = resolution_row.copy()

        combined_row = row_data.copy()
        combined_row.update({column: resolution_row.get(column, '') for column in RESOLUTION_COLUMNS if column not in {'video_id', 'position'}})
        rows.append(combined_row)

        status = str(resolution_row.get('resolution_status') or 'no_match')
        search_query_text = str(resolution_row.get('search_query') or row_data.get('track_raw') or '').strip()
        if status == 'resolved':
            summary['tracks_resolved'] += 1
            detail = f'{search_query_text} -> {resolution_row.get("resolved_video_id") or ""}'.strip()
        else:
            summary['tracks_unresolved'] += 1
            if status == 'unavailable':
                summary['tracks_unavailable'] += 1
            if status == 'ignored':
                detail = f'ignored suspicious line {search_query_text or row_data.get("track_raw") or ""}'.strip()
            else:
                detail = f'{status} {search_query_text}'.strip()
        if resolution_source == 'cache' and detail:
            detail = f'cached {detail}'
        _emit_progress(progress_callback, 'Resolving tracks', track_number, total_tracks, detail)

    if cache_dir and new_resolution_rows:
        resolution_cache_df = _upsert_resolution_rows(
            resolution_cache_df,
            resolution_dataframe(new_resolution_rows),
        )
        _write_cache_table(
            resolution_cache_path,
            resolution_cache_df,
            RESOLUTION_CACHE_COLUMNS,
            key_columns=['video_id', 'position'],
        )

    return scraped_tracks_dataframe(rows), summary


def analyze_resolved_tracks(
    track_df: pd.DataFrame,
    audio_cache_dir: str = 'data/audio_cache',
    feature_cache_dir: Optional[str] = None,
    cache_dir: Optional[str] = 'data/cache',
    metadata_workers: int = DEFAULT_METADATA_WORKERS,
    download_workers: int = 4,
    analysis_workers: int = 4,
    edge_seconds: float = 30.0,
    silence_top_db: float = 35.0,
    flow_profile: str = 'deep-dj',
    resource_profile: str = 'default',
    refresh_cache: bool = False,
    delete_audio_after_analysis: bool = True,
    reuse_cache_any_signature: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if track_df.empty:
        return track_df.copy(), {'tracks_analyzed': 0, 'tracks_with_features': 0, 'tracks_analysis_failed': 0}

    resolution_cache_path = _resolution_cache_path(cache_dir) if cache_dir else ''
    resolution_cache_df = (
        _compact_resolution_cache_rows(_read_cache_table(resolution_cache_path, RESOLUTION_CACHE_COLUMNS))
        if cache_dir else pd.DataFrame(columns=RESOLUTION_CACHE_COLUMNS)
    )

    resolved_tracks = track_df.loc[
        track_df['resolution_status'].fillna('').astype(str).eq('resolved')
        & track_df['resolved_video_id'].fillna('').astype(str).str.strip().ne('')
    ].copy()
    if resolved_tracks.empty:
        return track_df.copy(), {'tracks_analyzed': 0, 'tracks_with_features': 0, 'tracks_analysis_failed': 0}

    unique_tracks = (
        resolved_tracks[['resolved_video_id', 'resolved_title', 'resolved_artist', 'resolved_url']]
        .drop_duplicates(subset=['resolved_video_id'], keep='first')
        .rename(columns={
            'resolved_video_id': 'video_id',
            'resolved_title': 'title',
            'resolved_artist': 'artist',
            'resolved_url': 'url',
        })
        .reset_index(drop=True)
    )
    unique_tracks['uploader'] = ''
    unique_tracks['channel'] = ''
    unique_tracks['description'] = ''
    unique_tracks['tags'] = ''
    unique_tracks['category'] = ''
    unique_tracks = _enrich_resolved_tracks_with_metadata(
        unique_tracks,
        cache_dir=cache_dir,
        refresh=refresh_cache,
        metadata_workers=metadata_workers,
        progress_callback=progress_callback,
    )
    _emit_progress(progress_callback, 'Analyzing resolved tracks', 0, max(len(unique_tracks), 1), 'starting audio analysis')
    analyzed_unique_tracks = analyze_youtube_playlist_audio(
        unique_tracks,
        audio_cache_dir=audio_cache_dir,
        feature_cache_dir=feature_cache_dir,
        edge_seconds=edge_seconds,
        silence_top_db=silence_top_db,
        flow_profile=flow_profile,
        resource_profile=resource_profile,
        refresh_cache=refresh_cache,
        download_workers=download_workers,
        analysis_workers=analysis_workers,
        delete_audio_after_analysis=delete_audio_after_analysis,
        reuse_cache_any_signature=reuse_cache_any_signature,
        progress_callback=progress_callback,
    ).rename(columns={'video_id': 'resolved_video_id'})
    _emit_progress(progress_callback, 'Analyzing resolved tracks', len(unique_tracks), max(len(unique_tracks), 1), 'audio analysis complete')

    feature_columns = [
        column for column in analyzed_unique_tracks.columns
        if column not in {'resolved_video_id', 'title', 'artist', 'url'}
    ]
    merged = track_df.merge(
        analyzed_unique_tracks[['resolved_video_id'] + feature_columns],
        on='resolved_video_id',
        how='left',
    )

    failure_mask = (
        merged['resolution_status'].fillna('').astype(str).eq('resolved')
        & ~merged.apply(lambda row: _track_row_has_analysis(row, flow_profile=flow_profile), axis=1)
    )
    if failure_mask.any():
        merged.loc[failure_mask, 'resolution_status'] = 'analysis_failed'

    if cache_dir:
        updated_rows_df = resolution_dataframe(merged.reindex(columns=RESOLUTION_COLUMNS).to_dict(orient='records'))
        resolution_cache_df = _upsert_resolution_rows(resolution_cache_df, updated_rows_df)
        _write_cache_table(
            resolution_cache_path,
            resolution_cache_df,
            RESOLUTION_CACHE_COLUMNS,
            key_columns=['video_id', 'position'],
        )

    tracks_with_features = int(merged.apply(lambda row: _track_row_has_analysis(row, flow_profile=flow_profile), axis=1).sum())
    tracks_analysis_failed = int(failure_mask.sum())
    return merged, {
        'tracks_analyzed': int(len(unique_tracks)),
        'tracks_with_features': tracks_with_features,
        'tracks_analysis_failed': tracks_analysis_failed,
    }


def _track_row_has_analysis(row: pd.Series, flow_profile: str) -> bool:
    required = list(ANALYSIS_REQUIRED_COLUMNS)
    if flow_profile == 'deep-dj':
        required.extend(ANALYSIS_REQUIRED_DEEP_COLUMNS)
    for column in required:
        if column not in row.index or pd.isna(row.get(column)):
            return False
    return True


def _track_feature_columns(track_df: pd.DataFrame) -> list[str]:
    excluded = set(TRACK_COLUMNS)
    return sorted(column for column in track_df.columns if column not in excluded)


def _ordered_transition_columns(df: pd.DataFrame) -> list[str]:
    ordered_columns = [column for column in BASE_TRANSITION_COLUMNS if column in df.columns]
    ordered_columns.extend(sorted(column for column in df.columns if column not in ordered_columns))
    return ordered_columns


def build_training_transition_rows(
    track_df: pd.DataFrame,
    flow_profile: str = 'deep-dj',
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if track_df.empty:
        return pd.DataFrame(columns=BASE_TRANSITION_COLUMNS), {'positive_pairs': 0, 'pairs_skipped': 0}

    ordered_tracks = track_df.sort_values(['video_id', 'position'], kind='stable').reset_index(drop=True)
    feature_columns = _track_feature_columns(ordered_tracks)
    rows: list[dict[str, Any]] = []
    summary = {'positive_pairs': 0, 'pairs_skipped': 0}
    total_pairs = max(int(len(ordered_tracks) - ordered_tracks['video_id'].nunique()), 0)
    processed_pairs = 0
    _emit_progress(
        progress_callback,
        'Building training pairs',
        0,
        max(total_pairs, 1),
        f'assembling adjacent pairs from {ordered_tracks["video_id"].nunique()} source videos',
    )

    for _, group in ordered_tracks.groupby('video_id', sort=False):
        group = group.sort_values('position', kind='stable').reset_index(drop=True)
        for current_index in range(len(group) - 1):
            current_row = group.iloc[current_index]
            following_row = group.iloc[current_index + 1]
            if str(current_row.get('resolution_status') or '') != 'resolved':
                summary['pairs_skipped'] += 1
                processed_pairs += 1
                _emit_progress(progress_callback, 'Building training pairs', processed_pairs, total_pairs, 'skipped unresolved source track')
                continue
            if str(following_row.get('resolution_status') or '') != 'resolved':
                summary['pairs_skipped'] += 1
                processed_pairs += 1
                _emit_progress(progress_callback, 'Building training pairs', processed_pairs, total_pairs, 'skipped unresolved destination track')
                continue
            if not _track_row_has_analysis(current_row, flow_profile=flow_profile):
                summary['pairs_skipped'] += 1
                processed_pairs += 1
                _emit_progress(progress_callback, 'Building training pairs', processed_pairs, total_pairs, 'skipped missing analysis')
                continue
            if not _track_row_has_analysis(following_row, flow_profile=flow_profile):
                summary['pairs_skipped'] += 1
                processed_pairs += 1
                _emit_progress(progress_callback, 'Building training pairs', processed_pairs, total_pairs, 'skipped missing analysis')
                continue

            row = {
                'video_id': str(current_row['video_id']),
                'label': str(current_row.get('label') or DEFAULT_LABEL),
                'label_source': str(current_row.get('label_source') or DEFAULT_LABEL_SOURCE),
                'channel_handle': str(current_row.get('channel_handle') or ''),
                'channel_url': str(current_row.get('channel_url') or ''),
                'video_title': str(current_row.get('video_title') or ''),
                'video_url': str(current_row.get('video_url') or ''),
                'description_length': int(current_row.get('description_length') or 0),
                'from_position': int(current_row['position']),
                'to_position': int(following_row['position']),
                'from_timestamp': str(current_row['timestamp']),
                'to_timestamp': str(following_row['timestamp']),
                'from_timestamp_s': int(current_row['timestamp_s']),
                'to_timestamp_s': int(following_row['timestamp_s']),
                'transition_duration_s': max(int(following_row['timestamp_s']) - int(current_row['timestamp_s']), 0),
                'from_track_raw': str(current_row['track_raw']),
                'to_track_raw': str(following_row['track_raw']),
                'from_artist_guess': str(current_row.get('artist_guess') or ''),
                'from_title_guess': str(current_row.get('title_guess') or ''),
                'to_artist_guess': str(following_row.get('artist_guess') or ''),
                'to_title_guess': str(following_row.get('title_guess') or ''),
                'from_track_source': str(current_row.get('track_source') or ''),
                'to_track_source': str(following_row.get('track_source') or ''),
                'from_video_id': str(current_row.get('resolved_video_id') or ''),
                'to_video_id': str(following_row.get('resolved_video_id') or ''),
                'from_resolved_title': str(current_row.get('resolved_title') or ''),
                'from_resolved_artist': str(current_row.get('resolved_artist') or ''),
                'from_resolved_url': str(current_row.get('resolved_url') or ''),
                'from_resolved_duration_seconds': current_row.get('resolved_duration_seconds'),
                'to_resolved_title': str(following_row.get('resolved_title') or ''),
                'to_resolved_artist': str(following_row.get('resolved_artist') or ''),
                'to_resolved_url': str(following_row.get('resolved_url') or ''),
                'to_resolved_duration_seconds': following_row.get('resolved_duration_seconds'),
            }
            for column in feature_columns:
                row[f'from_{column}'] = current_row.get(column)
                row[f'to_{column}'] = following_row.get(column)
            rows.append(row)
            summary['positive_pairs'] += 1
            processed_pairs += 1
            _emit_progress(
                progress_callback,
                'Building training pairs',
                processed_pairs,
                total_pairs,
                f"{row['video_id']} {row['from_position']}->{row['to_position']}",
            )

    if not rows:
        return pd.DataFrame(columns=BASE_TRANSITION_COLUMNS), summary
    df = pd.DataFrame(rows)
    return df.reindex(columns=_ordered_transition_columns(df)).sort_values(['video_id', 'from_position'], kind='stable').reset_index(drop=True), summary


def scrape_training_transitions(
    channels: list[dict[str, Any]],
    cache_dir: str = 'data/cache',
    audio_cache_dir: str = 'data/audio_cache',
    feature_cache_dir: Optional[str] = None,
    max_videos: Optional[int] = None,
    max_search_results: int = MAX_SEARCH_RESULTS,
    metadata_workers: int = DEFAULT_METADATA_WORKERS,
    search_workers: int = DEFAULT_SEARCH_WORKERS,
    download_workers: int = 4,
    analysis_workers: int = 4,
    edge_seconds: float = 30.0,
    silence_top_db: float = 35.0,
    flow_profile: str = 'deep-dj',
    resource_profile: str = 'default',
    refresh_cache: bool = False,
    delete_audio_after_analysis: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    normalized_sources = [
        _normalize_training_source_config(source, default_label=DEFAULT_LABEL, default_label_source=DEFAULT_LABEL_SOURCE)
        for source in channels
    ]

    source_frames = []
    summary = {
        'channels_scanned': int(sum(1 for source in normalized_sources if str(source.get('source_type') or 'channel') == 'channel')),
        'videos_scanned': 0,
        'videos_with_tracklist': 0,
        'videos_skipped': 0,
        'tracks_parsed': 0,
        'tracks_resolved': 0,
        'tracks_unresolved': 0,
        'tracks_unavailable': 0,
        'tracks_analyzed': 0,
        'tracks_with_features': 0,
        'tracks_analysis_failed': 0,
        'positive_pairs': 0,
        'pairs_skipped': 0,
    }

    total_sources = max(len(normalized_sources), 1)
    for source_index, source in enumerate(normalized_sources, start=1):
        source_type = str(source.get('source_type') or 'channel')
        source_name = source.get('name') or channel_handle_from_url(source['url']) or extract_youtube_video_id(source['url']) or source['url']
        _emit_progress(
            progress_callback,
            'Scanning sources',
            source_index - 1,
            total_sources,
            source_name,
        )
        if source_type == 'video':
            source_df, source_summary = scrape_video_track_rows(
                video_url=source['url'],
                cache_dir=cache_dir,
                refresh=refresh_cache,
                label=source['label'],
                label_source=source['label_source'],
                progress_callback=progress_callback,
            )
        else:
            source_df, source_summary = scrape_channel_track_rows(
                channel_url=source['url'],
                max_videos=max_videos,
                cache_dir=cache_dir,
                refresh=refresh_cache,
                label=source['label'],
                label_source=source['label_source'],
                metadata_workers=metadata_workers,
                progress_callback=progress_callback,
            )
        if not source_df.empty:
            source_frames.append(source_df)
        for key, value in source_summary.items():
            summary[key] = summary.get(key, 0) + int(value)
        _emit_progress(
            progress_callback,
            'Scanning sources',
            source_index,
            total_sources,
            source_name,
        )

    source_track_df = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame(columns=SOURCE_TRACK_COLUMNS)
    if not source_track_df.empty:
        source_track_df = source_track_df.sort_values(['video_id', 'position'], kind='stable').reset_index(drop=True)

    resolved_track_df, resolve_summary = resolve_scraped_tracks(
        source_track_df,
        cache_dir=cache_dir,
        max_results=max_search_results,
        refresh=refresh_cache,
        search_workers=search_workers,
        progress_callback=progress_callback,
    )
    analyzed_track_df, analyze_summary = analyze_resolved_tracks(
        resolved_track_df,
        audio_cache_dir=audio_cache_dir,
        feature_cache_dir=feature_cache_dir,
        cache_dir=cache_dir,
        metadata_workers=metadata_workers,
        download_workers=download_workers,
        analysis_workers=analysis_workers,
        edge_seconds=edge_seconds,
        silence_top_db=silence_top_db,
        flow_profile=flow_profile,
        resource_profile=resource_profile,
        refresh_cache=refresh_cache,
        delete_audio_after_analysis=delete_audio_after_analysis,
        reuse_cache_any_signature=True,
        progress_callback=progress_callback,
    )
    training_df, pair_summary = build_training_transition_rows(
        analyzed_track_df,
        flow_profile=flow_profile,
        progress_callback=progress_callback,
    )
    for partial in [resolve_summary, analyze_summary, pair_summary]:
        for key, value in partial.items():
            summary[key] = summary.get(key, 0) + int(value)
    return training_df, summary


def scrape_channel_training_transitions(
    channel_url: str = DEFAULT_CHANNEL_URL,
    max_videos: Optional[int] = None,
    cache_dir: str = 'data/cache',
    audio_cache_dir: str = 'data/audio_cache',
    feature_cache_dir: Optional[str] = None,
    max_search_results: int = MAX_SEARCH_RESULTS,
    metadata_workers: int = DEFAULT_METADATA_WORKERS,
    search_workers: int = DEFAULT_SEARCH_WORKERS,
    download_workers: int = 4,
    analysis_workers: int = 4,
    edge_seconds: float = 30.0,
    silence_top_db: float = 35.0,
    flow_profile: str = 'deep-dj',
    resource_profile: str = 'default',
    refresh_cache: bool = False,
    delete_audio_after_analysis: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    return scrape_training_transitions(
        channels=[{'url': channel_url}],
        cache_dir=cache_dir,
        audio_cache_dir=audio_cache_dir,
        feature_cache_dir=feature_cache_dir,
        max_videos=max_videos,
        max_search_results=max_search_results,
        metadata_workers=metadata_workers,
        search_workers=search_workers,
        download_workers=download_workers,
        analysis_workers=analysis_workers,
        edge_seconds=edge_seconds,
        silence_top_db=silence_top_db,
        flow_profile=flow_profile,
        resource_profile=resource_profile,
        refresh_cache=refresh_cache,
        delete_audio_after_analysis=delete_audio_after_analysis,
        progress_callback=progress_callback,
    )


def write_training_transitions_csv(df: pd.DataFrame, out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if df.empty:
        pd.DataFrame(columns=COMPACT_TRANSITION_COLUMNS).to_csv(out_path, index=False, encoding='utf-8')
        return
    prepared = _sanitize_transition_export_text(df)
    prepared = _compact_transition_export_columns(prepared)
    prepared.to_csv(out_path, index=False, encoding='utf-8')


def _compact_transition_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.reindex(columns=COMPACT_TRANSITION_COLUMNS)
    present = [column for column in COMPACT_TRANSITION_COLUMNS if column in df.columns]
    return df.reindex(columns=present)


def _sanitize_transition_export_text(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    prepared = df.copy()
    description_columns = [column for column in prepared.columns if column == 'description' or column.endswith('_description')]
    for column in description_columns:
        prepared[column] = prepared[column].map(_flatten_multiline_export_text)
    return prepared


def _flatten_multiline_export_text(value: Any) -> str:
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except TypeError:
        pass

    text = str(value).replace('\r\n', '\n').replace('\r', '\n')
    if not text:
        return ''

    # Keep description payload single-line in CSV output while preserving paragraph boundaries.
    flattened_lines = [
        ' '.join(line.split())
        for line in text.split('\n')
        if line.strip()
    ]
    return ' | '.join(flattened_lines).strip()


def build_parser(config: dict, config_path: str, no_config: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='scrape positive transition training rows from one or more YouTube channels and analyze the resolved songs'
    )
    parser.add_argument('--config', default=config_path, help='Path to project TOML config')
    parser.add_argument('--no-config', action='store_true', default=no_config, help='Ignore the TOML config and use CLI/default values only')
    parser.add_argument('--channel-url', default=None, help='Single channel videos URL to scrape instead of the configured channel list')
    parser.add_argument('--out', default=get_config_value(config, 'training.output_path', DEFAULT_OUTPUT_PATH), help='Output CSV path')
    parser.add_argument('--errors-out', default=get_config_value(config, 'training.errors_path', ''), help='Path for per-session warning/error report; defaults to a timestamped file next to --out')
    parser.add_argument('--max-videos', type=int, default=get_config_value(config, 'training.max_videos', None), help='Limit the number of source videos scraped per channel')
    parser.add_argument('--max-search-results', type=int, default=int(get_config_value(config, 'training.max_search_results', MAX_SEARCH_RESULTS)), help='Number of YouTube search candidates to inspect for each scraped track')
    parser.add_argument('--metadata-workers', type=int, default=int(get_config_value(config, 'training.metadata_workers', DEFAULT_METADATA_WORKERS)), help='Concurrent worker count for source-video metadata fetches')
    parser.add_argument('--search-workers', type=int, default=int(get_config_value(config, 'training.search_workers', DEFAULT_SEARCH_WORKERS)), help='Concurrent worker count for YouTube track searches')
    parser.add_argument('--cache-dir', default=get_config_value(config, 'cache.root_dir', 'data/cache'), help='Directory for reusable metadata and feature caches')
    _add_bool_override(
        parser,
        true_flag='--refresh-cache',
        false_flag='--no-refresh-cache',
        dest='refresh_cache',
        default=bool(get_config_value(config, 'analysis.refresh_cache', False)),
        true_help='Ignore reusable caches and refetch/recompute',
        false_help='Reuse caches even if the config enables refresh mode',
    )
    parser.add_argument('--audio-cache', default=get_config_value(config, 'cache.audio_dir', 'data/audio_cache'), help='Directory for cached audio downloads')
    parser.add_argument('--edge-seconds', type=float, default=float(get_config_value(config, 'analysis.edge_seconds', 30.0)), help='Analyze first/last non-silent seconds')
    parser.add_argument('--silence-top-db', type=float, default=float(get_config_value(config, 'analysis.silence_top_db', 35.0)), help='Silence threshold for trimming (higher trims more)')
    parser.add_argument('--flow-profile', default=get_config_value(config, 'analysis.flow_profile', 'deep-dj'), choices=['standard', 'deep-dj'], help='Transition edge analysis depth')
    parser.add_argument('--resource-profile', default=get_config_value(config, 'analysis.resource_profile', 'default'), choices=['default', 'background'], help='Resource usage profile: `background` throttles audio analysis so it can run more gently in the background')
    parser.add_argument('--download-workers', type=int, default=int(get_config_value(config, 'analysis.download_workers', 4)), help='Concurrent worker count for YouTube audio downloads')
    parser.add_argument('--analysis-workers', type=int, default=int(get_config_value(config, 'analysis.analysis_workers', 4)), help='Concurrent worker count for CPU-heavy audio feature extraction')
    _add_bool_override(
        parser,
        true_flag='--delete-audio-after-analysis',
        false_flag='--keep-audio-cache',
        dest='delete_audio_after_analysis',
        default=bool(get_config_value(config, 'analysis.delete_audio_after_analysis', True)),
        true_help='Delete downloaded audio files after their features are persisted',
        false_help='Keep downloaded audio files after analysis',
    )
    parser.add_argument('--log-level', default=get_config_value(config, 'logging.level', 'INFO'), choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'], help='Verbosity')
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    config, config_path, no_config = _bootstrap_config(argv)
    parser = build_parser(config, config_path=config_path, no_config=no_config)
    args = parser.parse_args(argv)

    configure_cli_logging(getattr(logging, args.log_level))
    progress = CliProgressRenderer()
    run_started_at = datetime.now()
    summary: dict[str, int] = {}
    run_failed = False
    errors_report_path = str(args.errors_out or '').strip() or _default_errors_report_path(args.out, run_started_at)
    session_error_capture = SessionErrorCaptureHandler(level=logging.WARNING)
    root_logger = logging.getLogger()
    root_logger.addHandler(session_error_capture)
    try:
        channels = resolve_training_sources(config, channel_url=args.channel_url)
        progress.section('Training scrape', f'{len(channels)} source(s) -> {args.out}')
        progress.note(
            'Pipeline',
            'scan sources -> parse tracklists -> resolve tracks -> fetch metadata -> analyze audio -> build training pairs',
        )
        df, summary = scrape_training_transitions(
            channels=channels,
            cache_dir=args.cache_dir,
            audio_cache_dir=args.audio_cache,
            feature_cache_dir=os.path.join(args.cache_dir, 'audio_features.sqlite'),
            max_videos=args.max_videos,
            max_search_results=args.max_search_results,
            metadata_workers=args.metadata_workers,
            search_workers=args.search_workers,
            download_workers=args.download_workers,
            analysis_workers=args.analysis_workers,
            edge_seconds=args.edge_seconds,
            silence_top_db=args.silence_top_db,
            flow_profile=args.flow_profile,
            resource_profile=args.resource_profile,
            refresh_cache=args.refresh_cache,
            delete_audio_after_analysis=args.delete_audio_after_analysis,
            progress_callback=progress.update,
        )
        progress.section('Writing output', args.out)
        write_training_transitions_csv(df, args.out)
        progress.success(
            'Training scrape complete',
            f"{summary.get('positive_pairs', 0)} positive pairs from {summary.get('videos_with_tracklist', 0)} source videos",
        )
        print('Wrote training transition CSV to', args.out)
        print(format_scrape_summary_report(
            summary,
            output_path=args.out,
            errors_path=errors_report_path,
            warning_count=session_error_capture.warning_count,
            error_count=session_error_capture.error_count,
        ))
    except Exception:
        run_failed = True
        logger.exception('Training scrape failed')
        raise
    finally:
        run_finished_at = datetime.now()
        errors_report_written = False
        try:
            _write_session_errors_report(
                errors_report_path,
                entries=session_error_capture.entries,
                run_started_at=run_started_at,
                run_finished_at=run_finished_at,
                run_failed=run_failed,
            )
            errors_report_written = True
        except Exception as exc:  # pragma: no cover - disk/io dependent
            print(f'Failed to write session errors report to {errors_report_path}: {exc}')
        finally:
            root_logger.removeHandler(session_error_capture)
            progress.close()
        if errors_report_written:
            print('Wrote session errors report to', errors_report_path)


if __name__ == '__main__':
    main()
