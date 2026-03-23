from __future__ import annotations

import hashlib
import logging
import os
import platform
import socket
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOGGER = logging.getLogger(__name__)
_REMOTE_EJS_COMPONENT = 'ejs:github'
_RUNTIME_CANDIDATES = ('deno', 'node', 'quickjs', 'bun')
_GYAN_FFMPEG_ESSENTIALS_ZIP = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
_DENO_WINDOWS_ZIP_URLS = {
    'x86_64': 'https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip',
    'aarch64': 'https://github.com/denoland/deno/releases/latest/download/deno-aarch64-pc-windows-msvc.zip',
    'i686': 'https://github.com/denoland/deno/releases/latest/download/deno-i686-pc-windows-msvc.zip',
}
_FFMPEG_INSTALL_LOCK = threading.Lock()
_JS_RUNTIME_INSTALL_LOCK = threading.Lock()
_COOKIEFILE_CACHE_LOCK = threading.Lock()
_COOKIEFILE_SANITIZE_LOCK = threading.Lock()
_LOG_ONCE_LOCK = threading.Lock()
_COOKIEFILE_CACHE: dict[str, tuple[int, int, str | None]] = {}
_DENO_RUNTIME_INSTALL_FAILED = False
_DOWNLOAD_TIMEOUT_SECONDS = 30.0
_DOWNLOAD_PROGRESS_LOG_INTERVAL_SECONDS = 5.0
_EMITTED_LOG_KEYS: set[str] = set()


class _YtDlpLogger:
    def debug(self, message: str) -> None:
        _LOGGER.debug(str(message))

    def info(self, message: str) -> None:
        _LOGGER.info(str(message))

    def warning(self, message: str) -> None:
        _LOGGER.warning(str(message))

    def error(self, message: str) -> None:
        _LOGGER.debug(str(message))


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name) or '').strip().lower()
    if raw in {'1', 'true', 'yes', 'on'}:
        return True
    if raw in {'0', 'false', 'no', 'off'}:
        return False
    return bool(default)


def _log_once(level: str, key: str, message: str, *args: Any) -> None:
    with _LOG_ONCE_LOCK:
        if key in _EMITTED_LOG_KEYS:
            return
        _EMITTED_LOG_KEYS.add(key)
    getattr(_LOGGER, str(level))(message, *args)


def _youtube_player_skips_js(ydl_opts: dict[str, Any]) -> bool:
    extractor_args = ydl_opts.get('extractor_args') or {}
    if not isinstance(extractor_args, dict):
        return False
    youtube_args = extractor_args.get('youtube') or {}
    if not isinstance(youtube_args, dict):
        return False
    player_skip = youtube_args.get('player_skip')
    if isinstance(player_skip, str):
        return player_skip.strip().lower() == 'js'
    if isinstance(player_skip, (list, tuple, set)):
        return any(str(value).strip().lower() == 'js' for value in player_skip)
    return False


def _ensure_jsless_youtube_player_config(ydl_opts: dict[str, Any]) -> dict[str, Any]:
    if not _youtube_player_skips_js(ydl_opts):
        return ydl_opts

    prepared = dict(ydl_opts)
    extractor_args = dict(prepared.get('extractor_args') or {})
    youtube_args = dict(extractor_args.get('youtube') or {})
    player_client = youtube_args.get('player_client')
    if not player_client:
        youtube_args['player_client'] = ['android_vr']
    extractor_args['youtube'] = youtube_args
    prepared['extractor_args'] = extractor_args
    return prepared


def _default_yt_dlp_cache_setting() -> str | bool:
    if not _env_truthy('MAI_YTDLP_ENABLE_INTERNAL_CACHE', default=False):
        return False
    return str(_REPO_ROOT / 'data' / 'cache' / 'yt_dlp_internal')


def _filter_default_js_runtimes(runtimes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not runtimes:
        return {}
    if 'node' in runtimes and len(runtimes) == 1 and not _env_truthy('MAI_YTDLP_ALLOW_NODE_RUNTIME', default=False):
        deno_runtime = _ensure_windows_deno_runtime()
        if deno_runtime:
            _log_once(
                'warning',
                'node_to_deno_fallback',
                'Detected only Node.js for yt-dlp JS challenges; using auto-detected Deno runtime instead of Node.'
            )
            return deno_runtime
        _log_once(
            'warning',
            'node_runtime_skipped',
            'Detected only Node.js for yt-dlp JS challenges; skipping automatic js_runtimes due known provider crashes. '
            'Set MAI_YTDLP_ALLOW_NODE_RUNTIME=1 to re-enable Node explicitly.'
        )
        return {}
    return runtimes


def _normalize_windows_architecture_name(machine_name: str) -> str:
    normalized = str(machine_name or '').strip().lower().replace('-', '_')
    if normalized in {'amd64', 'x64', 'x86_64'}:
        return 'x86_64'
    if normalized in {'arm64', 'aarch64'}:
        return 'aarch64'
    if normalized in {'x86', 'i386', 'i686'}:
        return 'i686'
    return 'x86_64'


def _deno_windows_zip_url() -> str:
    configured = str(os.getenv('MAI_DENO_WINDOWS_ZIP_URL') or '').strip()
    if configured:
        return configured
    arch = _normalize_windows_architecture_name(platform.machine())
    return _DENO_WINDOWS_ZIP_URLS.get(arch, _DENO_WINDOWS_ZIP_URLS['x86_64'])


def _is_usable_js_runtime_binary(path: str, *, version_args: tuple[str, ...] = ('--version',), timeout_seconds: float = 10.0) -> bool:
    binary_path = str(path or '').strip()
    if not binary_path or not os.path.exists(binary_path):
        return False
    try:
        result = subprocess.run(
            [binary_path, *version_args],
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except Exception:
        return False
    return int(result.returncode) == 0


def _download_file_with_progress(
    url: str,
    destination_path: str,
    *,
    label: str,
    timeout_seconds: float = _DOWNLOAD_TIMEOUT_SECONDS,
    progress_log_interval_seconds: float = _DOWNLOAD_PROGRESS_LOG_INTERVAL_SECONDS,
) -> None:
    start = time.monotonic()
    previous_timeout = socket.getdefaulttimeout()
    last_log_at = start

    def _reporthook(block_count: int, block_size: int, total_size: int) -> None:
        nonlocal last_log_at
        now = time.monotonic()
        if (now - last_log_at) < float(progress_log_interval_seconds):
            return
        downloaded = int(max(0, block_count) * max(0, block_size))
        elapsed = max(now - start, 0.0)
        speed_mbps = (downloaded / elapsed / (1024.0 * 1024.0)) if elapsed > 0.0 else 0.0
        if total_size and total_size > 0:
            percent = min(100.0, max(0.0, (downloaded / float(total_size)) * 100.0))
            _LOGGER.warning(
                '%s download in progress: %.1f%% (%d/%d MB) at %.2f MB/s',
                label,
                percent,
                downloaded // (1024 * 1024),
                int(total_size) // (1024 * 1024),
                speed_mbps,
            )
        else:
            _LOGGER.warning(
                '%s download in progress: %d MB at %.2f MB/s',
                label,
                downloaded // (1024 * 1024),
                speed_mbps,
            )
        last_log_at = now

    try:
        socket.setdefaulttimeout(float(timeout_seconds))
        urllib.request.urlretrieve(url, destination_path, reporthook=_reporthook)
    finally:
        socket.setdefaulttimeout(previous_timeout)

    elapsed = max(time.monotonic() - start, 0.0)
    size_bytes = os.path.getsize(destination_path) if os.path.exists(destination_path) else 0
    _LOGGER.warning(
        '%s download complete: %d MB in %.1fs',
        label,
        int(size_bytes) // (1024 * 1024),
        elapsed,
    )


def _candidate_paths(raw_path: str) -> list[Path]:
    candidate = Path(str(raw_path).strip()).expanduser()
    if candidate.is_absolute():
        return [candidate]
    return [Path.cwd() / candidate, _REPO_ROOT / candidate]


def _normalize_cookie_lines(lines: list[str]) -> tuple[list[str], int, int]:
    normalized: list[str] = []
    repaired = 0
    dropped = 0
    index = 0
    while index < len(lines):
        line = str(lines[index]).rstrip('\r\n')
        if not line or line.startswith('#'):
            normalized.append(line)
            index += 1
            continue

        parts = line.split('\t')
        if len(parts) == 7:
            normalized.append(line)
            index += 1
            continue

        if len(parts) == 4 and index + 1 < len(lines):
            continuation = str(lines[index + 1]).rstrip('\r\n')
            continuation_parts = continuation.split('\t')
            combined = f'{line}{continuation}'
            if continuation.startswith('\t') and len(continuation_parts) == 4 and len(combined.split('\t')) == 7:
                normalized.append(combined)
                repaired += 1
                index += 2
                continue

        dropped += 1
        index += 1

    return normalized, repaired, dropped


def _sanitize_cookiefile_for_yt_dlp(source_path: Path) -> str | None:
    source_stat = source_path.stat()
    lines = source_path.read_text(encoding='utf-8', errors='replace').splitlines()
    normalized_lines, repaired, dropped = _normalize_cookie_lines(lines)
    data_lines = [line for line in normalized_lines if line and not line.startswith('#')]
    if not data_lines:
        _LOGGER.warning('Ignoring cookie file %s because it does not contain any valid Netscape cookie rows', source_path)
        return None
    has_netscape_header = any(line.strip().startswith('# Netscape HTTP Cookie File') for line in normalized_lines if line.strip())
    if repaired == 0 and dropped == 0 and has_netscape_header:
        return str(source_path)

    cache_dir = _REPO_ROOT / 'data' / 'cache' / 'yt_dlp'
    cache_dir.mkdir(parents=True, exist_ok=True)
    path_hash = hashlib.sha1(str(source_path.resolve(strict=False)).encode('utf-8')).hexdigest()[:12]
    sanitized_path = cache_dir / f'{source_path.stem}_{path_hash}_{source_stat.st_mtime_ns}_{source_stat.st_size}_sanitized.txt'
    with _COOKIEFILE_SANITIZE_LOCK:
        if sanitized_path.exists():
            return str(sanitized_path)
        sanitized_text = '# Netscape HTTP Cookie File\n' + '\n'.join(data_lines).rstrip('\n') + '\n'
        fd, temp_path = tempfile.mkstemp(prefix='mai_cookie_', suffix='.tmp', dir=str(cache_dir))
        os.close(fd)
        try:
            Path(temp_path).write_text(sanitized_text, encoding='utf-8')
            os.replace(temp_path, sanitized_path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        if repaired or dropped or not has_netscape_header:
            _LOGGER.warning(
                'Sanitized cookie file %s into %s (repaired=%d, dropped=%d)',
                source_path,
                sanitized_path,
                repaired,
                dropped,
            )
    return str(sanitized_path)


def resolve_yt_dlp_cookiefile(explicit_path: str | None = None) -> str | None:
    candidates: list[Path] = []
    if explicit_path is not None and str(explicit_path).strip():
        candidates.extend(_candidate_paths(str(explicit_path)))
    else:
        env_path = str(os.getenv('MAI_YTDLP_COOKIEFILE') or '').strip()
        if env_path:
            candidates.extend(_candidate_paths(env_path))
        else:
            candidates.append(Path.cwd() / 'cookies.txt')
            candidates.append(_REPO_ROOT / 'cookies.txt')

    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False))
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.is_file():
            candidate_stat = candidate.stat()
            with _COOKIEFILE_CACHE_LOCK:
                cached = _COOKIEFILE_CACHE.get(normalized)
                if cached is not None and cached[:2] == (candidate_stat.st_mtime_ns, candidate_stat.st_size):
                    return cached[2]
            resolved = _sanitize_cookiefile_for_yt_dlp(candidate)
            with _COOKIEFILE_CACHE_LOCK:
                _COOKIEFILE_CACHE[normalized] = (candidate_stat.st_mtime_ns, candidate_stat.st_size, resolved)
            return resolved
    return None


def _ensure_windows_deno_runtime(cache_root: str | None = None) -> dict[str, dict[str, Any]]:
    global _DENO_RUNTIME_INSTALL_FAILED

    if os.name != 'nt':
        return {}

    cache_root_path = str(cache_root or (_REPO_ROOT / 'data'))
    tools_dir = os.path.join(cache_root_path, 'tools', 'deno')
    os.makedirs(tools_dir, exist_ok=True)
    deno_exe = os.path.join(tools_dir, 'deno.exe')
    if _is_usable_js_runtime_binary(deno_exe):
        return {'deno': {'path': deno_exe}}
    if os.path.exists(deno_exe):
        _LOGGER.warning('Existing deno runtime at %s is unusable; redownloading a compatible build', deno_exe)
        try:
            os.remove(deno_exe)
        except OSError:
            pass
    if _DENO_RUNTIME_INSTALL_FAILED:
        return {}

    with _JS_RUNTIME_INSTALL_LOCK:
        if _is_usable_js_runtime_binary(deno_exe):
            return {'deno': {'path': deno_exe}}
        if _DENO_RUNTIME_INSTALL_FAILED:
            return {}
        zip_path = os.path.join(tools_dir, 'deno.zip')
        try:
            deno_zip_url = _deno_windows_zip_url()
            _LOGGER.warning('No JS runtime found for yt-dlp; downloading deno to %s', tools_dir)
            _download_file_with_progress(
                deno_zip_url,
                zip_path,
                label='deno runtime',
            )
            with zipfile.ZipFile(zip_path, 'r') as zf:
                deno_member = next((m for m in zf.namelist() if m.lower().endswith('/deno.exe') or m.lower() == 'deno.exe'), None)
                if not deno_member:
                    raise RuntimeError('deno zip did not contain deno.exe')
                zf.extract(deno_member, tools_dir)
            extracted_deno = os.path.join(tools_dir, deno_member.replace('/', os.sep))
            if os.path.normcase(extracted_deno) != os.path.normcase(deno_exe):
                os.replace(extracted_deno, deno_exe)
            if not _is_usable_js_runtime_binary(deno_exe):
                raise RuntimeError(f'downloaded deno binary is unusable at {deno_exe}')
            return {'deno': {'path': deno_exe}}
        except Exception as exc:
            _LOGGER.warning('failed to auto-install deno JS runtime: %r', exc)
            _DENO_RUNTIME_INSTALL_FAILED = True
            return {}


def ensure_yt_dlp_js_runtime(cache_root: str | None = None) -> dict[str, dict[str, Any]]:
    for runtime_name in _RUNTIME_CANDIDATES:
        runtime_path = shutil.which(runtime_name)
        if runtime_path:
            return {runtime_name: {'path': runtime_path}}
    return _ensure_windows_deno_runtime(cache_root=cache_root)


def _has_ffmpeg_and_ffprobe(ffmpeg_dir: str | None = None) -> bool:
    def which(name: str) -> str | None:
        if ffmpeg_dir:
            candidate = os.path.join(ffmpeg_dir, name)
            if os.path.exists(candidate):
                return candidate
        return shutil.which(name)

    return which('ffmpeg') is not None and which('ffprobe') is not None


def ensure_yt_dlp_ffmpeg_location(cache_root: str | None = None) -> str | None:
    if _has_ffmpeg_and_ffprobe():
        return None

    if os.name != 'nt':
        return None

    cache_root_path = str(cache_root or (_REPO_ROOT / 'data'))
    tools_dir = os.path.join(cache_root_path, 'tools', 'ffmpeg')
    bin_dir = os.path.join(tools_dir, 'bin')
    os.makedirs(bin_dir, exist_ok=True)

    ffmpeg_exe = os.path.join(bin_dir, 'ffmpeg.exe')
    ffprobe_exe = os.path.join(bin_dir, 'ffprobe.exe')
    if os.path.exists(ffmpeg_exe) and os.path.exists(ffprobe_exe):
        return bin_dir

    with _FFMPEG_INSTALL_LOCK:
        if os.path.exists(ffmpeg_exe) and os.path.exists(ffprobe_exe):
            return bin_dir
        zip_path = os.path.join(tools_dir, 'ffmpeg-release-essentials.zip')
        try:
            _LOGGER.warning('ffmpeg/ffprobe not found; downloading to %s', tools_dir)
            _download_file_with_progress(
                _GYAN_FFMPEG_ESSENTIALS_ZIP,
                zip_path,
                label='ffmpeg bundle',
            )
            with zipfile.ZipFile(zip_path, 'r') as zf:
                members = zf.namelist()
                ffmpeg_member = next((m for m in members if m.lower().endswith('/bin/ffmpeg.exe')), None)
                ffprobe_member = next((m for m in members if m.lower().endswith('/bin/ffprobe.exe')), None)
                if not ffmpeg_member or not ffprobe_member:
                    raise RuntimeError('ffmpeg zip did not contain expected executables')
                zf.extract(ffmpeg_member, tools_dir)
                zf.extract(ffprobe_member, tools_dir)
            extracted_ffmpeg = os.path.join(tools_dir, ffmpeg_member.replace('/', os.sep))
            extracted_ffprobe = os.path.join(tools_dir, ffprobe_member.replace('/', os.sep))
            os.replace(extracted_ffmpeg, ffmpeg_exe)
            os.replace(extracted_ffprobe, ffprobe_exe)
            return bin_dir
        except Exception as exc:
            _LOGGER.warning('failed to auto-install ffmpeg: %r', exc)
            return None


def apply_yt_dlp_auth_options(ydl_opts: dict[str, Any], cookiefile: str | None = None) -> dict[str, Any]:
    prepared = _ensure_jsless_youtube_player_config(dict(ydl_opts or {}))
    prepared.setdefault('logger', _YtDlpLogger())
    prepared.setdefault('cachedir', _default_yt_dlp_cache_setting())
    skip_js_player = _youtube_player_skips_js(prepared)
    if 'js_runtimes' not in prepared and not skip_js_player:
        detected_runtimes = _filter_default_js_runtimes(ensure_yt_dlp_js_runtime())
        if detected_runtimes:
            prepared['js_runtimes'] = detected_runtimes
    if not skip_js_player:
        remote_components = list(prepared.get('remote_components') or [])
        if _REMOTE_EJS_COMPONENT not in remote_components:
            remote_components.append(_REMOTE_EJS_COMPONENT)
        prepared['remote_components'] = remote_components
    ffmpeg_location = ensure_yt_dlp_ffmpeg_location()
    if ffmpeg_location:
        prepared.setdefault('ffmpeg_location', ffmpeg_location)
    resolved_cookiefile = resolve_yt_dlp_cookiefile(cookiefile)
    if resolved_cookiefile:
        prepared['cookiefile'] = resolved_cookiefile
    return prepared
