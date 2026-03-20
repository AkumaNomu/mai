from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOGGER = logging.getLogger(__name__)
_REMOTE_EJS_COMPONENT = 'ejs:github'
_RUNTIME_CANDIDATES = ('deno', 'node', 'quickjs', 'bun')
_GYAN_FFMPEG_ESSENTIALS_ZIP = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
_DENO_WINDOWS_ZIP_URL = 'https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip'
_FFMPEG_INSTALL_LOCK = threading.Lock()
_JS_RUNTIME_INSTALL_LOCK = threading.Lock()
_COOKIEFILE_CACHE_LOCK = threading.Lock()
_COOKIEFILE_SANITIZE_LOCK = threading.Lock()
_COOKIEFILE_CACHE: dict[str, tuple[int, int, str | None]] = {}


class _YtDlpLogger:
    def debug(self, message: str) -> None:
        _LOGGER.debug(str(message))

    def info(self, message: str) -> None:
        _LOGGER.info(str(message))

    def warning(self, message: str) -> None:
        _LOGGER.warning(str(message))

    def error(self, message: str) -> None:
        _LOGGER.debug(str(message))


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


def ensure_yt_dlp_js_runtime(cache_root: str | None = None) -> dict[str, dict[str, Any]]:
    for runtime_name in _RUNTIME_CANDIDATES:
        runtime_path = shutil.which(runtime_name)
        if runtime_path:
            return {runtime_name: {'path': runtime_path}}

    if os.name != 'nt':
        return {}

    cache_root_path = str(cache_root or (_REPO_ROOT / 'data'))
    tools_dir = os.path.join(cache_root_path, 'tools', 'deno')
    os.makedirs(tools_dir, exist_ok=True)
    deno_exe = os.path.join(tools_dir, 'deno.exe')
    if os.path.exists(deno_exe):
        return {'deno': {'path': deno_exe}}

    with _JS_RUNTIME_INSTALL_LOCK:
        if os.path.exists(deno_exe):
            return {'deno': {'path': deno_exe}}
        zip_path = os.path.join(tools_dir, 'deno.zip')
        try:
            _LOGGER.warning('No JS runtime found for yt-dlp; downloading deno to %s', tools_dir)
            urllib.request.urlretrieve(_DENO_WINDOWS_ZIP_URL, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                deno_member = next((m for m in zf.namelist() if m.lower().endswith('/deno.exe') or m.lower() == 'deno.exe'), None)
                if not deno_member:
                    raise RuntimeError('deno zip did not contain deno.exe')
                zf.extract(deno_member, tools_dir)
            extracted_deno = os.path.join(tools_dir, deno_member.replace('/', os.sep))
            if os.path.normcase(extracted_deno) != os.path.normcase(deno_exe):
                os.replace(extracted_deno, deno_exe)
            return {'deno': {'path': deno_exe}}
        except Exception as exc:
            _LOGGER.warning('failed to auto-install deno JS runtime: %r', exc)
            return {}


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
            urllib.request.urlretrieve(_GYAN_FFMPEG_ESSENTIALS_ZIP, zip_path)
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
    prepared = dict(ydl_opts or {})
    prepared.setdefault('logger', _YtDlpLogger())
    if 'js_runtimes' not in prepared:
        detected_runtimes = ensure_yt_dlp_js_runtime()
        if detected_runtimes:
            prepared['js_runtimes'] = detected_runtimes
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
