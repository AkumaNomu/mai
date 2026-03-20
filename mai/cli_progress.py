from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TextIO


ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
RESET = '\x1b[0m'
DIM = '\x1b[2m'
BOLD = '\x1b[1m'
FG_RED = '\x1b[31m'
FG_GREEN = '\x1b[32m'
FG_YELLOW = '\x1b[33m'
FG_BLUE = '\x1b[34m'
FG_MAGENTA = '\x1b[35m'
FG_CYAN = '\x1b[36m'
FG_WHITE = '\x1b[37m'
FG_BRIGHT_BLACK = '\x1b[90m'
FG_BRIGHT_GREEN = '\x1b[92m'
FG_BRIGHT_CYAN = '\x1b[96m'

LOG_LEVEL_STYLES = {
    logging.DEBUG: (FG_BRIGHT_BLACK, 'DEBUG'),
    logging.INFO: (FG_CYAN, 'INFO '),
    logging.WARNING: (FG_YELLOW, 'WARN '),
    logging.ERROR: (FG_RED, 'ERROR'),
    logging.CRITICAL: (FG_RED + BOLD, 'FATAL'),
}
NOTE_STYLES = {
    'info': (FG_CYAN, 'INFO '),
    'success': (FG_GREEN, 'DONE '),
    'warning': (FG_YELLOW, 'WARN '),
    'error': (FG_RED, 'FAIL '),
    'section': (FG_MAGENTA, 'STEP '),
}
BAR_COLORS = [FG_CYAN, FG_BLUE, FG_MAGENTA, FG_GREEN, FG_BRIGHT_CYAN]
HEARTBEAT_FRAMES = ('-', '\\', '|', '/')


def _enable_windows_vt_mode() -> None:
    if os.name != 'nt':
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        for handle_id in (-11, -12):
            handle = kernel32.GetStdHandle(handle_id)
            if handle in (0, -1):
                continue
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        return


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub('', str(text))


def _visible_len(text: str) -> int:
    return len(_strip_ansi(text))


class ColorLogFormatter(logging.Formatter):
    def __init__(self, use_color: bool = True) -> None:
        super().__init__(fmt='%(levelname)s: %(message)s')
        self.use_color = bool(use_color)

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        color, label = LOG_LEVEL_STYLES.get(record.levelno, (FG_WHITE, record.levelname[:5].upper().ljust(5)))
        timestamp = time.strftime('%H:%M:%S')
        if not self.use_color:
            return f'[{timestamp}] {label} {message}'
        return f'{DIM}[{timestamp}]{RESET} {color}{label}{RESET} {message}'


def configure_cli_logging(level: int, stream: TextIO = sys.stderr, use_color: bool = True) -> None:
    handler = logging.StreamHandler(stream)
    handler.setFormatter(ColorLogFormatter(use_color=use_color))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


@dataclass
class CliProgressRenderer:
    stream: TextIO = sys.stderr
    width: int = 28
    enabled: bool = True
    color: bool | None = None
    heartbeat_interval: float = 2.0
    _active_label: str | None = field(default=None, init=False)
    _last_bucket: int = field(default=-1, init=False)
    _last_current: int = field(default=-1, init=False)
    _last_total: int = field(default=-1, init=False)
    _last_detail: str = field(default='', init=False)
    _last_render_length: int = field(default=0, init=False)
    _progress_started_at: float = field(default=0.0, init=False)
    _last_progress_event_at: float = field(default=0.0, init=False)
    _last_heartbeat_at: float = field(default=0.0, init=False)
    _heartbeat_index: int = field(default=0, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _heartbeat_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _enable_windows_vt_mode()
        if self.color is None:
            self.color = bool(self.is_tty)
        if self.enabled and self.is_tty and self.heartbeat_interval > 0:
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name='mai-cli-progress', daemon=True)
            self._heartbeat_thread.start()

    @property
    def is_tty(self) -> bool:
        return bool(getattr(self.stream, 'isatty', lambda: False)())

    @property
    def supports_color(self) -> bool:
        return bool(self.color and self.is_tty)

    def note(self, label: str, detail: str = '', tone: str = 'info') -> None:
        if not self.enabled:
            return
        with self._lock:
            self._write_line(self._format_note(label, detail=detail, tone=tone))

    def section(self, label: str, detail: str = '') -> None:
        self.note(label, detail=detail, tone='section')

    def success(self, label: str, detail: str = '') -> None:
        self.note(label, detail=detail, tone='success')

    def warning(self, label: str, detail: str = '') -> None:
        self.note(label, detail=detail, tone='warning')

    def error(self, label: str, detail: str = '') -> None:
        self.note(label, detail=detail, tone='error')

    def update(self, label: str, current: int, total: int, detail: str = '') -> None:
        if not self.enabled:
            return
        total = max(int(total), 0)
        if total <= 0:
            return
        current = min(max(int(current), 0), total)
        detail = str(detail or '').strip()
        now = time.monotonic()

        with self._lock:
            if self._active_label != label or current < self._last_current:
                self._progress_started_at = now
                self._heartbeat_index = 0
                self._last_heartbeat_at = 0.0
            self._last_progress_event_at = now

            if self.is_tty:
                if self._active_label is not None and self._active_label != label:
                    print(file=self.stream, flush=True)
                    self._last_render_length = 0
                message = self._format_progress_message(label, current, total, detail, elapsed_seconds=now - self._progress_started_at)
                padding = ' ' * max(self._last_render_length - _visible_len(message), 0)
                print(f'\r{message}{padding}', end='', file=self.stream, flush=True)
                self._active_label = label
                self._last_render_length = _visible_len(message)
                self._last_current = current
                self._last_total = total
                self._last_detail = detail
                if current >= total:
                    print(file=self.stream, flush=True)
                    self._reset_progress_state()
                return

            bucket = int((current / float(total)) * 10)
            emit_step = total <= 20 and current != self._last_current
            emit_bucket = bucket != self._last_bucket
            emit_label_change = label != self._active_label
            emit_detail_change = detail != self._last_detail
            if not (emit_step or emit_bucket or emit_label_change or emit_detail_change or current in {0, 1, total}):
                return
            print(
                self._format_progress_message(label, current, total, detail, elapsed_seconds=now - self._progress_started_at),
                file=self.stream,
                flush=True,
            )
            self._active_label = label
            self._last_bucket = bucket
            self._last_current = current
            self._last_total = total
            self._last_detail = detail
            if current >= total:
                self._reset_progress_state()

    def close(self) -> None:
        self._stop_event.set()
        heartbeat_thread = self._heartbeat_thread
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=max(self.heartbeat_interval, 0.1) + 0.2)
        with self._lock:
            if self._active_label is not None and self.is_tty:
                print(file=self.stream, flush=True)
            self._reset_progress_state()

    def _reset_progress_state(self) -> None:
        self._active_label = None
        self._last_bucket = -1
        self._last_current = -1
        self._last_total = -1
        self._last_detail = ''
        self._last_render_length = 0
        self._progress_started_at = 0.0
        self._last_progress_event_at = 0.0
        self._last_heartbeat_at = 0.0
        self._heartbeat_index = 0

    def _write_line(self, message: str) -> None:
        if self._active_label is not None and self.is_tty:
            print(file=self.stream, flush=True)
            self._reset_progress_state()
        print(message, file=self.stream, flush=True)

    def _bar_color(self, label: str) -> str:
        index = sum(ord(char) for char in str(label)) % len(BAR_COLORS)
        return BAR_COLORS[index]

    def _format_note(self, label: str, detail: str = '', tone: str = 'info') -> str:
        color, prefix = NOTE_STYLES.get(tone, NOTE_STYLES['info'])
        timestamp = time.strftime('%H:%M:%S')
        detail_text = f' {DIM}|{RESET} {detail}' if detail else ''
        if not self.supports_color:
            return f'[{timestamp}] {prefix} {label}{(" | " + detail) if detail else ""}'
        return f'{DIM}[{timestamp}]{RESET} {color}{prefix}{RESET} {BOLD}{label}{RESET}{detail_text}'

    def _format_progress_message(
        self,
        label: str,
        current: int,
        total: int,
        detail: str,
        *,
        elapsed_seconds: float | None = None,
        heartbeat_frame: str = '',
    ) -> str:
        ratio = current / float(total)
        filled = min(self.width, max(0, int(round(self.width * ratio))))
        elapsed_text = f'{max(int(round(float(elapsed_seconds or 0.0))), 0)}s'
        spinner_text = heartbeat_frame or '*'
        bar_color = self._bar_color(label)
        if self.supports_color:
            filled_bar = f'{bar_color}{"#" * filled}{RESET}'
            remaining_bar = f'{FG_BRIGHT_BLACK}{"-" * (self.width - filled)}{RESET}'
            bar = filled_bar + remaining_bar
            timestamp = f'{DIM}[{time.strftime("%H:%M:%S")}]{RESET}'
            label_text = f'{BOLD}{label}{RESET}'
            count_text = f'{FG_WHITE}{current}/{total}{RESET}'
            percent_text = f'{FG_BRIGHT_CYAN}{ratio:.0%}{RESET}'
            detail_text = f' {DIM}|{RESET} {detail}' if detail else ''
            activity_text = f' {DIM}|{RESET} {FG_BRIGHT_BLACK}{spinner_text}{RESET} {FG_BRIGHT_BLACK}{elapsed_text}{RESET}'
            return f'{timestamp} {label_text} [{bar}] {count_text} {percent_text}{detail_text}{activity_text}'

        bar = '#' * filled + '-' * (self.width - filled)
        message = f'[{time.strftime("%H:%M:%S")}] {label}: [{bar}] {current}/{total} ({ratio:.0%})'
        if detail:
            message = f'{message} | {detail}'
        return f'{message} | {spinner_text} {elapsed_text}'

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(0.2):
            if not self.enabled or not self.is_tty or self.heartbeat_interval <= 0:
                continue
            with self._lock:
                if self._active_label is None or self._last_current < 0:
                    continue
                now = time.monotonic()
                if self._last_progress_event_at <= 0:
                    continue
                if now - self._last_progress_event_at < self.heartbeat_interval:
                    continue
                if self._last_heartbeat_at > 0 and now - self._last_heartbeat_at < self.heartbeat_interval:
                    continue
                self._heartbeat_index = (self._heartbeat_index + 1) % len(HEARTBEAT_FRAMES)
                message = self._format_progress_message(
                    self._active_label,
                    self._last_current,
                    max(self._last_total, 1),
                    self._last_detail,
                    elapsed_seconds=now - self._progress_started_at,
                    heartbeat_frame=HEARTBEAT_FRAMES[self._heartbeat_index],
                )
                padding = ' ' * max(self._last_render_length - _visible_len(message), 0)
                print(f'\r{message}{padding}', end='', file=self.stream, flush=True)
                self._last_render_length = _visible_len(message)
                self._last_heartbeat_at = now
