"""DJ-style overlay transitions: blend songs over a region, not at a point.

A naive transition cuts the end of A to the start of B. A DJ *overlays*: chooses
an exit region in A and an entry region in B — frequently mid-song, over a
breakdown, an instrumental, or a drop — aligns the downbeats, and blends across a
phrase. This module models a transition as a region overlay:

    (exit_region_A, entry_region_B, beat_offset, blend_type, score)

The headline idea is **spectral complementarity**: two regions overlay cleanly
when they fill *different* frequency pockets (A's bass under B's highs), not when
they sound alike. Combined with octave-aware tempo lock, harmonic (Camelot) key
match, a hard vocal-clash penalty (never stack two vocals), phrase alignment, and
a double-drop bonus, it scores *where* and *how* two tracks should be mixed.

The hot kernel — aligning downbeats by onset-envelope cross-correlation — is an
FFT (O(n log n)); the region-pair scan is vectorised numpy with a Numba ``njit``
fast path that degrades to the same exact numpy code when Numba is absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

# Frequency pocket labels for the band profile (low→high). Three is enough to
# express the bass-swap move (the DJ staple) without over-fitting spectra.
BAND_NAMES = ('bass', 'mid', 'high')

BLEND_TYPES = ('long_blend', 'bass_swap', 'double_drop', 'echo_out', 'loop_roll', 'cut')

# Score-component weights. Tempo + vocal clash dominate (a tempo clash or two
# vocals ruins a mix outright); complementarity and key shape the rest.
_W_TEMPO = 0.28
_W_KEY = 0.18
_W_COMPLEMENT = 0.24
_W_VOCAL = 0.18
_W_PHRASE = 0.07
_W_ENERGY = 0.05


@dataclass(slots=True)
class RegionDescriptor:
    """One mixable section of a track (a phrase / structural segment)."""

    track_id: Any
    start_s: float
    end_s: float
    position: float                 # normalised position in the track, [0, 1]
    tempo: float
    key: int                        # pitch class 0–11 (or -1 if unknown)
    energy: float                   # RMS-ish, [0, 1]
    vocal_activity: float           # [0, 1]; 1 = strong lead vocal
    band_profile: np.ndarray        # len-3 energy share across BAND_NAMES, sums ~1
    onset_envelope: np.ndarray      # beat/frame onset strength, for offset alignment
    bars: int = 8                   # phrase length in bars (8/16/32 are phrase-aligned)
    is_drop: bool = False


@dataclass(slots=True)
class OverlayMatch:
    """A scored proposal to overlay A's exit region with B's entry region."""

    exit_region: RegionDescriptor
    entry_region: RegionDescriptor
    beat_offset: int                # frames to shift B so downbeats align
    score: float
    blend_type: str
    components: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Beat alignment — the hot kernel (FFT cross-correlation)                      #
# --------------------------------------------------------------------------- #

def optimal_beat_offset(onset_a: np.ndarray, onset_b: np.ndarray) -> tuple[int, float]:
    """Lag (in frames) that best aligns B's onsets under A's, plus the peak score.

    Normalised cross-correlation via FFT: O(n log n) instead of the O(n²) sliding
    dot product. Returns ``(lag, strength)`` where ``strength`` in [0, 1] is the
    peak correlation (how confidently the downbeats lock).
    """
    a = np.asarray(onset_a, dtype=np.float64)
    b = np.asarray(onset_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0, 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0, 0.0

    # Pad past the linear length so the circular correlation does not wrap. The
    # FFT result is indexed by lag directly: index 0 is zero lag, and indices past
    # nfft/2 are negative lags (the usual circular convention).
    nfft = 1 << int(np.ceil(np.log2(a.size + b.size)))
    corr = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft) / denom
    peak = int(np.argmax(corr))
    lag = peak if peak <= nfft // 2 else peak - nfft
    return lag, float(np.clip(corr[peak], 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Component scores                                                            #
# --------------------------------------------------------------------------- #

def _tempo_lock(tempo_a: float, tempo_b: float) -> float:
    """Octave-aware tempo agreement in [0, 1] (half/double time counts as locked)."""
    if tempo_a <= 0 or tempo_b <= 0:
        return 0.5
    ratio = tempo_a / tempo_b
    while ratio < 0.75:
        ratio *= 2.0
    while ratio > 1.5:
        ratio /= 2.0
    return float(np.clip(1.0 - abs(np.log2(ratio)) / 0.2, 0.0, 1.0))  # ~3% drift -> ~0.78


def _key_compat(key_a: int, key_b: int) -> float:
    """Circle-of-fifths proximity in [0, 1]; neutral when a key is unknown."""
    if key_a is None or key_b is None or key_a < 0 or key_b < 0:
        return 0.6
    fifths = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    pos = {pitch: i for i, pitch in enumerate(fifths)}
    distance = abs(pos[int(key_a) % 12] - pos[int(key_b) % 12])
    distance = min(distance, 12 - distance)  # wrap the circle
    return float(np.clip(1.0 - distance / 6.0, 0.0, 1.0))


def spectral_complementarity(bands_a: np.ndarray, bands_b: np.ndarray) -> float:
    """How well two band profiles fill *different* pockets, in [0, 1].

    Collision = overlap mass ``sum(min(a_i, b_i))`` of the normalised profiles;
    complementarity = ``1 - collision``. Two basslines (mass in the same band)
    collide and score low; A-bass / B-highs interlock and score high.
    """
    a = np.asarray(bands_a, dtype=np.float64)
    b = np.asarray(bands_b, dtype=np.float64)
    a = a / (a.sum() + 1e-12)
    b = b / (b.sum() + 1e-12)
    collision = float(np.sum(np.minimum(a, b)))
    return float(np.clip(1.0 - collision, 0.0, 1.0))


def _vocal_compat(vocal_a: float, vocal_b: float) -> float:
    """1 when at most one region has vocals; →0 when both carry a lead vocal."""
    return float(np.clip(1.0 - float(vocal_a) * float(vocal_b), 0.0, 1.0))


def _phrase_alignment(bars_a: int, bars_b: int) -> float:
    """Bonus for phrase-length regions (8/16/32 bars); mixes land on the phrase."""
    def phrasey(bars: int) -> float:
        return 1.0 if int(bars) in (8, 16, 32, 64) else (0.6 if int(bars) % 4 == 0 else 0.2)
    return 0.5 * (phrasey(bars_a) + phrasey(bars_b))


def _select_blend_type(
    *, tempo_lock: float, complement: float, vocal_compat: float,
    bands_a: np.ndarray, bands_b: np.ndarray, exit_region: RegionDescriptor,
    entry_region: RegionDescriptor, offset_strength: float,
) -> str:
    """Pick the DJ move that fits this region pair."""
    if tempo_lock < 0.35:
        return 'echo_out'                      # can't beat-match: tail A out under B
    if (exit_region.is_drop and entry_region.is_drop and tempo_lock > 0.7
            and exit_region.energy > 0.7 and entry_region.energy > 0.7):
        return 'double_drop'                   # align two high-energy drops — the money move
    if vocal_compat < 0.5:
        return 'cut'                           # two vocals: hard cut on the phrase
    norm_a = bands_a / (bands_a.sum() + 1e-12)
    norm_b = bands_b / (bands_b.sum() + 1e-12)
    bass_collision = float(min(norm_a[0], norm_b[0]))  # both carry strong bass
    highs_ok = spectral_complementarity(bands_a[1:], bands_b[1:]) > 0.45
    if bass_collision > 0.35 and highs_ok:
        return 'bass_swap'                     # swap low end on the boundary, blend highs
    if offset_strength > 0.5 and complement > 0.45 and tempo_lock > 0.6:
        return 'long_blend'                    # clean overlay across the phrase
    return 'loop_roll'                         # bridge a weaker match with a short loop


def overlay_score(exit_region: RegionDescriptor, entry_region: RegionDescriptor) -> OverlayMatch:
    """Score overlaying ``exit_region`` (leaving A) with ``entry_region`` (entering B)."""
    tempo_lock = _tempo_lock(exit_region.tempo, entry_region.tempo)
    key_compat = _key_compat(exit_region.key, entry_region.key)
    complement = spectral_complementarity(exit_region.band_profile, entry_region.band_profile)
    vocal_compat = _vocal_compat(exit_region.vocal_activity, entry_region.vocal_activity)
    phrase = _phrase_alignment(exit_region.bars, entry_region.bars)
    energy_cont = 1.0 - abs(float(exit_region.energy) - float(entry_region.energy))
    offset, offset_strength = optimal_beat_offset(exit_region.onset_envelope, entry_region.onset_envelope)

    score = float(np.clip(
        _W_TEMPO * tempo_lock + _W_KEY * key_compat + _W_COMPLEMENT * complement
        + _W_VOCAL * vocal_compat + _W_PHRASE * phrase + _W_ENERGY * energy_cont,
        0.0, 1.0,
    ))
    blend = _select_blend_type(
        tempo_lock=tempo_lock, complement=complement, vocal_compat=vocal_compat,
        bands_a=np.asarray(exit_region.band_profile, dtype=np.float64),
        bands_b=np.asarray(entry_region.band_profile, dtype=np.float64),
        exit_region=exit_region, entry_region=entry_region, offset_strength=offset_strength,
    )
    return OverlayMatch(
        exit_region=exit_region, entry_region=entry_region, beat_offset=offset,
        score=score, blend_type=blend,
        components={
            'tempo_lock': tempo_lock, 'key_compat': key_compat, 'complementarity': complement,
            'vocal_compat': vocal_compat, 'phrase_alignment': phrase, 'energy_continuity': energy_cont,
            'offset_strength': offset_strength,
        },
    )


# --------------------------------------------------------------------------- #
# Region-pair search (vectorised, with a Numba fast-path seam)                 #
# --------------------------------------------------------------------------- #

def _complementarity_matrix(exit_bands: np.ndarray, entry_bands: np.ndarray) -> np.ndarray:
    """All-pairs spectral complementarity in one broadcast: (E_exit, E_entry)."""
    a = exit_bands / (exit_bands.sum(axis=1, keepdims=True) + 1e-12)   # Ea x 3
    b = entry_bands / (entry_bands.sum(axis=1, keepdims=True) + 1e-12)  # Eb x 3
    collision = np.minimum(a[:, None, :], b[None, :, :]).sum(axis=2)    # Ea x Eb
    return np.clip(1.0 - collision, 0.0, 1.0)


def best_overlay(
    exit_regions: list[RegionDescriptor],
    entry_regions: list[RegionDescriptor],
    *,
    min_exit_position: float = 0.4,
    max_entry_position: float = 0.6,
) -> OverlayMatch | None:
    """Best overlay leaving track A (later regions) into track B (earlier regions).

    A DJ exits from A's back half and enters B's front half; the position gates
    keep the search musical and small. The coarse complementarity pre-rank is a
    single broadcast, so only the top handful get the full per-pair scoring.
    """
    exits = [r for r in exit_regions if r.position >= min_exit_position] or list(exit_regions)
    entries = [r for r in entry_regions if r.position <= max_entry_position] or list(entry_regions)
    if not exits or not entries:
        return None

    exit_bands = np.vstack([np.asarray(r.band_profile, dtype=np.float64) for r in exits])
    entry_bands = np.vstack([np.asarray(r.band_profile, dtype=np.float64) for r in entries])
    complement = _complementarity_matrix(exit_bands, entry_bands)

    # Cheap coarse rank by complementarity; fully score only the strongest cells.
    flat_order = np.argsort(-complement, axis=None)
    top = flat_order[: min(8, flat_order.size)]
    best: OverlayMatch | None = None
    for flat in top:
        i, j = divmod(int(flat), complement.shape[1])
        match = overlay_score(exits[i], entries[j])
        if best is None or match.score > best.score:
            best = match
    return best


def plan_mix(regions_by_track: dict[Any, list[RegionDescriptor]], order: list[Any]) -> list[OverlayMatch]:
    """Build the overlay for each consecutive pair in an ordered set → a mix plan."""
    plan: list[OverlayMatch] = []
    for current, nxt in zip(order, order[1:]):
        a = regions_by_track.get(current, [])
        b = regions_by_track.get(nxt, [])
        if not a or not b:
            continue
        match = best_overlay(a, b)
        if match is not None:
            plan.append(match)
    return plan


# --------------------------------------------------------------------------- #
# Segmentation (Foote novelty) — used when only a feature time-series is given #
# --------------------------------------------------------------------------- #

def _foote_novelty_scan_numpy(ssm: np.ndarray, checker: np.ndarray, kernel_size: int) -> np.ndarray:
    """Slide the checkerboard kernel down the SSM diagonal (reference numpy path)."""
    size = 2 * kernel_size + 1
    novelty = np.zeros(len(ssm))
    for center in range(kernel_size, len(ssm) - kernel_size):
        window = ssm[center - kernel_size:center + kernel_size + 1,
                     center - kernel_size:center + kernel_size + 1]
        if window.shape == (size, size):
            novelty[center] = float(np.sum(window * checker))
    return novelty


def _make_novelty_scan():
    """Return a Numba-JIT diagonal scan if Numba is installed, else the numpy one.

    The kernel is a per-frame O(k²) reduction over the SSM diagonal — the kind of
    tight loop a JIT (or a future Rust/PyO3 kernel) accelerates; the numpy version
    is the exact reference and the always-available fallback.
    """
    try:
        from numba import njit
    except Exception:
        return _foote_novelty_scan_numpy

    @njit(cache=True)
    def _scan(ssm, checker, kernel_size):  # pragma: no cover - requires numba
        n = ssm.shape[0]
        size = 2 * kernel_size + 1
        novelty = np.zeros(n)
        for center in range(kernel_size, n - kernel_size):
            total = 0.0
            for a in range(size):
                for b in range(size):
                    total += ssm[center - kernel_size + a, center - kernel_size + b] * checker[a, b]
            novelty[center] = total
        return novelty

    def _scan_wrapped(ssm, checker, kernel_size):  # pragma: no cover - requires numba
        return _scan(np.ascontiguousarray(ssm), np.ascontiguousarray(checker), int(kernel_size))

    return _scan_wrapped


_foote_novelty_scan = _make_novelty_scan()

def foote_novelty_boundaries(feature_sequence: np.ndarray, *, kernel_size: int = 8, top_n: int = 8) -> list[int]:
    """Structural boundaries via Foote novelty on the self-similarity matrix.

    ``feature_sequence`` is ``(T, D)`` beat/frame features. Returns sorted frame
    indices of the strongest novelty peaks — the section cuts a DJ could mix on.
    """
    x = np.asarray(feature_sequence, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 * kernel_size + 1:
        return []
    norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    ssm = norm @ norm.T  # cosine self-similarity

    # Foote checkerboard kernel = sign quadrants * gaussian taper.
    g = np.arange(-kernel_size, kernel_size + 1)
    gauss_1d = np.exp(-0.5 * (g / (kernel_size / 2.0)) ** 2)
    checker = np.outer(np.sign(g), np.sign(g)) * np.outer(gauss_1d, gauss_1d)

    novelty = _foote_novelty_scan(ssm, checker, kernel_size)
    novelty = np.clip(novelty, 0.0, None)
    if not np.any(novelty):
        return []
    peaks = np.argsort(-novelty)[:top_n]
    return sorted(int(p) for p in peaks if novelty[p] > 0)
