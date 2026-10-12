"""Frozen, zero-phase low-pass for the post-stimulus window of the PRIME decider.

Why this exists
---------------
At 1 kHz the post-stimulus window was band-limited by the polyphase resampler (101-tap
Kaiser FIR, 500 Hz cutoff).  With ``post_sfreq`` = 5 kHz nothing resamples that window any
more, so it is broadband.  This module puts a 500 Hz low-pass back -- with 51 taps instead
of 101, so a blank edge is smeared by +-4.8 ms instead of +-10 ms.

The filter
----------
    h = scipy.signal.firwin(numtaps, cutoff_hz, window=("kaiser", 5.0), fs=sfreq)
    numtaps = 2 * round(halfwidth_ms * sfreq / 1000) + 1          # 51 at 5 kHz / 5 ms

Applied as ONE centred convolution (``scipy.ndimage.convolve1d``): zero delay, DC gain 1,
gain 0.93 at 400 Hz / 0.50 at 500 Hz / 0.07 at 600 Hz, reach +-5 ms (+-4.8 ms in practice:
at 5 kHz / 500 Hz the two outermost taps are exactly zero).  Epoch edges are padded by
replicating the edge sample, never with zeros.

What it must not be
-------------------
* ``lfilter``         causal: 25 samples = 5 ms delay, moves the N45 out of the 38-50 ms window.
* ``filtfilt``        doubles the reach to +-10 ms and squares the gain.
* ``mne`` ``filter`` its defaults design their own ~133-tap filter.
* zero padding        -40 % at the first sample of the epoch.

Where it goes
-------------
Directly AFTER the first ``fix_stim_artifact`` (``artifact_window_1``) -- so it only smears the
small edge of the blank (microvolts), never the TMS pulse (millivolts) -- and BEFORE the
post bad-channel check / interpolation / ICA, in BOTH ``preprocess_calibration`` and
``Preprocessor.preprocess_post``.

Calibrate -> freeze -> apply
----------------------------
Blink thresholds, MAD statistics, SOUND, SSP-SIR and the bad-channel AUC are all calibrated
on filtered data and then applied to single trials, so calibration and every live pulse must
use the *same* filter.  Therefore:

1. calibration: ``lp = PostLowpass.design(cutoff_hz, halfwidth_ms, post_sfreq)`` (None = off,
   or cutoff >= Nyquist, e.g. a 1 kHz run), apply it to all calibration epochs, and store
   ``lp.to_stamp()`` in ``calibration_params["post_lowpass"]`` (settings AND the taps);
2. live / replay / offline: ``lp = PostLowpass.from_bundle(calibration_params)`` -- built from
   the stamp only, never from the config.  No stamp (legacy bundle) -> None -> no filtering,
   so old bundles replay exactly as before.

The module needs only numpy and scipy; ``apply_to_epochs`` merely duck-types an MNE Epochs.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping

import numpy as np
import scipy
from scipy.ndimage import convolve1d
from scipy.signal import firwin

__all__ = ["PostLowpass", "STAMP_KEY", "half_width_samples"]

STAMP_KEY = "post_lowpass"   # key in calibration_params (the bundle)
_STAMP_VERSION = 1
# pad_mode -> scipy.ndimage mode.  'edge' replicates the edge sample (numpy 'edge');
# 'reflect' mirrors around it without repeating it (numpy 'reflect' == ndimage 'mirror').
_NDIMAGE_MODE = {"edge": "nearest", "reflect": "mirror"}
_PAD_TEXT = {"edge": "edge-replicating padding", "reflect": "mirror padding"}
_ATOL_SYMMETRY = 1e-12
_ATOL_DC_GAIN = 1e-9
_TAP_EPS = 1e-12             # taps below this fraction of the largest one do not count for the reach


def half_width_samples(halfwidth_ms: float, sfreq: float) -> int:
    """One-sided filter length in samples for a half-width in ms (rounded half up)."""
    return int(math.floor(float(halfwidth_ms) * 1e-3 * float(sfreq) + 0.5))


class PostLowpass:
    """Immutable, calibrated post-stimulus low-pass.

    Build it with :meth:`design` at calibration, and with :meth:`from_bundle` /
    :meth:`from_stamp` everywhere else.  Do not re-design it on the live path.
    """

    __slots__ = ("_taps", "_sfreq", "_cutoff_hz", "_halfwidth_ms", "_kaiser_beta", "_pad_mode")

    def __init__(self, taps, *, sfreq, cutoff_hz, halfwidth_ms, kaiser_beta=5.0, pad_mode="edge"):
        taps = np.array(taps, dtype=np.float64)          # private copy
        sfreq, cutoff_hz = float(sfreq), float(cutoff_hz)
        halfwidth_ms, kaiser_beta = float(halfwidth_ms), float(kaiser_beta)

        if not (math.isfinite(sfreq) and sfreq > 0):
            raise ValueError(f"sfreq must be a positive number, got {sfreq}")
        if not (math.isfinite(cutoff_hz) and 0 < cutoff_hz < sfreq / 2):
            raise ValueError(f"cutoff_hz must be in (0, {sfreq / 2:g}) Hz, got {cutoff_hz}")
        if pad_mode not in _NDIMAGE_MODE:
            raise ValueError(f"pad_mode must be one of {sorted(_NDIMAGE_MODE)}, got {pad_mode!r} "
                             "(zero padding is deliberately not offered)")
        if taps.ndim != 1 or taps.size < 3 or taps.size % 2 == 0:
            raise ValueError("taps must be a 1-D array of odd length >= 3 (zero phase needs a centre tap)")
        if not np.all(np.isfinite(taps)):
            raise ValueError("taps contain NaN/inf")
        if not np.allclose(taps, taps[::-1], rtol=0.0, atol=_ATOL_SYMMETRY):
            raise ValueError("taps are not symmetric -- not a zero-phase filter")
        if abs(taps.sum() - 1.0) > _ATOL_DC_GAIN:
            raise ValueError(f"taps sum to {taps.sum():.12g}, DC gain must be 1")
        half = (taps.size - 1) // 2
        if half != half_width_samples(halfwidth_ms, sfreq):
            raise ValueError(f"{taps.size} taps do not match halfwidth_ms={halfwidth_ms:g} at {sfreq:g} Hz "
                             f"(expected {2 * half_width_samples(halfwidth_ms, sfreq) + 1})")

        taps.setflags(write=False)
        self._taps = taps
        self._sfreq = sfreq
        self._cutoff_hz = cutoff_hz
        self._halfwidth_ms = halfwidth_ms
        self._kaiser_beta = kaiser_beta
        self._pad_mode = pad_mode

    # ------------------------------------------------------------------ construction

    @classmethod
    def design(cls, cutoff_hz, halfwidth_ms, sfreq, *, kaiser_beta=5.0, pad_mode="edge",
               on_nyquist="skip"):
        """Calibrate: derive the filter from (cutoff in Hz, half-width in ms) at ``sfreq``.

        Returns None when the filter is switched off (``cutoff_hz`` None/0) or -- with
        ``on_nyquist="skip"`` -- when the cutoff is not below Nyquist (e.g. ``post_sfreq`` 1 kHz,
        where the resampler already band-limits; a RuntimeWarning says so).  With
        ``on_nyquist="raise"`` that case is a ValueError instead.
        """
        if not cutoff_hz:
            return None
        if on_nyquist not in ("skip", "raise"):
            raise ValueError(f"on_nyquist must be 'skip' or 'raise', got {on_nyquist!r}")
        if halfwidth_ms is None:
            raise ValueError("halfwidth_ms is required when a cutoff is set")
        cutoff_hz, halfwidth_ms, sfreq = float(cutoff_hz), float(halfwidth_ms), float(sfreq)
        if not (math.isfinite(sfreq) and sfreq > 0):
            raise ValueError(f"sfreq must be a positive number, got {sfreq}")
        if not (math.isfinite(cutoff_hz) and cutoff_hz > 0):
            raise ValueError(f"cutoff_hz must be a positive number, got {cutoff_hz}")
        if cutoff_hz >= sfreq / 2:
            msg = (f"post low-pass cutoff {cutoff_hz:g} Hz is not below the Nyquist frequency "
                   f"{sfreq / 2:g} Hz of the post window (post_sfreq={sfreq:g} Hz)")
            if on_nyquist == "raise":
                raise ValueError(msg)
            warnings.warn(msg + "; low-pass skipped", RuntimeWarning, stacklevel=2)
            return None
        half = half_width_samples(halfwidth_ms, sfreq)
        if half < 1:
            raise ValueError(f"halfwidth_ms={halfwidth_ms:g} is less than one sample at {sfreq:g} Hz")
        taps = firwin(2 * half + 1, cutoff_hz, window=("kaiser", float(kaiser_beta)), fs=sfreq)
        return cls(taps, sfreq=sfreq, cutoff_hz=cutoff_hz, halfwidth_ms=halfwidth_ms,
                   kaiser_beta=kaiser_beta, pad_mode=pad_mode)

    @classmethod
    def from_stamp(cls, stamp):
        """Rebuild the frozen filter from ``to_stamp()`` output.  The stored taps are used as they
        are (never re-designed), after the same sanity checks as in the constructor."""
        if not isinstance(stamp, Mapping):
            raise ValueError(f"'{STAMP_KEY}' stamp must be a dict, got {type(stamp).__name__}")
        missing = [k for k in ("version", "cutoff_hz", "halfwidth_ms", "sfreq", "numtaps", "taps")
                   if k not in stamp]
        if missing:
            raise ValueError(f"'{STAMP_KEY}' stamp is missing {missing}")
        if int(stamp["version"]) != _STAMP_VERSION:
            raise ValueError(f"unsupported '{STAMP_KEY}' stamp version {stamp['version']} "
                             f"(this code reads version {_STAMP_VERSION})")
        taps = np.asarray(stamp["taps"], dtype=np.float64)
        if int(stamp["numtaps"]) != taps.size:
            raise ValueError(f"'{STAMP_KEY}' stamp says {stamp['numtaps']} taps but stores {taps.size}")
        return cls(taps, sfreq=stamp["sfreq"], cutoff_hz=stamp["cutoff_hz"],
                   halfwidth_ms=stamp["halfwidth_ms"], kaiser_beta=stamp.get("kaiser_beta", 5.0),
                   pad_mode=stamp.get("pad_mode", "edge"))

    @classmethod
    def from_bundle(cls, calibration_params):
        """The filter a bundle was calibrated with, or None for a bundle without the stamp
        (legacy bundles, or a session that ran without the filter)."""
        stamp = calibration_params.get(STAMP_KEY)
        return None if stamp is None else cls.from_stamp(stamp)

    @classmethod
    def check_bundle(cls, calibration_params, *, post_sfreq, n_times=None):
        """Fail loudly if the bundle's stamp does not fit the rate / epoch length it will run at.

        For ``Preprocessor._check_bundle_matches_rates``.  No stamp -> nothing to check.
        ``n_times`` is the length of the unprocessed post epoch (the one the filter sees).
        """
        lp = cls.from_bundle(calibration_params)     # raises on a malformed stamp
        if lp is None:
            return
        lp.check_rate(post_sfreq)
        if n_times is not None:
            lp.check_length(n_times)

    def to_stamp(self):
        """Plain-python/numpy dict for ``calibration_params[STAMP_KEY]`` (safe for np.save/pickle)."""
        return {
            "version": _STAMP_VERSION,
            "kind": "fir_zero_phase_lowpass",
            "design": "scipy.signal.firwin(numtaps, cutoff_hz, window=('kaiser', kaiser_beta), fs=sfreq); "
                      "single centred scipy.ndimage.convolve1d, edge-replicating padding",
            "cutoff_hz": self._cutoff_hz,
            "halfwidth_ms": self._halfwidth_ms,
            "sfreq": self._sfreq,
            "numtaps": int(self._taps.size),
            "kaiser_beta": self._kaiser_beta,
            "pad_mode": self._pad_mode,
            "taps": self._taps.copy(),          # authoritative: replay uses these taps, not a re-design
            "scipy_version": scipy.__version__,
        }

    def summary(self):
        """Compact settings record (no taps) for ``proc_info.npy`` and logs."""
        return {
            "cutoff_hz": self._cutoff_hz,
            "halfwidth_ms": self._halfwidth_ms,
            "numtaps": int(self._taps.size),
            "sfreq": self._sfreq,
            "window": "kaiser",
            "kaiser_beta": self._kaiser_beta,
            "pad_mode": self._pad_mode,
            "reach_ms": round(self.reach_ms, 3),
            "zero_phase": True,
        }

    def describe(self):
        """One line for the console / log (the same wording goes into the session printout)."""
        return (f"post low-pass: {self._cutoff_hz:g} Hz, {self._taps.size}-tap zero-phase FIR "
                f"(Kaiser beta={self._kaiser_beta:g}), reach +-{self.reach_ms:.1f} ms at "
                f"{self._sfreq:g} Hz, {_PAD_TEXT[self._pad_mode]}")

    def __repr__(self):
        return f"PostLowpass({self.describe()[len('post low-pass: '):]})"

    def __reduce__(self):                       # pickle / deepcopy through the validated stamp
        return (type(self).from_stamp, (self.to_stamp(),))

    # ------------------------------------------------------------------ properties

    @property
    def taps(self):
        return self._taps                       # read-only view

    @property
    def sfreq(self):
        return self._sfreq

    @property
    def cutoff_hz(self):
        return self._cutoff_hz

    @property
    def halfwidth_ms(self):
        return self._halfwidth_ms

    @property
    def kaiser_beta(self):
        return self._kaiser_beta

    @property
    def pad_mode(self):
        return self._pad_mode

    @property
    def numtaps(self):
        return int(self._taps.size)

    @property
    def half(self):
        """Samples from the centre tap to the end of the filter."""
        return (self._taps.size - 1) // 2

    @property
    def reach_samples(self):
        """Samples a sample can influence on each side (outermost near-zero taps ignored)."""
        significant = np.flatnonzero(np.abs(self._taps) > _TAP_EPS * np.abs(self._taps).max())
        return int(self.half - significant[0])

    @property
    def reach_ms(self):
        return 1e3 * self.reach_samples / self._sfreq

    # ------------------------------------------------------------------ use

    def gain(self, freqs_hz):
        """Real (zero-phase) amplitude response at ``freqs_hz``."""
        f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
        k = np.arange(-self.half, self.half + 1)
        return np.cos(2.0 * np.pi * np.outer(f, k) / self._sfreq) @ self._taps

    def apply(self, data, axis=-1):
        """Filter ``data`` along ``axis`` (time).  Returns a new array; ``data`` is not modified.

        Shape-independent: filtering a whole (n_trials, n_channels, n_times) calibration array
        and filtering each trial alone give bit-identical results per trial.
        """
        x = np.asarray(data)
        if x.dtype.kind != "f":
            x = x.astype(np.float64)
        n = x.shape[axis]
        if n < self.numtaps:
            raise ValueError(f"{n} samples along axis {axis} is shorter than the {self.numtaps}-tap low-pass")
        return convolve1d(x, self._taps, axis=axis, mode=_NDIMAGE_MODE[self._pad_mode])

    def apply_to_epochs(self, epochs):
        """Filter a preloaded MNE ``Epochs``/``EpochsArray`` in place (like ``fix_stim_artifact``)
        and return it.  Refuses epochs whose sampling rate differs from the one designed for."""
        self.check_rate(epochs.info["sfreq"])
        if not getattr(epochs, "preload", False):
            raise ValueError("epochs must be preloaded")
        data = epochs._data
        data[...] = self.apply(data, axis=-1)
        return epochs

    # ------------------------------------------------------------------ guards

    def check_rate(self, sfreq):
        if abs(float(sfreq) - self._sfreq) > 1e-6:
            raise ValueError(f"post low-pass was designed at {self._sfreq:g} Hz but the post window "
                             f"runs at {float(sfreq):g} Hz -- bundle and post_sfreq disagree")

    def check_length(self, n_times):
        if n_times < self.numtaps:
            raise ValueError(f"post epoch has {n_times} samples, fewer than the {self.numtaps}-tap low-pass")

    def same_as(self, other):
        """True if ``other`` is the same filter (identical taps and settings)."""
        return (isinstance(other, PostLowpass)
                and np.array_equal(self._taps, other._taps)
                and self._sfreq == other._sfreq
                and self._cutoff_hz == other._cutoff_hz
                and self._halfwidth_ms == other._halfwidth_ms
                and self._kaiser_beta == other._kaiser_beta
                and self._pad_mode == other._pad_mode)


if __name__ == "__main__":      # quick look at the default design
    _lp = PostLowpass.design(500.0, 5.0, 5000.0)
    print(_lp.describe())
    for _f in (100, 300, 400, 450, 500, 600, 750):
        print(f"  gain @ {_f:4d} Hz: {_lp.gain(_f)[0]:.4f}")
