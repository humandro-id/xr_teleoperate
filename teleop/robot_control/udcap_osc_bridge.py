"""
Shared OSC bridge for UDCAP haptic gloves.

Listens on a UDP/OSC port, decodes UDCAP per-finger flex bits
(4-bit per segment, max of 3 segments per finger plus a thumb 'spread'
channel) and writes a hand-specific motor array into two shared
multiprocessing Arrays (left and right) via a user-supplied `mapper`
callback.

This module is hand-agnostic: it knows nothing about Inspire, Brainco,
etc. Concrete controllers (`Inspire_Controller_UDCAP`,
`Brainco_Controller_UDCAP`, …) supply their own mapper that converts
the normalised UDCAP flex/spread values into their own motor convention.

The decoded UDCAP OSC addresses look like:

    /<prefix>/<prefix>/<prefix>/RightIndex11
    /<prefix>/<prefix>/<prefix>/LeftThumbspread2

where the last path component is "<Side><Finger><joint><bit>" or
"<Side><Finger>spread<bit>" with bit ∈ {1, 2, 4, 8} and
joint ∈ {1, 2, 3}. Each segment value is the sum of active bits, in
[0, 15].
"""

import os
import threading
import time

import numpy as np

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

# Set UDCAP_DEBUG=1 (or "true") in the env to dump raw flex/spread values
# every second to the console — useful to diagnose stuck sensors.
_DEBUG = os.environ.get("UDCAP_DEBUG", "0").lower() not in ("", "0", "false", "no")

try:
    from pythonosc import dispatcher as osc_dispatcher
    from pythonosc import osc_server
    _OSC_AVAILABLE = True
except ImportError:
    _OSC_AVAILABLE = False
    logger_mp.warning(
        "python-osc no disponible. Instalar con: pip install python-osc"
    )

# ── UDCAP OSC defaults ────────────────────────────────────────────────────
DEFAULT_UDCAP_HOST = "127.0.0.1"
DEFAULT_UDCAP_PORT = 9000

# Finger names as emitted by UDCAP (case-sensitive after Left/Right prefix)
UDCAP_FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")

# Per-finger segment selection — which of {j1, j2, j3} to use for the flex
# calculation, per hand. Some UDCAP units have stuck/dead sensors that
# differ between left and right (e.g. right thumb j2 stuck at 15), so this
# is configurable per side.
#
# Override at runtime with env vars (comma-separated segment indices):
#   UDCAP_LEFT_THUMB_SEGMENTS=1,2
#   UDCAP_RIGHT_THUMB_SEGMENTS=1,3
#   UDCAP_LEFT_INDEX_SEGMENTS=1,2,3
#   ...etc, pattern: UDCAP_<LEFT|RIGHT>_<FINGER>_SEGMENTS
DEFAULT_FINGER_SEGMENTS = {
    "left": {
        "Thumb":  (1, 2),
        "Index":  (1, 2, 3),
        "Middle": (1, 2, 3),
        "Ring":   (1, 2, 3),
        "Pinky":  (1, 2, 3),
    },
    "right": {
        # Right thumb j2 is stuck-high on this unit; use j1 + j3 only.
        "Thumb":  (1, 3),
        "Index":  (1, 2, 3),
        "Middle": (1, 2, 3),
        "Ring":   (1, 2, 3),
        "Pinky":  (1, 2, 3),
    },
}

# Per-finger gain applied after baseline subtraction; raise if curl feels weak.
DEFAULT_UDCAP_FINGER_GAIN = 1.0
# EMA on the per-finger flex (0=no smoothing/instant, 1=full latest value).
# Higher = more reactive (less lag) but slightly noisier.
DEFAULT_UDCAP_FLEX_SMOOTH = 0.7
# How often the OSC bits → motor array mapping runs (Hz).
DEFAULT_UDCAP_MAP_RATE_HZ = 50.0

# Auto-calibration: during the first N seconds we sample the raw flex and
# treat a percentile of that window as the "open-hand baseline". After
# that, every reading is shifted so that the baseline maps to 0 and the
# full 0..1 range is used for actual flexion → much better resolution for
# small finger curls. Hold your hands fully open while this runs.
DEFAULT_UDCAP_CALIB_SECONDS = 2.0
# Percentile of the calibration samples used as the baseline. Lower = more
# sensitive (smaller baseline) but more prone to false-positive flexion if
# the hand wasn't perfectly still while calibrating.
DEFAULT_UDCAP_CALIB_PERCENTILE = 75
# Hard cap on the learned baseline. If calibration accidentally captures a
# high value (user moved a finger, stuck sensor), this prevents the
# baseline from eating the full motion range.
_MAX_BASELINE = 0.4
# Floor for (1 - baseline) when rescaling, to avoid over-amplifying noise.
_MIN_USABLE_RANGE = 0.2


def _load_segments_from_env(defaults):
    """Build the {side: {finger: (segments,)}} table, applying any
    UDCAP_<LEFT|RIGHT>_<FINGER>_SEGMENTS env-var overrides."""
    out = {s: dict(defaults[s]) for s in ("left", "right")}
    for side in ("left", "right"):
        for finger in UDCAP_FINGERS:
            key = f"UDCAP_{side.upper()}_{finger.upper()}_SEGMENTS"
            val = os.environ.get(key)
            if not val:
                continue
            try:
                parsed = tuple(
                    int(x.strip()) for x in val.split(",")
                    if x.strip() and int(x.strip()) in (1, 2, 3)
                )
                if parsed:
                    out[side][finger] = parsed
                    logger_mp.info(
                        f"[UDCAPOSCBridge] env override {key}={parsed}"
                    )
            except ValueError:
                logger_mp.warning(
                    f"[UDCAPOSCBridge] could not parse {key}={val!r}"
                )
    return out


class UDCAPOSCBridge:
    """OSC server that decodes UDCAP finger-flex bits and writes
    hand-specific motor commands into shared Arrays via a user-supplied
    `mapper` callback.

    The `mapper` signature is::

        mapper(side, flex_dict, thumb_spread) -> list[float]

    where:
      - `side`         : "left" or "right"
      - `flex_dict`    : {"Thumb": 0..1, "Index": 0..1, ...} (1.0 = closed)
      - `thumb_spread` : float in [0, 1] from UDCAP thumb spread bits
      - returns       : list of length matching the shared Array

    Values are already EMA-smoothed when handed to the mapper.
    """

    def __init__(
        self,
        left_mapped_array,
        right_mapped_array,
        mapper,
        host=DEFAULT_UDCAP_HOST,
        port=DEFAULT_UDCAP_PORT,
        finger_gain=DEFAULT_UDCAP_FINGER_GAIN,
        flex_smooth=DEFAULT_UDCAP_FLEX_SMOOTH,
        map_rate_hz=DEFAULT_UDCAP_MAP_RATE_HZ,
        finger_segments=None,
        debug=None,
        calibration_seconds=DEFAULT_UDCAP_CALIB_SECONDS,
        log_tag="UDCAPOSCBridge",
    ):
        if not _OSC_AVAILABLE:
            raise RuntimeError(
                "python-osc is required for UDCAP. "
                "Install with: pip install python-osc"
            )

        self._left_arr = left_mapped_array
        self._right_arr = right_mapped_array
        self._mapper = mapper
        self._finger_gain = float(finger_gain)
        self._flex_smooth = float(flex_smooth)
        self._map_rate_hz = float(map_rate_hz)
        self._finger_segments = (
            _load_segments_from_env(DEFAULT_FINGER_SEGMENTS)
            if finger_segments is None
            else finger_segments
        )
        self._debug = _DEBUG if debug is None else bool(debug)
        self._last_debug_t = 0.0
        self._log_tag = log_tag

        # finger_bits[side][finger][segment][bit] = 0/1
        # segment ∈ {1, 2, 3, 'spread'}, bit ∈ {1, 2, 4, 8}
        self._bits_lock = threading.Lock()
        self._finger_bits = {
            s: {
                f: {k: {1: 0, 2: 0, 4: 0, 8: 0}
                    for k in (1, 2, 3, "spread")}
                for f in UDCAP_FINGERS
            }
            for s in ("left", "right")
        }

        # EMA state for each finger flex (kept in mapping thread only).
        self._flex_ema = {
            "left":  {f: 0.0 for f in UDCAP_FINGERS},
            "right": {f: 0.0 for f in UDCAP_FINGERS},
        }
        # EMA state for thumb spread.
        self._thumb_spread_ema = {"left": 0.0, "right": 0.0}

        # ── Auto-calibration state ──
        # Allow disabling with UDCAP_NO_CALIB=1 (then baseline stays at 0).
        # Override duration with UDCAP_CALIB_SECONDS=<float>.
        env_no_calib = os.environ.get("UDCAP_NO_CALIB", "0").lower()
        env_calib_s = os.environ.get("UDCAP_CALIB_SECONDS")
        if env_calib_s:
            try:
                calibration_seconds = float(env_calib_s)
            except ValueError:
                pass
        self._calib_seconds = float(calibration_seconds)
        self._calib_enabled = env_no_calib in ("", "0", "false", "no") \
            and self._calib_seconds > 0.0
        self._calibrated = not self._calib_enabled
        self._calib_started_t = None
        self._calib_samples = {
            "left":  {f: [] for f in UDCAP_FINGERS},
            "right": {f: [] for f in UDCAP_FINGERS},
        }
        self._spread_calib_samples = {"left": [], "right": []}
        self._baseline = {
            "left":  {f: 0.0 for f in UDCAP_FINGERS},
            "right": {f: 0.0 for f in UDCAP_FINGERS},
        }
        self._spread_baseline = {"left": 0.0, "right": 0.0}

        # ── OSC server ──
        # Try the requested host first; if bind fails (e.g. user passed the
        # remote PC's IP instead of one of the robot's interfaces), fall back
        # to 0.0.0.0 so the controller doesn't silently start without data.
        disp = osc_dispatcher.Dispatcher()
        disp.set_default_handler(self._osc_handler)
        try:
            self._server = osc_server.ThreadingOSCUDPServer((host, port), disp)
            bound_host = host
        except OSError as e:
            if host not in ("0.0.0.0", "", None):
                logger_mp.warning(
                    f"[{self._log_tag}] Cannot bind to {host}:{port} ({e}). "
                    f"Falling back to 0.0.0.0:{port}."
                )
                self._server = osc_server.ThreadingOSCUDPServer(
                    ("0.0.0.0", port), disp)
                bound_host = "0.0.0.0"
            else:
                raise
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        logger_mp.info(
            f"[{self._log_tag}] OSC listening on {bound_host}:{port}"
        )

        # ── Mapping timer (decodes bits → motor array @ map_rate_hz) ──
        self._running = True
        self._map_thread = threading.Thread(target=self._map_loop, daemon=True)
        self._map_thread.start()

    # ---- OSC handler -----------------------------------------------------

    def _osc_handler(self, addr, *args):
        """Decode one OSC address into (side, finger, segment, bit, value)."""
        parts = addr.split("/")
        if len(parts) < 4:
            return
        # The trailing element holds the side+finger+segment+bit token.
        nm = parts[-1] if parts[-1] else parts[-2]
        if nm.startswith("Right"):
            side, rest = "right", nm[5:]
        elif nm.startswith("Left"):
            side, rest = "left", nm[4:]
        else:
            return

        finger = next((x for x in UDCAP_FINGERS if rest.startswith(x)), None)
        if not finger:
            return
        rest = rest[len(finger):]

        if "spread" in rest:
            try:
                bit = int(rest.split("spread")[1])
            except ValueError:
                return
            key = "spread"
        else:
            if len(rest) != 2 or not rest.isdigit():
                return
            key = int(rest[0])
            bit = int(rest[1])
        if bit not in (1, 2, 4, 8):
            return
        if key != "spread" and key not in (1, 2, 3):
            return

        v = 1 if (args and args[0] in (True, 1, "True", "true")) else 0
        with self._bits_lock:
            self._finger_bits[side][finger][key][bit] = v

    # ---- bits → flex mapping --------------------------------------------

    def _segments_for(self, side, finger):
        """Return the segments tuple to use for `(side, finger)`."""
        side_map = self._finger_segments.get(side)
        if isinstance(side_map, dict) and finger in side_map:
            return side_map[finger]
        # Back-compat: allow flat {finger: tuple} dicts.
        flat = self._finger_segments.get(finger)
        if isinstance(flat, tuple):
            return flat
        return (1, 2, 3)

    def _decode_flex(self, raw_out=None):
        """Return {'left': {finger: flex01}, 'right': {finger: flex01}}.

        Per-finger flex = max(seg_value/15) over the segments configured
        in `self._finger_segments[side][finger]`. Values are raw (no
        baseline subtraction, no gain) and already clamped to [0, 1].
        1.0 = fully closed, 0.0 = fully open.

        If `raw_out` (dict) is provided, also fills it with per-segment
        raw values for debug logging:
            raw_out[side][finger] = [v_j1, v_j2, v_j3]  (each 0..15)
        """
        out = {"left": {}, "right": {}}
        with self._bits_lock:
            for s in ("left", "right"):
                if raw_out is not None:
                    raw_out[s] = {}
                for f in UDCAP_FINGERS:
                    segs = self._segments_for(s, f)
                    seg_vals = []
                    raw_segs = []
                    for j in (1, 2, 3):
                        b = self._finger_bits[s][f][j]
                        v = b[1] + b[2] * 2 + b[4] * 4 + b[8] * 8
                        raw_segs.append(v)
                        if j in segs:
                            seg_vals.append(v / 15.0)
                    if raw_out is not None:
                        raw_out[s][f] = raw_segs
                    flex = max(seg_vals) if seg_vals else 0.0
                    out[s][f] = float(np.clip(flex, 0.0, 1.0))
        return out

    def _apply_baseline(self, side, finger, raw_flex):
        """Subtract baseline and rescale, then apply gain. Returns 0..1."""
        b = self._baseline[side][finger]
        usable = max(1.0 - b, _MIN_USABLE_RANGE)
        v = (raw_flex - b) / usable
        v = max(0.0, v) * self._finger_gain
        return float(np.clip(v, 0.0, 1.0))

    def _apply_spread_baseline(self, side, raw_spread):
        b = self._spread_baseline[side]
        usable = max(1.0 - b, _MIN_USABLE_RANGE)
        v = (raw_spread - b) / usable
        return float(np.clip(max(0.0, v), 0.0, 1.0))

    def _decode_thumb_spread(self):
        """Return {'left': spread01, 'right': spread01} in [0, 1]."""
        out = {"left": 0.0, "right": 0.0}
        with self._bits_lock:
            for s in ("left", "right"):
                b = self._finger_bits[s]["Thumb"]["spread"]
                v = b[1] + b[2] * 2 + b[4] * 4 + b[8] * 8
                out[s] = float(np.clip(v / 15.0, 0.0, 1.0))
        return out

    def _map_loop(self):
        """Periodically translate UDCAP bits → motor shared Arrays."""
        period = 1.0 / self._map_rate_hz
        a = self._flex_smooth
        if self._calib_enabled:
            logger_mp.info(
                f"[{self._log_tag}] Calibrating open-hand baseline for "
                f"{self._calib_seconds:.1f}s — hold BOTH hands relaxed/open."
            )
        while self._running:
            t0 = time.time()
            raw = {} if self._debug else None
            flex = self._decode_flex(raw_out=raw)
            spread = self._decode_thumb_spread()

            # ── Calibration phase ──
            if not self._calibrated:
                if self._calib_started_t is None:
                    self._calib_started_t = t0
                for s in ("left", "right"):
                    for f in UDCAP_FINGERS:
                        self._calib_samples[s][f].append(flex[s][f])
                    self._spread_calib_samples[s].append(spread[s])
                if (t0 - self._calib_started_t) >= self._calib_seconds:
                    self._finalize_calibration()

            for side, target_arr in (
                ("left", self._left_arr),
                ("right", self._right_arr),
            ):
                # Apply baseline + gain BEFORE smoothing.
                for f in UDCAP_FINGERS:
                    adjusted = self._apply_baseline(side, f, flex[side][f])
                    self._flex_ema[side][f] = (
                        a * adjusted
                        + (1.0 - a) * self._flex_ema[side][f]
                    )
                adj_spread = self._apply_spread_baseline(side, spread[side])
                self._thumb_spread_ema[side] = (
                    a * adj_spread
                    + (1.0 - a) * self._thumb_spread_ema[side]
                )

                mapped = self._mapper(
                    side,
                    self._flex_ema[side],
                    self._thumb_spread_ema[side],
                )
                with target_arr.get_lock():
                    target_arr[:] = mapped

            if self._debug and (t0 - self._last_debug_t) >= 1.0:
                self._last_debug_t = t0
                self._print_debug(raw)

            sleep_t = max(0.0, period - (time.time() - t0))
            time.sleep(sleep_t)

    def _finalize_calibration(self):
        """Compute open-hand baseline from collected samples."""
        pct = DEFAULT_UDCAP_CALIB_PERCENTILE
        for s in ("left", "right"):
            for f in UDCAP_FINGERS:
                samples = self._calib_samples[s][f]
                if samples:
                    b = float(np.percentile(samples, pct))
                    # Cap to avoid eating the whole motion range if the
                    # calibration accidentally captured a high spike.
                    self._baseline[s][f] = min(b, _MAX_BASELINE)
            sp = self._spread_calib_samples[s]
            if sp:
                b = float(np.percentile(sp, pct))
                self._spread_baseline[s] = min(b, _MAX_BASELINE)
        # Free memory.
        self._calib_samples = None
        self._spread_calib_samples = None
        self._calibrated = True
        logger_mp.info(
            f"[{self._log_tag}] Calibration done (p{pct}, capped at "
            f"{_MAX_BASELINE}). "
            f"L baselines: { {f: round(self._baseline['left'][f], 2) for f in UDCAP_FINGERS} } "
            f"R baselines: { {f: round(self._baseline['right'][f], 2) for f in UDCAP_FINGERS} } "
            f"L spread baseline={self._spread_baseline['left']:.2f} "
            f"R spread baseline={self._spread_baseline['right']:.2f}"
        )

    def _print_debug(self, raw):
        """Pretty-print raw segment values, baseline, flex EMA and spread.

        Format per finger:  Thu:[ 0*/ 0*/ 0x] b=0.07 ema=0.00
        Where each segment shows its raw 0..15 value followed by:
          *  = segment is being used (configured in finger_segments)
          x  = segment is ignored
        b = baseline learned during open-hand calibration.
        ema = final smoothed value being sent to the hand (0..1).
        """
        try:
            status = "" if self._calibrated else " [CALIBRATING…]"
            for side in ("left", "right"):
                parts = []
                for f in UDCAP_FINGERS:
                    rs = raw[side][f] if raw else (0, 0, 0)
                    used = self._segments_for(side, f)
                    rs_str = "/".join(
                        f"{rs[i]:2d}" + ("*" if (i + 1) in used else "x")
                        for i in range(3)
                    )
                    parts.append(
                        f"{f[:3]}:[{rs_str}] "
                        f"b={self._baseline[side][f]:.2f} "
                        f"ema={self._flex_ema[side][f]:.2f}"
                    )
                parts.append(
                    f"Tspread={self._thumb_spread_ema[side]:.2f}"
                )
                logger_mp.info(
                    f"[{self._log_tag}][{side.upper()}]{status} "
                    + "  ".join(parts)
                )
        except Exception as e:
            logger_mp.warning(f"[{self._log_tag}] debug print error: {e}")

    # ---- lifecycle -------------------------------------------------------

    def shutdown(self):
        self._running = False
        try:
            self._server.shutdown()
        except Exception:
            pass
