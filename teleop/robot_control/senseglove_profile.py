"""
SenseGlove Nova 2 → Inspire configuration profiles.

Every tunable of the SenseGlove/Inspire pipeline (finger mapping, thumb
saturation, force feedback, touch thresholds, mount offset, timing and glove
namespaces) lives in a profile identified by an ID, e.g. "SG-003".
Profiles are defined in teleop/config/senseglove_profiles.yaml and selected
with --sg-profile. The resolved profile is stored in each recorded episode so
results can be compared per configuration.

A profile may declare `base: <other ID>` and only override what changes.
Anything not set falls back to DEFAULT_PROFILE (the original hand-tuned values).
"""

import copy
import hashlib
import json
import os

import yaml

DEFAULT_PROFILES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'config', 'senseglove_profiles.yaml')

DEFAULT_PROFILE = {
    'description': '',
    'gloves': {
        'left_ns': '/senseglove/glove00799/lh',
        'right_ns': '/senseglove/glove00768/rh',
    },
    'mapping': {
        # (open, closed) in rad for the four long fingers
        'finger_ranges': {
            'index_mcp':  [0.0, 1.2],
            'index_pip':  [0.0, 1.6],
            'index_dip':  [0.0, 1.5],
            'middle_mcp': [0.0, 1.1],
            'middle_pip': [0.0, 1.6],
            'middle_dip': [0.0, 1.4],
            'ring_mcp':   [0.0, 1.0],
            'ring_pip':   [0.0, 1.5],
            'ring_dip':   [0.0, 1.3],
            'pinky_mcp':  [0.0, 1.0],
            'pinky_pip':  [0.0, 1.5],
            'pinky_dip':  [0.0, 1.5],
        },
        'thumb': {
            # Nova 2 left/right abduction come out with opposite signs
            'yaw_invert': {'left': True, 'right': False},
            'adapt_inner_frac': 0.12,
            'adapt_min_span': 0.05,
            'pitch_close_frac': 0.28,
        },
    },
    'force_feedback': {
        'enabled': True,
        # Inspire force_act units (0-4096)
        'deadzone': 200,
        'max': 3300,
        # Multiplier on the 0-100 brake command; 0 disables that finger
        'gain': {'thumb': 1.0, 'index': 1.0, 'middle': 1.0, 'ring': 1.0},
    },
    'touch': {
        # Inspire tactile units → 0-100 vibration level
        'deadzone': 50,
        'max': 3300,
    },
    'vibration': {
        # Inspire touch → Nova 2 LRA motors
        'enabled': True,
        # Multiplier on the 0-100 level; 0 disables that motor
        'gain': {'thumb_tip': 1.0, 'index_tip': 1.0, 'palm_index': 1.0, 'palm_pinky': 1.0},
        'frequency_hz': 180.0,
        # Must be longer than 1 / timing.bridge_rate_hz to feel continuous
        'pulse_s': 0.1,
    },
    'inspire': {
        # One number for all motors or a list of 6 in Inspire motor order
        # [pinky, ring, middle, index, thumb_bend, thumb_rotation].
        # null keeps the value the hand has at power-on.
        # Each finger stops closing when force_act reaches force_set.
        'force_set': None,
        'speed_set': None,
        # Re-send period so a hand that restarts or connects late gets them too
        'resend_s': 2.0,
    },
    'mount_offset': {
        'left_euler_deg': [180.0, 180.0, 0.0],
        'right_euler_deg': [-180.0, 180.0, 0.0],
        'pos': [0.0, 0.0, 0.0],
    },
    'timing': {
        'bridge_rate_hz': 20.0,
        'control_fps': 100.0,
    },
}


INSPIRE_NUM_MOTORS = 6
# Valid register ranges of the Inspire FTP/DFTP
INSPIRE_SETTING_RANGES = {
    'force_set': (0, 3000),
    'speed_set': (0, 1000),
}


class SenseGloveProfileError(ValueError):
    pass


def _inspire_setting(value, name):
    """None, a number or 6 numbers → None or a list of 6 ints in range."""
    if value is None:
        return None
    values = [value] * INSPIRE_NUM_MOTORS if isinstance(value, (int, float)) else list(value)
    lo, hi = INSPIRE_SETTING_RANGES[name]
    if len(values) != INSPIRE_NUM_MOTORS or not all(
            isinstance(v, (int, float)) and lo <= v <= hi for v in values):
        raise SenseGloveProfileError(
            f"'inspire.{name}' must be null, a number or {INSPIRE_NUM_MOTORS} numbers "
            f"in [{lo}, {hi}], got {value!r}")
    return [int(round(v)) for v in values]


def _deep_merge(base, override, path=''):
    out = copy.deepcopy(base)
    for key, value in override.items():
        key_path = f'{path}.{key}' if path else key
        if key not in out:
            raise SenseGloveProfileError(f"Unknown profile key '{key_path}'")
        if isinstance(out[key], dict):
            if not isinstance(value, dict):
                raise SenseGloveProfileError(f"'{key_path}' must be a mapping")
            out[key] = _deep_merge(out[key], value, key_path)
        else:
            out[key] = value
    return out


class SenseGloveProfile:
    def __init__(self, profile_id, params, source_file=None):
        self.id = profile_id
        self.params = params
        self.source_file = source_file

    def __getitem__(self, key):
        return self.params[key]

    @property
    def description(self):
        return self.params.get('description', '')

    @property
    def params_hash(self):
        """Changes whenever the resolved values change, even if the ID is reused."""
        blob = json.dumps(self.params, sort_keys=True).encode('utf-8')
        return hashlib.sha1(blob).hexdigest()[:10]

    def glove_topics(self, side):
        """(joint_states, haptics_commands, vibration_commands) topics of one glove."""
        ns = self.params['gloves'][f'{side}_ns'].rstrip('/')
        return f'{ns}/joint_states', f'{ns}/haptics_commands', f'{ns}/vibration_commands'

    def inspire_settings(self):
        """(force_set, speed_set) as lists of 6 ints, or None to leave untouched."""
        cfg = self.params['inspire']
        return (_inspire_setting(cfg['force_set'], 'force_set'),
                _inspire_setting(cfg['speed_set'], 'speed_set'))

    def with_glove_ns(self, left_ns=None, right_ns=None):
        params = copy.deepcopy(self.params)
        if left_ns:
            params['gloves']['left_ns'] = left_ns
        if right_ns:
            params['gloves']['right_ns'] = right_ns
        return SenseGloveProfile(self.id, params, self.source_file)

    def to_info(self):
        return {
            'id': self.id,
            'description': self.description,
            'params_hash': self.params_hash,
            'source_file': self.source_file,
            'params': self.params,
        }


def _resolve(profile_id, raw_profiles, seen):
    if profile_id not in raw_profiles:
        available = ', '.join(sorted(raw_profiles)) or '(none)'
        raise SenseGloveProfileError(
            f"SenseGlove profile '{profile_id}' not found. Available: {available}")
    if profile_id in seen:
        chain = ' -> '.join(seen + [profile_id])
        raise SenseGloveProfileError(f"Circular 'base' in SenseGlove profiles: {chain}")

    raw = dict(raw_profiles[profile_id] or {})
    base_id = raw.pop('base', None)
    if base_id is None:
        base = DEFAULT_PROFILE
    else:
        base = _resolve(base_id, raw_profiles, seen + [profile_id])
    return _deep_merge(base, raw)


def list_profiles(path=DEFAULT_PROFILES_FILE):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data.get('profiles', {}) or {}


def load_profile(profile_id=None, path=DEFAULT_PROFILES_FILE):
    """Load and resolve a profile. profile_id=None uses the file's default_profile."""
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    raw_profiles = data.get('profiles', {}) or {}
    if profile_id is None:
        profile_id = data.get('default_profile')
        if profile_id is None:
            raise SenseGloveProfileError(
                f"No --sg-profile given and no 'default_profile' in {path}")

    params = _resolve(profile_id, raw_profiles, [])
    profile = SenseGloveProfile(profile_id, params, os.path.abspath(path))
    profile.inspire_settings()
    return profile
