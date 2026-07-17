# Declarative Settings Registry — Design (plan item 5.3)

Status: **proposal for review** (no code changes). 2026-07-16.

## Problem

Every user-visible setting currently lives in four-to-five parallel,
hand-synchronized systems (project memory calls this out as the
standing pattern):

1. a widget + default in `SettingsDialog.setup_ui` (1,642 lines),
2. a getter (`get_grid_color()` style),
3. `save_settings` (~276 lines, hand-built nested dict) and
   `restore_settings` (~333 lines, hand-read with fallbacks),
4. factory reset,
5. script-state save/restore (the recording feature captures its own
   copy of the state dict).

Concrete trace for one setting (`grid_color`): a canvas attribute
(`neolyzer.py:576`), a canvas setter (`:1150`), a QLineEdit + default
duplicated as a literal (`:10546`), a getter with the default
duplicated again (`:10583`), plus its save/restore/reset/script
entries. Adding one checkbox is a five-place edit; the July review
found the save/restore pair already drifted (missing keys, and the
schema version was frozen at '3.06' for two releases).

## Proposal

A single registry table, one row per setting:

```python
# settings_registry.py (or a SettingsDialog class attribute)
Setting = namedtuple('Setting', 'key widget_attr kind default')

SETTINGS = [
    Setting('display.grid_color',      'grid_color_edit',      'text',    '#888888'),
    Setting('display.projection',      'projection_combo',     'choice',  'Rectangular'),
    Setting('filters.mag_max',         'mag_max_spin',         'float',   22.0),
    Setting('milky_way.enabled',       'mw_enable_check',      'bool',    False),
    # ...
]
```

`kind` maps to one (get, set) implementation per widget class —
QLineEdit/text, QComboBox/choice, QCheckBox/bool, QSpinBox/int,
QDoubleSpinBox/float, color swatch, etc. Then:

- `save_settings`  = `{s.key: get(s) for s in SETTINGS}` + version
- `restore_settings` = loop with per-key fallback to `s.default`
  (unknown keys ignored, missing keys defaulted — forward and
  backward compatible by construction)
- factory reset = `set(s, s.default)` loop
- script state = same serializer, possibly a `script=False` flag on
  rows that shouldn't be captured in recordings
- migrations = explicit table `{old_key: new_key}` applied before the
  restore loop, keyed off the file's version string (now single-
  sourced in `src/version.py`)

The five systems collapse to one table plus ~50 lines of machinery.
New setting = one registry row (the widget itself still gets created
in `setup_ui`, which decomposing is plan item 6.2's business).

## Why not QSettings?

QSettings would replace only the JSON file, not the four-way
duplication, and would move state into platform-specific locations
(registry/plists) that are harder to inspect, sync, or ship with bug
reports. The JSON file at `~/.neolyzer/` (per plan 6.3) stays.

## Prototype scope (before committing to the pattern)

Convert **one group** — the Milky Way settings (~10 widgets, all four
systems, includes a color and a slider) — while leaving the legacy
dict entries in place for every other group. Acceptance: settings
file round-trips byte-identically for untouched groups; the Milky Way
group survives save → restore → factory reset → script record/replay;
suite + smoke test green. If the prototype holds, migrate group by
group (each its own commit), then delete the hand-written pairs.

## Risks

- `restore_settings` currently has order dependencies (some setters
  trigger `on_settings_changed` cascades); the registry loop must
  block signals during restore, as the current code does piecemeal.
- Script replay compatibility: old recordings store the legacy dict —
  the migration table must cover recorded state too (same code path).
