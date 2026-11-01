"""Read API for the NWB release of the LGN dataset (DANDI).  One file; needs only numpy, pandas and h5py (not pynwb).

    import lgndata_nwb as lgn                    # or: from lgn_tools import lgndata_nwb as lgn
    lgn.set_data_root('/path/to/dandiset')       # the folder that contains sub-Expt1 ... sub-Expt4  (after `dandi download DANDI:<id>`)
    cells = lgn.find_cells(modality='NAT', regime='unique')          # which cells have long natural-movie data?
    c = lgn.load_cell(1, 0, 3, cells.trial_key.iloc[0])               # one cell's spikes, frame times and stimulus for one recording
    r = lgn.load_repeat_cell(...)                                      # repeat recordings: spikes per repetition, ready for lgn.compute_psth

The function names and the returned fields are those of the lab's original reader (`lgndata_tools`), so analysis code written for it runs unchanged; only the
data root differs.  Differences, all deliberate: repeat recordings are always returned in the normalised `per_unit` form (as `load_repeat_trial` does);
`lib=` arguments are accepted and ignored (the stimulus arrays are read from the stimulus library, see below); free-text fields are the public wording of the release.

The stimulus ARRAYS are not in the recording files: all files share one library, the NWB file `sub-Expt1/sub-Expt1_ses-StimulusLibrary_image.nwb` of the Dandiset (it holds no
recordings; the analogue of the lab's `stim_library.h5`).  The trials tables hold only the reference (template name, first and last frame); `load_trial` / `load_cell` /
`load_repeat_*` read the frames from the library when you call them.  It is looked up under the data root; use `set_stimulus_library(path)` to point to a copy elsewhere.
Spikes, times, units and all other metadata work without it.

The files use a SYNTHETIC session timeline (see `notes` in any file): the reader subtracts each recording's `timeline_offset_s`, so every time it returns is
relative to that recording's first stimulus trigger, exactly as in the original dataset.

Experiment / penetration numbers: `experiment_number` = the digit of ExptN (1-4); `penetration_number` is 0-based in sorted penetration-name order within the experiment.
"""

from __future__ import annotations

import glob
import json
import os
import re

import h5py
import numpy as np
import pandas as pd

_FILE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_NAMES = ["Expt1", "Expt2", "Expt3", "Expt4"]
_STIMRES_S = 8.33911056439e-3
_default_root: str | None = None
_stim_library_path: str | None = None
_LIBRARY_GLOB = os.path.join("sub-*", "sub-*_ses-StimulusLibrary*.nwb")
_LIB_FILES: dict[str, h5py.File] = {}
_SLICE_CACHE: dict = {}           # (library path, dataset, start, stop) -> frames; the movie slices are large, so only the last few are kept
_SLICE_CACHE_MAX = 4


def set_data_root(path: str) -> None:
    """Folder that contains `sub-Expt1` ... `sub-Expt4` (the downloaded dandiset, or `nwb_release/` of the lab's repository)."""
    global _default_root
    _default_root = os.path.abspath(path)


def set_stimulus_library(path: str | None) -> None:
    """Use the stimulus library file at `path` (instead of the one found under the data root); None restores the default."""
    global _stim_library_path
    _stim_library_path = None if path is None else os.path.abspath(path)


def get_data_root() -> str:
    return _default_root if _default_root is not None else os.path.join(_FILE_ROOT, "nwb_release")


def _resolve_root(root):
    return root if root is not None else get_data_root()


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# NWB (HDF5) reading helpers
# ---------------------------------------------------------------------------------------------------------------------------------------------------

def _dec(ds):
    if h5py.check_string_dtype(ds.dtype) is not None:
        return list(ds.asstr()[()])
    return ds[()]


def _table(g, skip=()):
    """A DynamicTable group -> {column: list/array}; ragged columns (with an `_index` dataset) become lists of arrays."""
    cols = [c.decode() if isinstance(c, bytes) else c for c in g.attrs["colnames"]]
    out = {"id": g["id"][()]}
    for c in cols:
        if c in skip:
            continue
        data = _dec(g[c])
        if c + "_index" in g:
            ends = g[c + "_index"][()]
            starts = np.concatenate([[0], ends[:-1]])
            data = [np.asarray(data[a:b]) for a, b in zip(starts, ends)]
        out[c] = data
    return out


def _none_if(x, bad):
    return None if x == bad else x


def _library(root):
    path = _stim_library_path
    if path is None:
        hits = sorted(glob.glob(os.path.join(root, _LIBRARY_GLOB)))
        path = hits[0] if hits else os.path.join(root, "sub-Expt1", "sub-Expt1_ses-StimulusLibrary_image.nwb")
    if path not in _LIB_FILES:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"stimulus library not found: {path}. The stimulus frames are not in the recording files; download the file "
                                    "`sub-Expt1/sub-Expt1_ses-StimulusLibrary_image.nwb` from the Dandiset into the data root, or call set_stimulus_library(path).")
        _LIB_FILES[path] = h5py.File(path, "r")
    return path, _LIB_FILES[path]


def _library_slice(root, name, a, b):
    path, f = _library(root)
    key = (path, name, a, b)
    if key not in _SLICE_CACHE:
        arr = f["stimulus/templates"][name]["data"][a:b]
        if name == "MSEQ":
            arr = arr.reshape(arr.shape[0], -1)
        if len(_SLICE_CACHE) >= _SLICE_CACHE_MAX:
            _SLICE_CACHE.pop(next(iter(_SLICE_CACHE)))
        _SLICE_CACHE[key] = arr
    return _SLICE_CACHE[key]


def _nan_none(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)


class _Pen:
    """One opened penetration file with its small tables loaded; spike times and stimulus slices are read on demand."""

    def __init__(self, path):
        self.path = path
        self.root = os.path.dirname(os.path.dirname(path))        # the data root (folder of the sub-Expt*/ directories)
        self.h = h5py.File(path, "r")
        h = self.h
        self.trials = _table(h["intervals/trials"])
        self.units = _table(h["units"], skip=("spike_times", "electrodes"))
        self.reps = _table(h["intervals/repetitions"]) if "repetitions" in h["intervals"] else None
        sc = h["scratch/lgn_meta"]
        self.meta = json.loads(sc.asstr()[()] if isinstance(sc, h5py.Dataset) else sc["data"].asstr()[()])
        self.exp, self.pen = self.meta["experiment"], self.meta["penetration"]
        self.n_trials = len(self.trials["trial_key"])
        self.n_units = len(self.units["id"])
        self.row = {k: i for i, k in enumerate(self.trials["trial_key"])}
        order = np.argsort(np.asarray(self.trials["trial_index"]), kind="stable")
        self.rows_in_original_order = [int(i) for i in order]
        self.rep_rows = {}
        if self.reps is not None:
            for i, k in enumerate(self.reps["trial_key"]):
                self.rep_rows.setdefault(k, []).append(i)
            for k in self.rep_rows:
                self.rep_rows[k].sort(key=lambda i: self.reps["rep_index"][i])
        self._spk_idx = h["units/spike_times_index"][()]
        self._spk = {}

    def close(self):
        try:
            self.h.close()
        except Exception:
            pass

    # ---- spikes ----
    def spikes(self, i: int) -> np.ndarray:
        if i not in self._spk:
            a = 0 if i == 0 else int(self._spk_idx[i - 1])
            b = int(self._spk_idx[i])
            self._spk[i] = self.h["units/spike_times"][a:b]
        return self._spk[i]

    # ---- stimulus ----
    def template_slice(self, name, a, b):
        """Frames a:b of dataset `name` of the stimulus library (not stored in the NWB file)."""
        return _library_slice(self.root, name, a, b)

    def frame_abs_times(self, series):
        g = self.h["stimulus/presentation"][series]
        if "timestamps" in g:
            return g["timestamps"][()]
        n = g["data"].shape[0]
        return g["starting_time"][()] + np.arange(n) / g["starting_time"].attrs["rate"]


_PEN_CACHE: dict[str, _Pen] = {}


def close_all() -> None:
    for p in _PEN_CACHE.values():
        p.close()
    _PEN_CACHE.clear()
    _SLICE_CACHE.clear()
    for f in _LIB_FILES.values():
        f.close()
    _LIB_FILES.clear()


def _pen(path: str) -> _Pen:
    if path not in _PEN_CACHE:
        if len(_PEN_CACHE) >= 6:
            _PEN_CACHE.pop(next(iter(_PEN_CACHE))).close()
        _PEN_CACHE[path] = _Pen(path)
    return _PEN_CACHE[path]


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# 1. Numbered session access
# ---------------------------------------------------------------------------------------------------------------------------------------------------

_session_cache: dict[str, pd.DataFrame] = {}


def _files(root=None) -> dict[str, dict[str, str]]:
    root = _resolve_root(root)
    out = {}
    for e in EXPERIMENT_NAMES:
        d = {}
        for f in glob.glob(os.path.join(root, f"sub-{e}", f"sub-{e}_ses-*_ecephys*.nwb")):
            pen = re.search(r"_ses-(.+)_ecephys(?:\+image)?\.nwb$", os.path.basename(f)).group(1)
            d[pen] = f
        out[e] = dict(sorted(d.items()))
    if not any(out.values()):
        raise FileNotFoundError(
            f"No recording files (sub-Expt1/sub-Expt1_ses-*_ecephys.nwb, ...) found under {root!r}. Point the reader at the folder that contains sub-Expt1 ... sub-Expt4 "
            f"(the folder made by `dandi download DANDI:002028`): lgn.set_data_root('/full/path/to/002028'). A relative path is taken relative to the current working "
            f"directory, which is {os.getcwd()!r}.")
    return out


def build_session_table(root: str | None = None) -> pd.DataFrame:
    """One row per penetration: session_index, experiment_number (1-4), experiment_name, penetration_number (0-based within the experiment),
    penetration_name, session_date."""
    rows, si = [], 0
    for e, pens in _files(root).items():
        for pn, (pen, f) in enumerate(pens.items()):
            with h5py.File(f, "r") as h:
                sd = h["session_start_time"].asstr()[()][:10]
            rows.append({"session_index": si, "experiment_number": int(e[4:]), "experiment_name": e, "penetration_number": pn, "penetration_name": pen, "session_date": sd})
            si += 1
    return pd.DataFrame(rows)


def _cached_session_table(root=None):
    key = _resolve_root(root)
    if key not in _session_cache:
        _session_cache[key] = build_session_table(root)
    return _session_cache[key]


def name_for(experiment_number: int, penetration_number: int, root: str | None = None) -> tuple[str, str]:
    t = _cached_session_table(root)
    row = t[(t.experiment_number == experiment_number) & (t.penetration_number == penetration_number)]
    if len(row) == 0:
        raise KeyError(f"no penetration at experiment_number={experiment_number}, penetration_number={penetration_number}")
    r = row.iloc[0]
    return r.experiment_name, r.penetration_name


def number_for(experiment_name: str, penetration_name: str, root: str | None = None) -> tuple[int, int]:
    t = _cached_session_table(root)
    row = t[(t.experiment_name == experiment_name) & (t.penetration_name == penetration_name)]
    if len(row) == 0:
        raise KeyError(f"no penetration named {experiment_name}/{penetration_name}")
    r = row.iloc[0]
    return int(r.experiment_number), int(r.penetration_number)


def _open(experiment_number, penetration_number, root=None) -> _Pen:
    e, p = name_for(experiment_number, penetration_number, root)
    return _pen(_files(root)[e][p])


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# trial / unit reconstruction (NWB rows -> the dict fields of the original reader)
# ---------------------------------------------------------------------------------------------------------------------------------------------------

def _rms(tr, i):
    c = tr["contrast"][i]
    if len(c) > 1:
        d = {}
        for L in "XHL":
            v = tr["rms_" + L][i]
            if L in c and not np.isnan(v):
                d[L] = float(v)
        return d
    v = tr["contrast_rms"][i]
    return None if np.isnan(v) else float(v)


def _timing(tr, i):
    if tr["timing_dt_effective_s"][i] != tr["timing_dt_effective_s"][i] and tr["timing_refreshes_per_trigger"][i] < 0:
        return None
    pauses = json.loads(tr["timing_pauses"][i]) if tr["timing_pauses"][i] else []
    return {"refreshes_per_trigger": int(tr["timing_refreshes_per_trigger"][i]), "dt_nominal_s": float(tr["timing_dt_nominal_s"][i]),
            "dt_effective_s": float(tr["timing_dt_effective_s"][i]), "ppm_vs_nominal": float(tr["timing_ppm_vs_nominal"][i]),
            "n_triggers": int(tr["timing_n_triggers"][i]), "n_segments": int(tr["timing_n_segments"][i]),
            "pauses": [{"after_trigger_index": int(a), "extra_s": float(b)} for a, b in pauses],
            "resid_rms_us": float(tr["timing_resid_rms_us"][i]), "resid_max_us": float(tr["timing_resid_max_us"][i]), "trigger_resolution_s": 2.5e-05}


def _stim_ref(tr, i):
    name = tr["stim_template"][i]
    if not name or name.startswith("clip_"):
        return None
    d = {"family": name, "start": int(tr["stim_start"][i]), "stop": int(tr["stim_stop"][i])}
    if tr["stim_by_contrast"][i]:
        d["by_contrast"] = {L: [a, b] for L, a, b in json.loads(tr["stim_by_contrast"][i])}
    return d


def _stim(p: _Pen, tr, i):
    name = tr["stim_template"][i]
    if not name:
        return None
    if tr["stim_by_contrast"][i]:
        return {L: p.template_slice(name, a, b) for L, a, b in json.loads(tr["stim_by_contrast"][i])}
    return p.template_slice(name, int(tr["stim_start"][i]), int(tr["stim_stop"][i]))


def _rel_frame_times(p: _Pen, tr, i):
    s = tr["stimulus_series"][i]
    if not s:
        return None
    return p.frame_abs_times(s) - tr["timeline_offset_s"][i]


def _repeat_per_unit(p: _Pen, key):
    """Normalised per_unit list of a repeat recording: spikes per repetition (relative to the repetition start) and the per-rep fields."""
    tr, rp = p.trials, p.reps
    rows = p.rep_rows[key]
    starts = np.array([rp["start_time"][r] for r in rows])
    stops = np.array([rp["stop_time"][r] for r in rows])
    contrast = [rp["contrast"][r] for r in rows]
    dur = [float(rp["rep_duration"][r]) for r in rows]
    rel = [float(rp["rep_start_rel"][r]) for r in rows]
    rel = None if all(np.isnan(x) for x in rel) else rel
    bad = [set(json.loads(rp["invalid_units"][r])) for r in rows]
    out = []
    for i in range(p.n_units):
        sp = p.spikes(i)
        a = np.searchsorted(sp, starts, side="left")
        b = np.searchsorted(sp, stops, side="left")
        out.append({"spk_times": [sp[x:y] - s0 for x, y, s0 in zip(a, b, starts)], "contrast_per_rep": list(contrast), "rep_durations": list(dur),
                    "valid_rep": [i not in bd for bd in bad], "rep_start_times": None if rel is None else list(rel)})
    return out


def _unit_fields(p: _Pen):
    u = p.units
    n = p.n_units
    rq = []
    for s in u["response_quality"]:
        rq.append({c: {"rating": r, "flags": f, "raw": w} for c, r, f, w in json.loads(s)})
    return {
        "name": list(u["unit_name"]), "electrode": [int(x) for x in u["electrode_number"]], "subletter": list(u["subletter"]), "LGN_layer": list(u["LGN_layer"]),
        "xy_class": [x or None for x in u["xy_class"]], "xy_class_source": [x or None for x in u["xy_class_source"]],
        "xy_class_log_f2f1": [_nan_none(x) for x in u["xy_class_log_f2f1"]], "quality_mean": [_nan_none(x) for x in u["quality_mean"]],
        "quality_by_trial": [{a: b for a, b in json.loads(s)} for s in u["quality_by_trial"]], "response_quality": rq,
        "response_quality_note": [x or "" for x in u["response_quality_note"]], "unit_notes": [x or "" for x in u["unit_notes"]],
        "possible_duplicate_of": [[y for y in x.split(", ") if y] for x in u["possible_duplicate_of"]],
        "shared_input_with": [[y for y in x.split(", ") if y] for x in u["shared_input_with"]],
    }


def _quality_for_file(qbt: dict, pen: str, file_letter):
    """Spike-sorting quality (the lab's 1-6 scale; Expt3, Expt4 only) of a unit in the recording with this sequence letter (the lab's recording labels are inconsistent, the letter is the join key)."""
    if file_letter is None:
        return None
    for k, v in (qbt or {}).items():
        kk = k[len(pen):] if k.startswith(pen) else k
        if kk[:1] == file_letter:
            return v
    return None


def _trial_common(p: _Pen, i):
    tr = p.trials
    return {
        "modality": tr["modality"][i], "regime": tr["regime"][i], "code": tr["code"][i], "file_letter": tr["file_letter"][i] or None,
        "trial_order": _none_if(int(tr["trial_order"][i]), -1), "source_files": json.loads(tr["source_files"][i]), "duration_s": _nan_none(tr["duration_s"][i]),
        "contrast": tr["contrast"][i] or None, "contrast_rms": _rms(tr, i), "contrast_notes": tr["contrast_notes"][i] or None,
        "contrast_segments": [{"start": a, "stop": b, "contrast": L} for L, a, b in json.loads(tr["contrast_segments"][i])] if tr["contrast_segments"][i] else None,
        "coverage": json.loads(tr["coverage"][i]) if tr["coverage"][i] else None,
        "accepted_assumptions": json.loads(tr["accepted_assumptions"][i]), "notes": json.loads(tr["notes"][i]),
        "n_reps": _none_if(int(tr["n_reps"][i]), -1), "trig_per_rep": _none_if(int(tr["trig_per_rep"][i]), -1), "block": _none_if(int(tr["block"][i]), -1),
    }


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# 3. Trial-level access
# ---------------------------------------------------------------------------------------------------------------------------------------------------

def load_trial(experiment_number: int, penetration_number: int, trial_key: str, lib=None, root: str | None = None) -> dict:
    """One recording with all units.  Same fields as the original `load_trial`.  `stim`: the stimulus frames shown (None if the content is not identified; for
    the FF_LONG recordings a dict {contrast letter: samples}); `stim_ref` {family, start, stop[, by_contrast]} says where they sit in the template; `frame_times`
    (unique recordings) and `spk_times` are in seconds from the recording's first trigger.  Repeat recordings return `spk_times=None` and the normalised `per_unit`
    list (see `load_repeat_trial`)."""
    p = _open(experiment_number, penetration_number, root)
    if trial_key not in p.row:
        raise KeyError(trial_key)
    i = p.row[trial_key]
    tr = p.trials
    c = _trial_common(p, i)
    uf = _unit_fields(p)
    off = float(tr["timeline_offset_s"][i])
    repeat = c["regime"] == "repeat"
    spk, per_unit = None, None
    if repeat:
        per_unit = _repeat_per_unit(p, trial_key)
    elif tr["n_spikes_per_unit"][i][0] >= 0:
        lo, hi = tr["start_time"][i] - 1e-6, tr["stop_time"][i] + 1e-6
        spk = []
        for u in range(p.n_units):
            sp = p.spikes(u)
            a, b = np.searchsorted(sp, lo, side="left"), np.searchsorted(sp, hi, side="right")
            spk.append(sp[a:b] - off)
    bt = tr["block_triggers"][i]
    pen = p.pen
    return {
        "stim": _stim(p, tr, i), "stim_ref": _stim_ref(tr, i), "frame_times": _rel_frame_times(p, tr, i), "block_triggers": np.asarray(json.loads(bt)) if bt else None,
        "timing": _timing(tr, i), "spk_times": spk, "per_unit": per_unit, "unit_names": uf["name"],
        "xy_class": uf["xy_class"], "xy_class_source": uf["xy_class_source"], "xy_class_log_f2f1": uf["xy_class_log_f2f1"],
        "electrode": uf["electrode"], "subletter": uf["subletter"], "LGN_layer": uf["LGN_layer"], "unit_notes": uf["unit_notes"],
        "possible_duplicate_of": uf["possible_duplicate_of"], "shared_input_with": uf["shared_input_with"],
        "response_quality": uf["response_quality"], "quality_mean": uf["quality_mean"], "quality_by_trial": uf["quality_by_trial"],
        "quality": [_quality_for_file(q, pen, c["file_letter"]) for q in uf["quality_by_trial"]], "response_quality_note": uf["response_quality_note"],
        "eccentricity_x_deg": p.meta.get("eccentricity_x_deg"), "eccentricity_y_deg": p.meta.get("eccentricity_y_deg"), "penetration_notes": p.meta.get("penetration_notes"),
        "file_letter": c["file_letter"], "trial_order": c["trial_order"], "source_files": c["source_files"], "duration_s": c["duration_s"],
        "accepted_assumptions": c["accepted_assumptions"], "contrast": c["contrast"], "contrast_segments": c["contrast_segments"], "coverage": c["coverage"],
        "contrast_rms": c["contrast_rms"], "contrast_notes": c["contrast_notes"], "regime": c["regime"], "modality": c["modality"], "notes": c["notes"],
    }


_CELL_FIELDS = ["xy_class", "xy_class_source", "xy_class_log_f2f1", "electrode", "subletter", "LGN_layer", "unit_notes", "possible_duplicate_of", "shared_input_with",
                "response_quality", "quality_mean", "quality_by_trial", "quality", "response_quality_note"]
_SHARED_FIELDS = ["stim", "stim_ref", "frame_times", "block_triggers", "timing", "eccentricity_x_deg", "eccentricity_y_deg", "penetration_notes", "file_letter", "trial_order",
                  "source_files", "duration_s", "accepted_assumptions", "contrast", "contrast_segments", "coverage", "contrast_rms", "regime", "modality", "notes"]


def load_cell(experiment_number: int, penetration_number: int, cell_number: int, trial_key: str, lib=None, root: str | None = None) -> dict:
    """One cell's data for one recording (`trial_key` normally comes from a `find_cells()`/`cell_info()` row)."""
    t = load_trial(experiment_number, penetration_number, trial_key, root=root)
    out = {k: t[k] for k in _SHARED_FIELDS}
    out["spk_times"] = t["spk_times"][cell_number] if t["spk_times"] is not None else None
    out["per_unit"] = t["per_unit"][cell_number] if t["per_unit"] is not None else None
    out["cell_name"] = t["unit_names"][cell_number]
    for k in _CELL_FIELDS:
        out[k] = t[k][cell_number]
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# 4. Repeat recordings
# ---------------------------------------------------------------------------------------------------------------------------------------------------

def _frames_per_rep(experiment_name, code, n_reps):
    """Trigger-resolution frames per repetition by (experiment label, code); None where the structure is not established.  A stored per-recording value (`trig_per_rep`) takes precedence."""
    if experiment_name == "Expt1":
        if code in ("ff020", "ff055"):
            return 960
        if code in ("nr015", "nr040", "wr020", "wr055"):
            return 250
    if experiment_name == "Expt2":
        if code == "nr":
            return {24: 5400, 30: 600}.get(n_reps)
        if code in ("wr", "nr_pooled"):
            return 600
    if experiment_name == "Expt3":
        if code in ("wr", "nr", "wr1short", "fff_btrio"):
            return 600
        if code in ("fff_hl16", "ff", "fff"):
            return 1800
    if experiment_name == "Expt4":
        return 600
    return None


def classify_repeat_shape(t: dict) -> str:
    return "per_unit"


def load_repeat_trial(experiment_number: int, penetration_number: int, trial_key: str, lib=None, root: str | None = None) -> dict:
    """A repeat recording (`regime == 'repeat'`) in a uniform PSTH-ready shape: `per_unit[i] = {'spk_times': [rep0, rep1, ...] (seconds from each repetition's start),
    'contrast_per_rep', 'rep_durations', 'valid_rep', 'rep_start_times' (seconds from the recording's first trigger, None if not known)}`, plus the stimulus
    (`stim`/`stim_ref`), `dt` (stimulus frame period, s) and the unit annotations.  The repetition windows contain the stimulus content only; the ~0.175 s restart pauses
    are not part of any repetition (pause = diff(rep_start_times) - rep_durations)."""
    p = _open(experiment_number, penetration_number, root)
    i = p.row[trial_key]
    tr = p.trials
    c = _trial_common(p, i)
    if c["regime"] != "repeat":
        raise ValueError(f"{trial_key!r} is regime={c['regime']!r}, not 'repeat'")
    uf = _unit_fields(p)
    fpr = c["trig_per_rep"] or _frames_per_rep(p.exp, c["code"], c["n_reps"])
    dt = (_STIMRES_S * (1 if fpr == 960 else 2)) if fpr else None
    return {
        "shape": "per_unit", "per_unit": _repeat_per_unit(p, trial_key), "unit_names": uf["name"], "stim": _stim(p, tr, i), "stim_ref": _stim_ref(tr, i), "timing": _timing(tr, i), "dt": dt,
        "xy_class": uf["xy_class"], "xy_class_source": uf["xy_class_source"], "xy_class_log_f2f1": uf["xy_class_log_f2f1"], "electrode": uf["electrode"], "subletter": uf["subletter"],
        "LGN_layer": uf["LGN_layer"], "unit_notes": uf["unit_notes"], "possible_duplicate_of": uf["possible_duplicate_of"], "shared_input_with": uf["shared_input_with"],
        "response_quality": uf["response_quality"], "response_quality_note": uf["response_quality_note"], "quality_mean": uf["quality_mean"], "quality_by_trial": uf["quality_by_trial"],
        "quality": [_quality_for_file(q, p.pen, c["file_letter"]) for q in uf["quality_by_trial"]],
        "eccentricity_x_deg": p.meta.get("eccentricity_x_deg"), "eccentricity_y_deg": p.meta.get("eccentricity_y_deg"), "penetration_notes": p.meta.get("penetration_notes"),
        "modality": c["modality"], "code": c["code"], "file_letter": c["file_letter"], "trial_order": c["trial_order"], "source_files": c["source_files"], "duration_s": c["duration_s"],
        "accepted_assumptions": c["accepted_assumptions"], "contrast": c["contrast"], "contrast_rms": c["contrast_rms"], "regime": c["regime"], "notes": c["notes"],
    }


def load_repeat_cell(experiment_number: int, penetration_number: int, cell_number: int, trial_key: str, lib=None, root: str | None = None) -> dict:
    """`load_repeat_trial` sliced to one cell."""
    t = load_repeat_trial(experiment_number, penetration_number, trial_key, root=root)
    u = t["per_unit"][cell_number]
    out = {k: u[k] for k in ("spk_times", "contrast_per_rep", "rep_durations", "valid_rep", "rep_start_times")}
    for k in ("timing", "stim", "stim_ref", "dt", "eccentricity_x_deg", "eccentricity_y_deg", "penetration_notes", "modality", "code", "file_letter", "trial_order", "source_files",
              "duration_s", "accepted_assumptions", "contrast", "contrast_rms", "regime", "notes"):
        out[k] = t[k]
    out["cell_name"] = t["unit_names"][cell_number]
    for k in _CELL_FIELDS:
        out[k] = t[k][cell_number]
    return out


def compute_psth(repeat_cell: dict, binw: float = 0.02):
    """(bin_centers, rate_hz) from a `load_repeat_cell()`-style dict; averages over valid repetitions; bins span [0, median(rep_durations))."""
    trep = np.median(repeat_cell["rep_durations"])
    nbins = int(trep / binw)
    valid = [s for s, v in zip(repeat_cell["spk_times"], repeat_cell["valid_rep"]) if v]
    counts = np.zeros(nbins)
    for s in valid:
        c, _ = np.histogram(s, bins=nbins, range=(0, trep))
        counts += c
    return (np.arange(nbins) + 0.5) * binw, counts / max(len(valid), 1) / binw


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# 2. Cell-first search
# ---------------------------------------------------------------------------------------------------------------------------------------------------

_BOOKKEEPING_REGIMES = {"diagnostic", "n/a", "unknown"}


def _rms_scalar(rms):
    return float(rms) if isinstance(rms, (int, float)) else float("nan")


def _rms_levels(contrast, rms) -> str:
    if rms is None or contrast is None:
        return ""
    if isinstance(rms, dict):
        return " ".join(f"{k}={v:g}" for k, v in rms.items() if v is not None)
    return f"{contrast}={float(rms):g}"


def build_cell_trial_table(experiments: list[str] | None = None, root: str | None = None, valid_only: bool = True) -> pd.DataFrame:
    """One row per (experiment_number, penetration_number, cell_number, trial_key); `valid_only` drops diagnostic / discarded recordings.  Uses only the small tables of the files
    (no spike times are read), so it is fast."""
    rows = []
    for s in _cached_session_table(root).itertuples():
        if experiments is not None and s.experiment_name not in experiments:
            continue
        p = _open(s.experiment_number, s.penetration_number, root)
        tr = p.trials
        uf = _unit_fields(p)
        cell_of = {n: i for i, n in enumerate(uf["name"])}
        ecc = (p.meta.get("eccentricity_x_deg"), p.meta.get("eccentricity_y_deg"))
        ann = []
        for i in range(p.n_units):
            dups = uf["possible_duplicate_of"][i]
            ann.append({"electrode": uf["electrode"][i], "subletter": uf["subletter"][i], "LGN_layer": uf["LGN_layer"][i], "eccentricity_x_deg": ecc[0], "eccentricity_y_deg": ecc[1],
                        "unit_notes": uf["unit_notes"][i], "possible_duplicate_of": ", ".join(dups), "duplicate_redundant": any(cell_of.get(d, i) < i for d in dups),
                        "shared_input_with": ", ".join(uf["shared_input_with"][i]), "response_quality_note": uf["response_quality_note"][i]})
        for i in p.rows_in_original_order:
            c = _trial_common(p, i)
            if valid_only and c["regime"] in _BOOKKEEPING_REGIMES:
                continue
            sr = _stim_ref(tr, i)
            n_ft = int(tr["n_frame_times"][i])
            rep_n = c["n_reps"] if c["regime"] == "repeat" else None
            fpr = (c["trig_per_rep"] or _frames_per_rep(s.experiment_name, c["code"], rep_n)) if c["regime"] == "repeat" else None
            if rep_n is None and fpr and n_ft >= 0:
                rep_n = n_ft // fpr
            nsp = tr["n_spikes_per_unit"][i]
            has_stim = bool(tr["stim_template"][i])
            for cell in range(p.n_units):
                q = uf["quality_by_trial"][cell]
                rows.append({
                    "experiment_number": s.experiment_number, "penetration_number": s.penetration_number, "cell_number": cell, "experiment_name": s.experiment_name,
                    "penetration_name": s.penetration_name, "cell_name": uf["name"][cell], "trial_key": tr["trial_key"][i], "modality": c["modality"], "regime": c["regime"], "code": c["code"],
                    "file_letter": c["file_letter"], "trial_order": c["trial_order"], "source_files": ", ".join(c["source_files"]) if c["source_files"] else None, "duration_s": c["duration_s"],
                    "contrast": c["contrast"], "contrast_rms": _rms_scalar(c["contrast_rms"]), "contrast_rms_levels": _rms_levels(c["contrast"], c["contrast_rms"]),
                    "xy_class": uf["xy_class"][cell], "xy_class_source": uf["xy_class_source"][cell], "xy_class_log_f2f1": uf["xy_class_log_f2f1"][cell],
                    "quality_mean": uf["quality_mean"][cell], "quality": _quality_for_file(q, s.penetration_name, c["file_letter"]),
                    "response_quality": (uf["response_quality"][cell].get(c["code"]) or {}).get("rating"), "response_quality_flags": (uf["response_quality"][cell].get(c["code"]) or {}).get("flags", ""),
                    "n_spikes": None if nsp[cell] < 0 else int(nsp[cell]), "has_stim": has_stim,
                    "stim_family": sr["family"] if sr else None, "stim_start": sr["start"] if sr else None, "stim_stop": sr["stop"] if sr else None,
                    "n_frames": ((rep_n * fpr) if (rep_n is not None and fpr is not None) else (sr["stop"] - sr["start"]) if sr else (n_ft if n_ft >= 0 else None)),
                    "frames_per_rep": fpr, "n_reps": rep_n, **ann[cell]})
    return pd.DataFrame(rows)


def scan_all_trials(experiments: list[str] | None = None, root: str | None = None) -> pd.DataFrame:
    """One row per recording (summary table)."""
    rows = []
    for s in _cached_session_table(root).itertuples():
        if experiments is not None and s.experiment_name not in experiments:
            continue
        p = _open(s.experiment_number, s.penetration_number, root)
        for i in p.rows_in_original_order:
            c = _trial_common(p, i)
            rows.append({"experiment": s.experiment_name, "penetration": s.penetration_name, "trial_key": p.trials["trial_key"][i], "modality": c["modality"], "code": c["code"],
                         "regime": c["regime"], "contrast": c["contrast"], "contrast_rms": _rms_scalar(c["contrast_rms"]), "contrast_rms_levels": _rms_levels(c["contrast"], c["contrast_rms"]),
                         "stim_ref": _stim_ref(p.trials, i), "has_stim": bool(p.trials["stim_template"][i]), "n_units": p.n_units, "notes": " | ".join(c["notes"]),
                         "experiment_number": s.experiment_number, "penetration_number": s.penetration_number})
    return pd.DataFrame(rows)


_CONTRAST_ORDER = "XHL"
_CONTRAST_ALIAS = {"HC": "H", "LC": "L", "XC": "X"}


def find_cells(modality: str | None = None, regime: str | None = None, xy_class: str | None = None, contrast: str | None = None, contrast_exact: bool = False,
               min_spikes: int = 1, valid_only: bool = True, frame_thresh: int | None = None, exclude_duplicates: bool = False,
               table: pd.DataFrame | None = None, root: str | None = None, **filters) -> pd.DataFrame:
    """(1) "What cells have this type of data?"  `modality`: 'NAT', 'SBN', 'FF', 'FF_LONG', 'FF_BTRIO', 'mseq', 'gray' (case-insensitive); `regime`: 'unique', 'repeat',
    'trio' (pooled three-contrast white noise); `xy_class`: 'x' / 'y'; `contrast`: letters X (RMS 1.00) / H / L -- by default a recording matches when it CONTAINS every letter
    given ('H' also returns 'HL' and 'XHL'), `contrast_exact=True` requires equality.  `frame_thresh` keeps recordings whose stimulus spans at least that many frames.
    `exclude_duplicates=True` keeps one unit per group of units listed in `possible_duplicate_of` (default: every unit is returned; the column is informational, see the dataset notes).
    Any other column of `build_cell_trial_table()` can be filtered with `column=value`.  Returns one row per matching (cell, recording), including `trial_key`."""
    df = table if table is not None else build_cell_trial_table(root=root, valid_only=valid_only)
    mask = pd.Series(True, index=df.index)
    if modality is not None:
        mask &= df.modality.str.lower() == modality.lower()
    if regime is not None:
        mask &= df.regime == regime
    if xy_class is not None:
        mask &= df.xy_class == xy_class
    if contrast is not None:
        want = "".join(_CONTRAST_ALIAS.get(contrast, contrast))
        have = df.contrast.fillna("").astype(str)
        mask &= (have == "".join(sorted(want, key=_CONTRAST_ORDER.index))) if contrast_exact else have.apply(lambda c: bool(c) and set(want) <= set(c))
    if frame_thresh is not None:
        mask &= (df.stim_stop - df.stim_start) >= frame_thresh
    for col, val in filters.items():
        mask &= df[col] == val
    mask &= df.n_spikes.fillna(0) >= min_spikes
    if exclude_duplicates and "duplicate_redundant" in df:
        mask &= ~df.duplicate_redundant.fillna(False).astype(bool)
    return df[mask].reset_index(drop=True)


def cell_info(experiment_number: int, penetration_number: int, cell_number: int, valid_only: bool = True, table: pd.DataFrame | None = None, root: str | None = None) -> pd.DataFrame:
    """(2) "What data is available for this cell?" -- every recording the cell appears in."""
    df = table if table is not None else build_cell_trial_table(root=root, valid_only=valid_only)
    m = (df.experiment_number == experiment_number) & (df.penetration_number == penetration_number) & (df.cell_number == cell_number)
    return df[m].reset_index(drop=True)
