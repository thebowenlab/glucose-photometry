# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:57:11 2026

@author: Adams
"""
# photometry/metrics/onset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple
import numpy as np
from photometry.preprocess.time import infer_dt


@dataclass(slots=True)
class FrequencyOnsetConfig:
    event_time: float = 0.0
    pre_min: float = 30.0
    post_min: float = 60.0
    spec_win_min: float = 20.0
    spec_overlap: float = 0.99
    spec_scaling: str = "density"
    band_period_min: tuple[float, float] = (14.0, 30.0)
    n_pseudo: int = 200
    guard_min: float = 5.0
    alpha_local: float = 0.01
    seed: int | None = 0
    tail: str = "upper"
    cluster_tail: str = "upper"
    min_cycles: float = 1.5
    exclude_edges: bool = True
    detect_window: tuple[float, float] | None = (0.0, 10.0)
    compute_cluster: bool = False


def detect_frequency_onset_jitter(
    t,
    fp,
    *,
    cfg: FrequencyOnsetConfig | None = None,
    return_debug: bool = False,
) -> dict[str, Any]:
    cfg = cfg or FrequencyOnsetConfig()

    res = _event_locked_tf_jitter_core(
        t,
        fp,
        event_time=cfg.event_time,
        pre_min=cfg.pre_min,
        post_min=cfg.post_min,
        spec_win_min=cfg.spec_win_min,
        spec_overlap=cfg.spec_overlap,
        spec_scaling=cfg.spec_scaling,
        band_period_min=cfg.band_period_min,
        n_pseudo=cfg.n_pseudo,
        guard_min=cfg.guard_min,
        alpha_local=cfg.alpha_local,
        seed=cfg.seed,
        tail=cfg.tail,
        cluster_tail=cfg.cluster_tail,
        min_cycles=cfg.min_cycles,
        exclude_edges=cfg.exclude_edges,
        detect_window=cfg.detect_window,
        compute_cluster=cfg.compute_cluster,
        return_debug=return_debug,
    )

    if not res:
        return {
            "t_onset": np.nan,
            "onset_idx": -1,
            "onset_p": np.nan,
            "found_onset": False,
            "method": "freq_jitter",
            "detect_window": cfg.detect_window,
            "params": {
                "band_period_min": cfg.band_period_min,
                "n_pseudo": cfg.n_pseudo,
                "alpha_local": cfg.alpha_local,
            },
        }

    out = {
        "t_onset": res["first_sig_time_min"],
        "onset_idx": res["first_sig_idx"],
        "onset_p": res["first_sig_p"],
        "found_onset": bool(np.isfinite(res["first_sig_time_min"])),
        "method": "freq_jitter",
        "detect_window": res["detect_window"],
        "params": res["params"],
    }

    if return_debug:
        out.update({
            "tf_time_rel_min": res["tf_time_rel_min"],
            "tf_freq_cpm": res["tf_freq_cpm"],
            "p_time": res["p_time"],
            "band_power_event": res["band_power_event"],
        })
        if "cluster_pvalue" in res:
            out["cluster_pvalue"] = res["cluster_pvalue"]

    return out

def _event_locked_tf_jitter_core(
    t, fp, *,
    event_time: float,
    pre_min: float = 30.0,
    post_min: float = 60.0,
    spec_win_min: float = 20.0,
    spec_overlap: float = 0.99,
    spec_scaling: str = "density",
    band_period_min: tuple[float, float] = (10.0, 30.0),
    n_pseudo: int = 200,
    guard_min: float = 5.0,
    alpha_local: float = 0.01,
    seed: int | None = 0,
    tail: str = "upper",              # 'upper'|'lower'|'two-sided' (for per-bin p)
    cluster_tail: str = "upper",      # 'upper'|'two-sided' (for cluster formation)
    min_cycles: float = 1.5,          # require >= this many cycles over [−pre, +post]
    exclude_edges: bool = True,       # mask first/last ~½ window in time
    compute_cluster: bool = False,
    return_debug: bool = False,
    detect_window: tuple[float, float] | None = None,  # e.g. (0.0, 30.0)
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # --- prep
    t = np.asarray(t, float); fp = np.asarray(fp, float)
    if t.size < 512 or fp.size < 512:
        return out
    n = min(t.size, fp.size); t, fp = t[:n], fp[:n]
    dt = infer_dt(t)
    fs_cpm = 1.0 / dt

    # --- real event-locked segment
    t_rel, x_ev = _event_locked_segment(t, fp, event_time, pre_min, post_min)
    
    if t_rel.size == 0 or not _has_full_window(t_rel, pre_min, post_min):
        return out

    f_cpm, t_spec_rel, S_real = _build_spectrogram_lowf(
        x_ev, fs_cpm, spec_win_min, spec_overlap, spec_scaling
    )  # S_real: (F, T)

    # re-center TF time to be relative to event:
    tf_t_rel = (-pre_min + t_spec_rel).astype(float)  # minutes relative to event

    # --- restrict to a SINGLE requested band (PERIODS → FREQS) --------------
    if band_period_min is not None:
        min_period, max_period = band_period_min
        min_period = max(min_period, 1e-9)
        max_period = max(max_period, 1e-9)
        # convert to frequencies (cpm)
        fhi = 1.0 / min_period         # highest frequency (shortest period)
        flo = 1.0 / max_period         # lowest frequency (longest period)
        if flo > fhi:
            flo, fhi = fhi, flo
        fmask = (f_cpm >= flo) & (f_cpm <= fhi)
    else:
        # if None, keep all freqs (not recommended if you want a single band)
        fmask = np.ones_like(f_cpm, dtype=bool)

    Fkeep = np.where(fmask)[0]
    if Fkeep.size == 0:
        return out

    S_real_band = S_real[fmask, :]          # (F_keep, T)
    ff_keep = f_cpm[fmask]

    # ---- analysis mask (time + freq) ---------------------------------------
    # time mask: exclude first/last ~½ window to reduce edge leakage
    if exclude_edges and t_spec_rel.size > 1:
        halfw = 0.5 * spec_win_min
        tm = (tf_t_rel >= (-pre_min + halfw)) & (tf_t_rel <= (post_min - halfw))
    else:
        tm = np.ones_like(tf_t_rel, dtype=bool)

    # frequency mask: require sufficient cycles across window
    total_minutes = pre_min + post_min
    fm_cycles = (ff_keep * total_minutes) >= float(min_cycles)
    # 2D analysis mask
    AM = np.outer(fm_cycles, tm)  # (F_keep, T)

    # --- build pseudo onsets
    t0 = t[0] + pre_min
    t1 = t[-1] - post_min
    if t1 <= t0:
        return out

    rng = np.random.default_rng(seed) if (seed is not None) else np.random.default_rng()
    # draw pseudos uniformly, excluding guard window
    pseudos = []
    for _ in range(n_pseudo * 2):  # oversample then trim to n_pseudo
        cand = rng.uniform(t0, t1)
        if abs(cand - event_time) >= guard_min:
            pseudos.append(cand)
        if len(pseudos) >= n_pseudo:
            break
    pseudos = np.array(pseudos[:n_pseudo], float)
    if pseudos.size < max(25, n_pseudo // 2):  # ensure we have enough nulls
        return out

    # --- build pseudo spectrograms (band-limited) ---------------------------
    S_p_list = []
    for pe in pseudos:
        t_rel_p, x_p = _event_locked_segment(t, fp, pe, pre_min, post_min)
        if x_p.size == 0 or not _has_full_window(t_rel_p, pre_min, post_min):
            continue
        f_cpm_p, t_spec_p, S_p = _build_spectrogram_lowf(
            x_p, fs_cpm, spec_win_min, spec_overlap, spec_scaling
        )
        # align T to real
        T_target = S_real.shape[1]
        if S_p.shape[1] > T_target:
            S_p = S_p[:, :T_target]
        elif S_p.shape[1] < T_target:
            pad = np.repeat(S_p[:, [-1]], T_target - S_p.shape[1], axis=1)
            S_p = np.concatenate([S_p, pad], axis=1)
        S_p_list.append(S_p[fmask, :])
    if not S_p_list:
        return out
    S_p = np.stack(S_p_list, axis=0)  # (N, F_keep, T)
    N, Fk, Tbins = S_p.shape

    # --- per-bin p-map (within the *single band* only) ----------------------
    S_mu  = np.nanmean(S_p, axis=0)
    S_std = np.nanstd(S_p, axis=0, ddof=1) + 1e-12
    S_real_rep = np.broadcast_to(S_real_band[None, :, :], (N, Fk, Tbins))

    # counts
    ge = (S_p >= S_real_rep).sum(axis=0).astype(float)  # null >= real
    le = (S_p <= S_real_rep).sum(axis=0).astype(float)  # null <= real

    # upper-tail p: prob(null >= real)  -> small when real is LARGE
    p_upper = (1.0 + ge) / (N + 1.0)
    # lower-tail p: prob(null <= real)  -> small when real is SMALL
    p_lower = (1.0 + le) / (N + 1.0)

    if tail == "upper":
        p_map = p_upper
    elif tail == "lower":
        p_map = p_lower
    else:  # 'two-sided'
        p_map = 2.0 * np.minimum(p_upper, 1.0 - p_upper)
    p_map = np.clip(p_map, 1e-6, 1.0)

    # respect analysis mask: set outside-ROI to NaN
    p_map[~AM] = np.nan

    # --- NEW: collapse to band power over time + first significant epoch ----
    # Band power time series: mean PSD in band at each time bin
    band_power_real = np.nanmean(S_real_band, axis=0)        # shape (T,)
    band_power_null = np.nanmean(S_p, axis=1)                # shape (N, T)

    # Use the *most significant* p within the band at each time bin
    p_time = np.nanmin(p_map, axis=0)                        # (T,)

    # detection window [t_min, t_max] relative to event
    if detect_window is None:
        det_lo, det_hi = 0.0, post_min
    else:
        det_lo, det_hi = detect_window

    detect_mask = (tf_t_rel >= det_lo) & (tf_t_rel <= det_hi) & tm
    valid_mask  = detect_mask & ~np.isnan(p_time)

    sig_mask = valid_mask & (p_time <= alpha_local)

    if np.any(sig_mask):
        first_idx = int(np.where(sig_mask)[0][0])
        first_sig_time = float(tf_t_rel[first_idx])
        first_sig_p    = float(p_time[first_idx])
    else:
        first_idx = -1
        first_sig_time = np.nan
        first_sig_p    = np.nan

    # --- cluster-mass permutation (unchanged, but now within the band) ------
    Tmap_real = (S_real_band - S_mu) / S_std
    Tmap_real[~AM] = 0.0  # ensure masked bins don’t form clusters
   
    if compute_cluster:
        from scipy import ndimage as ndi

        def _max_mass_pos(Tmap, thr):
            # positive-only clusters
            mask = Tmap > thr
            if not np.any(mask):
                return 0.0
            labels, nlab = ndi.label(
                mask,
                structure=np.array([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]], dtype=np.uint8)
            )
            mass = 0.0
            for lab in range(1, nlab + 1):
                m = (labels == lab)
                mass = max(mass, float(Tmap[m].sum()))
            return mass
        # estimate a POSITIVE threshold from null Tmaps (sampled)
        take = max(10000 // max(1, N), 64)
        pool_pos = []
        S_sum = np.nansum(S_p, axis=0); S_sqsum = np.nansum(S_p**2, axis=0)
        for i in range(min(N, 64)):
            mu_loo = (S_sum - S_p[i]) / max(1, (N - 1))
            if N > 2:
                var_loo = ((S_sqsum - S_p[i]**2) - (N - 1) * mu_loo**2) / max(1, (N - 2))
                std_loo = np.sqrt(np.maximum(var_loo, 1e-12))
            else:
                std_loo = np.maximum(np.nanstd(S_p, axis=0, ddof=1), 1e-12)
            T_i = (S_p[i] - mu_loo) / std_loo
            T_i[AM == 0] = 0.0
            flat = T_i[T_i > 0].ravel()
            if flat.size:
                if flat.size > take:
                    pool_pos.append(rng.choice(flat, size=take, replace=False))
                else:
                    pool_pos.append(flat)
        pool_pos = np.concatenate(pool_pos) if pool_pos else np.maximum(Tmap_real, 0).ravel()
        thr_pos = np.percentile(pool_pos, 100 * (1 - alpha_local))

        if cluster_tail == "upper":
            max_mass_real = _max_mass_pos(Tmap_real, thr_pos)
            max_masses_null = []
            for i in range(N):
                mu_loo = (S_sum - S_p[i]) / max(1, (N - 1))
                if N > 2:
                    var_loo = ((S_sqsum - S_p[i]**2) - (N - 1) * mu_loo**2) / max(1, (N - 2))
                    std_loo = np.sqrt(np.maximum(var_loo, 1e-12))
                else:
                    std_loo = np.maximum(np.nanstd(S_p, axis=0, ddof=1), 1e-12)
                T_i = (S_p[i] - mu_loo) / std_loo
                T_i[AM == 0] = 0.0
                max_masses_null.append(_max_mass_pos(T_i, thr_pos))
            max_masses_null = np.asarray(max_masses_null, float)
        else:
            # fallback to original abs-cluster code (if you keep _label_clusters_gt_threshold)
            labels, masses, max_mass_real = _label_clusters_gt_threshold(
                Tmap_real, thr_pos, use_abs=True
            )
            max_masses_null = []
            for i in range(N):
                mu_loo = (S_sum - S_p[i]) / max(1, (N - 1))
                if N > 2:
                    var_loo = ((S_sqsum - S_p[i]**2) - (N - 1) * mu_loo**2) / max(1, (N - 2))
                    std_loo = np.sqrt(np.maximum(var_loo, 1e-12))
                else:
                    std_loo = np.maximum(np.nanstd(S_p, axis=0, ddof=1), 1e-12)
                T_i = (S_p[i] - mu_loo) / std_loo
                T_i[AM == 0] = 0.0
                max_masses_null.append(_max_mass_pos(T_i, thr_pos))
            max_masses_null = np.asarray(max_masses_null, float)
    
        p_cluster = float(
            (1.0 + (max_masses_null >= max_mass_real).sum()) /
            (max_masses_null.size + 1.0)
        )

    out.update({
    "first_sig_time_min": first_sig_time,
    "first_sig_idx": first_idx,
    "first_sig_p": first_sig_p,
    "detect_window": (det_lo, det_hi),
    "params": {
        "tail": tail,
        "cluster_tail": cluster_tail,
        "alpha_local": alpha_local,
        "n_pseudo": int(S_p.shape[0]),
        "min_cycles": float(min_cycles),
        "exclude_edges": bool(exclude_edges),
        "band_period_min": band_period_min,
        "pre_min": pre_min,
        "post_min": post_min,
        "spec_win_min": spec_win_min,
        "spec_overlap": spec_overlap,
        "detect_window": (det_lo, det_hi),
    }
})
    
    if compute_cluster:
        out.update({
            "cluster_pvalue": p_cluster,
            "cluster_mass_real": max_mass_real,
            "cluster_mass_null": max_masses_null,
        })
        
    if return_debug:
        out.update({
            "tf_time_rel_min": tf_t_rel,
            "tf_freq_cpm": ff_keep,
            "S_event": S_real_band,
            "pmap_two_sided": 2.0 * np.minimum(p_upper, 1.0 - p_upper),
            "pmap_one_sided_upper": p_upper,
            "Tmap_real": Tmap_real,
            "analysis_mask": AM,
            "band_power_event": band_power_real,
            "band_power_null": band_power_null,
            "p_time": p_time,
        })
    
    return out

def _has_full_window(t_rel: np.ndarray, pre_min: float, post_min: float, tol: float = 1e-9) -> bool:
    if t_rel.size == 0:
        return False
    return (t_rel[0] <= (-pre_min + tol)) and (t_rel[-1] >= (post_min - tol))

def _event_locked_segment(t: np.ndarray,
                          x: np.ndarray,
                          event_time: float,
                          pre_min: float,
                          post_min: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return x segment and its time axis (relative to event_time)."""
    t = np.asarray(t, float); x = np.asarray(x, float)
    t0 = event_time - pre_min
    t1 = event_time + post_min
    m = (t >= t0) & (t <= t1)
    if not np.any(m):
        return np.array([], dtype=float), np.array([], dtype=float)
    return t[m] - event_time, x[m]

def _label_clusters_gt_threshold(A: np.ndarray, thr: float, use_abs: bool = True):
    """Return cluster labels and per-cluster masses for 2D array A with threshold thr."""
    from scipy import ndimage as ndi
    if use_abs:
        mask = np.abs(A) > thr
        weights = np.abs(A)
    else:
        mask = A > thr
        weights = A
    if not np.any(mask):
        return mask.astype(np.int32), [], 0.0
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]], dtype=np.uint8)  # 4-connectivity
    labels, nlab = ndi.label(mask, structure=structure)
    masses = []
    for lab in range(1, nlab+1):
        m = (labels == lab)
        masses.append(float(weights[m].sum()))
    max_mass = max(masses) if masses else 0.0
    return labels, masses, max_mass

def _build_spectrogram_lowf(x: np.ndarray, fs_cpm: float,
                            spec_win_min: float, spec_overlap: float,
                            spec_scaling: str):
    """SciPy spectrogram wrapper that matches your low-f params."""
    from scipy.signal import spectrogram
    nperseg  = max(32, int(round(spec_win_min * fs_cpm)))
    noverlap = int(round(spec_overlap * nperseg))
    f_cpm, t_seg, Sxx = spectrogram(
        x, fs=fs_cpm, nperseg=nperseg, noverlap=noverlap,
        detrend=False, scaling=spec_scaling, mode="psd"
    )
    return f_cpm, t_seg, Sxx   # f in cycles/min, t in minutes, Sxx in PSD units