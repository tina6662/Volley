#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postprocess_inout_patched_v4.py  (X-DRIFT ENFORCED + auto click_court)

- Adds X-drift gating: drop candidates whose horizontal movement is too small.
- Threshold can be given as absolute pixels (--min-x-drift-px) and/or as a
  fraction of court width (--min-x-drift-pct). We take the MAX of the two.
- Court width sources, in order:
    1) court_json polygon width (max_x - min_x)
    2) video frame width (if --video_path provided and readable)
    3) robust x-range from data (5%~95% quantiles of smoothed x)
- Debug CSV now includes: x_pre_mean, x_post_mean, x_drift, x_range_local, x_thr

- NEW: If court.json is missing, auto-launch click_court to create it (if importable).
"""

import argparse, json, math, os, sys
import numpy as np
import pandas as pd
import cv2

# ----------------------- try to import click_court -----------------------
CLICK_COURT_MODULE = None
try:
    # case: tools/click_court.py
    from tools import click_court as _cc
    CLICK_COURT_MODULE = _cc
except Exception:
    try:
        # case: click_court.py in same directory / project root on PYTHONPATH
        import click_court as _cc
        CLICK_COURT_MODULE = _cc
    except Exception:
        CLICK_COURT_MODULE = None

# ========================= Utilities =========================

def movavg(a, k=5):
    k = max(1, int(k))
    if a is None or len(a)==0:
        return np.array([])
    a = np.asarray(a, dtype=float)
    pad = k//2
    ap = np.pad(a, (pad,pad), mode="edge")
    ker = np.ones(k, dtype=float)/k
    return np.convolve(ap, ker, mode="valid")

def guess_cols(df):
    names = {c.lower(): c for c in df.columns}
    # frame
    frame_col = names.get("frame", None)
    if frame_col is None:
        cand = [c for c in df.columns if c.lower().startswith("frame")]
        frame_col = cand[0] if cand else df.columns[0]

    # x, y
    x_alias = ["x","ball_x","cx","x_pos","xpix","center_x"]
    y_alias = ["y","ball_y","cy","y_pos","ypix","center_y"]
    x_col = next((names[a] for a in x_alias if a in names), None)
    y_col = next((names[a] for a in y_alias if a in names), None)
    if x_col is None:
        candx = [c for c in df.columns if "x" in c.lower()]
        x_col = candx[0] if candx else None
    if y_col is None:
        candy = [c for c in df.columns if "y" in c.lower()]
        y_col = candy[0] if candy else None
    return frame_col, x_col, y_col

def merge_nearby_frames(frames, win=12):
    frames = sorted(set(int(f) for f in frames))
    if not frames:
        return []
    merged = [frames[0]]
    for f in frames[1:]:
        if f - merged[-1] <= win:
            merged[-1] = max(merged[-1], f)
        else:
            merged.append(f)
    return merged

def _is_pair(p):
    return isinstance(p, (list, tuple)) and len(p)==2 and all(isinstance(x, (int,float)) for x in p)

def find_polygon_in_json(data):
    if isinstance(data, dict):
        for key in ["court_polygon","outer","polygon","court","bounds"]:
            if key in data:
                val = data[key]
                if isinstance(val, list) and len(val)>=4 and all(_is_pair(p) for p in val):
                    return val
                if isinstance(val, dict):
                    for v in val.values():
                        if isinstance(v, list) and len(v)>=4 and all(_is_pair(p) for p in v):
                            return v
        for v in data.values():
            poly = find_polygon_in_json(v)
            if poly is not None:
                return poly
    elif isinstance(data, list):
        if len(data)>=4 and all(_is_pair(p) for p in data):
            return data
        for v in data:
            poly = find_polygon_in_json(v)
            if poly is not None:
                return poly
    return None

def load_court_polygon(court_json_path):
    if not court_json_path:
        return None
    try:
        with open(court_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        poly = find_polygon_in_json(data)
        if poly is None:
            print(f"[court] Could not find a polygon in {court_json_path}. Expected a list of 4+ [x,y] points.")
            return None
        poly = [(int(round(p[0])), int(round(p[1]))) for p in poly]
        return poly
    except Exception as e:
        print(f"[court] Failed to load {court_json_path}: {e}")
        return None

def point_in_polygon(x, y, poly):
    if x is None or y is None or poly is None or len(poly)<3:
        return None
    inside = False
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1)%n]
        cond = ((y1 > y) != (y2 > y))
        if cond:
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside

def estimate_court_width_px(court_poly, video_path, x_s_full):
    # 1) court polygon
    if court_poly and len(court_poly)>=2:
        xs = [p[0] for p in court_poly]
        w = float(max(xs) - min(xs))
        if w > 0:
            return w, "court_json"
    # 2) video width
    if video_path:
        try:
            cap = cv2.VideoCapture(video_path)
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            cap.release()
            if w and w > 0:
                return float(w), "video_width"
        except Exception:
            pass
    # 3) robust x range from data (5%~95% quantiles)
    if x_s_full is not None and len(x_s_full)>0 and np.isfinite(x_s_full).any():
        xs = x_s_full[np.isfinite(x_s_full)]
        if xs.size >= 10:
            lo = np.percentile(xs, 5)
            hi = np.percentile(xs, 95)
            if hi > lo:
                return float(hi - lo), "data_quantile"
    return 0.0, "unknown"

# ========================= Stage 1: loose picks =========================

def stage1_candidates(frames, y, y_floor_pct=58.0, smooth_win=5, deriv_win=3,
                      pre_mean_win=3, post_mean_win=3, v_down_min=0.3, v_up_max=-0.3,
                      support_radius=2, support_min=1, min_gap=10, neighbor_expand=1, plateau_tol=4.0, verbose=False):
    y = y.copy().astype(float)
    y[y <= 0] = np.nan
    y_fill = pd.Series(y).interpolate(limit=30, limit_direction='both').to_numpy()
    y_s = movavg(y_fill, smooth_win)
    dy  = movavg(np.gradient(y_s), deriv_win)
    thr = float(np.nanpercentile(y_s, y_floor_pct))

    idx = []
    eps = 1e-6
    n = len(y_s)
    for i in range(1, n-1):
        if not np.isfinite(y_s[i]) or not np.isfinite(dy[i-1]) or not np.isfinite(dy[i+1]):
            continue
        pos_before = dy[i-1] > eps
        neg_after  = dy[i+1] < -eps
        near_zero  = abs(dy[i]) < 2e-3
        pos_ctx = pos_before or (np.nanmax(dy[max(0,i-2):i]) > eps)
        neg_ctx = neg_after  or (np.nanmin(dy[i+1:min(n,i+3)]) < -eps)
        sign_flip = (pos_before and neg_after) or (near_zero and pos_ctx and neg_ctx)
        if not sign_flip:
            continue
        if y_s[i] < thr:
            continue
        pre_mean = float(np.nanmean(dy[max(0,i-pre_mean_win):i]))
        post_mean = float(np.nanmean(dy[i+1:min(n, i+1+post_mean_win)]))
        if pre_mean < v_down_min:
            continue
        if post_mean > v_up_max:
            continue
        L = max(0, i-support_radius); R = min(n-1, i+support_radius)
        support = int(np.sum(y_s[L:R+1] >= y_s[i] - 1.5))
        if support < support_min:
            continue
        idx.append(i)

    picks = []
    last_f = -10**9
    for i in idx:
        f = int(frames[i])
        if f - last_f < min_gap:
            continue
        picks.append(i); last_f = f

    chosen = set()
    for i in picks:
        L = max(0, i-neighbor_expand); R = min(n-1, i+neighbor_expand)
        for k in range(L, R+1):
            chosen.add(int(frames[k]))
    out = sorted(chosen)
    if verbose:
        print(f"[stage1_loose] produced {len(out)} frames")
    return out

# ========================= Local window stats =========================

def window_stats(df, frames_arr, ys, center, W=15):
    i = max(0, center - W)
    j = min(len(df)-1, center + W)
    fr = frames_arr[i:j+1]
    y  = ys[i:j+1].copy().astype(float)
    y[y <= 0] = np.nan
    y_s = pd.Series(y).interpolate(limit=30, limit_direction='both').to_numpy()
    if np.any(~np.isfinite(y_s)):
        med = np.nanmedian(y_s)
        y_s[~np.isfinite(y_s)] = med
    y_s = movavg(y_s, 5)
    x = np.arange(len(y_s))
    coeff = np.polyfit(x, y_s, 2)
    concavity = coeff[0]
    residual = float(np.sqrt(np.mean((np.polyval(coeff, x)-y_s)**2)))
    if (fr==center).any():
        c = int(np.where(fr==center)[0][0])
    else:
        c = int(np.argmin(np.abs(fr-center)))
    pre = y_s[max(0,c-6):c+1]
    post = y_s[c:min(len(y_s), c+7)]
    pre_inc = np.sum(np.diff(pre) > 0)/max(1,len(pre)-1)
    post_dec= np.sum(np.diff(post) < 0)/max(1,len(post)-1)
    pre_drop = y_s[c] - np.min(pre) if len(pre)>0 else 0.0
    post_drop= y_s[c] - np.min(post) if len(post)>0 else 0.0
    return y_s, concavity, residual, pre_inc, post_dec, pre_drop, post_drop, c, fr

def window_observation_quality(df_ball, fr, W=6, teleport_px=80.0):
    cols = df_ball.columns.str.lower()
    dfb = df_ball.copy(); dfb.columns = cols
    frame_col = 'frame' if 'frame' in cols else cols[0]
    vis_col   = 'visibility' if 'visibility' in cols else None
    x_col     = 'x' if 'x' in cols else ( 'ball_x' if 'ball_x' in cols else ( 'cx' if 'cx' in cols else None))
    y_col     = 'y' if 'y' in cols else ( 'ball_y' if 'ball_y' in cols else ( 'cy' if 'cy' in cols else None))
    nb = dfb[(dfb[frame_col] >= fr - W) & (dfb[frame_col] <= fr + W)].copy()
    if vis_col is None:
        if x_col is None or y_col is None:
            nb['visible'] = 0.0
        else:
            nb['visible'] = ((nb[x_col] >= 0) & (nb[y_col] >= 0)).astype(float)
    else:
        nb['visible'] = (nb[vis_col] > 0).astype(float)
    vis_rate = float(nb['visible'].mean()) if len(nb) else 0.0
    teleports = 0
    if x_col is not None and y_col is not None:
        nbv = nb[nb['visible'] > 0].sort_values(frame_col)
        if len(nbv) >= 2:
            dx = nbv[x_col].diff().abs()
            dy = nbv[y_col].diff().abs()
            teleports = int(((dx > teleport_px) | (dy > teleport_px)).sum())
    return vis_rate, teleports

# ========================= Stage 2: filter + cluster =========================

def stage2_filter_and_select(df, frames_arr, ys, cand, W=15,
                             concavity_max=-0.05, pre_inc_min=0.85, post_dec_min=0.55,
                             pre_drop_min=18.0, post_drop_min=14.0,
                             vy_ratio_max=1.40, vy_down_stop=3.2, group_gap=6,
                             cluster_gap=60, cluster_select='dy_stop', require_down_stop=True, verbose=False,
                             residual_max=20.0, pre_range_min=10.0, post_range_min=8.0,
                             dir_win=5, require_no_dir_flip=True,
                             x_s_full=None, dx_full=None,
                             xdrift_thr_px=0.0):
    # Smooth y/dy
    y_full = ys.copy().astype(float); y_full[y_full<=0]=np.nan
    y_full = pd.Series(y_full).interpolate(limit=30, limit_direction='both').to_numpy()
    y_s_full = movavg(y_full, 5); dy_full = movavg(np.gradient(y_s_full), 3)

    feats = {}
    for f in cand:
        y_s, conc, resid, pre_inc, post_dec, pre_drop, post_drop, c, fr = window_stats(df, frames_arr, ys, f, W=W)
        idx_arr = np.where(frames_arr == f)[0]
        if len(idx_arr)==0:
            continue
        idx = int(idx_arr[0])
        pre = dy_full[max(0, idx - 6):idx]
        post = dy_full[idx + 1:min(len(dy_full), idx + 1 + 6)]
        pre_mean = float(np.nanmean(pre)) if len(pre) else 0.0
        post_mean = float(np.nanmean(post)) if len(post) else 0.0
        vy_ratio = (abs(post_mean) / pre_mean) if pre_mean > 1e-9 else np.inf
        pre_y = y_s[max(0, c - 6):c + 1]
        post_y = y_s[c:min(len(y_s), c + 7)]
        pre_range = float(np.nanmax(pre_y) - np.nanmin(pre_y)) if len(pre_y) > 0 else 0.0
        post_range = float(np.nanmax(post_y) - np.nanmin(post_y)) if len(post_y) > 0 else 0.0

        # -------- Direction flip (reworked; controlled only by --dir-win) --------
        pre_dir = float('nan'); post_dir = float('nan'); dir_flip = False
        if dx_full is not None and len(dx_full) == len(frames_arr):
            pre_dx  = dx_full[max(0, idx - dir_win):idx]
            post_dx = dx_full[idx + 1:min(len(dx_full), idx + 1 + dir_win)]

            def sgn(v):
                if not np.isfinite(v) or abs(v) < 1e-12: 
                    return 0
                return 1 if v > 0 else -1

            pre_dir  = float(np.nanmean(pre_dx))  if pre_dx.size  > 0 else float('nan')
            post_dir = float(np.nanmean(post_dx)) if post_dx.size > 0 else float('nan')

            flip_mean = (sgn(pre_dir) * sgn(post_dir) < 0)
            pre_has_pos  = (pre_dx.size  > 0) and np.isfinite(pre_dx).any()  and (np.nanmax(pre_dx)  > 0)
            pre_has_neg  = (pre_dx.size  > 0) and np.isfinite(pre_dx).any()  and (np.nanmin(pre_dx)  < 0)
            post_has_pos = (post_dx.size > 0) and np.isfinite(post_dx).any() and (np.nanmax(post_dx) > 0)
            post_has_neg = (post_dx.size > 0) and np.isfinite(post_dx).any() and (np.nanmin(post_dx) < 0)
            flip_any = (pre_has_pos and post_has_neg) or (pre_has_neg and post_has_pos)
            dir_flip = flip_mean or flip_any

        # ----------- NEW: X drift -----------
        x_pre_mean = x_post_mean = x_drift = x_range_local = float('nan')
        if x_s_full is not None and len(x_s_full)==len(frames_arr):
            pre_x = x_s_full[max(0, idx-dir_win):idx]
            post_x= x_s_full[idx+1:min(len(x_s_full), idx+1+dir_win)]
            if pre_x.size>0 and post_x.size>0:
                x_pre_mean  = float(np.nanmean(pre_x))
                x_post_mean = float(np.nanmean(post_x))
                x_drift = float(abs(x_post_mean - x_pre_mean))
            loc_x = x_s_full[max(0, idx-W):min(len(x_s_full), idx+W+1)]
            if loc_x.size>0:
                loc_x = loc_x[np.isfinite(loc_x)]
                if loc_x.size>0:
                    x_range_local = float(np.nanmax(loc_x) - np.nanmin(loc_x))

        feats[f] = dict(
            conc=conc, resid=resid, pre_inc=pre_inc, post_dec=post_dec,
            pre_drop=pre_drop, post_drop=post_drop, vy_ratio=vy_ratio,
            pre_range=pre_range, post_range=post_range, y_at=y_s[c],
            pre_dir=pre_dir, post_dir=post_dir, dir_flip=dir_flip,
            x_pre_mean=x_pre_mean, x_post_mean=x_post_mean,
            x_drift=x_drift, x_range_local=x_range_local,
        )

    passed = []
    for f, v in feats.items():
        passed_flag = (
            v["conc"] <= concavity_max and
            v.get("resid", 0.0) <= residual_max and
            v["pre_inc"] >= pre_inc_min and
            v["post_dec"] >= post_dec_min and
            v["pre_drop"] >= pre_drop_min and
            v["post_drop"] >= post_drop_min and
            v.get("pre_range", 0.0) >= pre_range_min and
            v.get("post_range", 0.0) >= post_range_min and
            v["vy_ratio"] <= vy_ratio_max
        )
        base = (v.get('y_at', float('inf')) >= y_abs_min) and (v.get('post_drop', float('-inf')) <= post_drop_max)
        looks_like_hit = (
            v.get('post_drop', 0.0) >= HIT_PD and
            v.get('vy_ratio', 0.0)  >= HIT_VYR and
            v.get('y_at', 1e9)      <= HIT_YUP
        )
        looks_like_dead = (
            v.get('post_drop', 1e9) <= DEAD_PD and
            v.get('vy_ratio', 1e9)  <= DEAD_VYR and
            v.get('pre_drop', 0.0)  >= DEAD_PRE
        )

        if require_no_dir_flip and v.get('dir_flip', False):
            passed_flag = False

        # ---- NEW: fail if x movement below threshold ----
        if xdrift_thr_px > 0 and math.isfinite(v.get("x_drift", float('nan'))):
            effective_drift = max(v.get("x_drift", 0.0), v.get("x_range_local", 0.0))
            if not (effective_drift >= xdrift_thr_px):
                passed_flag = False

        vis_rate, teleports = window_observation_quality(df_ball=df, fr=f, W=W, teleport_px=TP_PX)
        vis_ok  = (vis_rate >= MIN_VIS)
        tele_ok = (teleports <= MAX_TP)
        passed_flag = passed_flag and (base or looks_like_dead) and (not looks_like_hit) and vis_ok and tele_ok
        if passed_flag:
            passed.append(f)
    if verbose: print(f"[curve] passed shape/ratio+dir+xdrift: {len(passed)}")
    if not passed:
        return []

    # group -> clusters -> pick
    groups = [[passed[0]]]
    for f in passed[1:]:
        if f - groups[-1][-1] <= group_gap:
            groups[-1].append(f)
        else:
            groups.append([f])
    clusters = [groups[0][:]]
    for g in groups[1:]:
        if g[0] - clusters[-1][-1] <= cluster_gap:
            clusters[-1].extend(g)
        else:
            clusters.append(g[:])

    out = []
    for g in clusters:
        cand_idx = []
        for fr in g:
            idx = int(np.where(frames_arr==fr)[0][0])
            cand_idx.append((fr, dy_full[idx], y_s_full[idx]))
        if cluster_select == 'dy_stop':
            pick = None
            for fr, d, yv in sorted(cand_idx, key=lambda t: t[0]):
                if d >= vy_down_stop:
                    pick = fr
            if pick is None:
                guarded = [fr for (fr, d, yv) in cand_idx if feats.get(fr,{}).get('pre_range',0)>=pre_range_min
                           and feats.get(fr,{}).get('post_range',0)>=post_range_min
                           and feats.get(fr,{}).get('resid',0)<= residual_max]
                if guarded:
                    pick = max(guarded, key=lambda fr: feats[fr]['y_at'])
                elif require_down_stop:
                    continue
                else:
                    pick = sorted(cand_idx, key=lambda t: t[2])[-1][0]
        elif cluster_select == 'min_y':
            pick = sorted(cand_idx, key=lambda t: t[2])[0][0]
        elif cluster_select == 'max_y':
            pick = sorted(cand_idx, key=lambda t: t[2])[-1][0]
        elif cluster_select == 'earliest':
            pick = sorted(cand_idx, key=lambda t: t[0])[0][0]
        elif cluster_select == 'latest':
            pick = sorted(cand_idx, key=lambda t: t[0])[-1][0]
        else:
            pick = sorted(cand_idx, key=lambda t: t[2])[-1][0]
        out.append(pick)

    # Debug features
    debug_rows = []
    for f, v in feats.items():
        vv = v.copy(); vv['frame'] = f; vv['x_thr'] = float(xdrift_thr_px)
        debug_rows.append(vv)
    try:
        os.makedirs("output", exist_ok=True)
        pd.DataFrame(debug_rows).to_csv("output/landing_debug_feats_v3.csv", index=False)
    except Exception:
        pass

    final = sorted(out)
    if verbose: print(f"[curve] selected per cluster: {final}")
    return final

# ========================= XY helpers =========================

def valid_pair(xv, yv):
    try:
        xv = float(xv); yv = float(yv)
    except Exception:
        return False
    if not (math.isfinite(xv) and math.isfinite(yv)):
        return False
    if int(round(xv)) == -1 or int(round(yv)) == -1:
        return False
    if xv <= 0 or yv <= 0:
        return False
    return True

def find_nearest_xy(frame_idx, xy_map, window):
    if window <= 0:
        return (None, float('nan'), float('nan'))
    if frame_idx in xy_map:
        xv, yv = xy_map[frame_idx]
        if valid_pair(xv, yv):
            return (frame_idx, xv, yv)
    for d in range(1, window+1):
        for cand in (frame_idx - d, frame_idx + d):
            if cand in xy_map:
                xv, yv = xy_map[cand]
                if valid_pair(xv, yv):
                    return (cand, xv, yv)
    return (None, float('nan'), float('nan'))

# ========================= Overlay HUD =========================

def overlay_video(video_path, out_video, keep_rows, court_poly, args):
    import math
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[emit] cannot open video: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    outv = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

    persist_frames = max(1, int(round(fps * args.persist_sec)))

    # 事件列表（依 frame 排序，帶 idx）
    events = []
    for idx, r in enumerate(keep_rows.sort_values("frame").itertuples(index=False)):
        start = int(r.frame)
        end   = start + persist_frames
        events.append({
            "idx": idx+1,
            "start": start, "end": end, "frame": int(r.frame),
            "x": r.x, "y": r.y, "in_out": r.in_out,
            "orig_frame": int(getattr(r,"orig_frame", r.frame)),
            "xy_from_frame": int(getattr(r,"xy_from_frame", r.frame)),
            "time_sec": float(getattr(r,"time_sec", 0.0)),
        })

    # 畫面用色與字型
    BLUE=(255,0,0); GREEN=(0,255,0); RED=(0,0,255); YEL=(0,255,255)
    WHITE=(255,255,255); BLACK=(0,0,0); GRAY=(64,64,64)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 球場多邊形
    poly_np = None
    if court_poly:
        poly_np = np.array(court_poly, dtype=np.int32).reshape((-1,1,2))

    # ------------ History HUD 畫表格 ------------
    def draw_history_table(frame, cur_frame):
        hud_width = int(0.2 * w)
        panel_x1 = w - hud_width
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x1, 0), (w, h), BLACK, thickness=-1)
        cv2.addWeighted(overlay, args.hud_alpha, frame, 1-args.hud_alpha, 0, frame)

        cv2.putText(frame, "Landing History", (panel_x1 + 10, 28),
                    font, args.hud_font_scale+0.2, WHITE, 2)

        cols = [c.strip() for c in args.hud_cols.split(",") if c.strip()]
        col_titles = {
            "idx":"#", "frame":"F", "time":"t(s)", "in_out":"IN/OUT",
            "x":"X", "y":"Y", "orig_frame":"F0", "xy_from_frame":"FXY"
        }
        col_widths = {"idx":46, "frame":76, "time":72, "in_out":96, "x":70, "y":70,
                      "orig_frame":78, "xy_from_frame":88}

        x_positions = {}
        xcur = panel_x1 + 10
        for c in cols:
            x_positions[c] = xcur
            xcur += col_widths.get(c, 80)

        y0 = 52
        row_h = max(18, int(22 * args.hud_font_scale))
        for c in cols:
            title = col_titles.get(c, c.upper())
            cv2.putText(frame, title, (x_positions[c], y0), font, args.hud_font_scale, YEL, 1)
        y = y0 + int(row_h*0.9)
        cv2.line(frame, (panel_x1+8, y), (w-8, y), GRAY, 1)

        hist = [e for e in events if e["frame"] <= cur_frame]
        total = len(hist)
        max_rows = max(4, args.hud_max_rows)
        shown = hist[-max_rows:]

        for e in shown:
            vals = {
                "idx":   str(e["idx"]),
                "frame": str(e["frame"]),
                "time":  f"{e['time_sec']:.2f}",
                "in_out": e["in_out"].upper(),
                "x":     ("" if not math.isfinite(e["x"]) else str(int(e["x"]))),
                "y":     ("" if not math.isfinite(e["y"]) else str(int(e["y"]))),
                "orig_frame": str(e["orig_frame"]),
                "xy_from_frame": str(e["xy_from_frame"]),
            }
            color = GREEN if e["in_out"]=="in" else (RED if e["in_out"]=="out" else YEL)
            y += row_h
            for c in cols:
                cv2.putText(frame, vals.get(c, ""), (x_positions[c], y),
                            font, args.hud_font_scale, color, 1)

        cv2.putText(frame, f"Total: {total}", (panel_x1 + 10, h - 12),
                    font, args.hud_font_scale, WHITE, 1)

    # 主循環
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if poly_np is not None:
            cv2.polylines(frame, [poly_np], isClosed=True, color=BLUE, thickness=2)

        active = [e for e in events if e["start"] <= fidx <= e["end"]]
        for e in active:
            x, y = e["x"], e["y"]
            label = e["in_out"]
            color = GREEN if label=="in" else (RED if label=="out" else YEL)
            if pd.notna(x) and pd.notna(y):
                xi, yi = int(x), int(y)
                cv2.circle(frame, (xi, yi), 14, color, 3)
                cv2.putText(frame, f"{label.upper()} {e['frame']} ({e['time_sec']:.2f}s)",
                            (xi+16, yi-16), font, 0.7, color, 2)

        if getattr(args, "hud", False):
            if getattr(args, "hud_mode", "history") == "history":
                draw_history_table(frame, fidx)
            else:
                hud_width = int(0.28 * w)
                panel_x1 = w - hud_width
                overlay = frame.copy()
                cv2.rectangle(overlay, (panel_x1, 0), (w, h), BLACK, thickness=-1)
                cv2.addWeighted(overlay, args.hud_alpha, frame, 1-args.hud_alpha, 0, frame)
                ycur = 24
                cv2.putText(frame, "Landings", (panel_x1 + 10, ycur),
                            font, args.hud_font_scale+0.1, WHITE, 2)
                ycur += 18
                lst = [e for e in events if e["frame"] <= fidx]
                lst = lst[-args.hud_max_rows:]
                for e in lst:
                    color = GREEN if e["in_out"]=="in" else (RED if e["in_out"]=="out" else YEL)
                    info = f"#{e['frame']}  {e['time_sec']:.2f}s  {e['in_out']}"
                    cv2.putText(frame, info, (panel_x1 + 10, ycur),
                                font, args.hud_font_scale, color, 1)
                    ycur += int(22 * args.hud_font_scale)

        outv.write(frame); fidx += 1

    cap.release(); outv.release()
    print(f"Saved video with overlays -> {out_video}")

# ========================= Blacklist parse =========================

def parse_blacklist(args):
    blk = set()
    if args.blacklist_frames:
        for tok in args.blacklist_frames.split(","):
            tok = tok.strip()
            if tok.isdigit():
                blk.add(int(tok))
    if args.blacklist_ranges:
        for seg in args.blacklist_ranges.split(";"):
            seg = seg.strip()
            if "-" in seg:
                a,b = seg.split("-",1)
                if a.strip().isdigit() and b.strip().isdigit():
                    lo, hi = int(a), int(b)
                    if lo > hi: lo, hi = hi, lo
                    blk.update(range(lo, hi+1))
    return blk

# ========================= Main =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y-abs-min", type=float, default=640.0)
    ap.add_argument("--post-drop-max", type=float, default=55.0)
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--court_json", default=None)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--video_path", default=None)
    ap.add_argument("--out_video", default=None)
    ap.add_argument("--stage1_loose", action="store_true")
    ap.add_argument("--y_floor_pct", type=float, default=58.0)
    ap.add_argument("--skip_final_filter", action="store_true")
    ap.add_argument("--emit_candidates_only", action="store_true")
    ap.add_argument("--curve_filter_on", action="store_true")
    ap.add_argument("--curve_win", type=int, default=15)
    ap.add_argument("--concavity_max", type=float, default=-0.05)
    ap.add_argument("--pre_inc_min", type=float, default=0.8)
    ap.add_argument("--post_dec_min", type=float, default=0.45)
    ap.add_argument("--pre_drop_min", type=float, default=12.0)
    ap.add_argument("--post_drop_min", type=float, default=10.0)
    ap.add_argument("--vy_ratio_max", type=float, default=2.2)
    ap.add_argument("--vy_down_stop", type=float, default=2.8)
    ap.add_argument("--group_gap", type=int, default=6)
    ap.add_argument("--cluster_gap", type=int, default=60)
    ap.add_argument("--cluster_select", default='dy_stop', choices=['dy_stop','min_y','max_y','earliest','latest'])
    ap.add_argument("--require_down_stop", action="store_true", default=True)
    ap.add_argument("--keep_frames", default=None)
    ap.add_argument("--whitelist_only", action="store_true")
    ap.add_argument("--min_event_gap", type=int, default=12)
    ap.add_argument("--snap_xy_window", type=int, default=3)
    ap.add_argument("--snap_replace_frame", action="store_true")
    ap.add_argument("--persist_sec", type=float, default=1.5)
    ap.add_argument("--hud", action="store_true")
    ap.add_argument("--hud_max_rows", type=int, default=18)
    ap.add_argument("--hud_font_scale", type=float, default=0.55)
    ap.add_argument("--hud_alpha", type=float, default=0.35)
    ap.add_argument("--blacklist_frames", default=None)
    ap.add_argument("--blacklist_ranges", default=None)
    ap.add_argument("--residual_max", type=float, default=30.0)
    ap.add_argument("--pre_range_min", type=float, default=6.0)
    ap.add_argument("--post_range_min", type=float, default=6.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--min-vis-rate", type=float, default=0.6)
    ap.add_argument("--max-teleports", type=int, default=1)
    ap.add_argument("--teleport-px", type=float, default=80.0)
    ap.add_argument("--hit-post-drop-min", type=float, default=30.0)
    ap.add_argument("--hit-vy-ratio-min", type=float, default=1.8)
    ap.add_argument("--hit-y-upper", type=float, default=640.0)
    ap.add_argument("--dead-post-drop-max", type=float, default=6.0)
    ap.add_argument("--dead-vy-ratio-max", type=float, default=0.50)
    ap.add_argument("--dead-pre-drop-min", type=float, default=20.0)
    # Direction filter knobs
    ap.add_argument("--dir-win", type=int, default=3, help="Half-window to average dx before/after candidate")
    ap.add_argument("--allow-dir-flip", action="store_true", help="Do NOT drop on direction flips")
    # NEW: X-drift knobs
    ap.add_argument("--min-x-drift-px", type=float, default=0.0, help="Minimum required x movement in pixels")
    ap.add_argument("--min-x-drift-pct", type=float, default=0.0, help="Minimum required x movement as fraction of court width (e.g., 0.5 for half court)")
    ap.add_argument("--hud_mode", choices=["history","classic"], default="history",
                help="history: 右側表格列出過去落地點；classic: 舊的簡易清單")
    ap.add_argument("--hud_cols", default="idx,frame,time,in_out,x,y",
                help="表格欄位，逗號分隔；可用: idx,frame,time,in_out,x,y,orig_frame,xy_from_frame")

    args = ap.parse_args()
    global y_abs_min, post_drop_max
    y_abs_min = getattr(args, 'y_abs_min', 730.0)
    post_drop_max = getattr(args, 'post_drop_max', 40.0)
    global MIN_VIS, MAX_TP, TP_PX, HIT_PD, HIT_VYR, HIT_YUP, DEAD_PD, DEAD_VYR, DEAD_PRE
    MIN_VIS  = getattr(args, 'min_vis_rate', 0.60)
    MAX_TP   = getattr(args, 'max_teleports', 1)
    TP_PX    = getattr(args, 'teleport_px', 80.0)
    HIT_PD   = getattr(args, 'hit_post_drop_min', 45.0)
    HIT_VYR  = getattr(args, 'hit_vy_ratio_min', 2.4)
    HIT_YUP  = getattr(args, 'hit_y_upper', 750.0)
    DEAD_PD  = getattr(args, 'dead_post_drop_max', 6.0)
    DEAD_VYR = getattr(args, 'dead_vy_ratio_max', 0.50)
    DEAD_PRE = getattr(args, 'dead_pre_drop_min', 20.0)

    forced_set = set()

    # --- auto-run click_court if court.json is missing and module exists ---
    need_court = (not args.court_json) or (not os.path.exists(args.court_json))
    if need_court and CLICK_COURT_MODULE is not None:
        if not args.video_path:
            print("[auto] court.json 不存在，且未提供 --video_path，無法啟動 click_court。")
        else:
            out_path = args.court_json or os.path.join("output", "court.json")
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            print(f"[auto] court.json not found → launching click_court to create: {out_path}")
            _saved_argv = sys.argv[:]
            try:
                sys.argv = ["click_court.py", "--video", args.video_path, "--out", out_path]
                if hasattr(CLICK_COURT_MODULE, "main"):
                    CLICK_COURT_MODULE.main()
                else:
                    print("[auto] click_court module has no main(); skip.")
            finally:
                sys.argv = _saved_argv
            args.court_json = out_path

    df = pd.read_csv(args.csv_path)
    frame_col, x_col, y_col = guess_cols(df)
    frames = df[frame_col].astype(int).values
    y = pd.to_numeric(df[y_col], errors="coerce").astype(float).values

    # Prepare smooth x and dx for direction / drift checks
    names = {c.lower(): c for c in df.columns}
    _x_col = None
    for k in ["x","ball_x","cx","x_pos","xpix","center_x"]:
        if k in names: _x_col = names[k]; break
    if _x_col is None:
        for c in df.columns:
            if 'x' in c.lower():
                _x_col = c; break

    x_s_full = dx_full = None
    if _x_col is not None:
        x_full = pd.to_numeric(df[_x_col], errors="coerce").astype(float).values
        x_full[x_full<=0] = np.nan
        x_full = pd.Series(x_full).interpolate(limit=30, limit_direction='both').to_numpy()
        x_s_full = movavg(x_full, 5)
        dx_full = movavg(np.gradient(x_s_full), 3)

    # FPS guess
    fps_guess = 30.0
    if args.video_path:
        try:
            _cap = cv2.VideoCapture(args.video_path)
            _fps = _cap.get(cv2.CAP_PROP_FPS)
            if _fps and _fps > 0:
                fps_guess = float(_fps)
            _cap.release()
        except Exception:
            pass

    court_poly = load_court_polygon(args.court_json) if args.court_json else None
    if args.court_json and court_poly is None:
        print("[court] Warning: court.json provided but polygon not found; in/out will be 'unknown'.")

    # === NEW: compute x-drift threshold in pixels ===
    court_w_px, cw_src = estimate_court_width_px(court_poly, args.video_path, x_s_full)
    thr_px_from_pct = (args.min_x_drift_pct * court_w_px) if (court_w_px > 0 and args.min_x_drift_pct > 0) else 0.0
    xdrift_thr_px = max(float(args.min_x_drift_px or 0.0), float(thr_px_from_pct or 0.0))
    if xdrift_thr_px > 0:
        print(f"[xdrift] court_width={court_w_px:.1f} ({cw_src}), min-x-drift={xdrift_thr_px:.1f}px "
              f"(px={args.min_x_drift_px}, pct={args.min_x_drift_pct})")

    # Stage 1
    cand_frames = []
    if args.stage1_loose or args.curve_filter_on or True:
        stage1 = stage1_candidates(frames, y, y_floor_pct=args.y_floor_pct, verbose=args.verbose)
        cand_frames = sorted(set(cand_frames).union(stage1))

    # Helper to collect rows
    def build_rows(frames_list, forced_set=None):
        xy_map = {}
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            for _, r in df.iterrows():
                try:
                    f = int(r[frame_col])
                except Exception:
                    continue
                xv = pd.to_numeric(pd.Series([r[x_col]])).iloc[0]
                yv = pd.to_numeric(pd.Series([r[y_col]])).iloc[0]
                xv = float(xv) if pd.notna(xv) else np.nan
                yv = float(yv) if pd.notna(yv) else np.nan
                if pd.notna(xv) and (int(round(xv)) == -1 or xv <= 0): xv = np.nan
                if pd.notna(yv) and (int(round(yv)) == -1 or yv <= 0): yv = np.nan
                xy_map[f] = (xv, yv)

        used_frames = set()
        out_rows = []
        for f in frames_list:
            detected_f = int(f)
            xv = yv = np.nan
            use_frame = detected_f
            xy_from_frame = detected_f
            snapped = False

            if detected_f in xy_map:
                xv, yv = xy_map[detected_f]

            if not (pd.notna(xv) and pd.notna(yv)) and args.snap_xy_window > 0:
                nf, nx, ny = find_nearest_xy(detected_f, xy_map, args.snap_xy_window)
                if nf is not None and pd.notna(nx) and pd.notna(ny):
                    snapped = True
                    xy_from_frame = int(nf)
                    if args.snap_replace_frame:
                        use_frame = int(nf)
                    xv, yv = nx, ny

            label = "unknown"
            if pd.notna(xv) and pd.notna(yv) and court_poly is not None:
                inside = point_in_polygon(float(xv), float(yv), court_poly)
                if inside is True:
                    label = "in"
                elif inside is False:
                    label = "out"

            if use_frame in used_frames:
                continue
            used_frames.add(use_frame)

            def f2t(fr):
                return float(fr) / fps_guess if fps_guess > 0 else np.nan

            row = {
                "frame": int(use_frame),
                "x": xv, "y": yv, "in_out": label,
                "orig_frame": int(detected_f),
                "xy_from_frame": int(xy_from_frame),
                "time_sec": f2t(use_frame),
                "orig_time_sec": f2t(detected_f),
                "xy_from_time_sec": f2t(xy_from_frame),
                "snapped_xy": bool(snapped),
                "replaced_frame": bool(args.snap_replace_frame and snapped),
                "forced": False
            }
            out_rows.append(row)

        df_out = pd.DataFrame(out_rows)
        if df_out.empty or "frame" not in df_out.columns:
            return pd.DataFrame(columns=[
                "frame","x","y","in_out",
                "orig_frame","xy_from_frame",
                "time_sec","orig_time_sec","xy_from_time_sec",
                "snapped_xy","replaced_frame","forced"
            ])
        df_out = df_out.sort_values("frame").reset_index(drop=True)
        if forced_set:
            df_out['forced'] = df_out['frame'].apply(lambda f: int(f) in forced_set)
        return df_out

    # emit-only / skip-final
    if args.skip_final_filter or args.emit_candidates_only:
        all_frames = merge_nearby_frames(cand_frames, win=args.min_event_gap)
        if args.keep_frames:
            wl = sorted({int(x) for x in args.keep_frames.split(",") if x.strip().isdigit()})
            if args.whitelist_only:
                all_frames = wl
            else:
                all_frames = merge_nearby_frames(sorted(set(all_frames).union(wl)), win=args.min_event_gap) if all_frames else wl

        blk = parse_blacklist(args)
        if blk:
            all_frames = [f for f in all_frames if int(f) not in blk]

        out_df = build_rows(all_frames, forced_set)
        out_df.to_csv(args.out_csv, index=False)
        print(f"Saved {len(out_df)} landings -> {args.out_csv}")
        if args.video_path and args.out_video:
            overlay_video(args.video_path, args.out_video, out_df, court_poly, args)
        print("Final frames:", out_df["frame"].tolist())
        return

    # Stage 2
    final_frames = cand_frames
    if args.curve_filter_on:
        final_frames = stage2_filter_and_select(
            df=df, frames_arr=frames, ys=y, cand=cand_frames, W=args.curve_win,
            concavity_max=args.concavity_max, pre_inc_min=args.pre_inc_min, post_dec_min=args.post_dec_min,
            pre_drop_min=args.pre_drop_min, post_drop_min=args.post_drop_min,
            vy_ratio_max=args.vy_ratio_max, vy_down_stop=args.vy_down_stop,
            group_gap=args.group_gap, cluster_gap=args.cluster_gap,
            cluster_select=args.cluster_select, require_down_stop=args.require_down_stop,
            verbose=args.verbose,
            residual_max=args.residual_max, pre_range_min=args.pre_range_min, post_range_min=args.post_range_min,
            dir_win=args.dir_win, require_no_dir_flip=(not args.allow_dir_flip),
            x_s_full=x_s_full, dx_full=dx_full,
            xdrift_thr_px=xdrift_thr_px
        )

    if args.keep_frames:
        wl = sorted({int(x) for x in args.keep_frames.split(",") if x.strip().isdigit()})
        if args.whitelist_only:
            final_frames = wl
        else:
            final_frames = merge_nearby_frames(sorted(set(final_frames).union(wl)), win=args.min_event_gap) if final_frames else wl

    blk = parse_blacklist(args)
    if blk:
        final_frames = [f for f in final_frames if int(f) not in blk]

    out_df = build_rows(final_frames, forced_set)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved {len(out_df)} landings -> {args.out_csv}")
    print("Final frames:", out_df["frame"].tolist())
    if args.video_path and args.out_video:
        overlay_video(args.video_path, args.out_video, out_df, court_poly, args)

if __name__ == "__main__":
    main()
