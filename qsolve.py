"""qsolve.py -- constrained "minimise cosine" solver for the quant maker.

Finds the GUI-expressible assignment that maximises whole-model cosine
under a target file size:
  * one tier per weight group (the present quantizable groups of the model)
  * optional layer bumps: all quantizable tensors of the selected layers
    raised by N tiers (clamped at the ladder top)

Method (aggregated to the group level):
  1. Dinkelbach outer loop + separable (mu, nu) Lagrangian inner over the
     groups, using measured per-group S/Q/C from the cosine table.
  2. Steepest-ascent polish over the joint space (group tier +/-1, group
     pair raise/lower, add/remove bump layer) -- O(n) per move via
     precomputed per-layer group-tier sums. Bump depth is fixed at 1.

Model-agnostic: the group list and layer count come from the table/context
(no per-arch knowledge). Budget accounting uses the table's MEASURED bytes
(C + c0 + overhead) -- c0 already bakes the frozen tensors exactly.
"""
import re

import numpy as np

import qmodel as q

_LAY_RE = re.compile(r"^blk\.(\d+)\.")


def _matrices(t):
    """Per-group (n,K) and per-layer (n_layer+1,n,K) S/Q/C sums."""
    groups = [g for g in t["GROUPS"]]
    nlay = int(t["n_layer"]) + 1
    K = len(t["ladder"])
    n = len(groups)
    gi = {g: i for i, g in enumerate(groups)}
    SG = np.zeros((n, K)); QG = np.zeros((n, K)); CG = np.zeros((n, K))
    SL = np.zeros((nlay, n, K)); QL = np.zeros((nlay, n, K))
    CL = np.zeros((nlay, n, K))
    pos, upg, S, Q, C = t["pos"], t["upg"], t["S"], t["Q"], t["C"]
    for i, name in enumerate(t["names"]):
        if not upg[i]:
            continue
        g = gi.get(t["groups"][i])
        if g is None:
            continue
        SG[g] += S[i]; QG[g] += Q[i]; CG[g] += C[i]
        m = _LAY_RE.match(name)
        if m is None or int(m.group(1)) >= nlay:
            # Embed / Output (no blk.L) or out-of-range layer: never bumped
            continue
        L = int(m.group(1))
        SL[L, g] += S[i]; QL[L, g] += Q[i]; CL[L, g] += C[i]
    return SG, QG, CG, SL, QL, CL


def solve(target_gb, use_imatrix=False):
    """Return (result_dict, None) or (None, error_message)."""
    t = q._load_cos("imatrix" if use_imatrix else "plain")
    if t is None:
        return None, "cosine table not found"
    ladder = t["ladder"]; K = len(ladder)
    A, s0, q0 = t["A_total"], t["s0"], t["q0"]
    SG, QG, CG, SL, QL, CL = _matrices(t)
    groups = [g for g in t["GROUPS"]]
    n = len(groups)
    AR = np.arange(n)
    nlay = int(t["n_layer"]) + 1

    # byte budget for group tiers: target - frozen (c0) - metadata. c0
    # already bakes every frozen tensor exactly, so no per-tensor adjust.
    frozen = t["c0"]
    B = target_gb * 1e9 - frozen - t["overhead"]
    # per-group floor = each group at its OWN smallest tier (col 0 may not
    # be the true min when imatrix-required IQ tiers fall back to a K tier).
    min_cost = float(CG.min(axis=1).sum())
    if B < min_cost:
        return None, (f"target {target_gb:g} GB is below the minimum "
                      f"({(min_cost + frozen + t['overhead'])/1e9:.2f} GB, "
                      "smallest tier per group)")

    def cosval(Su, Qu):
        return (Su + s0) / np.sqrt(A * (Qu + q0))

    # ---------------- phase 1: Dinkelbach + Lagrangian over groups -------
    def H(mu):
        """max_t sum_g [S - mu Q - nu C]  s.t. C<=B  -> (ti, value, cost)."""
        W = SG - mu * QG

        def pick(nu):
            M = W - nu * CG
            ti = M.argmax(1)
            return ti, float(M.max(1).sum()), float(CG[AR, ti].sum())

        if pick(0.0)[2] <= B:
            return pick(0.0)
        hi = 1e-6
        while pick(hi)[2] > B and hi < 1e12:
            hi *= 10.0
        lo = 0.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if pick(mid)[2] > B:
                lo = mid
            else:
                hi = mid
        best = None
        for nu in (hi, hi * 0.5, hi * 2.0, lo):
            ti, v, c = pick(nu)
            if c <= B and (best is None or v > best[1]):
                best = (ti, v, c)
        return best

    x = np.full(n, ladder.index("Q4_K"), dtype=np.int64)
    lam = cosval(float(SG[AR, x].sum()), float(QG[AR, x].sum()))
    xbest = x.copy()
    for _ in range(30):
        kappa = lam * np.sqrt(A)
        Qcur = float(QG[AR, xbest].sum())
        mu0 = kappa * kappa / (4.0 * max(Qcur, 1e-9))
        grid = mu0 * np.logspace(-2.5, 2.5, 200)
        gv = np.full(len(grid), -np.inf)
        gx = [None] * len(grid)
        for k, mu in enumerate(grid):
            r = H(float(mu))
            if r is None or r[2] > B:
                continue
            g = -mu * q0 - kappa * kappa / (4.0 * mu) + r[1]
            gv[k] = g
            gx[k] = r[0]
        k0 = int(np.argmax(gv))
        if gx[k0] is None:
            break

        def geval(mu):
            r = H(mu)
            if r is None or r[2] > B:
                return -np.inf, None
            return -mu * q0 - kappa * kappa / (4.0 * mu) + r[1], r[0]

        a = max(0, k0 - 3); b = min(len(grid) - 1, k0 + 3)
        mlo, mhi = float(grid[a]), float(grid[b])
        gr = (np.sqrt(5) - 1) / 2
        c_ = mhi - gr * (mhi - mlo); d_ = mlo + gr * (mhi - mlo)
        fc, xfc = geval(c_); fd, xfd = geval(d_)
        for _ in range(35):
            if fc < fd:
                mlo, c_, fc, xfc = c_, d_, fd, xfd
                c_ = mhi - gr * (mhi - mlo); fc, xfc = geval(c_)
            else:
                mhi, d_, fd, xfd = d_, c_, fc, xfc
                d_ = mlo + gr * (mhi - mlo); fd, xfd = geval(d_)
            if mhi - mlo < 1e-7 * mhi:
                break
        _, xi = max((fc, xfc), (fd, xfd), (gv[k0], gx[k0]),
                    key=lambda z: z[0])
        if xi is None:
            break
        xbest = xi
        newlam = cosval(float(SG[AR, xi].sum()), float(QG[AR, xi].sum()))
        if abs(newlam - lam) < 1e-11 * lam:
            lam = newlam
            break
        lam = newlam

    # ---------------- phase 2: polish over (groups, bumps)  (fixed 1) ----
    x = xbest.copy()
    bump = np.zeros(nlay, dtype=bool)
    steps = 1

    def bump_sums():
        return (SL[bump].sum(0), QL[bump].sum(0), CL[bump].sum(0))

    def eval_state(xv, N1S, N1Q, N1C, st):
        t1 = np.minimum(xv + st, K - 1)
        Sg = (SG[AR, xv] - N1S[AR, xv] + N1S[AR, t1])
        Qg = (QG[AR, xv] - N1Q[AR, xv] + N1Q[AR, t1])
        Cg = (CG[AR, xv] - N1C[AR, xv] + N1C[AR, t1])
        return float(Sg.sum()), float(Qg.sum()), float(Cg.sum())

    N1S, N1Q, N1C = bump_sums()
    Su, Qu, Ct = eval_state(x, N1S, N1Q, N1C, steps)
    cur = cosval(Su, Qu)

    def better(cos2, c2):
        return c2 <= B and (cos2 > cur + 1e-12
                            or (abs(cos2 - cur) <= 1e-12
                                and c2 < Ct - 1.0))

    improved = True
    while improved:
        improved = False
        best = None          # (cos, cost, kind, payload)

        def consider(c2, ccost, kind, payload):
            nonlocal best
            if better(c2, ccost) and (best is None or c2 > best[0]):
                best = (c2, ccost, kind, payload)

        for g in range(n):
            for d in (-1, 1):
                nt = x[g] + d
                if 0 <= nt < K:
                    x2 = x.copy(); x2[g] = nt
                    s2 = eval_state(x2, N1S, N1Q, N1C, steps)
                    consider(cosval(s2[0], s2[1]), s2[2], "x", x2)
        for g in range(n):
            if x[g] >= K - 1:
                continue
            for h in range(n):
                if h == g or x[h] <= 0:
                    continue
                x2 = x.copy(); x2[g] += 1; x2[h] -= 1
                s2 = eval_state(x2, N1S, N1Q, N1C, steps)
                consider(cosval(s2[0], s2[1]), s2[2], "x", x2)
        for L in range(nlay):
            b2 = bump.copy(); b2[L] = not b2[L]
            N2S, N2Q, N2C = (SL[b2].sum(0), QL[b2].sum(0), CL[b2].sum(0))
            s2 = eval_state(x, N2S, N2Q, N2C, steps)
            consider(cosval(s2[0], s2[1]), s2[2], "b", b2)

        if best is None:
            break
        improved = True
        kind, payload = best[2], best[3]
        if kind == "x":
            x = payload
        elif kind == "b":
            bump = payload
            N1S, N1Q, N1C = bump_sums()
        Su, Qu, Ct = eval_state(x, N1S, N1Q, N1C, steps)
        cur = cosval(Su, Qu)

    tiers = {groups[g]: ladder[int(x[g])] for g in range(n)}
    layers = sorted(int(L) for L in range(nlay) if bump[L])
    size = (Ct + frozen + t["overhead"]) / 1e9
    return ({
        "tiers": tiers, "layers": layers, "steps": int(steps),
        "cos": float(cur), "dev": 1.0 - float(cur), "size_gb": size,
        "method": "group+bump",
    }, None)


def solve_min_size(target_cos, use_imatrix=False, tol_gb=0.05):
    """Smallest size in the group+bump space achieving cos >= target_cos.

    cos(solution(B)) is monotone non-decreasing in the byte budget B, so
    binary-search B between the all-IQ1_S minimum and a feasible upper
    bracket, then return the (re)solved assignment at the top of the range.
    """
    t = q._load_cos("imatrix" if use_imatrix else "plain")
    if t is None:
        return None, "cosine table not found"
    if not (0.0 < target_cos < 1.0):
        return None, f"target cosine {target_cos} outside (0,1)"

    SG, QG, CG, SL, QL, CL = _matrices(t)
    frozen = t["c0"]
    lo = (float(CG.min(axis=1).sum()) + frozen + t["overhead"]) / 1e9

    res, err = solve(lo, use_imatrix)
    if res is None:
        return None, err
    if res["cos"] >= target_cos:
        res["method"] += " (min-size)"
        return res, None          # even the smallest build beats the target

    hi = 12.0
    while True:
        res, err = solve(hi, use_imatrix)
        if res is None:
            return None, err
        if res["cos"] >= target_cos:
            break
        hi *= 1.6
        if hi > 40.0:
            return None, (f"cannot reach cos {target_cos:.6f} "
                          f"(best below {hi/1.6:g} GB is "
                          f"{res['cos']:.6f})")
    while hi - lo > tol_gb:
        mid = (lo + hi) / 2
        res_mid, err = solve(mid, use_imatrix)
        if res_mid is None:
            return None, err
        if res_mid["cos"] >= target_cos:
            hi, res = mid, res_mid
        else:
            lo = mid
    res["method"] += " (min-size)"
    return res, None
