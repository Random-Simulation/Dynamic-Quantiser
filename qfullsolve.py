"""qfullsolve.py -- FULL per-tensor "minimise cosine" solver.

Unconstrained: every quantizable tensor of the model picks its own tier.
Method: Dinkelbach outer loop + separable (mu, nu) Lagrangian inner;
multi-start local search with single-tier and pair-swap moves; final
polish with from-scratch verification. Both the plain and imatrix cosine
tables are supported; nothing is written or printed by the solve itself.

Frozen tensors (the model's not-quantizable set) keep their SOURCE type
in the returned assignment, so the result is a complete whole-model
schema. Fully model-agnostic: the frozen set and source types come from
the table context.
"""
import numpy as np

import qmodel as q

# Hard-coded cap: at most this fraction of the quantized bytes may use an IQ
# ("non-fast") tier -- the slower-to-dequant grid-search types. A move that
# would push the IQ byte ratio above this is rejected, so the solver trades a
# little cosine to stay within it. On the fast K ladder (no IQ tiers) every
# IQ byte is 0 and the cap is a no-op, so fast-table solves are unchanged.
IQ_RATIO_CAP = 0.20


def _lagrange(B, S, Q, C, AR, s0, q0, A, K, x0):
    """Dinkelbach on (S+s0)/sqrt(Q+q0); inner separable Lagrangian."""

    def cosval(Su, Qu):
        return (Su + s0) / np.sqrt(A * (Qu + q0))

    def H(mu):
        W = S - mu * Q

        def pick(nu):
            M = W - nu * C
            ti = M.argmax(1)
            return ti, float(M.max(1).sum()), float(C[AR, ti].sum())

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

    n = len(x0)
    lam = cosval(float(S[AR, x0].sum()), float(Q[AR, x0].sum()))
    xbest = x0.copy()
    for _ in range(30):
        kappa = lam * np.sqrt(A)
        Qcur = float(Q[AR, xbest].sum())
        mu0 = kappa * kappa / (4.0 * max(Qcur, 1e-9))
        grid = mu0 * np.logspace(-2.5, 2.5, 200)
        gv = np.full(len(grid), -np.inf)
        gx = [None] * len(grid)
        for k, mu in enumerate(grid):
            r = H(float(mu))
            if r is None or r[2] > B:
                continue
            gv[k] = -mu * q0 - kappa * kappa / (4.0 * mu) + r[1]
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
        newlam = cosval(float(S[AR, xi].sum()), float(Q[AR, xi].sum()))
        if abs(newlam - lam) < 1e-11 * lam:
            lam = newlam
            break
        lam = newlam
    return xbest, cosval(float(S[AR, xbest].sum()),
                         float(Q[AR, xbest].sum()))


def _single_pass(x, Su, Qu, Ci, iqB, c, B, S, Q, C, I, AR, s0, q0, A, K):
    def cosval(Su, Qu):
        return (Su + s0) / np.sqrt(A * (Qu + q0))
    for _ in range(6000):
        Sall = Su + S - S[AR, x][:, None]
        Qall = Qu + Q - Q[AR, x][:, None]
        Call = Ci + C - C[AR, x][:, None]
        Iall = iqB + I - I[AR, x][:, None]
        cval = (Sall + s0) / np.sqrt(A * (Qall + q0))
        # size-feasible, keeps the IQ byte ratio within the cap, and improves
        mask = ((Call <= B) & (Iall <= IQ_RATIO_CAP * Call)
                & (cval > c + 1e-12))
        mask[AR, x] = False
        if not mask.any():
            break
        idx = int(np.argmax(np.where(mask, cval, -np.inf)))
        i, t = divmod(idx, K)
        Su += float(S[i, t] - S[i, x[i]])
        Qu += float(Q[i, t] - Q[i, x[i]])
        Ci += float(C[i, t] - C[i, x[i]])
        iqB += float(I[i, t] - I[i, x[i]])
        x[i] = t
        c = float(cval.flat[idx])
    return x, Su, Qu, Ci, iqB, c


def _swap_pass(x, Su, Qu, Ci, iqB, c, B, S, Q, C, I, AR, s0, q0, A, K, rng,
               rounds=40, per=400):
    m = len(x)

    def cosval(Su, Qu):
        return (Su + s0) / np.sqrt(A * (Qu + q0))
    for _ in range(rounds):
        moved = False
        for _s in range(per):
            i = int(rng.integers(m)); j = int(rng.integers(m))
            if i == j:
                continue
            xi, xj = x[i], x[j]
            if xi >= K - 1 or xj == 0:
                continue
            tis = np.arange(xi + 1, K); tjs = np.arange(0, xj)
            dC = ((C[i, tis] - C[i, xi])[:, None]
                  + (C[j, tjs] - C[j, xj])[None, :])
            dI = ((I[i, tis] - I[i, xi])[:, None]
                  + (I[j, tjs] - I[j, xj])[None, :])
            dS = ((S[i, tis] - S[i, xi])[:, None]
                  + (S[j, tjs] - S[j, xj])[None, :])
            dQ = ((Q[i, tis] - Q[i, xi])[:, None]
                  + (Q[j, tjs] - Q[j, xj])[None, :])
            nc = np.where((Ci + dC) <= B,
                          (Su + dS + s0) / np.sqrt(A * (Qu + dQ + q0)),
                          -np.inf)
            # gate candidates that would push the IQ byte ratio over the cap
            nc = np.where((iqB + dI) <= IQ_RATIO_CAP * (Ci + dC), nc, -np.inf)
            idx = int(nc.argmax()); pi, pj = divmod(idx, len(tjs))
            if nc.flat[idx] > c + 1e-12:
                ti, tj = int(tis[pi]), int(tjs[pj])
                Su += float(S[i, ti] - S[i, xi] + S[j, tj] - S[j, xj])
                Qu += float(Q[i, ti] - Q[i, xi] + Q[j, tj] - Q[j, xj])
                Ci += float(C[i, ti] - C[i, xi] + C[j, tj] - C[j, xj])
                iqB += float(I[i, ti] - I[i, xi] + I[j, tj] - I[j, xj])
                x[i], x[j] = ti, tj
                c = float(nc.flat[idx]); moved = True
        if not moved:
            break
    return x, Su, Qu, Ci, iqB, c


def _local(B, S, Q, C, I, AR, s0, q0, A, K, m, restarts=16, seed=0):
    rng = np.random.default_rng(seed)
    # uniform all-tier starts that are size- AND IQ-ratio-feasible (the all-K
    # starts are 0% IQ; all-IQ starts are 100% and would be dead, so skip)
    starts = [np.full(m, t, np.int64) for t in range(K)
              if (float(C[:, t].sum()) <= B
                  and float(I[:, t].sum()) <= IQ_RATIO_CAP * float(C[:, t].sum()))]
    for _ in range(restarts):
        x = rng.integers(0, K, m)
        guard = 0
        while float(C[AR, x].sum()) > B and guard < 200000:
            guard += 1
            i = int(rng.integers(m))
            if x[i] > 0:
                x[i] -= 1
        Ci = float(C[AR, x].sum())
        if Ci <= B and float(I[AR, x].sum()) <= IQ_RATIO_CAP * Ci:
            starts.append(x.copy())
    bestx, bestc = None, -1.0
    for x in starts:
        Su = float(S[AR, x].sum()); Qu = float(Q[AR, x].sum())
        Ci = float(C[AR, x].sum()); iqB = float(I[AR, x].sum())
        c = (Su + s0) / np.sqrt(A * (Qu + q0))
        x, Su, Qu, Ci, iqB, c = _single_pass(
            x, Su, Qu, Ci, iqB, c, B, S, Q, C, I, AR, s0, q0, A, K)
        x, Su, Qu, Ci, iqB, c = _swap_pass(
            x, Su, Qu, Ci, iqB, c, B, S, Q, C, I, AR, s0, q0, A, K, rng)
        x, Su, Qu, Ci, iqB, c = _single_pass(
            x, Su, Qu, Ci, iqB, c, B, S, Q, C, I, AR, s0, q0, A, K)
        Ci2 = float(C[AR, x].sum())
        if Ci2 <= B and float(I[AR, x].sum()) <= IQ_RATIO_CAP * Ci2:
            c2 = (float(S[AR, x].sum()) + s0) / np.sqrt(
                A * (float(Q[AR, x].sum()) + q0))
            if c2 > bestc:
                bestc, bestx = c2, x.copy()
    return bestx, bestc


def solve(target_gb, use_imatrix=False, restarts=16):
    """Full per-tensor solve. Returns (result, None) or (None, error).

    result: assignment {name: tier} for ALL tensors, cos, dev,
    size_gb (measured ref bytes), tier_counts, method.
    """
    t = q._load_cos("imatrix" if use_imatrix else "plain")
    if t is None:
        return None, "cosine table not found"
    ladder = t["ladder"]; K = len(ladder)
    upg = t["upg"]; m = int(upg.sum())
    S = t["S"][upg].astype(np.float64)
    Q = t["Q"][upg].astype(np.float64)
    C = t["C"][upg].astype(np.float64)
    AR = np.arange(m)
    # IQ byte-matrix: I[i, t] = C[i, t] if tier t is an IQ (non-fast) tier,
    # else 0. Drives the hard IQ_RATIO_CAP (max fraction of quantized bytes
    # that may be IQ). On the fast K ladder every entry is 0 -> cap is a no-op.
    is_iq = np.array([tier.startswith("IQ") for tier in ladder], bool)
    I = np.where(is_iq[None, :], C, 0.0)
    s0, q0, A = t["s0"], t["q0"], t["A_total"]
    B = target_gb * 1e9 - t["overhead"] - t["c0"]
    # Absolute floor = each tensor at its OWN smallest tier. This is NOT
    # column 0: a tier's column can be larger than another tier's for a
    # given tensor (shape fallback, or an uncovered-tier Q4_K column), so
    # col 0 alone is not the true per-tensor floor (e.g. Q2_K can be the
    # smallest column of a tensor whose col 0 is a Q4_K-sized fallback).
    min_bytes = float(C.min(axis=1).sum())
    minsize = (min_bytes + t["c0"] + t["overhead"]) / 1e9
    if B < min_bytes:
        return None, (f"target {target_gb:g} GB is below the minimum "
                      f"({minsize:.2f} GB, smallest tier per tensor)")

    x0 = np.full(m, ladder.index("Q4_K"), np.int64)
    xlag, clag = _lagrange(B, S, Q, C, AR, s0, q0, A, K, x0)
    # both the size budget and the IQ byte cap must hold
    lag_feas = (float(C[AR, xlag].sum()) <= B
                and float(I[AR, xlag].sum())
                <= IQ_RATIO_CAP * float(C[AR, xlag].sum()))
    if lag_feas:
        clag = (float(S[AR, xlag].sum()) + s0) / np.sqrt(
            A * (float(Q[AR, xlag].sum()) + q0))
    xloc, cloc = _local(B, S, Q, C, I, AR, s0, q0, A, K, m, restarts)
    loc_feas = (xloc is not None and float(C[AR, xloc].sum()) <= B
                and float(I[AR, xloc].sum())
                <= IQ_RATIO_CAP * float(C[AR, xloc].sum()))

    cands = ([(clag, xlag, "lagrange")] if lag_feas else []) + \
            ([(cloc, xloc, "local")] if loc_feas else [])
    if not cands:
        return None, "no feasible solution"
    c, x, tag = max(cands, key=lambda z: z[0])

    # polish the winner; accept only on strict improvement + feasibility
    rng = np.random.default_rng(1)
    Su = float(S[AR, x].sum()); Qu = float(Q[AR, x].sum())
    Ci = float(C[AR, x].sum()); iqB = float(I[AR, x].sum())
    c = (Su + s0) / np.sqrt(A * (Qu + q0))
    x, Su, Qu, Ci, iqB, c = _single_pass(
        x.copy(), Su, Qu, Ci, iqB, c, B, S, Q, C, I, AR, s0, q0, A, K)
    x, Su, Qu, Ci, iqB, c = _swap_pass(
        x, Su, Qu, Ci, iqB, c, B, S, Q, C, I, AR, s0, q0, A, K, rng,
        rounds=60, per=600)
    x, Su, Qu, Ci, iqB, c = _single_pass(
        x, Su, Qu, Ci, iqB, c, B, S, Q, C, I, AR, s0, q0, A, K)
    true_c = (float(S[AR, x].sum()) + s0) / np.sqrt(
        A * (float(Q[AR, x].sum()) + q0))
    Ci2 = float(C[AR, x].sum())
    if (Ci2 <= B and float(I[AR, x].sum()) <= IQ_RATIO_CAP * Ci2
            and true_c > c + 1e-12):
        c, x = true_c, x
        tag += "+polish"

    src_type = t.get("src_type")
    assignment = {}
    upg_idx = 0   # AR is identity over the compressed upg array
    for i, n in enumerate(t["names"]):
        if upg[i]:
            assignment[n] = ladder[int(x[upg_idx])]
            upg_idx += 1
        elif src_type is not None:
            assignment[n] = str(src_type[i])   # frozen: keep source type
        else:
            assignment[n] = "F32"

    from collections import Counter
    counts = Counter(assignment.values())
    gcounts = Counter((t["groups"][i], assignment[n])
                      for i, n in enumerate(t["names"]))
    group_counts = {}
    for (g, tier), ncnt in gcounts.items():
        group_counts.setdefault(g, {})[tier] = ncnt
    size = (float(C[AR, x].sum()) + t["c0"] + t["overhead"]) / 1e9
    Ci_tot = float(C[AR, x].sum())
    iq_frac = (float(I[AR, x].sum()) / Ci_tot) if Ci_tot > 0 else 0.0
    return ({
        "assignment": assignment,
        "cos": float(c), "dev": 1.0 - float(c), "size_gb": size,
        "tier_counts": dict(counts), "group_counts": group_counts,
        "iq_frac": iq_frac,
        "method": tag,
    }, None)
