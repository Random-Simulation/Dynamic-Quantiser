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


def _single_pass(x, Su, Qu, Ci, c, B, S, Q, C, AR, s0, q0, A, K):
    def cosval(Su, Qu):
        return (Su + s0) / np.sqrt(A * (Qu + q0))
    for _ in range(6000):
        Sall = Su + S - S[AR, x][:, None]
        Qall = Qu + Q - Q[AR, x][:, None]
        Call = Ci + C - C[AR, x][:, None]
        cval = (Sall + s0) / np.sqrt(A * (Qall + q0))
        mask = (Call <= B) & (cval > c + 1e-12)
        mask[AR, x] = False
        if not mask.any():
            break
        idx = int(np.argmax(np.where(mask, cval, -np.inf)))
        i, t = divmod(idx, K)
        Su += float(S[i, t] - S[i, x[i]])
        Qu += float(Q[i, t] - Q[i, x[i]])
        Ci += float(C[i, t] - C[i, x[i]])
        x[i] = t
        c = float(cval.flat[idx])
    return x, Su, Qu, Ci, c


def _swap_pass(x, Su, Qu, Ci, c, B, S, Q, C, AR, s0, q0, A, K, rng,
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
            dS = ((S[i, tis] - S[i, xi])[:, None]
                  + (S[j, tjs] - S[j, xj])[None, :])
            dQ = ((Q[i, tis] - Q[i, xi])[:, None]
                  + (Q[j, tjs] - Q[j, xj])[None, :])
            nc = np.where((Ci + dC) <= B,
                          (Su + dS + s0) / np.sqrt(A * (Qu + dQ + q0)),
                          -np.inf)
            idx = int(nc.argmax()); pi, pj = divmod(idx, len(tjs))
            if nc.flat[idx] > c + 1e-12:
                ti, tj = int(tis[pi]), int(tjs[pj])
                Su += float(S[i, ti] - S[i, xi] + S[j, tj] - S[j, xj])
                Qu += float(Q[i, ti] - Q[i, xi] + Q[j, tj] - Q[j, xj])
                Ci += float(C[i, ti] - C[i, xi] + C[j, tj] - C[j, xj])
                x[i], x[j] = ti, tj
                c = float(nc.flat[idx]); moved = True
        if not moved:
            break
    return x, Su, Qu, Ci, c


def _local(B, S, Q, C, AR, s0, q0, A, K, m, restarts=16, seed=0):
    rng = np.random.default_rng(seed)
    starts = [np.full(m, t, np.int64) for t in range(K)
              if float(C[:, t].sum()) <= B]
    for _ in range(restarts):
        x = rng.integers(0, K, m)
        guard = 0
        while float(C[AR, x].sum()) > B and guard < 200000:
            guard += 1
            i = int(rng.integers(m))
            if x[i] > 0:
                x[i] -= 1
        if float(C[AR, x].sum()) <= B:
            starts.append(x.copy())
    bestx, bestc = None, -1.0
    for x in starts:
        Su = float(S[AR, x].sum()); Qu = float(Q[AR, x].sum())
        Ci = float(C[AR, x].sum())
        c = (Su + s0) / np.sqrt(A * (Qu + q0))
        x, Su, Qu, Ci, c = _single_pass(x, Su, Qu, Ci, c, B, S, Q, C, AR,
                                        s0, q0, A, K)
        x, Su, Qu, Ci, c = _swap_pass(x, Su, Qu, Ci, c, B, S, Q, C, AR,
                                      s0, q0, A, K, rng)
        x, Su, Qu, Ci, c = _single_pass(x, Su, Qu, Ci, c, B, S, Q, C, AR,
                                        s0, q0, A, K)
        if float(C[AR, x].sum()) <= B:
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
    s0, q0, A = t["s0"], t["q0"], t["A_total"]
    B = target_gb * 1e9 - t["overhead"] - t["c0"]
    minsize = (float(C[:, 0].sum()) + t["c0"] + t["overhead"]) / 1e9
    if B < float(C[:, 0].sum()):
        return None, (f"target {target_gb:g} GB is below the minimum "
                      f"({minsize:.2f} GB, all tensors IQ1_S)")

    x0 = np.full(m, ladder.index("Q4_K"), np.int64)
    xlag, clag = _lagrange(B, S, Q, C, AR, s0, q0, A, K, x0)
    lag_feas = float(C[AR, xlag].sum()) <= B
    if lag_feas:
        clag = (float(S[AR, xlag].sum()) + s0) / np.sqrt(
            A * (float(Q[AR, xlag].sum()) + q0))
    xloc, cloc = _local(B, S, Q, C, AR, s0, q0, A, K, m, restarts)
    loc_feas = xloc is not None and float(C[AR, xloc].sum()) <= B

    cands = ([(clag, xlag, "lagrange")] if lag_feas else []) + \
            ([(cloc, xloc, "local")] if loc_feas else [])
    if not cands:
        return None, "no feasible solution"
    c, x, tag = max(cands, key=lambda z: z[0])

    # polish the winner; accept only on strict improvement + feasibility
    rng = np.random.default_rng(1)
    Su = float(S[AR, x].sum()); Qu = float(Q[AR, x].sum())
    Ci = float(C[AR, x].sum())
    c = (Su + s0) / np.sqrt(A * (Qu + q0))
    x, Su, Qu, Ci, c = _single_pass(x.copy(), Su, Qu, Ci, c, B, S, Q, C,
                                    AR, s0, q0, A, K)
    x, Su, Qu, Ci, c = _swap_pass(x, Su, Qu, Ci, c, B, S, Q, C, AR, s0, q0,
                                  A, K, rng, rounds=60, per=600)
    x, Su, Qu, Ci, c = _single_pass(x, Su, Qu, Ci, c, B, S, Q, C, AR,
                                    s0, q0, A, K)
    true_c = (float(S[AR, x].sum()) + s0) / np.sqrt(
        A * (float(Q[AR, x].sum()) + q0))
    if float(C[AR, x].sum()) <= B and true_c > c + 1e-12:
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
    return ({
        "assignment": assignment,
        "cos": float(c), "dev": 1.0 - float(c), "size_gb": size,
        "tier_counts": dict(counts), "group_counts": group_counts,
        "method": tag,
    }, None)
