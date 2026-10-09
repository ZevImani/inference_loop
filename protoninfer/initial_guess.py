"""Initial momentum guesses: convex-hull direction, |p| from deposited energy, or random."""
import numpy as np
from scipy.spatial import ConvexHull

# ── Convex-hull initial guess ──────────────────────────────────────────────────

_HULL_BACKGROUND_THR = 0.05
_HULL_IMAGE_CENTER   = np.array([32.0, 32.0])
_HULL_DEFAULT_SCALE  = 16.5   # MeV/pixel; override via momentum_scale arg

# Power-law calibration: |p| = _PMAG_PL_SCALE * pixel_sum ** _PMAG_PL_EXPONENT
# Fitted on edep_protons64_v2 validation set via log-log regression.
# Exponent ~0.52 reflects that pixel_sum ∝ track range ∝ |p|^~1.7 (Bethe-Bloch),
# so the inverse mapping has exponent ~1/1.7 ≈ 0.59, empirically ~0.52 here.
_PMAG_PL_SCALE    = 41.025
_PMAG_PL_EXPONENT = 0.5177


def _max_area_triangle(pts):
    """O(n³) largest inscribed triangle among convex-hull vertices (typically 5–30)."""
    n = len(pts)
    best, best_area = (0, 1, 2), -1.0
    for i in range(n):
        for j in range(i + 1, n):
            ab = pts[j] - pts[i]
            for k in range(j + 1, n):
                ac = pts[k] - pts[i]
                area = abs(ab[0] * ac[1] - ab[1] * ac[0])
                if area > best_area:
                    best_area = area
                    best = (i, j, k)
    return best, best_area * 0.5


def _hull_guess_pixels(img, momentum_scale):
    """
    Extract two-track (px, py) guesses from the convex hull of a detector image.

    Algorithm:
      1. Threshold lit pixels and compute convex hull.
      2. Find the max-area inscribed triangle → vertex + two track tips.
      3. Corner closest to image centre is the interaction vertex.
      4. Pixel directions map to momentum:  px = Δrow * scale,  py = Δcol * scale.

    Returns a dict with keys guess1, guess2 (MeV arrays), dir1, dir2, vertex,
    or None on failure (too few pixels, degenerate hull, etc.).
    """
    img_thr = np.array(img, dtype=np.float64)
    img_thr[img_thr < _HULL_BACKGROUND_THR] = 0.0

    ys, xs = np.where(img_thr > 0)
    if len(xs) < 4:
        return None

    pts = np.column_stack([xs.astype(float), ys.astype(float)])   # (col, row)

    try:
        hull = ConvexHull(pts)
    except Exception:
        return None

    hverts = pts[hull.vertices]
    if len(hverts) < 3:
        return None

    (i, j, k), _ = _max_area_triangle(hverts)
    triangle = hverts[[i, j, k]]

    dists  = np.linalg.norm(triangle - _HULL_IMAGE_CENTER, axis=1)
    v_idx  = int(np.argmin(dists))
    t1_idx = (v_idx + 1) % 3
    t2_idx = (v_idx + 2) % 3

    vertex = triangle[v_idx]
    dir1   = triangle[t1_idx] - vertex   # (dcol, drow)
    dir2   = triangle[t2_idx] - vertex

    # px ∝ Δrow, py ∝ Δcol  (empirically verified mapping from calibration)
    guess1 = np.array([dir1[1] * momentum_scale, dir1[0] * momentum_scale])
    guess2 = np.array([dir2[1] * momentum_scale, dir2[0] * momentum_scale])

    return dict(guess1=guess1, guess2=guess2, dir1=dir1, dir2=dir2, vertex=vertex)


def image_pmag_estimate(img):
    """
    Estimate |p| from total above-threshold pixel sum via a power-law calibration.

    Uses |p| = _PMAG_PL_SCALE * pixel_sum ** _PMAG_PL_EXPONENT, fitted by log-log
    regression on the edep_protons64_v2 validation set.  The exponent (~0.52) accounts
    for the nonlinear Bethe-Bloch range-momentum relationship; this reduces std(ratio)
    from ~0.09 (linear) to ~0.006 on the validation set.

    Parameters
    ----------
    img : (H, W) array-like

    Returns
    -------
    float — estimated |p| in MeV
    """
    if hasattr(img, 'cpu'):
        img = img.detach().cpu().numpy()
    img = np.asarray(img, dtype=np.float32)
    pixel_sum = float(img[img > _HULL_BACKGROUND_THR].sum())
    return _PMAG_PL_SCALE * pixel_sum ** _PMAG_PL_EXPONENT


def convex_hull_initial_guess(img, n_tracks=1, momentum_scale=None, pz_init=0.0,
                               use_pmag_estimate=False):
    """
    Estimate initial momentum guess(es) from a detector image using convex hull.

    For single-track, returns the direction of the dominant (longer) track tip.
    For two-track, calls the hull algorithm on the combined sum image and returns
    both track directions.

    When use_pmag_estimate=True, image_pmag_estimate() is used to set |p| from
    total energy deposition via a power-law calibration.  The hull-derived (px, py)
    direction is preserved; pz is set to sqrt(max(|p|^2 - px^2 - py^2, 0)), giving
    a non-zero starting pz when the energy estimate implies |p| > |pt|.
    pz_init is ignored when use_pmag_estimate=True.

    Parameters
    ----------
    img               : (H, W) array-like  — detector image (sum image for 2-track)
    n_tracks          : int  — 1 or 2
    momentum_scale    : float or None  — MeV/pixel; defaults to 15.0 MeV/pixel
    pz_init           : float  — initial pz in MeV when use_pmag_estimate=False (default 0)
    use_pmag_estimate : bool   — use power-law |p| estimator (default False)

    Returns
    -------
    List of n_tracks (px, py, pz) tuples, or None if hull construction fails
    (caller should fall back to random initialisation).
    """
    if momentum_scale is None:
        momentum_scale = _HULL_DEFAULT_SCALE

    if hasattr(img, 'cpu'):
        img = img.detach().cpu().numpy()
    img = np.asarray(img, dtype=np.float32)

    result = _hull_guess_pixels(img, momentum_scale)
    if result is None:
        return None

    if use_pmag_estimate:
        pmag_est = image_pmag_estimate(img)
        def _pz(px, py):
            return float(np.sqrt(max(pmag_est**2 - float(px)**2 - float(py)**2, 0.0)))
    else:
        def _pz(px, py):
            return float(pz_init)

    if n_tracks == 1:
        d1 = np.linalg.norm(result['dir1'])
        d2 = np.linalg.norm(result['dir2'])
        guess = result['guess1'] if d1 >= d2 else result['guess2']
        return [(float(guess[0]), float(guess[1]), _pz(*guess))]

    g1, g2 = result['guess1'], result['guess2']
    return [
        (float(g1[0]), float(g1[1]), _pz(*g1)),
        (float(g2[0]), float(g2[1]), _pz(*g2)),
    ]


def random_momentum_init(p_min=100, p_max=500, component_range=300, seed=None):
    """Sample a random (px, py, pz) with |p| in [p_min, p_max]."""
    rng = np.random.default_rng(seed)
    component_range = max(component_range, p_max / np.sqrt(3))   # keep p_max reachable
    while True:
        p = rng.uniform(-component_range, component_range, size=3)
        mag = np.linalg.norm(p)
        if p_min <= mag <= p_max:
            return tuple(p.tolist()), mag


def perturb_momentum(p, rng, pmag_scale=1.0, pmag_jitter=0.0, angle_jitter_deg=0.0):
    """
    Randomly perturb a starting (px, py, pz) guess.

    |p| is multiplied by pmag_scale * exp(N(0, pmag_jitter)) (log-normal, so it stays
    positive), and the direction is rotated by an angle ~ N(0, angle_jitter_deg) about
    a random axis perpendicular to it. That rotation is in 3D, so a pz=0 hull guess
    picks up a random-sign pz.
    """
    p = np.asarray(p, dtype=np.float64)
    mag = float(np.linalg.norm(p))
    if mag == 0.0:
        return tuple(p.tolist())
    u = p / mag

    if angle_jitter_deg > 0:
        w = rng.normal(size=3)
        w -= w.dot(u) * u
        w /= np.linalg.norm(w)
        theta = np.deg2rad(rng.normal(0.0, angle_jitter_deg))
        u = np.cos(theta) * u + np.sin(theta) * w

    mag *= pmag_scale
    if pmag_jitter > 0:
        mag *= float(np.exp(rng.normal(0.0, pmag_jitter)))
    return tuple((mag * u).tolist())

