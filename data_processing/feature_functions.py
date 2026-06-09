import cv2
import numpy as np


def compute_glcm_entropy(gray_crop):
    gray = (gray_crop / 8).astype(np.uint8)
    levels = 32
    glcm = np.zeros((levels, levels), dtype=np.float64)
    h, w = gray.shape
    for i in range(h):
        row = gray[i, :]
        np.add.at(glcm, (row[:-1], row[1:]), 1)
    total = glcm.sum()
    if total == 0:
        return 0.0
    glcm /= total
    return float(-np.sum(glcm * np.log(glcm + 1e-10)))


def compute_rts(gray_crop):
    h, w = gray_crop.shape
    cy, cx = h // 2, w // 2
    max_r = min(cx, cy, w - cx - 1, h - cy - 1)
    if max_r < 3:
        return 0.0
    n_angles = 24
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    profiles = np.full((n_angles, max_r), np.nan, dtype=np.float64)
    for i, angle in enumerate(angles):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rs = np.arange(max_r)
        xs = np.round(cx + rs * cos_a).astype(int)
        ys = np.round(cy + rs * sin_a).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        profiles[i, valid] = gray_crop[ys[valid], xs[valid]]
    mean_profile = np.nanmean(profiles, axis=0)
    corrs = []
    for i in range(n_angles):
        p = profiles[i, :]
        valid = ~np.isnan(p) & ~np.isnan(mean_profile)
        if valid.sum() < 3:
            continue
        c = np.corrcoef(p[valid], mean_profile[valid])[0, 1]
        if not np.isnan(c):
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0


def compute_gabor_edge_density(gray_crop):
    ks = 21
    sigma = 3.0
    thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    lamda = 5.0
    gamma = 0.5
    responses = []
    for th in thetas:
        kernel = cv2.getGaborKernel((ks, ks), sigma, th, lamda, gamma, psi=0)
        filtered = cv2.filter2D(gray_crop, cv2.CV_64F, kernel)
        responses.append(np.abs(filtered))
    return float(np.mean(responses))


def compute_log_zero_crossings(gray_crop):
    lap = cv2.Laplacian(gray_crop, cv2.CV_64F)
    nz = np.sum((lap[:-1, :-1] * lap[1:, 1:] < 0).astype(float))
    return float(np.log10(nz + 1))


def compute_lbp_ri(gray_crop):
    from skimage.feature import local_binary_pattern
    h, w = gray_crop.shape
    if h < 3 or w < 3:
        return 0.0
    lbp = local_binary_pattern(gray_crop, P=8, R=1, method='ror')
    return float(len(np.unique(lbp))) / 256.0


def compute_dct_high_energy(gray_crop):
    f = gray_crop.astype(float)
    h, w = f.shape
    hp, wp = (h + 1) // 2 * 2, (w + 1) // 2 * 2
    fpad = np.zeros((hp, wp), dtype=f.dtype)
    fpad[:h, :w] = f
    dct = cv2.dct(fpad / 255.0)
    tdct = np.sum(np.abs(dct)) + 1e-10
    n = max(1, min(hp, wp) // 8)
    return float(np.sum(np.abs(dct[-n:, -n:])) / tdct)


def compute_wavelet_approx(gray_crop):
    f = gray_crop.astype(float)
    h, w = f.shape
    hp, wp = (h + 1) // 2 * 2, (w + 1) // 2 * 2
    fpad = np.zeros((hp, wp), dtype=f.dtype)
    fpad[:h, :w] = f
    if min(hp, wp) < 4:
        return 0.0
    r, c = fpad.shape
    fp = fpad[:r - r % 2, :c - c % 2]
    L = (fp[:, ::2] + fp[:, 1::2]) / 2
    H = (fp[:, ::2] - fp[:, 1::2]) / 2
    LL = (L[::2, :] + L[1::2, :]) / 2
    LH = (L[::2, :] - L[1::2, :]) / 2
    HL = (H[::2, :] + H[1::2, :]) / 2
    HH = (H[::2, :] - H[1::2, :]) / 2
    te = np.sum(LL**2) + np.sum(LH**2) + np.sum(HL**2) + np.sum(HH**2) + 1e-10
    return float(np.sum(LL**2) / te)


def align_crop(gray_crop):
    """Rotate crop so the dominant object is horizontal (if angled)."""
    h, w = gray_crop.shape
    if h < 10 or w < 10:
        return gray_crop
    blurred = cv2.GaussianBlur(gray_crop, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray_crop
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    angle = rect[2]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 2.0:
        return gray_crop
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray_crop, M, (w, h), flags=cv2.INTER_CUBIC)
    return rotated


def robust_mask(gray_crop):
    """Get binary mask using Otsu, falling back to Canny-based if Otsu gives poor shape."""
    blurred = cv2.GaussianBlur(gray_crop, (5, 5), 0)
    otsu_val, bin_ot = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_ot = cv2.morphologyEx(bin_ot, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cs_ot, _ = cv2.findContours(bin_ot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs_ot:
        return bin_ot
    c = max(cs_ot, key=cv2.contourArea)
    ca = cv2.contourArea(c)
    per = cv2.arcLength(c, True)
    circ_ot = 4 * np.pi * ca / (per ** 2) if per > 0 else 0

    # Canny-based mask for low-contrast images
    edges = cv2.Canny(blurred, max(1, int(otsu_val * 0.3)), min(255, int(otsu_val * 1.5)))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs_ed, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs_ed:
        c2 = max(cs_ed, key=cv2.contourArea)
        ca2 = cv2.contourArea(c2)
        per2 = cv2.arcLength(c2, True)
        circ_ed = 4 * np.pi * ca2 / (per2 ** 2) if per2 > 0 else 0
        if circ_ed > circ_ot + 0.15 and circ_ed > 0.3:
            mask = np.zeros_like(gray_crop)
            cv2.drawContours(mask, [c2], -1, 255, -1)
            return mask
    return bin_ot


def compute_hole_count(gray_crop):
    binary = robust_mask(gray_crop)
    inv = cv2.bitwise_not(binary)
    h_contours, _ = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return float(len(h_contours) - 1) if len(h_contours) > 1 else 0.0
