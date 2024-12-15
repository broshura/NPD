import numpy as np
import math


def force(ibeam, sbeam):
    """
    Calculate force in a 3D orthogonal space.

    Parameters:
        ibeam: Incident beam coefficients in the format [[np.array(a1), np.array(b1)], [L, M], max(L)].
        sbeam: Scattered beam coefficients in the format [[np.array(p1), np.array(q1)], [L, M], max(L)].
    Returns:
        fx, fy, fz, tx, ty, tz, sx, sy, sz: Components of force, torque, and spin.
    """
    fx = fy = fz = 0

    # Extract coefficients and indices from ibeam and sbeam
    a, b = ibeam[0]
    p, q = sbeam[0]
    n, m = ibeam[1]
    nmax = ibeam[2]

    b = 1j * b
    q = 1j * q

    # Extend 1D arrays into a shape suitable for calculations
    addv = np.zeros((2 * nmax + 3,))

    at = np.hstack([a, addv])
    bt = np.hstack([b, addv])
    pt = np.hstack([p, addv])
    qt = np.hstack([q, addv])

    ci = n * (n+1) + m

    np1 = 2 * n + 2
    cinp1 = ci + np1
    cinp1mp1 = ci + np1 + 1
    cinp1mm1 = ci + np1 - 1
    cimp1 = ci + 1

    kimp = m > n - 1

    amp1 = at[cimp1]
    bmp1 = bt[cimp1]
    pmp1 = pt[cimp1]
    qmp1 = qt[cimp1]

    amp1[kimp] = 0
    bmp1[kimp] = 0
    pmp1[kimp] = 0
    qmp1[kimp] = 0

    a = a[ci]
    b = b[ci]
    p = p[ci]
    q = q[ci]

    Az = m / n / (n + 1) * np.imag(-(a) * np.conj(b) + np.conj(q) * (p))
    Bz = 1 / (n + 1) * np.emath.sqrt(n * (n - m + 1) * (n + m + 1) * (n + 2) / (2 * n + 3) / (2 * n + 1)) \
         * np.imag(at[cinp1] * np.conj(a) + bt[cinp1] * np.conj(b) - pt[cinp1] * np.conj(p) - qt[cinp1] * np.conj(q))

    fz = 2 * np.sum(Az + Bz)

    Axy = 1j / n / (n + 1) * np.emath.sqrt((n - m) * (n + m + 1)) \
          * (np.conj(pmp1) * q - np.conj(amp1) * b - np.conj(qmp1) * p + np.conj(bmp1) * a)

    Bxy = 1j / (n + 1) * np.emath.sqrt(n * (n + 2)) / np.emath.sqrt((2 * n + 1) * (2 * n + 3)) \
          * (np.emath.sqrt((n + m + 1) * (n + m + 2)) \
             * (p * np.conj(pt[cinp1mp1]) + q * np.conj(qt[cinp1mp1]) - a * np.conj(at[cinp1mp1]) - b * np.conj(
                bt[cinp1mp1])) \
             + np.emath.sqrt((n - m + 1) * (n - m + 2)) \
             * (pt[cinp1mm1] * np.conj(p) + qt[cinp1mm1] * np.conj(q) - at[cinp1mm1] * np.conj(a) - bt[
                cinp1mm1] * np.conj(b)))

    fxy = np.sum(Axy + Bxy)
    fx = np.real(fxy)
    fy = np.imag(fxy)

    return fx, fy, fz
