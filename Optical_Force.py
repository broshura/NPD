import numpy as np

def forcetorque(ibeam, sbeam, position=None, rotation=None, coherent=False, progress_callback=None):
    """
    FORCETORQUE calculate force in a 3D orthogonal space.
    If the beam shape coefficients are in the original coordinates,
    this outputs the force in 3D Cartesian coordinates.

    Units are beam power. Force results should be multiplied by n/c
    assuming the beam coefficients already have the correct units for power.

    Parameters:
    ibeam: Incident beam
    sbeam: Scattered beam
    position: Optional, 3xN array for multiple force calculations
    rotation: Optional, 3x3N array for multiple calculations
    coherent: Optional, if True, beams are added after translation
    progress_callback: Optional, function to call with progress

    Returns:
    fx, fy, fz: Force components in x, y, z directions
    """
    fx = 0
    fy = 0
    fz = 0

    a, b = ibeam[0]
    p, q = sbeam[0]
    n, m = ibeam[1]
    nmax = ibeam[2]
    b = 1j * b
    q = 1j * q

    addv = np.zeros((2 * nmax + 3, 1))
    at = np.vstack([a, np.tile(addv, (1, a.shape[1]))])
    bt = np.vstack([b, np.tile(addv, (1, b.shape[1]))])
    pt = np.vstack([p, np.tile(addv, (1, p.shape[1]))])
    qt = np.vstack([q, np.tile(addv, (1, q.shape[1]))])
    ci = n * (n + 1) + m

    np1 = 2 * n + 2
    cinp1 = ci + np1
    cinp1mp1 = ci + np1 + 1
    cinp1mm1 = ci + np1 - 1
    cimp1 = ci + 1

    kimp = (m > n - 1)
    anp1 = at[cinp1, :]
    bnp1 = bt[cinp1, :]
    pnp1 = pt[cinp1, :]
    qnp1 = qt[cinp1, :]
    anp1mp1 = at[cinp1mp1, :]
    bnp1mp1 = bt[cinp1mp1, :]
    pnp1mp1 = pt[cinp1mp1, :]
    qnp1mp1 = qt[cinp1mp1, :]
    anp1mm1 = at[cinp1mm1, :]
    bnp1mm1 = bt[cinp1mm1, :]
    pnp1mm1 = pt[cinp1mm1, :]
    qnp1mm1 = qt[cinp1mm1, :]
    amp1 = at[cimp1, :]
    bmp1 = bt[cimp1, :]
    pmp1 = pt[cimp1, :]
    qmp1 = qt[cimp1, :]
    amp1[kimp, :] = 0
    bmp1[kimp, :] = 0
    pmp1[kimp, :] = 0
    qmp1[kimp, :] = 0
    a = a[ci, :]
    b = b[ci, :]
    p = p[ci, :]
    q = q[ci, :]

    Az = m / n / (n + 1) * np.imag(-(a) * np.conj(b) + np.conj(q) * (p))
    Bz = 1 / (n + 1) * np.sqrt(n * (n - m + 1) * (n + m + 1) * (n + 2) / (2 * n + 3) / (2 * n + 1)) * np.imag(
        anp1 * np.conj(a) + bnp1 * np.conj(b) - (pnp1) * np.conj(p) - (qnp1) * np.conj(q))
    fz = 2 * np.sum(Az + Bz)

    Axy = 1j / n / (n + 1) * np.sqrt((n - m) * (n + m + 1)) * (
            np.conj(pmp1) * q - np.conj(amp1) * b - np.conj(qmp1) * p + np.conj(bmp1) * a)
    Bxy = 1j / (n + 1) * np.sqrt(n * (n + 2)) / np.sqrt((2 * n + 1) * (2 * n + 3)) * (
            np.sqrt((n + m + 1) * (n + m + 2)) * (p * np.conj(pnp1mp1) + q * np.conj(qnp1mp1) - a * np.conj(anp1mp1) - b * np.conj(bnp1mp1)) +
            np.sqrt((n - m + 1) * (n - m + 2)) * (pnp1mm1 * np.conj(p) + qnp1mm1 * np.conj(q) - anp1mm1 * np.conj(a) - bnp1mm1 * np.conj(b)))
    fxy = np.sum(Axy + Bxy)
    fx = np.real(fxy)
    fy = np.imag(fxy)

    return np.array([fx, fy, fz])

# Example usage:
# fx, fy, fz = forcetorque(ibeam, sbeam, position=position, rotation=rotation, coherent=True, progress_callback=progress_callback)