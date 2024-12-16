import numpy as np

def forcetorque(ibeam, sbeam):
    """
    Calculate force, torque, and spin in a 3D orthogonal space.

    Parameters:
        ibeam: List containing incident beam coefficients [[a1, b1], [L, M], max(L)].
        sbeam: List containing scattered beam coefficients [[p1, q1], [L, M], max(L)].

    Returns:
        Tuple containing forces, torques, and spins in 3D.
    """

    # Initialize outputs
    fx, fy, fz = 0, 0, 0

    # Parse beam coefficients
    a, b = ibeam[0]  # Incident field coefficients
    L, M = ibeam[1]  # Mode indices
    p, q = sbeam[0]  # Scattered field coefficients
    
    nmax = ibeam[2]  # Maximum mode index

    # Convert b and q to complex
    b = 1j * np.array(b)
    q = 1j * np.array(q)

    # Extend coefficients with zeros to match size requirements
    addv = np.zeros(2 * nmax + 3, dtype=complex)

    at = np.concatenate([a, addv])
    bt = np.concatenate([b, addv])
    pt = np.concatenate([p, addv])
    qt = np.concatenate([q, addv])

    # Combined index for modes
    ci = L * (L + 1) // 2 + M
    #ci = L * (L +1 ) + M
    
    # Auxiliary indices
    np1 = 2 * L + 2
    cinp1 = ci + np1
    cinp1mp1 = ci + np1 + 1
    cinp1mm1 = ci + np1 - 1
    cimp1 = ci + 1

    # For m+1, check boundary conditions
    kimp = (M > L - 1)
    


    # Select coefficients
    anp1 = at[cinp1]
    bnp1 = bt[cinp1]
    pnp1 = pt[cinp1]
    qnp1 = qt[cinp1]

    anp1mp1 = at[cinp1mp1]
    bnp1mp1 = bt[cinp1mp1]
    pnp1mp1 = pt[cinp1mp1]
    qnp1mp1 = qt[cinp1mp1]

    anp1mm1 = at[cinp1mm1]
    bnp1mm1 = bt[cinp1mm1]
    pnp1mm1 = pt[cinp1mm1]
    qnp1mm1 = qt[cinp1mm1]

    amp1 = at[cimp1]
    bmp1 = bt[cimp1]
    pmp1 = pt[cimp1]
    qmp1 = qt[cimp1]

    # Handle boundary conditions
    amp1[kimp] = 0
    bmp1[kimp] = 0
    pmp1[kimp] = 0
    qmp1[kimp] = 0

    # Original coefficients
    a = a[ci]
    b = b[ci]
    p = p[ci]
    q = q[ci]

    # Calculate forces
    Az = M / L / (L + 1) * np.imag(-a * np.conj(b) + np.conj(q) * p)
    Bz = 1 / (L + 1) * np.sqrt(L * (L - M + 1) * (L + M + 1) * (L + 2) / (2 * L + 3) / (2 * L + 1)) * \
         np.imag(anp1 * np.conj(a) + bnp1 * np.conj(b) - pnp1 * np.conj(p) - qnp1 * np.conj(q))
    fz = 2 * np.sum(Az + Bz)

    Axy = 1j / L / (L + 1) * np.sqrt((L - M) * (L + M + 1)) * \
          (np.conj(pmp1) * q - np.conj(amp1) * b - np.conj(qmp1) * p + np.conj(bmp1) * a)
    Bxy = 1j / (L + 1) * np.sqrt(L * (L + 2)) / np.sqrt((2 * L + 1) * (2 * L + 3)) * \
          (np.sqrt((L + M + 1) * (L + M + 2)) * (p * np.conj(pnp1mp1) + q * np.conj(qnp1mp1) - a * np.conj(anp1mp1) - b * np.conj(bnp1mp1)) + \
           np.sqrt((L - M + 1) * (L - M + 2)) * (pnp1mm1 * np.conj(p) + qnp1mm1 * np.conj(q) - anp1mm1 * np.conj(a) - bnp1mm1 * np.conj(b)))

    fxy = np.sum(Axy + Bxy)
    fx = np.real(fxy)
    fy = np.imag(fxy)

    return fx, fy, fz
