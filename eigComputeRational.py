import numpy as np
from numpy.linalg import norm
from birdseye import eye


@eye
def eigComputeRational(i, E, t, rho, tol=1e-10, maxIter=2000):
    """
    The status values returned are:
    0 - No problems
    1 - mu < 0 was encountered in some iteration
    2 - mu > delta(i+1) was encountered in some iteration
    3 - Maximum number of iterations exceeded
    """

    # Compute delta
    delta = (E - E[i]) / rho
    N = len(delta)

    # Initialize some looping variables
    converged = False
    iters = 0
    status = 0

    # The case for i = N is different, since many terms drop out of the
    # interpolation expression making its computation easier
    if i == N - 1:
        # Initialize the value of mu
        mu = 1

        while not converged and iters < maxIter:
            if np.isinf(mu) or np.isnan(mu):
                print("Oooops")
                return

            iters += 1

            # Compute some helper values
            psi, phi, Dpsi, _ = evaluateRationals(mu, delta, t, i, N)
            mu += ((1 + psi) / Dpsi) * psi

            # Guard the mu values
            if mu < 0:
                mu = tol
                status = 1

            w = 1 + phi + psi

            if np.abs(w) <= tol * N * (1 + np.abs(psi) + np.abs(phi)):
                converged = True

    else:
        # Initialize the value of mu
        inds = np.concatenate((np.arange(i), np.arange(i + 2, N)))
        trest = t[inds].dot(t[inds] / (delta[inds] - delta[i + 1]))

        b = delta[i + 1] + (t[i]**2 + t[i + 1]**2) / (1 + trest)
        c = (delta[i + 1] * t[i]**2) / (1 + trest)

        mu1 = b / 2 - np.sqrt(b**2 - 4 * c) / 2
        mu2 = b / 2 + np.sqrt(b**2 - 4 * c) / 2

        # Pick the smallest value for mu > 0
        if mu1 < tol and mu2 > tol:
            mu = mu2
        elif mu1 > tol and mu2 < tol:
            mu = mu1
        else:
            mu = min(np.abs(mu1), np.abs(mu2))

        # Now iterate by constructing interpolating rationals
        while not converged and iters < maxIter:
            if np.isinf(mu) or np.isnan(mu):
                print("Oooops")
                return

            iters += 1
            # Compute some helper values
            D = delta[i + 1] - mu
            psi, phi, Dpsi, Dphi = evaluateRationals(mu, delta, t, i, N)
            if np.isinf(Dpsi) or np.isinf(Dphi):
                print("Oooops")
                return

            c = 1 + phi - D * Dphi
            a = (D * (1 + phi) + psi**2 / Dpsi) / c + psi / Dpsi
            w = 1 + phi + psi
            b = (D * w * psi) / (Dpsi * c)

            # Update mu
            mu += 2 * b / (a + np.sqrt(a**2 - 4 * b))

            # Guard the mu values
            if mu < 0:
                mu = tol
                status = 1

            if mu > delta[i + 1]:
                mu = delta[i + 1] - tol
                status = 2

            if np.abs(w) <= tol * N * (1 + np.abs(psi) + np.abs(phi)):
                converged = True

        if iters == maxIter:
            status = 3

    return mu, status

@eye
def evaluateRationals(x, delta, u, i, N):
    iL = np.arange(i + 1)
    iU = np.arange(i + 1, N)

    psi = u[iL].dot(u[iL] / (delta[iL] - x))
    phi = u[iU].dot(u[iU] / (delta[iU] - x))
    Dpsi = u[iL].dot(u[iL] / (delta[iL] - x)**2)
    Dphi = u[iU].dot(u[iU] / (delta[iU] - x)**2)
    return psi, phi, Dpsi, Dphi
