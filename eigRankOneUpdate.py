import numpy as np
from numpy.linalg import norm
from eigComputeRational import eigComputeRational


def eigRankOneUpdate(V, E, t, rho, acc=1e-12, verbose=False):
    """
    EIGRANKONEUPDATE   Compute eigendecomposition of a rank-1 perturbation
    of a matrix with known eigendecomposition. Given a matrix with known
    eigendecomposition, A = V*E*V', EIGRANKONEUPDATE efficiently calculates
    the eigendecomposition of the perturbed matrix, Ap = (A + rho*u*u'),
    where rho is some scalar and u is a vector.

    NOTE: The algorithm takes t = V'*u as the input rather than u directly.
    i.e., we assume that Ap is expressed as Ap = V*(diag(E) + rho*t*t')*V'

    [W, F] = eigRankOneUpdate(V, E, t, rho) returns the eigendecomposition
    of Ap = W*F*W' using eigComputeRational to compute the eigenvalues from
    the secular equation. If a reduced eigen-decompositon is provided, a
    reduced update is calculated i.e., when V is N x r and E is r x 1, with
    r < N. This is a rank-preserving update.

    For details on computing eigenvectors, see
    "Rank-One Modification of the Symmetric Eigenproblem",
    J. R. Bunch, C. P. Nielsen & D. C. Sorensen,
    Numer. Math., 31, 31-48 (1978).

    For further details on stability, see
    "A Stable and Efficient Algorithm for the Rank-one Modification of the
    Symmetric Eigenvalue Problem", M. Gu & S. C. Eisenstat,
    Research Report YALEU/DCS/RR-916 (1992).

    For implementation details, see
    "Matrix Algorithms Volume II: Eigensystems",
    G. W. Stewart, Chapter 3.1, SIAM (2001).

    Input:
        E - array
        V - matrix
    """

    # Constant to test if fractions of t can be set to zero
    Gt = 10

    # Get the eigenvalues and clean them up
    inds = np.argsort(E)
    lambdas = E[inds]
    V = V[:, inds]
    N = len(lambdas)
    tNorm = norm(t)

    # Round to accuracy so that we can detect unique eigenvalues properly
    maxLambda = np.max(np.abs(lambdas))
    tol = N * acc * np.sqrt(maxLambda)
    lambdas[np.abs(lambdas) < tol] = 0

    # Now, get the repeated eigenvalues, and replace the corresponding
    # eigenvectors V(lambda) with V(lambda)*H, where H is a Householder matrix
    # designed to introduce sparsity into the linear combination space i.e.,
    # make t look like [a, 0, 0, b, 0, c, 0, 0, 0, ...., 0]'
    uniqueEigs, uIndex = np.unique(lambdas, return_inverse=True)
    d = len(uniqueEigs)

    for i in range(d):
        inds = np.where(uIndex == i)[0]
        multiplicity = len(inds)

        if multiplicity > 1:
            v = t[inds]
            v_norm = norm(v)

            if rho < 0:
                t[inds] = np.concatenate(
                    (np.array([v_norm]), np.zeros(multiplicity - 1)))
                v[0] += v_norm
            else:
                t[inds] = np.concatenate(
                    (np.zeros(multiplicity - 1), np.array([-v_norm])))
                v[-1] += v_norm

            v_norm = norm(v)
            if v_norm > Gt * (maxLambda + tNorm**2) * acc:
                V[:, inds] -= 2 * V[:, inds].dot(np.outer(v, v)) / v_norm**2

    # Remove very small values of t
    tI = np.where(np.abs(t) <= Gt * (maxLambda / tNorm + tNorm) * acc)[0]
    t[tI] = 0

    # Get the indices of the non-zero values of t
    tI = np.where(t)[0]
    Ebar = lambdas[tI]
    tBar = t[tI]

    # Compute the all the eigenvalues, and the eigenvectors corresponding to
    # the components t(i) != 0. The components for t(i) == 0 remain unchanged
    Nt = len(tI)
    mu = np.zeros(Nt)

    if rho > 0:
        for i in range(Nt):
            mu[i], status = eigComputeRational(i, Ebar, tBar, rho, acc)
            if verbose:
                print("status: {}".format(status))

    else:
        Ef = -Ebar[::-1]
        tf = tBar[::-1]

        for i in range(Nt):
            mu[i], status = eigComputeRational(Nt - (i + 1), Ef, tf, -rho, acc)
            if verbose:
                print("status: {}".format(status))

    # Ebar = Ebar[::-1]
    Fbar = Ebar + rho * mu
    # Clean up Fbar
    Fbar[np.abs(Fbar) < acc] = 0

    # Insert the new eigenvalues of the perturbed matrix
    F = lambdas
    F[tI] = Fbar

    # Correct the entries of t as follows to make the eigenvector computation
    # more stable (formula 3.3 in Gu and Eisentat).
    # Actually we use the algorithm on pag 180 of the book Eygensystems:
    #              ____________________________________________________________
    #             / i-1                   N-1
    #            /  ___  ( F(j) - E(i) )  ___   ( F(j) - E(i) )
    # t(i) =    /   | |  ---------------  | |  ---------------- ( F(N) - E(i) )
    #       \  /    | |  ( E(j) - E(i) )  | |  ( E(j+1) - E(i) )
    #        \/    j = 1                 j = i
    #

    tBarNum = Fbar.reshape(-1, 1) - Ebar
    # add identity to prevent 0 on diagonal
    # (prevents 0 when doing the product of rows/column)
    tBarDen = Ebar.reshape(-1, 1) - Ebar + np.eye(Nt)
    # the actual product under square root
    tBar = np.sign(tBar) * np.sqrt(np.sign(rho) * (tBarNum / tBarDen).prod(axis=0))

    # Compute the eigenvectors using the corrected entries of t
    W = np.copy(V)
    for i in range(Nt):
        W[:, tI[i]] = (V[:, tI].dot(tBar / (Fbar[i] - Ebar))).flatten()

    # normalize by columns
    W /= norm(W, axis=0)

    i, j = np.where(np.isnan(W))
    W[i, j] = V[i, j]

    # Sort the eigenvalues and eigenvectors to return
    inds = np.argsort(F)
    F = F[inds]
    W = W[:, inds]

    return F, W


if __name__ == "__main__":
    # a = np.arange(1, 4)
    # b = np.arange(2, 5)
    # A = np.outer(a, a)
    # E, V = np.linalg.eig(A)

    V = np.array([[0.89443, 0.35857, 0.26726],
                  [-0.44721, 0.71714, 0.53452],
                  [0.00000, -0.59761, 0.80178]])
    E = np.array([0, 0, 14])

    b = np.array([1, 0, 1])
    rho = 0.1

    # # standard calculation
    # B = rho * np.outer(b, b)
    # A = np.add(A, B)
    # e, v = np.linalg.eig(A)
    # print(e)
    # print(v)
    # print(v.dot(np.diag(e)).dot(v.T))

    print("\n")
    t = V.T.dot(b)
    e, v = eigRankOneUpdate(V, E, t, rho)
    print(e)
    print(v)
    print(v.dot(np.diag(e)).dot(v.T))
