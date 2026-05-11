import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import null_space
from scipy.optimize import minimize_scalar
from scipy.sparse import diags, eye, kron, csr_matrix, SparseEfficiencyWarning


def _build_Dphi_Dtheta(
    Np: int,
    Nt: int
) -> tuple[csr_matrix, csr_matrix]:
    """
    Build sparse finite-difference matrices for angular derivatives.

    The angular distribution is stored as a flattened one-dimensional array
    with length Np * Nt. The original two-dimensional grid has Np points in
    the phi direction and Nt points in the theta direction.

    This function returns two sparse matrices, Dphi and Dtheta, both with
    shape (Np * Nt, Np * Nt). Multiplying these matrices by the flattened
    distribution approximates the derivatives with respect to phi and theta.

    Boundary connections are included in the matrices. The phi-direction
    boundary uses a folded periodic connection, where the theta index is
    reversed across the phi boundary. The theta-direction boundary is handled
    with a periodic-type finite-difference stencil.

    Parameters
    ----------
    Np : int
        Number of grid points in the phi direction.
    Nt : int
        Number of grid points in the theta direction.

    Returns
    -------
    Dphi : csr_matrix of shape (Np * Nt, Np * Nt)
        Sparse finite-difference matrix for the phi derivative.
    Dtheta : csr_matrix of shape (Np * Nt, Np * Nt)
        Sparse finite-difference matrix for the theta derivative.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        
        N = Np * Nt
        dphi, dtheta = np.pi/Np, np.pi/Nt

        # build Dphi
        Dphi = (
            -    diags(np.ones((Np - 2) * Nt), offsets= 2 * Nt, shape=(N, N), format="lil")
            +8 * diags(np.ones((Np - 1) * Nt), offsets=     Nt, shape=(N, N), format="lil")
            -8 * diags(np.ones((Np - 1) * Nt), offsets=-    Nt, shape=(N, N), format="lil")
            +    diags(np.ones((Np - 2) * Nt), offsets=-2 * Nt, shape=(N, N), format="lil")
        )

        submat = np.eye(Nt)[:, ::-1]

        Dphi[  0:  Nt, (Np-1)*Nt:            N ] =-8 * submat
        Dphi[  0:  Nt, (Np-2)*Nt:(Np - 1) * Nt ] =     submat
        Dphi[ Nt:2*Nt, (Np-1)*Nt:            N ] =     submat
        Dphi[ (Np-1)*Nt:            N,  0:  Nt ] = 8 * submat
        Dphi[ (Np-2)*Nt:(Np - 1) * Nt,  0:  Nt ] =-    submat
        Dphi[ (Np-1)*Nt:            N, Nt:2*Nt ] =-    submat
        
        Dphi = (Dphi / (12*dphi)).tocsr()

        # build Dtheta
        submat = (
            -    diags(np.ones(Nt - 2), offsets= 2, shape=(Nt, Nt), format="lil")
            +8 * diags(np.ones(Nt - 1), offsets= 1, shape=(Nt, Nt), format="lil")
            -8 * diags(np.ones(Nt - 1), offsets=-1, shape=(Nt, Nt), format="lil")
            +    diags(np.ones(Nt - 2), offsets=-2, shape=(Nt, Nt), format="lil")
        )

        submat[   0, Nt-1] = -8
        submat[   0, Nt-2] =  1
        submat[   1, Nt-1] =  1
        submat[Nt-1,    0] =  8
        submat[Nt-1,    1] = -1
        submat[Nt-2,    0] = -1

        submat = (submat / (12*dtheta)).tocsr()
        Dtheta = kron(eye(Np, format="csr"), submat, format="csr")

        return Dphi, Dtheta
    

def _psieqfun(
    x: float,
    psieq: NDArray[np.float64],
    dS: NDArray[np.float64],
    Nt: int,
    k: int,
) -> float:
    """
    Evaluate smoothness of a linear combination of two null-space vectors.
    The objective is the maximum jump at phi-block boundaries.
    """
    psi_tmp = psieq[:, k] + x * psieq[:, 1 - k]
    norm_psi = np.sum(psi_tmp * dS)
    psi_tmp = psi_tmp / norm_psi

    jump = np.abs(np.diff(psi_tmp))
    return np.max(jump[Nt - 1::Nt])


def get_psi_equilibrium(
    gradu: ArrayLike,
    r: float,
    Dr: float,
    Np: int=40,
    Nt: int=40,
) -> NDArray[np.float64]:
    
    Gamma = (r**2 - 1.0) / (r**2 + 1.0)

    gradu = np.asarray(gradu, dtype=float).reshape(3, 3).T

    # =========================
    # geometry and operators
    # =========================
    N = Nt * Np
    dphi, dtheta = np.pi/Np, np.pi/Nt
    phi = np.arange(dphi/2, np.pi, dphi)
    theta = np.arange(dtheta/2, np.pi, dtheta)

    PPmat, TTmat = np.meshgrid(phi, theta, indexing="ij")
    PP = PPmat.reshape(N)
    TT = TTmat.reshape(N)

    sinPHI = np.sin(PP)
    cosPHI = np.cos(PP)
    sinTHETA = np.sin(TT)
    cosTHETA = np.cos(TT)
    sinTHETAinv = 1.0/sinTHETA

    dS = sinTHETA * dphi * dtheta

    # p
    p = np.zeros((N, 3), dtype=float)
    p[:, 0] = sinTHETA * cosPHI         # x
    p[:, 1] = cosTHETA                  # y
    p[:, 2] = sinTHETA * sinPHI         # z

    # ePHI
    ePHI = np.zeros((N, 3), dtype=float)
    ePHI[:, 0] = -sinPHI                # x
    ePHI[:, 1] = 0.0                    # y
    ePHI[:, 2] = cosPHI                 # z

    # eTHETA
    eTHETA = np.zeros((N, 3), dtype=float)
    eTHETA[:, 0] = cosTHETA * cosPHI    # x
    eTHETA[:, 1] = -sinTHETA            # y
    eTHETA[:, 2] = cosTHETA * sinPHI    # z

    Dphi, Dtheta = _build_Dphi_Dtheta(Np, Nt)

    # =========================
    # prepare system matrix
    # =========================
    # Jeffery's equation
    WTET = 0.5 * np.hstack([gradu.T - gradu, gradu.T + gradu])
    WpEp = p @ WTET
    Ep = WpEp[:, 3:6]
    pEp = np.sum(p * Ep, axis=1)
    
    pdot = WpEp[:, 0:3] + Gamma * (Ep - p * pEp[:, None])
    
    # Smoluchowki equation
    phidotsinTHETA = np.sum(pdot * ePHI, axis=1)
    thetadot = np.sum(pdot * eTHETA, axis=1)

    Aphi = Dphi @ (
        diags(sinTHETAinv, format="csr") @ (Dr * Dphi)
        - diags(phidotsinTHETA, format="csr")
        )
    Atheta = Dtheta @ (
        diags(sinTHETA, format="csr") @ (Dr * Dtheta)
        - diags(thetadot * sinTHETA, format="csr")
        )

    res = diags(sinTHETAinv, format="csr") @ (Aphi + Atheta)

    # =========================
    # solve
    # =========================
    psieq = null_space(res.toarray())

    if psieq.shape[1] == 1:
        psi = np.abs(psieq[:, 0])

    elif psieq.shape[1] == 2:
        if np.max(np.abs(np.diff(psieq[:, 0]))) > np.max(np.abs(np.diff(psieq[:, 1]))):
            k = 0
        else:
            k = 1

        opt = minimize_scalar(
            lambda x: _psieqfun(x, psieq, dS, Nt, k),
            bracket=(-10.0, 10.0),
            method="brent",
            )

        psi = np.abs(psieq[:, k] + opt.x * psieq[:, 1 - k])

    else:
        raise RuntimeError(
            f"Unexpected null-space dimension: {psieq.shape[1]}"
            )

    norm_psi = np.sum(psi * dS)
    psi = psi / norm_psi

    psi = psi.reshape(Np,Nt)

    return psi