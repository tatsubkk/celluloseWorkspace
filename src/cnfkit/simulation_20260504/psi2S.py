from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator

def _psi_pt2ba(
    psi: ArrayLike,
    Na: int = 40,
    Nb: int = 40,
) -> NDArray[np.float64]:
    """
    Interpolate psi(phi, theta) onto a psi(beta, alpha) grid.
    """
    # ---------------------------------
    # original angular grids
    # ---------------------------------
    psi = np.asarray(psi, dtype=float)

    Np, Nt = psi.shape
    dphi, dtheta = np.pi/Np, np.pi/Nt
    phi = np.arange(dphi/2, np.pi, dphi)
    theta = np.arange(dtheta/2, np.pi, dtheta)

    # ---------------------------------
    # target angular grids
    # ---------------------------------
    dalpha, dbeta = np.pi/Na, np.pi/Nb
    alpha = np.arange(dalpha/2, np.pi, dalpha)
    beta = np.arange(dbeta/2, np.pi, dbeta)
    
    BBmat, AAmat = np.meshgrid(beta, alpha, indexing="ij")

    # ---------------------------------
    # convert alpha-beta grid to phi-theta grid
    # ---------------------------------
    p1 = np.sin(AAmat) * np.cos(BBmat)
    p2 = np.sin(AAmat) * np.sin(BBmat)
    p3 = np.cos(AAmat)

    theta_new = np.arccos(p2)

    eps = 1e-15
    arg = p1 / np.maximum(np.sin(theta_new), eps)
    arg = np.clip(arg, -1.0, 1.0)

    phi_new = np.arccos(arg)

    is_lower_half = p3 < 0
    phi_new[is_lower_half] = np.pi - np.abs(phi_new[is_lower_half])
    theta_new[is_lower_half] = np.pi - np.abs(theta_new[is_lower_half])

    phi_new = np.clip(phi_new, phi[0], phi[-1])
    theta_new = np.clip(theta_new, theta[0], theta[-1])

    # ---------------------------------
    # interpolator: psi(phi, theta)
    # ---------------------------------
    interp = RegularGridInterpolator(
        (phi, theta),
        psi,
        bounds_error=False,
        fill_value=np.nan,
    )

    points = np.column_stack([
        phi_new.reshape(-1),
        theta_new.reshape(-1),
    ])

    psi_ba = interp(points).reshape(Nb, Na)

    return psi_ba


def psi2psiproj(
    psi: ArrayLike,
    ) -> NDArray[np.float64]:
    """
    project psi(phi,theta) into psi_proj(phi)
    """
    Nt = psi.shape[-1]
    dtheta = np.pi/Nt
    theta = np.arange(dtheta/2, np.pi, dtheta)

    psi_proj = np.sum(psi*np.sin(theta), axis=-1) * dtheta
    return psi_proj


from .psi_flow import get_psi_along_streamline
def SAXS_along_centerline(
    gradu_track_path: str | Path,
    model: str,
    model_params: dict[str, float],
    x_targets: ArrayLike,
    Np: int = 40,
    Nt: int = 40,
    Na: int = 40,
    Nb: int = 40,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        ]:
    # --------------------------------------------
    # calculate psi on each streamline and project
    # --------------------------------------------
    # The scattering angle (ci) is discretized with the same resolution as phi
    dci = np.pi/Nb
    ci = np.arange(dci/2, np.pi, dci)

    Nz = max(
        int(p.stem.replace("gradu_track", ""))
        for p in Path(gradu_track_path).glob("gradu_track*.csv")
        ) + 1
    
    psiproj_list = []
    z_mat = []
    for i in range(0,Nz):
        print(f"track_{i}")
        psi_targets, Info = get_psi_along_streamline(
            gradu_track_path=gradu_track_path,
            gradu_track_idx=i,
            model = model,
            model_params = model_params,
            x_targets=x_targets,
            Np=Np,
            Nt=Nt,
            )
        psiproj_targets = []

        for psi in psi_targets:
            psi_ba = _psi_pt2ba(psi,Na,Nb)
            psiproj = psi2psiproj(psi_ba)
            psiproj_targets.append(psiproj)
        psiproj_list.append(psiproj_targets)
        z_mat.append(Info["z_targets"])
    
    psiproj_list = np.asarray(psiproj_list, dtype=float)
    z_mat = np.asarray(z_mat, dtype=float)

    # -------------------------------------
    # get weight matrix and average psiproj
    # -------------------------------------
    weights = np.empty_like(z_mat)
    weights[0, :] = z_mat[1, :] / 2
    weights[1:, :] = np.diff(z_mat, axis=0)
    weights[-1,:] *= 0.5

    psiproj_true = np.sum(psiproj_list * weights[:, :, None], axis=0)

    # ---------------------------------
    # Convert psiproj into S
    # ---------------------------------
    kernel = 1.5 * (np.cos(ci) ** 2) - 0.5
    ci_weight = np.abs(np.sin(ci))

    # true order
    norm_true = np.sum(psiproj_true * ci_weight[None, :], axis=1, keepdims=True) * dci
    psiproj_true /= norm_true
    S_true = np.sum(psiproj_true * kernel[None, :] * ci_weight[None, :], axis=1) * dci

    # postprocessed order
    psiproj_min = np.min(psiproj_true)
    psiproj_pp = psiproj_true - psiproj_min
    norm_pp = np.sum(psiproj_pp * ci_weight[None, :], axis=1, keepdims=True) * dci
    psiproj_pp /= norm_pp
    S_pp = np.sum(psiproj_pp * kernel[None, :] * ci_weight[None, :], axis=1) * dci

    return psiproj_true, S_true, psiproj_pp, S_pp, x_targets, ci
