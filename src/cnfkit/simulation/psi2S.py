from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

def psi2psiproj(
    psi: ArrayLike,
    ) -> NDArray[np.float64]:
    """
    project psi(phi,theta) into psi_proj(phi)
    """
    Nt = psi.shape[-1]
    dtheta = np.pi/Nt
    theta = np.arange(dtheta/2, np.pi, dtheta)

    psiproj = np.sum(psi*np.sin(theta), axis=-1) * dtheta
    return psiproj


from .psi_flow import get_psi_along_streamline
def SAXS_along_centerline(
    gradu_track_path: str | Path,
    model: str,
    model_params: dict[str, float],
    x_targets: ArrayLike,
    Np: int = 40,
    Nt: int = 40,
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
    # The scattering angle (chi) is discretized with the same resolution as phi
    dchi = np.pi/Np
    chi = np.arange(-np.pi/2+dchi/2, np.pi/2, dchi)

    Nz = max(
        int(p.stem.replace("gradu_track", ""))
        for p in Path(gradu_track_path).glob("gradu_track*.csv")
        ) + 1
    
    psiproj_list = []
    z_mat = []
    for i in range(0,Nz):
        psi_targets, Info = get_psi_along_streamline(
            gradu_track_path=gradu_track_path,
            gradu_track_idx=i,
            model = model,
            model_params = model_params,
            x_targets=x_targets,
            Np=Np,
            Nt=Nt,
            )
        psiproj_list.append(psi2psiproj(psi_targets))
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

    kernel = 1.5 * (np.cos(chi) ** 2) - 0.5
    ci_weight = np.abs(np.sin(chi))

    # true order
    norm_true = np.sum(psiproj_true * ci_weight[None, :], axis=1, keepdims=True) * dchi
    psiproj_true /= norm_true
    S_true = np.sum(psiproj_true * kernel[None, :] * ci_weight[None, :], axis=1) * dchi

    # postprocessed order
    psiproj_min = np.min(psiproj_true)
    psiproj_pp = psiproj_true - psiproj_min
    norm_pp = np.sum(psiproj_pp * ci_weight[None, :], axis=1, keepdims=True) * dchi
    psiproj_pp /= norm_pp
    S_pp = np.sum(psiproj_pp * kernel[None, :] * ci_weight[None, :], axis=1) * dchi

    return psiproj_true, S_true, psiproj_pp, S_pp, x_targets, chi
