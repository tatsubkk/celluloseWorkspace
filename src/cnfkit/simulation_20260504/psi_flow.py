import pathlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.sparse import SparseEfficiencyWarning, csr_matrix, diags, eye, kron
from tqdm import tqdm


def _build_Dphi_Dtheta(
    Np: int,
    Nt: int,
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


def get_psi_along_streamline(
    gradu_track_path,
    gradu_track_idx,
    model,
    model_params,
    x_targets,
    Np: int = 40,
    Nt: int = 40,
    set_initial_state=True,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """
    Solve the Smoluchowski equation along one streamline
    and return Psi at target x positions.

    psi shape = (Np, Nt)
    axis 0 : phi   (azimuthal angle, 0 → π)
    axis 1 : theta (polar angle, 0 → π)
    """

    if model == "C1":
        try:
            r = model_params["r"]
            Dr = model_params["Dr"]
        except KeyError as e:
            raise KeyError(f"C1 requires parameter '{e.args[0]}' in model_params") from None
        Gamma = (r**2 - 1.0) / (r**2 + 1.0)
        
        # =========================
        # read streamline data
        # =========================

        streamline_path = pathlib.Path(gradu_track_path)
        csv_path = streamline_path / f"gradu_track{gradu_track_idx}.csv"
    
        df = pd.read_csv(csv_path)

        t_data = df["t"].to_numpy(dtype=float)
        x_data = df["x"].to_numpy(dtype=float)
        y_data = df["y"].to_numpy(dtype=float)
        z_data = df["z"].to_numpy(dtype=float)
        gradu_data = df[
            [
                "Uxx", "Uxy", "Uxz",
                "Uyx", "Uyy", "Uyz",
                "Uzx", "Uzy", "Uzz",
            ]
        ].to_numpy(dtype=float)

        x_targets = np.asarray(x_targets, dtype=float)
        if np.any(np.diff(x_targets) <= 0):
            raise ValueError("x_targets must be strictly increasing.")

        # =========================
        # interpolation
        # =========================
        # t(x) interpolator
        x_to_t = interp1d(
            x_data,
            t_data,
            kind="linear",
            bounds_error=True
        )
        t_targets = np.asarray(x_to_t(x_targets), dtype=float)
        t_start = float(t_data[0])
        t_end = float(np.max(t_targets))

        # y(x) interpolator
        x_to_y = interp1d(
            x_data,
            y_data,
            kind="linear",
            bounds_error=True
        )
        y_targets = np.asarray(x_to_y(x_targets), dtype=float)

        # z(x) interpolator
        x_to_z = interp1d(
            x_data,
            z_data,
            kind="linear",
            bounds_error=True
        )
        z_targets = np.asarray(x_to_z(x_targets), dtype=float)

        # gradu(t) interpolator
        gradu_interp = interp1d(
            t_data,
            gradu_data,
            axis=0,
            kind="linear",
            bounds_error=True
        )

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

        Dphi, Dtheta = _build_Dphi_Dtheta(Np,Nt)

        # =========================
        # prepare system matrix
        # =========================
        pbar = tqdm(
            total=t_end - t_start,
            desc=f"gradu_track{gradu_track_idx}",
            unit="t",
            mininterval=0.5,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}, {postfix}",
            )
        t_seen = {"value": t_start}

        def rhs(t, Psi):
            pbar.update(t - t_seen["value"])
            t_seen["value"] = t
            pbar.set_postfix_str(f"t={t:.3g}")
        
            Psi = np.maximum(Psi, 0.0)

            # Jeffery's equation
            graduRaw = gradu_interp(t)
            gradu = graduRaw.reshape(3, 3).T

            WTET = 0.5 * np.hstack((gradu.T - gradu, gradu.T + gradu))
            WpEp = p @ WTET
            Ep = WpEp[:, 3:6]
            pEp = np.sum(p * Ep, axis=1)

            pdot = WpEp[:, 0:3] + Gamma * (Ep - p * pEp[:, None])

            # Smoluchowski equation
            phidotsinTHETA = np.sum(pdot * ePHI, axis=1)
            thetadot = np.sum(pdot * eTHETA, axis=1)
            
            AphiPsi = Dphi @ (
                 sinTHETAinv * (Dphi @ (Dr * Psi))
                 - phidotsinTHETA * Psi
                 )
            
            AthetaPsi = Dtheta @ (
                 sinTHETA * (Dtheta @ (Dr * Psi))
                 - thetadot * sinTHETA * Psi
                 )
            
            res = (AphiPsi + AthetaPsi) * sinTHETAinv
            return res

        # =========================
        # solve
        # =========================
        # initial state
        if set_initial_state == True:
            # run until equilibrium under gradu at the start point
            from cnfkit.simulation.psi_equilibrium import get_psi_equilibrium
            Psi0 = get_psi_equilibrium(
                gradu=gradu_interp(0),
                r=r,
                Dr=Dr,
                Np=Np,
                Nt=Nt,
                )
            Psi0 = Psi0.reshape(Np * Nt)
        else:
            # isotropic
            Psi0 = np.full(N, 1.0 / (2.0 * np.pi), dtype=float)

        sol = solve_ivp(
            rhs,
            t_span=(t_start, t_end),
            y0=Psi0,
            t_eval=t_targets,
            method="BDF",
            rtol=1e-3,
            atol=1e-4,
            vectorized=False,
            )
        if not sol.success:
            raise RuntimeError(f"ODE solve failed: {sol.message}")
        
        pbar.update(t_end - t_seen["value"])
        pbar.close()

        # =========================
        # return
        # =========================
        psi_targets = sol.y.T
        psi_targets = np.maximum(psi_targets, 0.0)
        psi_targets = psi_targets.reshape(len(x_targets), Np, Nt)
    
        psi_info = {
            "Np": Np,
            "Nt": Nt,
            "t_targets": t_targets,
            "x_targets": x_targets,
            "y_targets": y_targets,
            "z_targets": z_targets,
            }

        return psi_targets, psi_info
    
    elif model == "C2":
        try:
            r = model_params["r"]
            Dr_fast = model_params["Dr_fast"]
            Dr_slow = model_params["Dr_slow"]
            alpha_fast = model_params["alpha_fast"]
        except KeyError as e:
            raise KeyError(f"C2 requires parameter '{e.args[0]}' in model_params") from None
        
        psi_targets_fast, psi_info = get_psi_along_streamline(
            gradu_track_path = gradu_track_path,
            gradu_track_idx = gradu_track_idx,
            model = "C1",
            model_params = {"r":r, "Dr":Dr_fast},
            x_targets = x_targets,
            Np = Np,
            Nt = Nt,
            )
    
        psi_targets_slow, _ = get_psi_along_streamline(
            gradu_track_path = gradu_track_path,
            gradu_track_idx = gradu_track_idx,
            model = "C1",
            model_params = {"r":r, "Dr":Dr_slow},
            x_targets = x_targets,
            Np = Np,
            Nt = Nt,
            )
    
        psi_targets = (
            alpha_fast * psi_targets_fast
            + (1.0 - alpha_fast) * psi_targets_slow
        )

        return psi_targets, psi_info
    
    elif model == "C3":
        try:
            r = model_params["r"]
            Dr_fast = model_params["Dr_fast"]
            Dr_slow = model_params["Dr_slow"]
            alpha_fast = model_params["alpha_fast"]
            alpha_slow = model_params["alpha_slow"]
        except KeyError as e:
            raise KeyError(f"C3 requires parameter '{e.args[0]}' in model_params") from None
        
        psi_targets_fast, psi_info = get_psi_along_streamline(
            gradu_track_path = gradu_track_path,
            gradu_track_idx = gradu_track_idx,
            model = "C1",
            model_params = {"r":r, "Dr":Dr_fast},
            x_targets = x_targets,
            Np = Np,
            Nt = Nt,
            )
    
        psi_targets_slow, _ = get_psi_along_streamline(
            gradu_track_path = gradu_track_path,
            gradu_track_idx = gradu_track_idx,
            model = "C1",
            model_params = {"r":r, "Dr":Dr_slow},
            x_targets = x_targets,
            Np = Np,
            Nt = Nt,
            )
    
        psi_targets_iso = np.full_like(psi_targets_fast, 1.0 / (2.0 * np.pi))

        psi_targets = (
            alpha_fast * psi_targets_fast
            + alpha_slow * psi_targets_slow
            + (1-alpha_fast-alpha_slow) * psi_targets_iso
            )
        
        return psi_targets, psi_info
    
    elif model == "C3r":
        try:
            r_fast = model_params["r_fast"]
            r_slow = model_params["r_slow"]
            Dr_fast = model_params["Dr_fast"]
            Dr_slow = model_params["Dr_slow"]
            alpha_fast = model_params["alpha_fast"]
            alpha_slow = model_params["alpha_slow"]
        except KeyError as e:
            raise KeyError(f"C3r requires parameter '{e.args[0]}' in model_params") from None
        
        psi_targets_fast, psi_info = get_psi_along_streamline(
            gradu_track_path = gradu_track_path,
            gradu_track_idx = gradu_track_idx,
            model = "C1",
            model_params = {"r":r_fast, "Dr":Dr_fast},
            x_targets = x_targets,
            Np = Np,
            Nt = Nt,
            )
    
        psi_targets_slow, _ = get_psi_along_streamline(
            gradu_track_path = gradu_track_path,
            gradu_track_idx = gradu_track_idx,
            model = "C1",
            model_params = {"r":r_slow, "Dr":Dr_slow},
            x_targets = x_targets,
            Np = Np,
            Nt = Nt,
            )
    
        psi_targets_iso = np.full_like(psi_targets_fast, 1.0 / (2.0 * np.pi))

        psi_targets = (
            alpha_fast * psi_targets_fast
            + alpha_slow * psi_targets_slow
            + (1-alpha_fast-alpha_slow) * psi_targets_iso
            )
        
        return psi_targets, psi_info
    
    else:
        valid_models = ["C1", "C2", "C3", "C3r"]
        raise ValueError(
            f"Unknown model: {model!r}. "
            f"model must be one of {valid_models}."
        )