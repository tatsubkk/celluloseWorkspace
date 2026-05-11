import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def plot_psi_2d(
    psi,
    ax=None,
    show_colorbar=True
    ):
    psi = np.asarray(psi, dtype=float)

    Np, Nt = psi.shape
    dphi, dtheta = np.pi/Np, np.pi/Nt

    phi = np.arange(-np.pi/2+dphi/2, np.pi/2, dphi)
    theta = np.arange(dtheta/2, np.pi, dtheta)

    PPmat, TTmat = np.meshgrid(phi, theta, indexing="xy")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.pcolormesh(
        PPmat,
        TTmat,
        psi.T,
        shading="auto",
    )

    ax.set_xlabel("phi")
    ax.set_ylabel("theta")
    ax.set_title("Psi (2D)")

    ax.set_aspect('equal', adjustable='box')

    if show_colorbar:
        fig.colorbar(im, ax=ax)

    return fig, ax, im


def plot_psi_on_sphere(
    psi,
    ax=None,
    show_colorbar=True
    ):

    def sphericalise(
        x, y, z, c,
        max_angle: float = np.pi/10,
        max_iter: int = 20,
        ):
        """
        Refine a point cloud on the unit sphere by subdividing long edges.
        """
        for _ in range(max_iter):
            points = np.column_stack([x, y, z])
            triangles = ConvexHull(points).simplices

            # unique edges from triangular faces
            edges = np.vstack([
                triangles[:, [0, 1]],
                triangles[:, [0, 2]],
                triangles[:, [1, 2]],
            ])
            edges = np.unique(np.sort(edges, axis=1), axis=0)

            # check angular distance on the unit sphere
            p0 = points[edges[:, 0]]
            p1 = points[edges[:, 1]]
            dot_val = np.sum(p0 * p1, axis=1)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            angles = np.arccos(dot_val)

            long_mask = angles > max_angle
            if not np.any(long_mask):
                break

            long_edges = edges[long_mask]

            # midpoints on the sphere
            mid_points = 0.5 * (points[long_edges[:, 0]] + points[long_edges[:, 1]])
            mid_points /= np.linalg.norm(mid_points, axis=1, keepdims=True)
            mid_c = 0.5 * (c[long_edges[:, 0]] + c[long_edges[:, 1]])

            # append all new points at once
            x = np.concatenate([x, mid_points[:, 0]])
            y = np.concatenate([y, mid_points[:, 1]])
            z = np.concatenate([z, mid_points[:, 2]])
            c = np.concatenate([c, mid_c])

        return x, y, z, c

    psi = np.asarray(psi, dtype=float)
    Psi = psi.ravel()

    Np, Nt = psi.shape
    dphi, dtheta = np.pi/Np, np.pi/Nt
    phi = np.arange(-np.pi/2+dphi/2, np.pi/2, dphi)
    theta = np.arange(dtheta/2, np.pi, dtheta)

    PPmat, TTmat = np.meshgrid(phi, theta, indexing="ij")
    PP = PPmat.ravel()
    TT = TTmat.ravel()
    
    # full sphere duplication
    az = np.concatenate([PP, np.pi + PP])
    el = np.concatenate([TT, np.pi - TT])
    c = np.concatenate([Psi, Psi])

    # coordinate transformation
    x = np.sin(el) * np.cos(az)
    y = np.sin(el) * np.sin(az)
    z = np.cos(el)

    x, y, z, c = sphericalise(x, y, z, c)

    points = np.column_stack([x, y, z])
    triangles = ConvexHull(points).simplices
    face_c = c[triangles].mean(axis=1)

    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    surf = ax.plot_trisurf(
        x, y, z,
        triangles=triangles,
        linewidth=0.0,
        antialiased=True,
        shade=False,
    )

    surf.set_array(face_c)
    surf.set_clim(np.min(c), np.max(c))

    # setting
    ax.plot([0,1.1],[0,0],[0,0],"r", linewidth=2)
    ax.plot([0,0],[0,1.1],[0,0],"g", linewidth=2)
    ax.plot([0,0],[0,0],[0,1.1],"b", linewidth=2)

    ax.text(1.2, 0.0, 0.0, "x", fontsize=14, color="r")
    ax.text(0.0, 1.2, 0.0, "y", fontsize=14, color="g")
    ax.text(0.0, 0.0, 1.2, "z", fontsize=14, color="b")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)

    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=30, azim=45)

    if show_colorbar:
        fig.colorbar(surf, ax=ax, shrink=0.7)

    return fig, ax, surf


def plot_psiproj(
    psiproj: ArrayLike,
    chi: ArrayLike,
    x_targets: ArrayLike,
    indices: list[int] | None = None,
    ncols: int = 4,
    ylim: tuple[float, float] | None = (0, 3),
):
    psiproj = np.asarray(psiproj, dtype=float)
    chi = np.asarray(chi, dtype=float)
    x_targets = np.asarray(x_targets, dtype=float)

    if indices is None:
        indices = list(range(len(x_targets)))

    Nx_plot = len(indices)
    nrows = int(np.ceil(Nx_plot / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
    )

    for plot_idx, data_idx in enumerate(indices):
        ax = axes[plot_idx // ncols, plot_idx % ncols]

        ax.plot(chi, psiproj[data_idx], color="blue", label="psiproj")

        ax.set_title(f"x = {x_targets[data_idx]:.4f}")
        ax.set_xlim(-np.pi / 2, np.pi / 2)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.grid(True)

        if plot_idx == 0:
            ax.legend()

    for j in range(Nx_plot, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    plt.tight_layout()

    return fig, axes