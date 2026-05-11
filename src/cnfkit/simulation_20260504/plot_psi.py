import numpy as np
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
    phi = np.arange(dphi/2, np.pi, dphi)
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
    def sphericalise(x, y, z, c):
        x2 = np.asarray(x).reshape(-1).tolist()
        y2 = np.asarray(y).reshape(-1).tolist()
        z2 = np.asarray(z).reshape(-1).tolist()
        c2 = np.asarray(c).reshape(-1).tolist()

        split_line = True
        while split_line:
            pts = np.column_stack([x2, y2, z2])
            tri = ConvexHull(pts).simplices

            edges = np.vstack([
                tri[:, [0, 1]],
                tri[:, [0, 2]],
                tri[:, [1, 2]],
            ])
            edges = np.unique(np.sort(edges, axis=1), axis=0)

            split_line = False

            for i, j in edges:
                ax0, ay0, az0, ac0 = x2[i], y2[i], z2[i], c2[i]
                bx0, by0, bz0, bc0 = x2[j], y2[j], z2[j], c2[j]

                dot_val = np.clip(ax0*bx0 + ay0*by0 + az0*bz0, -1.0, 1.0)

                if np.arccos(dot_val) > np.pi / 10:
                    split_line = True

                    px = 0.5*(ax0 + bx0)
                    py = 0.5*(ay0 + by0)
                    pz = 0.5*(az0 + bz0)

                    mag = np.sqrt(px**2 + py**2 + pz**2)
                    px, py, pz = px/mag, py/mag, pz/mag

                    pc = 0.5*(ac0 + bc0)

                    x2.append(px)
                    y2.append(py)
                    z2.append(pz)
                    c2.append(pc)

        return np.array(x2), np.array(y2), np.array(z2), np.array(c2)

    psi = np.asarray(psi, dtype=float)

    Np, Nt = psi.shape
    dphi, dtheta = np.pi/Np, np.pi/Nt
    phi = np.arange(dphi/2, np.pi, dphi)
    theta = np.arange(dtheta/2, np.pi, dtheta)

    PPmat, TTmat = np.meshgrid(phi, theta, indexing="ij")
    PPmat = PPmat.reshape(-1)
    TTmat = TTmat.reshape(-1)
    Psi = psi.reshape(-1)

    # === full sphere duplication (固定) ===
    az = np.concatenate([PPmat, np.pi + PPmat])
    el = np.concatenate([TTmat, np.pi - TTmat])
    c = np.concatenate([Psi, Psi])

    # === 座標変換 ===
    x = np.sin(el) * np.cos(az)
    y = np.cos(el)
    z = np.sin(el) * np.sin(az)

    x, y, z, c = sphericalise(x, y, z, c)

    pts = np.column_stack([x, y, z])
    tri = ConvexHull(pts).simplices
    face_c = c[tri].mean(axis=1)

    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    surf = ax.plot_trisurf(
        x, y, z,
        triangles=tri,
        linewidth=0.0,
        antialiased=True,
        shade=False,
    )

    surf.set_array(face_c)
    surf.set_clim(np.min(c), np.max(c))

    # 軸
    ax.plot([0,1.1],[0,0],[0,0],"r", linewidth=2)
    ax.plot([0,0],[0,1.1],[0,0],"g", linewidth=2)
    ax.plot([0,0],[0,0],[0,1.1],"b", linewidth=2)

    # 軸ラベル（ちょい浮かせる）
    ax.text(1.2, 0.0, 0.0, "x", fontsize=14, color="r")
    ax.text(0.0, 1.2, 0.0, "y", fontsize=14, color="g")
    ax.text(0.0, 0.0, 1.2, "z", fontsize=14, color="b")

    # 軸範囲固定（これやらないと歪む）
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)

    # 比率固定（超重要）
    ax.set_box_aspect((1,1,1))

    ax.view_init(elev=30, azim=45)

    if show_colorbar:
        fig.colorbar(surf, ax=ax, shrink=0.7)

    return fig, ax, surf