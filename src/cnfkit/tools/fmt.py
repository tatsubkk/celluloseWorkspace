import shutil
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def get_latest_time(case_path):
    times = []

    for d in case_path.iterdir():
        if not d.is_dir():
            continue
        try:
            t = float(d.name)
            times.append((t, d.name))
        except ValueError:
            continue

    if not times:
        raise RuntimeError("No time directories found")

    return max(times)[1]


def foam2fmt(
        case_path,
        read_time = None,
        scalar_list = None,
        vector_list = None,
        ):
        
    scalar_list = ["alpha.water"] if scalar_list is None else scalar_list
    vector_list = ["U"] if vector_list is None else vector_list
    
    # =========================
    # read geoMetadata.json
    # =========================

    case_path = Path(case_path)
    meta_path = case_path / "geoMetadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"geoMetadata.json not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    reso = int(meta["reso"])
    height = float(meta["height"])
    width = float(meta["width"])
    x_inlet = float(meta["core"]["x_inlet"])
    x_outlet = float(meta["core"]["x_outlet"])

    cellN_core = round(reso*(x_outlet-x_inlet)) * round(reso*height) * round(reso*width)
    print(f"core cells = {cellN_core}")

    # =========================
    # set the time to read
    # =========================

    if read_time is None:
        read_time = get_latest_time(case_path)
    else:
        read_time = str(read_time)

    time_dir = case_path / read_time
    if not time_dir.is_dir():
        raise FileNotFoundError(f"Specified time not found: {read_time}")

    out_dir = case_path / "fmt"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    # =========================
    # start reading the case
    # =========================

    print(f'Start reading "{case_path.name}/{read_time}"')

    start = 23 # assumes current OpenFOAM field header length

    for scalar_field in scalar_list:
        field_path = time_dir / scalar_field
        with open(field_path, "r", encoding="utf-8") as f:
            vals = [float(line.strip()) for line in f.readlines()[start:start + cellN_core]]

        name = "alpha" if scalar_field == "alpha.water" else scalar_field.replace(".", "_")
        np.savetxt(out_dir / f"{name}.csv", np.array(vals), delimiter=",")
        print(f" Saved {out_dir / f'{name}.csv'}")

    for vector_field in vector_list:
        field_path = time_dir / vector_field
        with open(field_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[start:start + cellN_core]

        vecs = np.array(
            [line.strip().lstrip("(").rstrip(")").split() for line in lines],
            dtype=float
        )

        base = vector_field.replace(".", "_")
        np.savetxt(out_dir / f"{base}_x.csv", vecs[:, 0], delimiter=",")
        np.savetxt(out_dir / f"{base}_y.csv", vecs[:, 1], delimiter=",")
        np.savetxt(out_dir / f"{base}_z.csv", vecs[:, 2], delimiter=",")
        print(f" Saved {out_dir / f'{base}_x.csv'}")
        print(f" Saved {out_dir / f'{base}_y.csv'}")
        print(f" Saved {out_dir / f'{base}_z.csv'}")


def read_fmt(case_path, fields=None):

    case_path = Path(case_path)
    fmt_dir = case_path / "fmt"
    meta_path = case_path / "geoMetadata.json"

    # -------------------------
    # check required files
    # -------------------------
    if not fmt_dir.is_dir():
        raise FileNotFoundError(f"fmt directory not found: {fmt_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"geoMetadata.json not found: {meta_path}")

    # -------------------------
    # load metadata
    # -------------------------
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    reso = int(meta["reso"])
    height = float(meta["height"])
    width = float(meta["width"])
    x_inlet = float(meta["core"]["x_inlet"])
    x_outlet = float(meta["core"]["x_outlet"])

    # -------------------------
    # reconstruct grid size
    # -------------------------
    nx = int(round(reso * (x_outlet - x_inlet)))
    ny = int(round(reso * height))
    nz = int(round(reso * width))
    shape = (nx, ny, nz)

    data = {}

    # -------------------------
    # determine which fields to load
    # -------------------------
    if fields is None:
        # load all CSV files
        csv_files = sorted(fmt_dir.glob("*.csv"))
    else:
        # allow single string input
        fields = [fields] if isinstance(fields, str) else fields
        csv_files = [fmt_dir / f"{name}.csv" for name in fields]

    # -------------------------
    # read and reshape data
    # -------------------------
    for csv_path in csv_files:
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path.name} not found in fmt directory")

        name = csv_path.stem

        arr = np.loadtxt(csv_path, delimiter=",")
        if arr.size != nx * ny * nz:
            raise ValueError(
                f"Size mismatch in {csv_path.name}: "
                f"expected {nx * ny * nz}, got {arr.size}"
            )

        # reshape into 3D array (nx, ny, nz)
        data[name] = arr.reshape(shape)

    return {
        "fields": data,
        "meta": meta
    }


def plot_slice(
    fmt,
    field_name,
    axis="x",
    index=None,
    coord=None,
    ax=None,
    cmap="coolwarm",
    colorbar=True,
):
    fields = fmt["fields"]
    meta = fmt["meta"]

    if field_name not in fields:
        raise KeyError(f'Field "{field_name}" not found')

    arr = fields[field_name]

    reso = int(meta["reso"])
    height = float(meta["height"])
    width = float(meta["width"])
    x_inlet = float(meta["core"]["x_inlet"])
    x_outlet = float(meta["core"]["x_outlet"])

    nx = int(round(reso * (x_outlet - x_inlet)))
    ny = int(round(reso * height))
    nz = int(round(reso * width))

    dx = (x_outlet - x_inlet) / nx
    dy = height / ny
    dz = width / nz
    
    x = x_inlet + (np.arange(nx) + 0.5) * dx
    y = -height/2 + (np.arange(ny) + 0.5) * dy
    z = -width/2 + (np.arange(nz) + 0.5) * dz

    if (index is None) == (coord is None):
        raise ValueError('Specify exactly one of "index" or "coord"')

    if axis == "x":
        coords = x
        n = nx
    elif axis == "y":
        coords = y
        n = ny
    elif axis == "z":
        coords = z
        n = nz
    else:
        raise ValueError('axis must be "x", "y", or "z"')
    
    def _fmt_coord(v):
        return f"{v:.3f}".rstrip("0").rstrip(".")

    if coord is not None:
        show_coord = coord
        index = int(np.argmin(np.abs(coords - coord)))
    else:
        show_coord = coords[index]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if axis == "x":
        # yz plane
        img = arr[index, :, :]
        im = ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            extent=[-width/2, width/2, -height/2, height/2],
            cmap=cmap,
        )
        ax.set_xlabel("z")
        ax.set_ylabel("y")
        title_coord = x[index]
        ax.set_title(f"{field_name} at x = {_fmt_coord(show_coord)}")

    elif axis == "y":
        # zx plane
        img = arr[:, index, :]
        im = ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            extent=[x_inlet, x_outlet, -width/2, width/2],
            cmap=cmap,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        title_coord = y[index]
        ax.set_title(f"{field_name} at y = {_fmt_coord(show_coord)}")

    elif axis == "z":
        # yx plane
        img = arr[:, :, index]
        im = ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            extent=[x_inlet, x_outlet, -height/2, height/2],
            cmap=cmap,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        title_coord = z[index]
        ax.set_title(f"{field_name} at z = {_fmt_coord(show_coord)}")

    if colorbar:
        fig.colorbar(im, ax=ax)

    return ax
