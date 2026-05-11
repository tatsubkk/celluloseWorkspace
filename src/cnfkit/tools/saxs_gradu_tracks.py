import shutil
from shutil import which
import json
from pathlib import Path
from tqdm import tqdm

import subprocess

import numpy as np
import pandas as pd

def run_command(cmd,
                cwd=None,
                log_file=None,
                show_foam_error=False,
                ):
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            text=True,
            capture_output=True,
        )

    except subprocess.CalledProcessError as e:
        if show_foam_error:
            print("\nCommand failed:")
            print(" ".join(cmd))

            print("\nWorking directory:")
            print(cwd)

            print("\nSTDOUT:")
            print(e.stdout)

            print("\nSTDERR:")
            print(e.stderr)

        if log_file:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(e.stdout or "")
                f.write(e.stderr or "")
        raise

    if log_file:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(result.stdout or "")
            f.write(result.stderr or "")

    return result


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


def make_saxs_gradu_tracks_dict(fields,points):
    fields_str = " ".join(fields)
    points_str = "\n".join(
        f"({p[0]:.3f}e-3 {p[1]:.3f}e-3 {p[2]:.3f}e-3)"
        for p in points
    )

    return f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      postProcessDict;
}}

functions
{{
    readU
    {{
        type readFields;
        libs ("libfieldFunctionObjects.so");
        fields ({fields_str});
    }}

    saxs_gradu_tracks
    {{
        type streamLine;
        libs ("libfieldFunctionObjects.so");

        U U;
        direction forward;
        cloud particleTracks;

        fields ({fields_str});
        setFormat       csv;
        lifeTime        10000;
        seedSampleSet
        {{
            type        cloud;
            axis        xyz;
            points
            (
{points_str}
            );
        }}

        trackLength     1e-3;
        interpolationScheme cellPoint;
    }}
}}
"""


def get_saxs_gradu_tracks(
        case_path,
        read_time = None,
        n_tracks = 10,
        show_foam_error=False,
        ):
    
    if which("postProcess") is None:
        raise RuntimeError(
            "OpenFOAM environment is not loaded.\n"
            "Run:\n"
            " openfoam\n"
            "before running this script."
        )
    
    print("[1/4] Prepare\n")

    case_path = Path(case_path)
    meta_path = case_path / "geoMetadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"geoMetadata.json not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    height = float(meta["height"])
    x_inlet = float(meta["core"]["x_inlet"])

    # =========================
    # Set the time to read
    # =========================

    if read_time is None:
        read_time = get_latest_time(case_path)
    else:
        read_time = str(read_time)

    time_dir = case_path / read_time
    if not time_dir.is_dir():
        raise FileNotFoundError(f"Specified time not found: {read_time}")
    
    # =========================
    # Delete unnecessary dirs
    # =========================

    for target in time_dir.iterdir():
        if target.is_file() and target.name.startswith("grad"):
            target.unlink()

    target = case_path / "saxs_gradu_tracks"
    if target.exists():
        shutil.rmtree(target)

    # =========================
    # Get velocity gradients
    # =========================
    
    print("[2/4] Get velocity gradients")
    print(" Getting grad(U)")
    run_command(
        ["postProcess", "-time", read_time, "-func", "grad(U)"],
        cwd=case_path,
        show_foam_error=show_foam_error,
    )

    print(" Renaming grad(U) as gradU")
    src = time_dir / "grad(U)"
    dst = time_dir / "gradU"
    if src.exists():
        src.rename(dst)
    else:
        raise FileNotFoundError("grad(U) not found")
    
    print(" Separating gradU into each component\n")
    run_command(
        ["postProcess", "-time", read_time, "-func", "components(gradU)"],
        cwd=case_path,
        show_foam_error=show_foam_error,
    )

    # =========================
    # Extract values along streamlines
    # =========================

    print("[3/4] Extract values along streamlines")
    print(" Writing temporary dictfile for Openfoam postprocess functions")
    fields = [
        "U",
        "gradUxx", "gradUxy", "gradUxz",
        "gradUyx", "gradUyy", "gradUyz",
        "gradUzx", "gradUzy", "gradUzz",
    ]
    points = [
        [x_inlet + 0.5, 0, (height / (2*n_tracks)) * i]
        for i in range(n_tracks)
        ]

    dict_str = make_saxs_gradu_tracks_dict(fields, points)
    
    dict_path = case_path / "system" / "saxs_gradu_tracks_dict"
    with open(dict_path, "w") as f:
        f.write(dict_str)

    print(" Excuting the postprocess functions\n")
    run_command(
        ["postProcess", "-time", read_time, "-dict", "system/saxs_gradu_tracks_dict"],
        cwd=case_path,
        show_foam_error=show_foam_error,
    )

    # =========================
    # Assemble Extracted streamlinedata into into final csv file
    # =========================

    print("[4/4] Assemble Extracted streamlinedata into into final csv file")
    print(" Preparing base dataframe")
    pp_dir = case_path / "postProcessing" / "sets" / "saxs_gradu_tracks" / str(read_time)
    if not pp_dir.is_dir():
        raise FileNotFoundError(f"streamline output directory not found: {pp_dir}")
    
    df = pd.read_csv(pp_dir / "U_track0.csv") # contains x,y,z,U_x,U_y,U_z

    grad_names = [
        "gradUxx", "gradUxy", "gradUxz",
        "gradUyx", "gradUyy", "gradUyz",
        "gradUzx", "gradUzy", "gradUzz",
        ]
    for name in grad_names:
        df_temp = pd.read_csv(pp_dir / f"{name}_track0.csv")
        df[name] = df_temp[name]
    
    df = df.rename(columns={
        "U_x": "Ux", "U_y": "Uy", "U_z": "Uz",
        "gradUxx": "Uxx", "gradUxy": "Uxy", "gradUxz": "Uxz",
        "gradUyx": "Uyx", "gradUyy": "Uyy", "gradUyz": "Uyz",
        "gradUzx": "Uzx", "gradUzy": "Uzy", "gradUzz": "Uzz",
        })
    
    df = df.astype(np.float64)

    # remove duplicates
    diff = df["x"].diff()
    mask_drop = diff.abs() < 1e-10
    mask_drop.iloc[0] = False
    df = df[~mask_drop].reset_index(drop=True)

    # split streamlines
    diff = df["x"].diff()
    mask_split = diff.abs() > 1.0e-3
    mask_split.iloc[0] = True

    starts_pos = np.flatnonzero(mask_split)
    ends_pos = np.r_[starts_pos[1:] - 1, len(df) - 1]

    output_dir = case_path / "saxs_gradu_tracks"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(" Creating csv on each streamline\n")
    for idx in tqdm(range(len(starts_pos)), desc="Writing streamlines"):
        df_temp = df.iloc[starts_pos[idx]:ends_pos[idx] + 1].copy()

        x = df_temp["x"].to_numpy()
        y = df_temp["y"].to_numpy()
        z = df_temp["z"].to_numpy()

        Ux = df_temp["Ux"].to_numpy()
        Uy = df_temp["Uy"].to_numpy()
        Uz = df_temp["Uz"].to_numpy()

        U = np.sqrt(Ux**2 + Uy**2 + Uz**2)
        ds = np.sqrt( (x[1:]-x[:-1])**2 +(y[1:]-y[:-1])**2 + (z[1:]-z[:-1])**2 )
        dt = ds * (1/U[1:] + 1/U[:-1]) / 2
        time = np.r_[0, np.cumsum(dt)]
        df_temp["t"] = time

        order = [
            "t", "x", "y", "z",
            "Ux",  "Uy",  "Uz",
            "Uxx", "Uxy", "Uxz",
            "Uyx", "Uyy", "Uyz",
            "Uzx", "Uzy", "Uzz",
            ]
        df_temp = df_temp[order]
        df_temp.to_csv(
            output_dir / f"gradu_track{idx}.csv",
            index=False,
            float_format="%.8e"
            )
    
    print(f"=== Successfully made CSV data!! ===\n")

    print("Cleaning postProcessing outputs\n")
    shutil.rmtree(case_path / "postProcessing")

    print("finish")
