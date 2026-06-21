#!/usr/bin/env python3
"""
Package render outputs into a workshop data directory (e.g. data/kelvin/).
Reads data/<dataset>/renders/; writes staged training files into --out-dir (same dataset folder).
Canonical volume grids: renders/<stem>/volume_stage/volume.raw → lattice_XX.npz (via neural_xray raw_to_npy.py).

Expects render_projections.sh layout:
  renders/<stem>/images/proj_XX.png  (proj index matches azimuth list for intermediates)
  renders/<stem>/transforms.json

Staged images use train_XX.png / eval_XX.png (required by multi-camera-dataparser
eval_mode filename+modulo). Intermediate timesteps: train 0°/90°, eval 225°.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _timestep_from_stem(stem: str) -> int:
    m = re.search(r"_t(\d+)$", stem)
    if not m:
        raise ValueError(f"Cannot parse timestep from stem: {stem}")
    return int(m.group(1))


def _sorted_stems(renders_dir: Path) -> list[str]:
    stems = []
    for p in renders_dir.iterdir():
        if p.is_dir() and (p / "transforms.json").exists():
            stems.append(p.name)
    return sorted(stems, key=_timestep_from_stem)


def _load_transforms(path: Path) -> dict:
    return json.loads(path.read_text())


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _proj_index_for_azimuth(azimuth: float, azimuth_list: list[float], tol: float = 1e-3) -> int:
    for i, a in enumerate(azimuth_list):
        if abs((a % 360) - (azimuth % 360)) < tol or abs((a % 360) - (azimuth % 360) - 360) < tol:
            return i
    raise ValueError(f"Azimuth {azimuth} not in render list {azimuth_list}")


def _frame_at_azimuth(frames: list[dict], azimuth: float, azimuth_list: list[float]) -> dict:
    idx = _proj_index_for_azimuth(azimuth, azimuth_list)
    # proj_00.png is first frame when angles are rendered in list order
    name = f"proj_{idx:02d}.png"
    for fr in frames:
        if Path(fr["file_path"]).name == name:
            return fr
    if idx < len(frames):
        return frames[idx]
    raise IndexError(f"No frame for azimuth {azimuth} (expected {name})")


def _rewrite_frame(fr: dict, images_subdir: str, time: float, file_name: str | None = None) -> dict:
    nf = dict(fr)
    nf["file_path"] = f"{images_subdir}/{file_name or Path(fr['file_path']).name}"
    nf["time"] = round(time, 4)
    return nf


def _proj_name_to_train(name: str) -> str:
    m = re.match(r"proj_(\d+)\.png$", name)
    if m:
        return f"train_{int(m.group(1)):02d}.png"
    return name


def _copy_and_rename_proj_to_train(src_dir: Path, dst_dir: Path) -> None:
    """multi-camera-dataparser filename+modulo expects train_XX.png / eval_XX.png."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for png in sorted(src_dir.glob("proj_*.png")):
        idx = int(png.stem.split("_")[1])
        shutil.copy2(png, dst_dir / f"train_{idx:02d}.png")
    train_00 = dst_dir / "train_00.png"
    if train_00.exists():
        shutil.copy2(train_00, dst_dir / "eval_00.png")


def _rewrite_canonical_frames(frames: list[dict], images_subdir: str, time: float) -> list[dict]:
    out = [
        _rewrite_frame(fr, images_subdir, time, file_name=_proj_name_to_train(Path(fr["file_path"]).name))
        for fr in frames
    ]
    if out:
        eval_fr = dict(out[0])
        eval_fr["file_path"] = f"{images_subdir}/eval_00.png"
        out.append(eval_fr)
    return out


def _stage_intermediate_images(
    src_images: Path,
    dst_dir: Path,
    train_azimuths: list[float],
    eval_azimuth: float,
    azimuth_list: list[float],
    frames: list[dict],
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for train_idx, az in enumerate(train_azimuths):
        fr = _frame_at_azimuth(frames, az, azimuth_list)
        src = src_images / Path(fr["file_path"]).name
        shutil.copy2(src, dst_dir / f"train_{train_idx:02d}.png")
    _copy_eval_at_azimuth(src_images, dst_dir, eval_azimuth, azimuth_list, frames)


def _convert_volume_raw_to_npz(
    raw_path: Path,
    npz_path: Path,
    resolution: int,
    workshop_root: Path,
) -> None:
    """Convert xray_projection_render volume.raw to compressed npz for fast VoxelGrid loading."""
    script = workshop_root / "neural_xray/nerf_data/scripts/raw_to_npy.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"{raw_path} missing — re-run render_projections.sh (voxel export) for this timestep"
        )
    res = (resolution, resolution, resolution)
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(raw_path),
            "--resolution",
            *map(str, res),
            "--output",
            str(npz_path),
        ],
        check=True,
    )


def _copy_eval_at_azimuth(
    src_images: Path,
    dst_dir: Path,
    azimuth: float,
    azimuth_list: list[float],
    frames: list[dict],
) -> None:
    fr = _frame_at_azimuth(frames, azimuth, azimuth_list)
    src = src_images / Path(fr["file_path"]).name
    shutil.copy2(src, dst_dir / "eval_00.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders-dir", type=Path, required=True)
    parser.add_argument(
        "--yaml-dir",
        type=Path,
        default=None,
        help="Optional; unused for training grids (volume comes from renders volume.raw)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--volume-res",
        type=int,
        default=1024,
        help="Voxel resolution of volume.raw (must match VOLUME_RES in render_projections.sh)",
    )
    parser.add_argument(
        "--intermediate-azimuths",
        default="0,90,45,135,180,225,270,315",
        help="Must match INTERMEDIATE_AZIMUTHAL_ANGLES used in render_files.sh",
    )
    parser.add_argument(
        "--dynamic-train-azimuths",
        default="0,90",
        help="Perpendicular views used in transforms_00_to_T (intermediate timesteps)",
    )
    parser.add_argument(
        "--dynamic-eval-azimuth",
        type=float,
        default=225.0,
        help="View copied to eval_00.png for intermediate timesteps",
    )
    args = parser.parse_args()
    workshop_root = Path(__file__).resolve().parents[1]

    azimuth_list = _parse_float_list(args.intermediate_azimuths)
    train_azimuths = _parse_float_list(args.dynamic_train_azimuths)

    stems = _sorted_stems(args.renders_dir)
    if len(stems) < 2:
        raise SystemExit(f"Need at least 2 rendered timesteps under {args.renders_dir}")

    t0_stem, t1_stem = stems[0], stems[-1]
    t0, t1 = _timestep_from_stem(t0_stem), _timestep_from_stem(t1_stem)
    if t1 == t0:
        raise SystemExit("First and last timestep are identical")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for stem, tag, t_val in ((t0_stem, "00", 0.0), (t1_stem, f"{t1:02d}", 1.0)):
        src_render = args.renders_dir / stem
        img_dir = args.out_dir / f"images_{tag}"
        _copy_and_rename_proj_to_train(src_render / "images", img_dir)

        tr = _load_transforms(src_render / "transforms.json")
        frames = _rewrite_canonical_frames(tr["frames"], f"images_{tag}", t_val)
        tr_out = {k: v for k, v in tr.items() if k != "frames"}
        tr_out["frames"] = frames
        (args.out_dir / f"transforms_{tag}.json").write_text(json.dumps(tr_out, indent=2))

        volume_raw = src_render / "volume_stage" / "volume.raw"
        lattice_npz = args.out_dir / f"lattice_{tag}.npz"
        if lattice_npz.exists():
            print(f"  skip {lattice_npz.name} (already exists)")
        else:
            print(f"  {volume_raw.name} -> {lattice_npz.name}")
            _convert_volume_raw_to_npz(volume_raw, lattice_npz, args.volume_res, workshop_root)

    combined = _load_transforms(args.renders_dir / t0_stem / "transforms.json")
    all_frames: list[dict] = []

    for stem in stems:
        step = _timestep_from_stem(stem)
        t_norm = (step - t0) / (t1 - t0)
        tag = f"{step:02d}"
        src_render = args.renders_dir / stem
        img_dir = args.out_dir / f"images_{tag}"
        tr = _load_transforms(src_render / "transforms.json")

        if stem in (t0_stem, t1_stem):
            _copy_and_rename_proj_to_train(src_render / "images", img_dir)
            all_frames.extend(_rewrite_canonical_frames(tr["frames"], f"images_{tag}", t_norm))
        else:
            _stage_intermediate_images(
                src_render / "images",
                img_dir,
                train_azimuths,
                args.dynamic_eval_azimuth,
                azimuth_list,
                tr["frames"],
            )
            for train_idx, az in enumerate(train_azimuths):
                fr = _frame_at_azimuth(tr["frames"], az, azimuth_list)
                all_frames.append(
                    _rewrite_frame(fr, f"images_{tag}", t_norm, file_name=f"train_{train_idx:02d}.png")
                )
            fr_eval = _frame_at_azimuth(tr["frames"], args.dynamic_eval_azimuth, azimuth_list)
            all_frames.append(
                _rewrite_frame(fr_eval, f"images_{tag}", t_norm, file_name="eval_00.png")
            )

    combined["frames"] = all_frames
    out_name = f"transforms_{t0:02d}_to_{t1:02d}.json"
    (args.out_dir / out_name).write_text(json.dumps(combined, indent=2))

    print(f"Staged dataset at {args.out_dir}")
    print(f"  canonical: all projections + lattice_*.npz on {t0_stem} and {t1_stem}")
    print(f"  dynamic intermediates: train azimuths {train_azimuths}, eval {args.dynamic_eval_azimuth}°")
    print(f"  {len(all_frames)} frames in {out_name}")


if __name__ == "__main__":
    main()
