#!/usr/bin/env python3
"""
Evaluation: cross-section comparison (vel_6 / vel_12 / spatiotemporal_mix / GT) + mixing α(t).

Uses pipeline.eval_along_plane() from the nerfstudio-xray exporter infrastructure.

Outputs:
  eval_xsections.png   — 4 rows × N timesteps of x-z density cross-sections
  eval_mixing.png      — spatiotemporal mixing coefficient α(t)

Usage (from repo root):
    python scripts/eval_kelvin.py
    python scripts/eval_kelvin.py --timesteps 0 0.2 0.4 0.6 0.8 1.0 --resolution 128
"""
import sys
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def setup_paths(nx_root: Path):
    sys.path.insert(0, str(nx_root / 'nerfstudio'))
    sys.path.insert(0, str(nx_root / 'nerfstudio-xray' / 'nerf-xray'))


def load_pipeline(config_yml: Path, device: str):
    from nerfstudio.utils.eval_utils import eval_setup
    _, pipeline, _, _ = eval_setup(config_yml, eval_num_rays_per_chunk=256,
                                   test_mode='inference')
    pipeline = pipeline.to(device)
    pipeline.eval()
    return pipeline


def xz_slice(pipeline, time: float, resolution: int, rhomax: float) -> np.ndarray:
    """x-z cross-section at centre y (distance=0) via pipeline.eval_along_plane."""
    arr = pipeline.eval_along_plane(
        target='field',
        plane='xz',
        distance=0.0,
        engine='numpy',
        resolution=resolution,
        rhomax=rhomax,
        time=time,
    )
    return arr  # (resolution, resolution) float in [0, rhomax_normalised]


def gt_xz_slice(npz: Path, out_res: int) -> np.ndarray:
    """Load GT volume npz, extract x-z slice at centre y, downsample to out_res²."""
    vol = np.load(npz)['vol']           # (N, N, N) uint8, axes: (x, y, z)
    N = vol.shape[1]
    sl = vol[:, N // 2, :].astype(np.float32) / 255.0   # (N, N)
    if sl.shape[0] != out_res:
        sl = zoom(sl, out_res / sl.shape[0], order=1)
    return sl


def find_gt_npz(data_dir: Path, timestep: float) -> Path:
    """Find the GT npz file closest to the requested timestep.

    Naming convention: lattice_XX.npz where XX = step index (0–20, even steps).
    Normalised time = step / 20.
    """
    step = round(timestep * 20)           # e.g. t=0.5 → step 10
    # round to nearest even step that exists
    step = round(step / 2) * 2
    step = max(0, min(20, step))
    candidate = data_dir / f'lattice_{step:02d}.npz'
    if candidate.exists():
        return candidate
    # fallback to endpoints
    return sorted(data_dir.glob('lattice_*.npz'))[0 if timestep < 0.5 else -1]


@torch.no_grad()
def get_mixing_curve(pipeline, n_times: int = 51, n_pts: int = 4096, device: str = 'cuda'):
    """Sample mean ± std of α(t) across random spatial positions."""
    t_vals = torch.linspace(0, 1, n_times)
    # Sample positions roughly in the structure extent
    positions = ((torch.rand(n_pts, 3) * 2 - 1) * 0.75).to(device)
    means, stds = [], []
    for t in t_vals:
        times = t.view(1, 1).expand(n_pts, 1).to(device)
        alpha = pipeline.model.field_weighing.get_mixing_coefficient(positions, times, step=0)
        alpha = alpha.squeeze().cpu().float()
        means.append(alpha.mean().item())
        stds.append(alpha.std().item())
    return t_vals.numpy(), np.array(means), np.array(stds)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/kelvin'))
    p.add_argument('--data-dir',   type=Path, default=Path('data/kelvin'))
    p.add_argument('--nx-root',    type=Path, default=Path('neural_xray'))
    p.add_argument('--timesteps',  type=float, nargs='+',
                   default=[0.0, 0.2, 0.5, 0.8, 1.0])
    p.add_argument('--resolution', type=int, default=128,
                   help='Cross-section resolution in pixels (default: 128)')
    p.add_argument('--rhomax',     type=float, default=None,
                   help='Density normalisation (auto-detected from first model if not set)')
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-xsec',  type=Path, default=Path('outputs/kelvin/eval_xsections.png'))
    p.add_argument('--out-mix',   type=Path, default=Path('outputs/kelvin/eval_mixing.png'))
    p.add_argument('--models', nargs='+',
                   choices=['vel_6', 'vel_9', 'spatiotemporal_mix'],
                   default=['vel_6', 'vel_9', 'spatiotemporal_mix'],
                   help='Which models to evaluate (default: all three)')
    return p.parse_args()


def main():
    args = parse_args()
    setup_paths(args.nx_root)

    timesteps = args.timesteps
    n_t = len(timesteps)

    configs = {
        'vel_6':              args.output_dir / 'xray_vfield/vel_6/config.yml',
        'vel_9':              args.output_dir / 'xray_vfield/vel_9/config.yml',
        'spatiotemporal_mix': args.output_dir / 'spatiotemporal_mix/vel_9/config.yml',
    }

    # ── GT slices ─────────────────────────────────────────────────────────────
    gt_slices: List[np.ndarray] = []
    for t in timesteps:
        npz = find_gt_npz(args.data_dir, t)
        actual_t = int(npz.stem.split('_')[1]) / 20
        print(f'GT t={t:.2f} → {npz.name}  (actual t={actual_t:.2f})')
        gt_slices.append(gt_xz_slice(npz, args.resolution))

    # Update row label to reflect actual GT source
    gt_label = 'GT (FEM volume)'

    # ── NeRF slices ──────────────────────────────────────────────────────────
    model_slices = {}
    mix_data = None
    rhomax = args.rhomax  # may be None → auto-detect from first model

    for name, config_yml in configs.items():
        if name not in args.models:
            model_slices[name] = [np.zeros((args.resolution, args.resolution)) for _ in timesteps]
            continue
        if not config_yml.exists():
            print(f'[skip] {name}: {config_yml} not found')
            model_slices[name] = [np.zeros((args.resolution, args.resolution))
                                  for _ in timesteps]
            continue

        print(f'\nLoading {name}  ({config_yml})')
        pipeline = load_pipeline(config_yml, args.device)

        # Auto-detect rhomax from a single query at t=0 if not set
        if rhomax is None:
            probe = pipeline.eval_along_plane(
                target='field', plane='xz', distance=0.0,
                engine='numpy', resolution=args.resolution, rhomax=1.0, time=0.0,
            )
            rhomax = float(probe.max()) if probe.max() > 0 else 1.0
            print(f'  rhomax auto-detected: {rhomax:.4f}')

        slices = []
        for t in timesteps:
            print(f'  t={t:.2f}')
            slices.append(xz_slice(pipeline, t, args.resolution, rhomax))
        model_slices[name] = slices

        if name == 'spatiotemporal_mix':
            print('  computing mixing curve...')
            mix_data = get_mixing_curve(pipeline, device=args.device)

        del pipeline
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    # ── Figure 1: cross-sections ──────────────────────────────────────────────
    print('\nPlotting cross-sections...')
    label_map = {'vel_6': 'vel_6', 'vel_9': 'vel_9', 'spatiotemporal_mix': 'spatiotemporal mix'}
    rows = [(label_map[m], model_slices[m]) for m in ['vel_6', 'vel_9', 'spatiotemporal_mix']
            if m in args.models]
    rows.append((gt_label, gt_slices))
    n_rows = len(rows)

    fig, axes = plt.subplots(n_rows, n_t,
                             figsize=(2.6 * n_t, 2.7 * n_rows),
                             squeeze=False)
    fig.suptitle('x-z cross-sections at centre y', fontsize=13, y=1.01)

    gt_vmax = max(sl.max() for sl in gt_slices)

    for r, (label, slices) in enumerate(rows):
        vmax = gt_vmax if r == n_rows - 1 else 1.0   # NeRF slices already rhomax-normalised
        for c, (t, sl) in enumerate(zip(timesteps, slices)):
            ax = axes[r, c]
            ax.imshow(sl.T, origin='lower', cmap='hot', vmin=0, vmax=vmax, aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f't = {t:.2f}', fontsize=10)
        axes[r, 0].set_ylabel(label, fontsize=9, labelpad=4)

    plt.tight_layout()
    fig.savefig(args.out_xsec, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {args.out_xsec}')

    # ── Figure 2: mixing coefficient ─────────────────────────────────────────
    if mix_data is not None:
        print('Plotting mixing coefficients...')
        t_vals, means, stds = mix_data
        fig2, ax = plt.subplots(figsize=(6, 4))
        ax.fill_between(t_vals, means - stds, means + stds,
                        alpha=0.25, color='steelblue', label='±1 std (spatial)')
        ax.plot(t_vals, means, color='steelblue', lw=2, label=r'mean $\alpha(t)$')
        ax.plot(t_vals, t_vals, 'k--', lw=1, alpha=0.5, label='ideal  ($\\alpha = t$)')
        ax.axhline(0.5, color='0.6', lw=0.5, ls=':')
        ax.set_xlabel('Normalised time  $t$', fontsize=12)
        ax.set_ylabel(r'Mixing coefficient  $\alpha$', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(r'$\alpha = 0$ → forward canonical,  $\alpha = 1$ → backward canonical',
                     fontsize=10)
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig2.savefig(args.out_mix, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f'Saved → {args.out_mix}')


if __name__ == '__main__':
    main()
