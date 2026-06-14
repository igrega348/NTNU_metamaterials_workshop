import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor('#0f1117')

DARK  = '#0f1117'
PANEL = '#1a1d27'
GRID  = '#2a2d3a'
WHITE = '#e8e8f0'
DIM   = '#6b6d80'
BLUE  = '#4a9eff'
GREEN = '#3ddc84'
RED   = '#ff5555'
AMBER = '#ffb347'
PURPLE= '#c084fc'
TEAL  = '#2dd4bf'

def ax_dark(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)

def arrow(ax, x0, y0, dx, dy, color, lw=1.5, hs=8, hw=4):
    ax.annotate('', xy=(x0+dx, y0+dy), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=f'->', color=color,
                        lw=lw, mutation_scale=hs))

def basis_arrows(ax, cx, cy, R, scale=0.32, labels=True, alpha=1.0):
    """Draw X/Y/Z+ axes from (cx,cy). R is 2x2 rotation in 2D for the view."""
    dirs = {'X+': (np.array([1,0]), RED),
            'Y+': (np.array([0,1]), GREEN),
            'Z+': (np.array([0,0]), BLUE)}  # Z handled separately as vertical
    # We'll show X and Z+ in 2D as a simplified isometric view
    # X goes right, Z+ goes up
    ax.annotate('', xy=(cx+scale, cy), xytext=(cx,cy),
        arrowprops=dict(arrowstyle='->', color=RED, lw=1.8, mutation_scale=10))
    ax.annotate('', xy=(cx, cy+scale), xytext=(cx,cy),
        arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.8, mutation_scale=10))
    ax.annotate('', xy=(cx-scale*0.6, cy-scale*0.5), xytext=(cx,cy),
        arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.8, mutation_scale=10))
    if labels:
        ax.text(cx+scale+0.02, cy, 'X+', color=RED,   fontsize=8, va='center')
        ax.text(cx+0.02, cy+scale+0.02, 'Z+', color=BLUE,  fontsize=8)
        ax.text(cx-scale*0.6-0.04, cy-scale*0.5-0.02, 'Y+', color=GREEN, fontsize=8, ha='right')

# ─── title ───────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97, 'OOP normed_corr metric bug — coordinate frame mismatch',
         ha='center', va='top', color=WHITE, fontsize=15, fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1  left: World frame + camera comparison
# ══════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_axes([0.02, 0.56, 0.30, 0.38])
ax_dark(ax1)
ax1.set_xlim(-1.1, 1.1); ax1.set_ylim(-1.1, 1.1)
ax1.set_aspect('equal'); ax1.axis('off')
ax1.set_title('World frame  (Go renderer / object.json)', color=WHITE, fontsize=10, pad=6)

# Origin
ax1.plot(0, 0, 'o', color=WHITE, ms=4)

# World Z+ axis (up)
ax1.annotate('', xy=(0, 0.9), xytext=(0,0),
    arrowprops=dict(arrowstyle='->', color=BLUE, lw=2, mutation_scale=12))
ax1.text(0.04, 0.88, 'World Z+  (up)', color=BLUE, fontsize=9)

# World X axis
ax1.annotate('', xy=(0.9, 0), xytext=(0,0),
    arrowprops=dict(arrowstyle='->', color=RED, lw=2, mutation_scale=12))
ax1.text(0.92, 0.03, 'X+', color=RED, fontsize=9)

# Spheres at roughly correct relative positions (schematic)
s1 = np.array([-0.38, 0.18])   # [-0.5,-0.5,0.2] projected
s2 = np.array([ 0.38,-0.28])   # [ 0.5, 0.5,-0.3] projected
for s, lbl in [(s1, '●  [-0.5, -0.5,  0.2]'), (s2, '●  [ 0.5,  0.5, -0.3]')]:
    circ = plt.Circle(s, 0.10, color=TEAL, alpha=0.35, zorder=3)
    ax1.add_patch(circ)
    ax1.plot(*s, 'o', color=TEAL, ms=5, zorder=4)
ax1.text(s1[0]+0.13, s1[1]+0.02, '●  [-0.5, -0.5,  0.2]', color=TEAL, fontsize=8)
ax1.text(s2[0]+0.13, s2[1]+0.02, '●  [ 0.5,  0.5, -0.3]', color=TEAL, fontsize=8)

ax1.text(-1.05, -1.05, 'object.json\nsphere centers\n(world space)',
         color=TEAL, fontsize=8, va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a3a3a', edgecolor=TEAL, alpha=0.8))

# ── In-plane camera (bottom-left of ax1) ──────────────────────────────────
cam_ip = np.array([-0.80, -0.55])   # camera position in schematic
ax1.plot(*cam_ip, 's', color=AMBER, ms=9, zorder=5)
ax1.text(cam_ip[0], cam_ip[1]-0.13, 'In-plane\ncamera', color=AMBER, fontsize=8, ha='center')

# View direction toward origin
vdir = -cam_ip / np.linalg.norm(cam_ip)
ax1.annotate('', xy=cam_ip+vdir*0.35, xytext=cam_ip,
    arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.4, mutation_scale=8))

# Camera +Y = world Z+ (shown as upward arrow)
ax1.annotate('', xy=cam_ip+np.array([0, 0.32]), xytext=cam_ip,
    arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.8, mutation_scale=9))
ax1.text(cam_ip[0]+0.05, cam_ip[1]+0.32, 'cam +Y\n= Z+  ✓', color=BLUE, fontsize=7.5)

# ── OOP camera (top-left) ──────────────────────────────────────────────────
cam_oop = np.array([-0.55, 0.72])
ax1.plot(*cam_oop, 's', color=PURPLE, ms=9, zorder=5)
ax1.text(cam_oop[0]-0.12, cam_oop[1]+0.08, 'OOP camera\n(elevated)', color=PURPLE, fontsize=8, ha='center')

vdir2 = -cam_oop / np.linalg.norm(cam_oop)
ax1.annotate('', xy=cam_oop+vdir2*0.35, xytext=cam_oop,
    arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.4, mutation_scale=8))

# Camera +Y tilted (not Z+)
cam_y_tilted = np.array([0.40, 0.60])
cam_y_tilted /= np.linalg.norm(cam_y_tilted)
ax1.annotate('', xy=cam_oop+cam_y_tilted*0.32, xytext=cam_oop,
    arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.8, mutation_scale=9))
ax1.text(cam_oop[0]+0.02, cam_oop[1]+0.37, 'cam +Y\n≠ Z+ !', color=PURPLE, fontsize=7.5)

# arc showing tilt angle
theta1 = 90
theta2 = 90 - 30
arc = Arc(cam_oop, 0.22, 0.22, angle=0, theta1=theta2, theta2=theta1, color=AMBER, lw=1.2)
ax1.add_patch(arc)
ax1.text(cam_oop[0]+0.13, cam_oop[1]+0.17, '~4°', color=AMBER, fontsize=8)

ax1.text(0.0, -1.05,
    'Both cameras use world Z+ as the LookAt up parameter.\n'
    'But for elevated cameras, corrected up ⊥ view dir ≠ Z+.',
    color=DIM, fontsize=7.5, ha='center', va='bottom', style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1  middle: auto_orient step
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_axes([0.35, 0.56, 0.30, 0.38])
ax_dark(ax2)
ax2.set_xlim(-0.05, 1.05); ax2.set_ylim(-0.05, 1.05)
ax2.axis('off')
ax2.set_title('auto_orient_and_center_poses  (nerfstudio dataparser)', color=WHITE, fontsize=10, pad=6)

def panel_box(ax, x, y, w, h, title, tc, bc):
    rect = mpatches.FancyBboxPatch((x,y), w, h,
        boxstyle='round,pad=0.01', linewidth=1.5,
        edgecolor=tc, facecolor=bc, zorder=2)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h-0.025, title, ha='center', va='top',
            color=tc, fontsize=8.5, fontweight='bold', zorder=3)

# ── In-plane path ─────────────────────────────────────────────────────────
panel_box(ax2, 0.02, 0.65, 0.45, 0.30, 'In-plane cameras', AMBER, '#2a2000')
ax2.text(0.245, 0.84, 'mean(cam +Y) = [0, 0, 1]', color=WHITE, fontsize=8, ha='center', zorder=3)
ax2.text(0.245, 0.77, 'already Z+  →  T = Identity', color=GREEN, fontsize=8, ha='center', zorder=3)
ax2.text(0.245, 0.70, 'No rotation applied', color=DIM, fontsize=7.5, ha='center', zorder=3)

ax2.annotate('', xy=(0.245, 0.56), xytext=(0.245, 0.65),
    arrowprops=dict(arrowstyle='->', color=GREEN, lw=2, mutation_scale=10))

panel_box(ax2, 0.02, 0.40, 0.45, 0.15, '', GREEN, '#0a2a10')
ax2.text(0.245, 0.515, 'NeRF frame  =  World frame  ✓', color=GREEN, fontsize=8.5, ha='center', fontweight='bold', zorder=3)

# ── OOP path ──────────────────────────────────────────────────────────────
panel_box(ax2, 0.52, 0.65, 0.45, 0.30, 'OOP cameras', PURPLE, '#1e1030')
ax2.text(0.745, 0.84, 'mean(cam +Y) = [0.068, 0.015, 0.998]', color=WHITE, fontsize=7.5, ha='center', zorder=3)
ax2.text(0.745, 0.77, '~4° off Z+  →  applies R ≈ 4°', color=RED, fontsize=8, ha='center', zorder=3)
ax2.text(0.745, 0.70, '(method="up" default)', color=DIM, fontsize=7.5, ha='center', zorder=3)

ax2.annotate('', xy=(0.745, 0.56), xytext=(0.745, 0.65),
    arrowprops=dict(arrowstyle='->', color=RED, lw=2, mutation_scale=10))

panel_box(ax2, 0.52, 0.40, 0.45, 0.15, '', RED, '#2a0a0a')
ax2.text(0.745, 0.515, 'NeRF frame  ≠  World frame  ✗', color=RED, fontsize=8.5, ha='center', fontweight='bold', zorder=3)

# ── why ───────────────────────────────────────────────────────────────────
ax2.text(0.5, 0.34, 'Why? auto_orient sees mean camera-up ≈ 4° off Z+\n'
         'and "corrects" it — not knowing the world frame is already valid.',
         color=DIM, fontsize=8, ha='center', va='top', style='italic')

# ── dataparser_transforms.json outcome ────────────────────────────────────
panel_box(ax2, 0.02, 0.07, 0.45, 0.20, 'dataparser_transforms.json', GREEN, '#0a1a0a')
ax2.text(0.245, 0.21, 'transform = Identity', color=GREEN, fontsize=9, ha='center', zorder=3)
ax2.text(0.245, 0.15, '(in-plane)', color=DIM, fontsize=8, ha='center', zorder=3)

panel_box(ax2, 0.52, 0.07, 0.45, 0.20, 'dataparser_transforms.json', RED, '#1a0a0a')
ax2.text(0.745, 0.21, 'transform = R  (4° rotation)', color=RED, fontsize=9, ha='center', zorder=3)
ax2.text(0.745, 0.15, '(OOP)', color=DIM, fontsize=8, ha='center', zorder=3)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1  right: the metric comparison
# ══════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_axes([0.68, 0.56, 0.30, 0.38])
ax_dark(ax3)
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.set_title('normed_corr metric evaluation  (get_eval_density_loss)', color=WHITE, fontsize=10, pad=6)

# Same query positions box
panel_box(ax3, 0.1, 0.82, 0.8, 0.13, '', BLUE, '#0a1a2a')
ax3.text(0.5, 0.915, 'Query positions  pos ∈ [−1, 1]³  (NeRF space)', color=BLUE, fontsize=8.5, ha='center', zorder=3)

# pred density
ax3.annotate('', xy=(0.25, 0.68), xytext=(0.25, 0.82),
    arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.5, mutation_scale=9))
panel_box(ax3, 0.02, 0.52, 0.45, 0.16, 'pred_density', PURPLE, '#1e1030')
ax3.text(0.245, 0.66, 'field(pos)', color=WHITE, fontsize=8.5, ha='center', zorder=3)
ax3.text(0.245, 0.59, 'NeRF was trained in\nNeRF frame  →  correct', color=GREEN, fontsize=7.5, ha='center', zorder=3)
ax3.text(0.245, 0.525, '  sees ● at R·p', color=PURPLE, fontsize=8, ha='center', zorder=3, fontstyle='italic')

# gt density
ax3.annotate('', xy=(0.75, 0.68), xytext=(0.75, 0.82),
    arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.5, mutation_scale=9))
panel_box(ax3, 0.52, 0.52, 0.45, 0.16, 'gt_density  (BROKEN)', RED, '#2a0a0a')
ax3.text(0.745, 0.66, 'obj.density(pos)', color=WHITE, fontsize=8.5, ha='center', zorder=3)
ax3.text(0.745, 0.59, 'object.json in world\nframe, no transform', color=RED, fontsize=7.5, ha='center', zorder=3)
ax3.text(0.745, 0.525, '  sees ● at p  ← wrong frame!', color=RED, fontsize=8, ha='center', zorder=3, fontstyle='italic')

# correlation
ax3.plot([0.245, 0.5], [0.52, 0.42], color=GRID, lw=1, ls='--')
ax3.plot([0.745, 0.5], [0.52, 0.42], color=GRID, lw=1, ls='--')
panel_box(ax3, 0.15, 0.28, 0.70, 0.14, '', RED, '#1a0808')
ax3.text(0.5, 0.385, 'corr(R·p location,  p location)  →  low!', color=RED, fontsize=8, ha='center', zorder=3)
ax3.text(0.5, 0.305, 'spheres appear shifted by ~4°  →  0.789', color=AMBER, fontsize=8.5, ha='center', fontweight='bold', zorder=3)

# shift illustration (small)
ax3.text(0.5, 0.23, '≈ 0.05 unit shift at r=0.7,  ~⅓ sphere radius', color=DIM, fontsize=7.5, ha='center')

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2  — fixes
# ══════════════════════════════════════════════════════════════════════════════
ax4 = fig.add_axes([0.02, 0.04, 0.45, 0.46])
ax_dark(ax4)
ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
ax4.axis('off')
ax4.set_title('Fix A — compensate in metric  (canonical_pipeline.py)', color=WHITE, fontsize=10, pad=6)

# Same query box
panel_box(ax4, 0.05, 0.84, 0.90, 0.12, '', BLUE, '#0a1a2a')
ax4.text(0.5, 0.91, 'pos ∈ [−1, 1]³  (NeRF space)', color=BLUE, fontsize=9, ha='center', zorder=3)

# pred arrow
ax4.annotate('', xy=(0.22, 0.72), xytext=(0.22, 0.84),
    arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.5, mutation_scale=9))

# new: invert rotation for gt
ax4.annotate('', xy=(0.78, 0.72), xytext=(0.78, 0.84),
    arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.5, mutation_scale=9))

panel_box(ax4, 0.52, 0.57, 0.43, 0.15, 'world_pos  (fixed)', GREEN, '#0a2010')
ax4.text(0.735, 0.705, 'world_pos = pos @ R', color=WHITE, fontsize=8.5, ha='center', zorder=3,
         fontfamily='monospace')
ax4.text(0.735, 0.635, 'inverse rotation\nNeRF → world', color=GREEN, fontsize=7.5, ha='center', zorder=3)

ax4.annotate('', xy=(0.735, 0.57), xytext=(0.735, 0.57+0.12),
    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5, mutation_scale=9))

panel_box(ax4, 0.02, 0.57, 0.45, 0.15, 'pred_density', PURPLE, '#1e1030')
ax4.text(0.245, 0.705, 'field(pos)', color=WHITE, fontsize=8.5, ha='center', zorder=3)
ax4.text(0.245, 0.635, 'NeRF frame  ✓', color=GREEN, fontsize=8, ha='center', zorder=3)

panel_box(ax4, 0.52, 0.37, 0.43, 0.20, 'gt_density  (fixed)', GREEN, '#0a2010')
ax4.text(0.735, 0.535, 'obj.density(world_pos)', color=WHITE, fontsize=8, ha='center', zorder=3)
ax4.text(0.735, 0.47, 'world frame  ✓', color=GREEN, fontsize=8, ha='center', zorder=3)
ax4.text(0.735, 0.39, '  sees ● at R·p  (correct!)', color=GREEN, fontsize=8, ha='center', zorder=3)

panel_box(ax4, 0.02, 0.37, 0.45, 0.20, 'pred_density', PURPLE, '#1e1030')
ax4.text(0.245, 0.47, 'sees ● at R·p', color=PURPLE, fontsize=8, ha='center', zorder=3)

ax4.plot([0.245, 0.5], [0.37, 0.27], color=GRID, lw=1, ls='--')
ax4.plot([0.735, 0.5], [0.37, 0.27], color=GRID, lw=1, ls='--')

panel_box(ax4, 0.15, 0.13, 0.70, 0.14, '', GREEN, '#0a2010')
ax4.text(0.5, 0.225, 'corr(R·p,  R·p)  →  1.0', color=GREEN, fontsize=9, ha='center', fontweight='bold', zorder=3)
ax4.text(0.5, 0.155, 'normed_corr = 0.982  ✓', color=GREEN, fontsize=9, ha='center', fontweight='bold', zorder=3)

ax4.text(0.5, 0.04,
    'R comes from dataparser_transforms.json (loaded at eval time).\n'
    'Safe: when R=Identity, pos @ I = pos — no-op for in-plane case.',
    color=DIM, fontsize=7.5, ha='center', style='italic')

# ── fix B ─────────────────────────────────────────────────────────────────
ax5 = fig.add_axes([0.52, 0.04, 0.45, 0.46])
ax_dark(ax5)
ax5.set_xlim(0, 1); ax5.set_ylim(0, 1)
ax5.axis('off')
ax5.set_title('Fix B — disable spurious rotation  (multi_camera_dataparser.py)', color=WHITE, fontsize=10, pad=6)

panel_box(ax5, 0.05, 0.78, 0.90, 0.18, 'Before (default = "up")', RED, '#1a0808')
ax5.text(0.5, 0.91, 'orientation_method = "up"', color=RED, fontsize=9, ha='center', zorder=3, fontfamily='monospace')
ax5.text(0.5, 0.845, 'Sees mean cam-up ≈ 4° off Z+ →', color=WHITE, fontsize=8, ha='center', zorder=3)
ax5.text(0.5, 0.795, 'applies R → NeRF frame ≠ World frame', color=RED, fontsize=8, ha='center', zorder=3)

ax5.annotate('', xy=(0.5, 0.65), xytext=(0.5, 0.78),
    arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5, mutation_scale=12))
ax5.text(0.5, 0.72, 'one-line fix', color=GREEN, fontsize=8, ha='center', style='italic')

panel_box(ax5, 0.05, 0.48, 0.90, 0.17, 'After (fixed = "none")', GREEN, '#0a2010')
ax5.text(0.5, 0.615, 'orientation_method = "none"', color=GREEN, fontsize=9, ha='center', zorder=3, fontfamily='monospace')
ax5.text(0.5, 0.555, 'No rotation applied → T = Identity', color=WHITE, fontsize=8, ha='center', zorder=3)
ax5.text(0.5, 0.495, 'NeRF frame  =  World frame  ✓', color=GREEN, fontsize=8, ha='center', zorder=3)

# consequence
ax5.text(0.5, 0.43, '↓', color=BLUE, fontsize=14, ha='center')

panel_box(ax5, 0.05, 0.27, 0.90, 0.15, '', BLUE, '#0a1020')
ax5.text(0.5, 0.38, 'metric works correctly without any compensation', color=WHITE, fontsize=8.5, ha='center', zorder=3)
ax5.text(0.5, 0.31, 'dataparser_transforms.json  →  Identity  for all datasets', color=BLUE, fontsize=8, ha='center', zorder=3)

ax5.text(0.5, 0.20,
    'Why is "none" correct here?\n'
    'The Go renderer explicitly sets world up = Z+ and the scene\n'
    'is centered at origin. The coordinate frame is already well-defined.',
    color=DIM, fontsize=8, ha='center', style='italic', va='top')

ax5.text(0.5, 0.07,
    'Both fixes are applied. Fix B prevents the problem;\n'
    'Fix A guards against it re-appearing from any future orientation step.',
    color=DIM, fontsize=7.5, ha='center', style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# flow arrows between panels (approx)
# ══════════════════════════════════════════════════════════════════════════════
for x in [0.33, 0.66]:
    fig.add_artist(mpatches.FancyArrowPatch(
        (x, 0.75), (x+0.02, 0.75),
        transform=fig.transFigure,
        arrowstyle='->', color=GRID,
        mutation_scale=15, lw=1.5))

plt.savefig('/tmp/oop_schematic.png', dpi=140, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print('Saved /tmp/oop_schematic.png')
