#!/usr/bin/env python3
"""Burner test: compare old X-based vs new arc-length-based interpolation.

Run:    python test_interp.py
Delete after verifying.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from src.preprocessing.interpolation import lin_interpolation


# ── old method kept inline for comparison ────────────────────────────────────

def lin_interpolation_old(data, length):
    x = data[:, 0]
    y = data[:, 1]
    xd = np.linspace(x[0], x[-1], length)
    y_interp = interp1d(x, y, kind='linear', bounds_error=False, fill_value="extrapolate")(xd)
    z = [data[0, 2]] * length
    phase = data[:, 3]
    phase_interp = interp1d(x, phase, kind='linear', bounds_error=False, fill_value="extrapolate")(xd)
    return np.vstack((xd, y_interp, z, phase_interp)).T


# ── synthetic trajectory ──────────────────────────────────────────────────────
# Path that goes mostly along X but has a pronounced Y-bulge in the middle.
# This is where the two methods diverge most visibly: X-sampling crowds points
# near the peak of the bulge (where dx/ds is smallest), arc-length sampling
# stays uniform in physical space.

N_RAW = 80
t = np.linspace(0, 1, N_RAW)

x_raw = t * 5.0                                          # 0 → 5 m
y_raw = 0.6 * np.sin(np.pi * t)                         # smooth arc, max 0.6 m
z_raw = np.full(N_RAW, 1.2)                             # constant tag height
phase_raw = 20 * np.pi * t + 2.5 * np.sin(4 * np.pi * t)  # realistic-ish unwrapped phase

data = np.column_stack([x_raw, y_raw, z_raw, phase_raw])

LENGTH = 60  # number of output samples

old = lin_interpolation_old(data, LENGTH)
new = lin_interpolation(data, LENGTH)


# ── sanity checks (printed) ───────────────────────────────────────────────────

def arc_spacings(res):
    return np.sqrt(np.diff(res[:, 0])**2 + np.diff(res[:, 1])**2)

old_sp = arc_spacings(old)
new_sp = arc_spacings(new)

total_arc = np.sqrt(np.diff(x_raw)**2 + np.diff(y_raw)**2).sum()
expected_step = total_arc / (LENGTH - 1)

print("=== Spacing check ===")
print(f"Total trajectory arc length : {total_arc:.4f} m")
print(f"Expected step (arc-length)  : {expected_step:.5f} m")
print(f"Old (X-based)  mean={old_sp.mean():.5f}  std={old_sp.std():.5f} m")
print(f"New (arc-len)  mean={new_sp.mean():.5f}  std={new_sp.std():.6f} m  ← should be ~0")

print("\n=== Endpoint check (new) ===")
print(f"  X: raw=[{x_raw[0]:.3f}, {x_raw[-1]:.3f}]  interp=[{new[0,0]:.3f}, {new[-1,0]:.3f}]")
print(f"  Y: raw=[{y_raw[0]:.3f}, {y_raw[-1]:.3f}]  interp=[{new[0,1]:.3f}, {new[-1,1]:.3f}]")
print(f"  Phase: raw=[{phase_raw[0]:.3f}, {phase_raw[-1]:.3f}]  interp=[{new[0,3]:.3f}, {new[-1,3]:.3f}]")


# ── plots ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Arc-length vs X-based interpolation", fontsize=14)

# 1 — trajectory in XY with sample markers
ax = axes[0, 0]
ax.plot(x_raw, y_raw, 'k-', lw=1.5, label='raw trajectory', zorder=1)
ax.scatter(old[:, 0], old[:, 1], s=40, color='tab:blue',
           zorder=3, label='X-sampled', alpha=0.8)
ax.scatter(new[:, 0], new[:, 1], s=40, color='tab:orange',
           zorder=3, label='arc-length sampled', marker='x', linewidths=1.5)
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_title('Sample positions on trajectory')
ax.legend(); ax.set_aspect('equal')

# 2 — inter-sample arc-length spacing
ax = axes[0, 1]
ax.plot(old_sp, color='tab:blue',
        label=f'X-sampled   σ = {old_sp.std():.5f} m')
ax.plot(new_sp, color='tab:orange',
        label=f'arc-length  σ = {new_sp.std():.2e} m')
ax.axhline(expected_step, color='gray', ls=':', lw=1, label=f'ideal step = {expected_step:.5f} m')
ax.set_xlabel('pair index'); ax.set_ylabel('distance between consecutive samples (m)')
ax.set_title('Inter-sample spacing  (lower σ → more uniform)')
ax.legend()

# 3 — phase signal
ax = axes[1, 0]
# map raw indices to a common x-axis (sample index in output)
raw_idx = np.linspace(0, LENGTH - 1, N_RAW)
ax.plot(raw_idx, phase_raw, 'k-', lw=1.5, alpha=0.6, label='raw phase')
ax.plot(old[:, 3], color='tab:blue', label='X-sampled phase')
ax.plot(new[:, 3], color='tab:orange', ls='--', label='arc-length phase')
ax.set_xlabel('output sample index'); ax.set_ylabel('phase (rad)')
ax.set_title('Phase signal')
ax.legend()

# 4 — X values of output samples (shows that arc-length output is not uniform in X)
ax = axes[1, 1]
ax.plot(old[:, 0], color='tab:blue', label='X-sampled: X uniform by construction')
ax.plot(new[:, 0], color='tab:orange', label='arc-length: X not uniform (expected)')
ax.set_xlabel('output sample index'); ax.set_ylabel('X (m)')
ax.set_title('X values of output samples')
ax.legend()

plt.tight_layout()
plt.show()
