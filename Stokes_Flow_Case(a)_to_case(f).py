import numpy as np
import matplotlib.pyplot as plt

def compute_streamfunction(case):
    # Case-specific parameters
    if case == 'a':
        R_inner, ε, Ω1, Ω2 = 0.25, 0.5, 0.0, 1.0
    elif case == 'b':
        R_inner, ε, Ω1, Ω2 = 0.5, 0.5, 0.0, 1.0
    elif case == 'c':
        R_inner, ε, Ω1, Ω2 = 0.5, 0.5, 1.0, 0.0
    elif case == 'd':
        R_inner, ε, Ω1, Ω2 = 0.5, 0.25, 1.0, 0.0
    elif case == 'e':
        R_inner, ε, Ω1, Ω2 = 0.3, 0.1, 1.0, -4.0
    elif case == 'f':
        R_inner, ε, Ω1, Ω2 = 0.3, 0.1, 1.0, 4.0
    else:
        raise ValueError("Invalid case")

    R_outer = 1.0
    center_outer = np.array([0.0, 0.0])
    center_inner = np.array([ε, 0.0])

    # Grid setup
    N = 500
    x = np.linspace(-1.6, 1.6, N)
    y = np.linspace(-1.6, 1.6, N)
    X, Y = np.meshgrid(x, y)

    r_outer = np.sqrt((X - center_outer[0])**2 + (Y - center_outer[1])**2)
    r_inner = np.sqrt((X - center_inner[0])**2 + (Y - center_inner[1])**2)

    # Fluid domain
    fluid_mask = (r_outer < R_outer) & (r_inner > R_inner)

    # Streamfunction from outer cylinder
    theta_outer = np.arctan2(Y - center_outer[1], X - center_outer[0])
    ψ_outer = Ω1 * (r_outer**2 - R_inner**2) * np.sin(theta_outer)

    # Streamfunction from inner cylinder
    theta_inner = np.arctan2(Y - center_inner[1], X - center_inner[0])
    ψ_inner = Ω2 * (R_inner**2 / r_inner) * np.sin(theta_inner)

    # Total streamfunction
    ψ = np.where(fluid_mask, ψ_outer + ψ_inner, np.nan)

    return X, Y, ψ, center_outer, R_outer, center_inner, R_inner

# Plot all 6 cases in 3x2 grid
fig, axes = plt.subplots(3, 2, figsize=(12, 16))
cases = ['a', 'b', 'c', 'd', 'e', 'f']
titles = {
    'a': '(a) Ω1 = 0, Ω2 ≠ 0, R=0.25, ε=0.5',
    'b': '(b) Ω1 = 0, Ω2 ≠ 0, R=0.5, ε=0.5',
    'c': '(c) Ω1 ≠ 0, Ω2 = 0, R=0.5, ε=0.5',
    'd': '(d) Ω1 ≠ 0, Ω2 = 0, R=0.5, ε=0.25',
    'e': '(e) Ω1 = 1, Ω2 = -4, R=0.3, ε=0.1',
    'f': '(f) Ω1 = 1, Ω2 = 4, R=0.3, ε=0.1'
}

for ax, case in zip(axes.flat, cases):
    X, Y, ψ, c_out, R_out, c_in, R_in = compute_streamfunction(case)
    levels = np.linspace(np.nanmin(ψ), np.nanmax(ψ), 60)
    ax.contour(X, Y, ψ, levels=levels, cmap='viridis')
    
    # Draw cylinders
    outer = plt.Circle(c_out, R_out, color='black', fill=False, linewidth=2)
    inner = plt.Circle(c_in, R_in, color='black', fill=False, linewidth=2)
    ax.add_patch(outer)
    ax.add_patch(inner)

    ax.set_aspect('equal')
    ax.set_xlim([-1.6, 1.6])
    ax.set_ylim([-1.6, 1.6])
    ax.set_title(titles[case], fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.suptitle("Stokes Flow Streamlines Between Eccentric Rotating Cylinders\nCases (a) to (f)", fontsize=16, y=1.02)
plt.show()
