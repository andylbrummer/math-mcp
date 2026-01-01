#!/usr/bin/env python3
"""Generate quantum demo videos with sensor lines and potential overlays."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path

mpl.use("Agg")

OUTPUT_DIR = Path(__file__).parent.parent / "docs/static/img/demos"


def create_gaussian_wavepacket(grid_size, position, momentum, width):
    """Create Gaussian wavepacket."""
    nx, ny = grid_size
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    px, py = position
    kx, ky = momentum

    psi = np.exp(-((xx - px)**2 + (yy - py)**2) / (2 * width**2))
    psi = psi * np.exp(1j * (kx * xx + ky * yy))
    psi = psi / np.sqrt(np.sum(np.abs(psi)**2))
    return psi


def create_slit_potential(grid_size, slit_config):
    """Create slit potential barrier."""
    nx, ny = grid_size
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # Barrier from x=80-90
    barrier = (xx > 80) & (xx < 90)

    if slit_config == "single":
        # Single slit: y=103-153 open
        blocked = barrier & ((yy < 103) | (yy > 153))
    elif slit_config == "double":
        # Double slit: openings at y=103-118 and y=138-153
        blocked = barrier & ((yy < 103) | ((yy > 118) & (yy < 138)) | (yy > 153))
    elif slit_config == "triple":
        # Triple slit: openings at y=103-118, y=123-133, y=138-153
        blocked = barrier & ((yy < 103) | ((yy > 118) & (yy < 123)) |
                            ((yy > 133) & (yy < 138)) | (yy > 153))
    else:
        blocked = np.zeros_like(xx, dtype=bool)

    potential = np.where(blocked, 1e10, 0.0)
    return potential


def create_lattice_potential(grid_size, lattice_type, depth, spacing, width):
    """Create lattice potential with Gaussian point centers."""
    nx, ny = grid_size
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    potential = np.zeros((nx, ny))

    if lattice_type == "square":
        for cx in np.arange(spacing / 2, nx, spacing):
            for cy in np.arange(spacing / 2, ny, spacing):
                r2 = (xx - cx)**2 + (yy - cy)**2
                potential += depth * np.exp(-r2 / (2 * width**2))
    elif lattice_type in ("hexagonal", "triangular"):
        cx_values = list(np.arange(spacing / 2, nx, spacing * np.sqrt(3) / 2))
        for row, cx in enumerate(cx_values):
            offset = (spacing / 2) if row % 2 else 0
            for cy in np.arange(spacing / 2 + offset, ny, spacing):
                r2 = (xx - cx)**2 + (yy - cy)**2
                potential += depth * np.exp(-r2 / (2 * width**2))

    return potential


def solve_schrodinger_2d(psi, potential, n_steps, dt):
    """Solve 2D Schrödinger equation using split-step Fourier method."""
    nx, ny = psi.shape

    # Momentum space grids
    kx = np.fft.fftfreq(nx, d=1.0) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=1.0) * 2 * np.pi
    kxx, kyy = np.meshgrid(kx, ky, indexing="ij")
    k2 = kxx**2 + kyy**2

    # Propagators
    exp_v = np.exp(-0.5j * potential * dt)
    exp_t = np.exp(-0.5j * k2 * dt)

    trajectory = [np.abs(psi)**2]
    store_every = max(1, n_steps // 100)

    for step in range(n_steps):
        # Split-step: V/2 -> T -> V/2
        psi = exp_v * psi
        psi = np.fft.ifft2(exp_t * np.fft.fft2(psi))
        psi = exp_v * psi

        if step % store_every == 0:
            trajectory.append(np.abs(psi)**2)

    return trajectory


def render_video(trajectory, potential, output_path, sensor_line=None,
                 show_potential=True, fps=20):
    """Render animation with optional sensor line and potential overlay."""

    if sensor_line is not None:
        fig, (ax, ax_sensor) = plt.subplots(
            1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [2, 1]}
        )
        fig.patch.set_facecolor("#0a0a1a")
        ax_sensor.set_facecolor("#0a0a1a")
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax_sensor = None

    fig.patch.set_facecolor("#0a0a1a")
    ax.set_facecolor("#0a0a1a")

    vmax = np.percentile([np.max(t) for t in trajectory], 95)
    im = ax.imshow(
        trajectory[0].T, origin="lower", cmap="viridis",
        vmin=0, vmax=vmax, aspect="equal"
    )

    # Show potential overlay
    if show_potential and potential is not None:
        pot_max = np.max(potential)
        if pot_max > 0:
            ax.contour(
                potential.T, levels=[pot_max * 0.5],
                colors=["red"], alpha=0.7, linewidths=2
            )

    # Sensor line setup
    sensor_accum = None
    sensor_line_plot = None
    sensor_max = None

    if sensor_line is not None and ax_sensor is not None:
        ax.axvline(x=sensor_line, color="yellow", linestyle="--", linewidth=2, alpha=0.8)
        ny = trajectory[0].shape[1]
        sensor_accum = np.zeros(ny)

        # Pre-compute max for fixed scale
        temp_accum = np.zeros(ny)
        for frame in trajectory:
            temp_accum = temp_accum + frame[sensor_line, :]
        sensor_max = np.max(temp_accum) * 1.1

        sensor_line_plot, = ax_sensor.plot(np.zeros(ny), np.arange(ny), color="cyan", linewidth=2)
        ax_sensor.set_xlim(0, sensor_max)
        ax_sensor.set_ylim(0, ny)
        ax_sensor.set_xlabel("Accumulated Intensity", color="white", fontsize=12)
        ax_sensor.set_ylabel("y", color="white", fontsize=12)
        ax_sensor.set_title("Detector", color="white", fontsize=14)
        ax_sensor.tick_params(colors="white")
        for spine in ax_sensor.spines.values():
            spine.set_color("#333")

    ax.set_xlabel("x", color="white", fontsize=12)
    ax.set_ylabel("y", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.set_title("Probability Density |ψ|²", color="white", fontsize=14)
    for spine in ax.spines.values():
        spine.set_color("#333")

    def animate(frame_idx):
        nonlocal sensor_accum
        im.set_array(trajectory[frame_idx].T)
        elements = [im]

        if sensor_line is not None and sensor_line_plot is not None:
            current_intensity = trajectory[frame_idx][sensor_line, :]
            sensor_accum = sensor_accum + current_intensity
            sensor_line_plot.set_xdata(sensor_accum)
            elements.append(sensor_line_plot)

        return elements

    anim = animation.FuncAnimation(fig, animate, frames=len(trajectory), interval=1000/fps, blit=True)

    # Save
    anim.save(str(output_path), writer="pillow", fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_slit_demo(slit_type):
    """Generate a slit diffraction demo."""
    print(f"\nGenerating {slit_type} slit demo...")

    grid_size = (256, 256)
    potential = create_slit_potential(grid_size, slit_type)
    psi = create_gaussian_wavepacket(grid_size, [40, 128], [0.2, 0], 35)

    print("  Running simulation...")
    trajectory = solve_schrodinger_2d(psi, potential, 1400, 0.1)

    print("  Rendering video...")
    output_path = OUTPUT_DIR / f"{slit_type}_slit.gif"
    render_video(trajectory, potential, output_path, sensor_line=220, show_potential=True)

    return output_path


def generate_bragg_demo(lattice_type):
    """Generate a Bragg scattering demo."""
    print(f"\nGenerating Bragg {lattice_type} demo...")

    grid_size = (256, 256)
    potential = create_lattice_potential(grid_size, lattice_type, depth=100, spacing=25, width=3)
    psi = create_gaussian_wavepacket(grid_size, [40, 128], [0.25, 0], 25)

    print("  Running simulation...")
    trajectory = solve_schrodinger_2d(psi, potential, 1200, 0.1)

    print("  Rendering video...")
    output_path = OUTPUT_DIR / f"bragg_{lattice_type}.gif"
    render_video(trajectory, potential, output_path, sensor_line=None, show_potential=True)

    return output_path


def convert_to_webm(gif_path):
    """Convert GIF to webm."""
    import subprocess
    webm_path = gif_path.with_suffix(".webm")
    webp_path = gif_path.with_suffix(".webp")

    # Convert to webm
    subprocess.run([
        "ffmpeg", "-y", "-i", str(gif_path),
        "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0",
        "-pix_fmt", "yuva420p", "-auto-alt-ref", "0",
        str(webm_path)
    ], capture_output=True)

    # Create webp thumbnail from middle frame
    subprocess.run([
        "ffmpeg", "-y", "-i", str(gif_path),
        "-vf", "select=eq(n\\,50)", "-vframes", "1",
        str(webp_path)
    ], capture_output=True)

    print(f"  Converted to: {webm_path.name}, {webp_path.name}")

    # Clean up GIF
    gif_path.unlink()


def main():
    """Generate all quantum demos."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Quantum Demos with Sensor Lines")
    print("=" * 60)

    # Slit demos
    for slit_type in ["single", "double", "triple"]:
        gif_path = generate_slit_demo(slit_type)
        convert_to_webm(gif_path)

    # Bragg demos
    for lattice_type in ["square", "hexagonal", "triangular"]:
        gif_path = generate_bragg_demo(lattice_type)
        convert_to_webm(gif_path)

    print("\n" + "=" * 60)
    print("All demos generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
