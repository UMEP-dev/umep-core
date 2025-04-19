import timeit

import matplotlib.pyplot as plt
import numpy as np
from umep import common
from umep.functions.svf_functions import svfForProcessing153
from umep.rustalgos import shadowing, skyview
from umep.util.SEBESOLWEIGCommonFiles.shadowingfunction_wallheight_23 import shadowingfunction_wallheight_23


def test_shadowing():
    # Test shadowingfunction_wallheight_23 vs shadowingfunction_wallheight_25 for speed
    repeats = 3
    dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp = make_test_arrays(resolution=1)

    def run_py():
        shadowingfunction_wallheight_23(
            dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp * np.pi / 180.0
        )

    times_py = timeit.repeat(run_py, number=1, repeat=repeats)
    print_timing_stats("shadowingfunction_wallheight_23", times_py)

    def run_rust():
        shadowing.shadowingfunction_wallheight_25(
            dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp * np.pi / 180.0, None, None
        )

    times_rust = timeit.repeat(run_rust, number=1, repeat=repeats)
    print_timing_stats("shadowing.shadowingfunction_wallheight_25", times_rust)

    # Print relative speed as percentage
    relative_speed(times_py, times_rust)

    # Run Python version
    vegsh, sh, vbshvegsh, wallsh, wallsun, wallshve, facesh, facesun = shadowingfunction_wallheight_23(
        dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp * np.pi / 180.0
    )
    result_py = {
        "veg_shadow_map": vegsh,
        "bldg_shadow_map": sh,
        "vbshvegsh": vbshvegsh,
        "wallsh": wallsh,
        "wallsun": wallsun,
        "wallshve": wallshve,
        "facesh": facesh,
        "facesun": facesun,
    }
    # Run Rust version
    result_rust = shadowing.shadowingfunction_wallheight_25(
        dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp * np.pi / 180.0, None, None
    )
    key_map = {
        "veg_shadow_map": "veg_shadow_map",
        "bldg_shadow_map": "bldg_shadow_map",
        "vbshvegsh": "vbshvegsh",
        "wallsh": "wallsh",
        "wallsun": "wallsun",
        "wallshve": "wallshve",
        "facesh": "facesh",
        "facesun": "facesun",
    }
    # Compare results
    compare_svf_results(result_py, result_rust, key_map, atol=0.01)
    # Plot visual residuals
    plot_visual_residuals(sh, result_rust.bldg_shadow_map, title_prefix="Building Shadows")


def test_svf():
    # Test svfForProcessing153 vs skyview.calculate_svf_153 for speed
    repeats = 1
    dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp = make_test_arrays(resolution=2)

    def run_py():
        svfForProcessing153(dsm, vegdsm, vegdsm2, scale, 1)

    times_py = timeit.repeat(run_py, number=1, repeat=repeats)
    print_timing_stats("svfForProcessing153", times_py)

    def run_rust():
        skyview.calculate_svf_153(dsm, vegdsm, vegdsm2, scale, True)

    times_rust = timeit.repeat(run_rust, number=1, repeat=repeats)
    print_timing_stats("skyview.calculate_svf_153", times_rust)

    # Print relative speed as percentage
    relative_speed(times_py, times_rust)

    # Run Python version
    result_py = svfForProcessing153(dsm, vegdsm, vegdsm2, scale, 1)
    # Run Rust version
    result_rust = skyview.calculate_svf_153(dsm, vegdsm, vegdsm2, scale, True)
    # Compare results
    key_map = {
        "svf": "svf",
        "svfE": "svf_east",
        "svfS": "svf_south",
        "svfW": "svf_west",
        "svfN": "svf_north",
        "svfveg": "svf_veg",
        "svfEveg": "svf_veg_east",
        "svfSveg": "svf_veg_south",
        "svfWveg": "svf_veg_west",
        "svfNveg": "svf_veg_north",
        "svfaveg": "svf_aniso_veg",
        "svfEaveg": "svf_aniso_veg_east",
        "svfSaveg": "svf_aniso_veg_south",
        "svfWaveg": "svf_aniso_veg_west",
        "svfNaveg": "svf_aniso_veg_north",
        "shmat": "shadow_matrix",
        "vegshmat": "veg_shadow_matrix",
        "vbshvegshmat": "vbshvegsh_matrix",
    }
    compare_svf_results(result_py, result_rust, key_map, atol=0.01)
    # plot visual residuals of svf
    plot_visual_residuals(result_py["svf"], result_rust.svf, title_prefix="SVF")


def make_test_arrays(
    resolution,
    dsm_path="demos/data/athens/DSM_{res}m.tif",
    veg_dsm_path="demos/data/athens/CDSM_{res}m.tif",
    wall_hts_path="demos/data/athens/walls_{res}m/wall_hts.tif",
    wall_aspect_path="demos/data/athens/walls_{res}m/wall_aspects.tif",
):
    dsm, dsm_transf, _crs = common.load_raster(dsm_path.format(res=resolution), bbox=None)
    vegdsm, _transf, _crs = common.load_raster(veg_dsm_path.format(res=resolution), bbox=None)
    vegdsm2 = np.zeros(dsm.shape, dtype=np.float32)  # Ensure float32
    azi = 45.0
    alt = 30.0
    scale = 1 / dsm_transf.a
    vegmax = vegdsm.max()
    amaxvalue = dsm.max() - dsm.min()
    amaxvalue = np.maximum(amaxvalue, vegmax)
    bush = np.zeros(dsm.shape, dtype=np.float32)  # Ensure float32
    wall_hts, _transf, _crs = common.load_raster(wall_hts_path.format(res=resolution), bbox=None)
    wall_asp, _transf, _crs = common.load_raster(wall_aspect_path.format(res=resolution), bbox=None)

    # Convert all loaded arrays to float32
    dsm = dsm.astype(np.float32)
    vegdsm = vegdsm.astype(np.float32)
    wall_hts = wall_hts.astype(np.float32)
    wall_asp = wall_asp.astype(np.float32)

    return dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp


# Calculate and print per-array right percentage
def pct(a, b, atol=0.001):
    if a is None or b is None:
        return float("nan")
    # Ensure shapes match before comparison
    if a.shape != b.shape:
        return f"Shape mismatch: {a.shape} vs {b.shape}"
    return 100.0 * np.isclose(a, b, atol=atol, rtol=0, equal_nan=True).sum() / a.size


def compare_svf_results(result_py, result_rust, key_map, atol=0.1):
    print("\n--- SVF Comparison ---")
    for py_key, rust_attr in key_map.items():
        py_val = result_py.get(py_key)
        rust_val = getattr(result_rust, rust_attr, None)
        match_pct = pct(py_val, rust_val, atol=atol)
        mean_diff = np.abs(py_val - rust_val).mean() if py_val is not None and rust_val is not None else float("nan")
        range_diff = np.abs(py_val - rust_val).max() if py_val is not None and rust_val is not None else float("nan")
        print(
            f"{py_key:<15} vs {rust_attr:<20} right: {match_pct:.2f} mean diff: {mean_diff:.3f} range: {range_diff:.2f}"
        )


def print_timing_stats(func_name, times):
    """Prints the min, max, and average timing statistics for a function."""
    if not times:
        print(f"\n{func_name}: No timing data available.")
        return
    min_time = min(times)
    max_time = max(times)
    avg_time = sum(times) / len(times)
    print(f"\n{func_name}: min={min_time:.3f}s, max={max_time:.3f}s, avg={avg_time:.3f}s")


def relative_speed(times_py, times_rust):
    """Calculates and prints how many times faster the Rust version is compared to Python."""
    rust_avg = sum(times_rust) / len(times_rust)
    py_avg = sum(times_py) / len(times_py)
    speedup_factor = py_avg / rust_avg
    print(f"\nRelative speed: Rust version is {speedup_factor:.2f} times faster than Python for given data.")


def plot_visual_residuals(
    py_array,
    rust_array,
    title_prefix="Visual",
    cmap="viridis",
    cmap_residuals="coolwarm",
    tick_fontsize="xx-small",
    colorbar_shrink=0.6,
):
    # check shape
    if py_array.shape != rust_array.shape:
        print(f"Error: Input arrays have different shapes: {py_array.shape} vs {rust_array.shape}")
        return

    residuals = rust_array - py_array

    # Determine the symmetric range for the residuals colormap
    max_abs_residual = np.abs(residuals).max()

    fig, axes = plt.subplots(3, 1, figsize=(6, 12))  # 3 rows, 1 column

    # Plot Array 1 (Python)
    im1 = axes[0].imshow(py_array, cmap=cmap)
    cbar1 = fig.colorbar(im1, ax=axes[0], shrink=colorbar_shrink)
    cbar1.ax.tick_params(labelsize=tick_fontsize)
    axes[0].set_title(f"{title_prefix} - Array 1 (Python)")
    axes[0].axis("off")

    # Plot Array 2 (Rust)
    im2 = axes[1].imshow(rust_array, cmap=cmap)
    cbar2 = fig.colorbar(im2, ax=axes[1], shrink=colorbar_shrink)
    cbar2.ax.tick_params(labelsize=tick_fontsize)
    axes[1].set_title(f"{title_prefix} - Array 2 (Rust)")
    axes[1].axis("off")

    # Plot Residuals with centered colormap
    im3 = axes[2].imshow(residuals, cmap=cmap_residuals, vmin=-max_abs_residual, vmax=max_abs_residual)
    cbar3 = fig.colorbar(im3, ax=axes[2], shrink=colorbar_shrink)
    cbar3.ax.tick_params(labelsize=tick_fontsize)
    axes[2].set_title(f"{title_prefix} - Residuals (Rust - Python)")
    axes[2].axis("off")

    plt.tight_layout()  # Adjust layout to prevent overlapping titles/labels
    plt.savefig(f"tests/rustalgos/{title_prefix.lower().replace(' ', '_')}_residuals.png", dpi=150)
