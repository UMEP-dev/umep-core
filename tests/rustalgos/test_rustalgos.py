import timeit

import numpy as np
from umep import common
from umep.functions.svf_functions import svfForProcessing153
from umep.rustalgos import shadowing, skyview
from umep.util.SEBESOLWEIGCommonFiles.shadowingfunction_wallheight_23 import shadowingfunction_wallheight_23


def make_test_arrays(
    resolution,
    dsm_path="demos/data/athens/DSM_{res}m.tif",
    veg_dsm_path="demos/data/athens/CDSM_{res}m.tif",
    wall_hts_path="demos/data/athens/walls_{res}m/wall_hts.tif",
    wall_aspect_path="demos/data/athens/walls_{res}m/wall_aspects.tif",
):
    dsm, dsm_transf, _crs = common.load_raster(dsm_path.format(res=resolution), bbox=None)
    vegdsm, _transf, _crs = common.load_raster(veg_dsm_path.format(res=resolution), bbox=None)
    vegdsm2 = np.zeros(dsm.shape)
    azi = 45.0
    alt = 30.0
    scale = 1 / dsm_transf.a
    vegmax = vegdsm.max()
    amaxvalue = dsm.max() - dsm.min()
    amaxvalue = np.maximum(amaxvalue, vegmax)
    bush = np.zeros(dsm.shape)
    wall_hts, _transf, _crs = common.load_raster(wall_hts_path.format(res=resolution), bbox=None)
    wall_asp, _transf, _crs = common.load_raster(wall_aspect_path.format(res=resolution), bbox=None)

    return dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp


# Calculate and print per-array right percentage
def pct(a, b, atol=0.001):
    if a is None or b is None:
        return "N/A"
    # Ensure shapes match before comparison
    if a.shape != b.shape:
        return f"Shape mismatch: {a.shape} vs {b.shape}"
    return 100.0 * np.isclose(a, b, atol=atol, rtol=0, equal_nan=True).sum() / a.size


def compare_svf_results(result_py, result_rust, key_map, atol=0.1):
    print("\n--- SVF Comparison ---")
    all_match = True
    for py_key, rust_attr in key_map.items():
        py_val = result_py.get(py_key)
        rust_val = getattr(result_rust, rust_attr, None)
        match_pct = pct(py_val, rust_val, atol=atol)
        mean_diff = np.abs(py_val - rust_val).mean() if py_val is not None and rust_val is not None else float("nan")
        print(f"{py_key:<15} vs {rust_attr:<20} right: {match_pct} mean diff: {mean_diff:.3f}")
        if isinstance(match_pct, (float, int)) and match_pct < 100.0:
            all_match = False
    assert all_match, "Not all SVF arrays matched perfectly."


def test_shadowing():
    repeats = 3
    dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp = make_test_arrays(resolution=1)

    def run_py():
        shadowingfunction_wallheight_23(
            dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp * np.pi / 180.0
        )

    times_py = timeit.repeat(run_py, number=1, repeat=repeats)
    print(
        f"test_shadowing_wallheight_23: min={min(times_py):.3f}s, max={max(times_py):.3f}s, avg={sum(times_py) / len(times_py):.3f}s"
    )

    def run_rust():
        shadowing.shadowingfunction_wallheight_25(
            dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp * np.pi / 180.0, None, None
        )

    times_rust = timeit.repeat(run_rust, number=1, repeat=repeats)
    print(
        f"test_shadowing_wallheight_25: min={min(times_rust):.3f}s, max={max(times_rust):.3f}s, avg={sum(times_rust) / len(times_rust):.3f}s"
    )

    vegsh, sh, vbshvegsh, wallsh, wallsun, wallshve, facesh, facesun = shadowingfunction_wallheight_23(
        dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp * np.pi / 180.0
    )
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
    compare_svf_results(result_py, result_rust, key_map, atol=0.1)


# v40
# test_shadowing_wallheight_23: min=1.150s, max=1.518s, avg=1.339s
# test_shadowing_wallheight_25: min=0.323s, max=0.351s, avg=0.333s


def test_svf():
    repeats = 1

    dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp = make_test_arrays(resolution=2)

    def run_py():
        svfForProcessing153(dsm, vegdsm, vegdsm2, scale, 1)

    times_py = timeit.repeat(run_py, number=1, repeat=repeats)
    print(
        f"svfForProcessing153: min={min(times_py):.3f}s, max={max(times_py):.3f}s, avg={sum(times_py) / len(times_py):.3f}s"
    )

    def run_rust():
        skyview.calculate_svf_153(dsm, vegdsm, vegdsm2, scale, True)

    times_rust = timeit.repeat(run_rust, number=1, repeat=repeats)
    print(
        f"calculate_svf_153: min={min(times_rust):.3f}s, max={max(times_rust):.3f}s, avg={sum(times_rust) / len(times_rust):.3f}s"
    )

    result_py = svfForProcessing153(dsm, vegdsm, vegdsm2, scale, 1)
    result_rust = skyview.calculate_svf_153(dsm, vegdsm, vegdsm2, scale, True)

    # Map Python keys to Rust attribute names
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

    compare_svf_results(result_py, result_rust, key_map, atol=0.1)


# v40
# svfForProcessing153: min=33.068s, max=33.068s, avg=33.068s
# calculate_svf_153: min=13.082s, max=13.082s, avg=13.082s

# v41
# svfForProcessing153: min=33.686s, max=33.686s, avg=33.686s
# calculate_svf_153: min=11.272s, max=11.272s, avg=11.272s

# v43
# svfForProcessing153: min=34.552s, max=34.552s, avg=34.552s
# calculate_svf_153: min=10.588s, max=10.588s, avg=10.588s
