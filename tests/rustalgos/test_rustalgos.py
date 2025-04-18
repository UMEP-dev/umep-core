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


"""
v69 f32 change
test_shadowing_wallheight_23: min=1.225s, max=1.425s, avg=1.313s
test_shadowing_wallheight_25: min=0.107s, max=0.125s, avg=0.115s

--- SVF Comparison ---
veg_shadow_map  vs veg_shadow_map       right: 99.73175542406311 mean diff: 0.003
bldg_shadow_map vs bldg_shadow_map      right: 100.0 mean diff: 0.000
vbshvegsh       vs vbshvegsh            right: 99.59196252465483 mean diff: 0.004
wallsh          vs wallsh               right: 100.0 mean diff: 0.000
wallsun         vs wallsun              right: 100.0 mean diff: 0.000
wallshve        vs wallshve             right: 100.0 mean diff: 0.000
facesh          vs facesh               right: 100.0 mean diff: 0.000
facesun         vs facesun              right: 100.0 mean diff: 0.000
"""


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

    # Run Python version with float64
    result_py = svfForProcessing153(dsm, vegdsm, vegdsm2, scale, 1)
    # Run Rust version with float32
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


"""
# v43
# svfForProcessing153: min=34.552s, max=34.552s, avg=34.552s
# calculate_svf_153: min=10.588s, max=10.588s, avg=10.588s

v69 f32 change

svfForProcessing153: min=34.866s, max=34.866s, avg=34.866s
calculate_svf_153: min=1.851s, max=1.851s, avg=1.851s

--- SVF Comparison ---
svf             vs svf                  right: 99.98076923076923 mean diff: 0.030
svfE            vs svf_east             right: 99.61834319526628 mean diff: 0.034
svfS            vs svf_south            right: 94.60996055226825 mean diff: 0.039
svfW            vs svf_west             right: 87.69970414201184 mean diff: 0.058
svfN            vs svf_north            right: 98.4354043392505 mean diff: 0.029
svfveg          vs svf_veg              right: 98.36390532544378 mean diff: 0.044
svfEveg         vs svf_veg_east         right: 96.99802761341223 mean diff: 0.045
svfSveg         vs svf_veg_south        right: 95.95660749506904 mean diff: 0.022
svfWveg         vs svf_veg_west         right: 94.6198224852071 mean diff: 0.043
svfNveg         vs svf_veg_north        right: 96.37721893491124 mean diff: 0.027
svfaveg         vs svf_aniso_veg        right: 98.5103550295858 mean diff: 0.041
svfEaveg        vs svf_aniso_veg_east   right: 97.79339250493096 mean diff: 0.043
svfSaveg        vs svf_aniso_veg_south  right: 95.02810650887574 mean diff: 0.025
svfWaveg        vs svf_aniso_veg_west   right: 92.61193293885601 mean diff: 0.047
svfNaveg        vs svf_aniso_veg_north  right: 97.0645956607495 mean diff: 0.026
shmat           vs shadow_matrix        right: N/A mean diff: nan
vegshmat        vs veg_shadow_matrix    right: N/A mean diff: nan
vbshvegshmat    vs vbshvegsh_matrix     right: N/A mean diff: nan
"""
