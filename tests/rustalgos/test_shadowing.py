import timeit

import numpy as np
from umep import common
from umep.rustalgos import shadowing
from umep.util.SEBESOLWEIGCommonFiles.shadowingfunction_wallheight_23 import shadowingfunction_wallheight_23


def make_test_arrays(
    dsm_path="demos/data/athens/DSM.tif",
    veg_dsm_path="demos/data/athens/CDSM.tif",
    wall_hts_path="demos/data/athens/walls/wall_hts.tif",
    wall_aspect_path="demos/data/athens/walls/wall_aspects.tif",
):
    dsm, _transf, _crs = common.load_raster(dsm_path, bbox=None)
    vegdsm, _transf, _crs = common.load_raster(veg_dsm_path, bbox=None)
    vegdsm2 = np.zeros(dsm.shape)
    azi = 45.0
    alt = 30.0
    scale = 1.0
    vegmax = vegdsm.max()
    amaxvalue = dsm.max() - dsm.min()
    amaxvalue = np.maximum(amaxvalue, vegmax)
    bush = np.zeros(dsm.shape)
    wall_hts, _transf, _crs = common.load_raster(wall_hts_path, bbox=None)
    wall_asp, _transf, _crs = common.load_raster(wall_aspect_path, bbox=None)

    return dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp


def test_shadowing_wallheight_23():
    repeats = 3

    dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp = make_test_arrays()

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

    # Calculate and print per-array right percentage
    def pct(a, b):
        return 100.0 * np.isclose(a, b, atol=0.001).sum() / a.size

    print(f"veg_shadow_map right: {pct(vegsh, result_rust.veg_shadow_map):.1f}%")
    print(f"bldg_shadow_map right: {pct(sh, result_rust.bldg_shadow_map):.1f}%")
    print(f"vbshvegsh right: {pct(vbshvegsh, result_rust.vbshvegsh):.1f}%")
    print(f"wallsh right: {pct(wallsh, result_rust.wallsh):.1f}%")
    print(f"wallsun right: {pct(wallsun, result_rust.wallsun):.1f}%")
    print(f"wallshve right: {pct(wallshve, result_rust.wallshve):.1f}%")
    print(f"facesh right: {pct(facesh, result_rust.facesh):.1f}%")
    print(f"facesun right: {pct(facesun, result_rust.facesun):.1f}%")


# test_shadowing_wallheight_23 (10 runs): min=1.039s, max=1.107s, avg=1.070s
# test_shadowing_wallheight_23 (10 runs): min=0.672s, max=0.710s, avg=0.689s

# v30
# test_shadowing_wallheight_23 (10 runs): min=1.196s, max=1.430s, avg=1.318s
# test_shadowing_wallheight_25 (10 runs): min=0.740s, max=0.835s, avg=0.783s

# v31
# test_shadowing_wallheight_23 (10 runs): min=1.112s, max=1.214s, avg=1.165s
# test_shadowing_wallheight_25 (10 runs): min=0.736s, max=0.807s, avg=0.767s

# v32
# test_shadowing_wallheight_23 (10 runs): min=1.150s, max=1.302s, avg=1.246s
# test_shadowing_wallheight_25 (10 runs): min=0.320s, max=0.326s, avg=0.323s

# v33
# test_shadowing_wallheight_23 (10 runs): min=1.236s, max=1.371s, avg=1.286s
# test_shadowing_wallheight_25 (10 runs): min=0.318s, max=0.341s, avg=0.333s

# v34
# test_shadowing_wallheight_23 (10 runs): min=1.182s, max=1.328s, avg=1.231s
# test_shadowing_wallheight_25 (10 runs): min=0.314s, max=0.322s, avg=0.319s

# v36 - x4 resolution
# test_shadowing_wallheight_23 (10 runs): min=4.336s, max=4.695s, avg=4.530s
# test_shadowing_wallheight_25 (10 runs): min=1.131s, max=1.225s, avg=1.164s


def test_shadowing():
    shadowing.test_shadowing()
