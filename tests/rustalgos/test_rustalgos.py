import timeit

import numpy as np
from umep import common
from umep.functions.svf_functions import svfForProcessing153
from umep.rustalgos import shadowing, skyview
from umep.util.SEBESOLWEIGCommonFiles.shadowingfunction_wallheight_23 import shadowingfunction_wallheight_23


def make_test_arrays(
    dsm_path="demos/data/athens/DSM.tif",
    veg_dsm_path="demos/data/athens/CDSM.tif",
    wall_hts_path="demos/data/athens/walls/wall_hts.tif",
    wall_aspect_path="demos/data/athens/walls/wall_aspects.tif",
):
    dsm, dsm_transf, _crs = common.load_raster(dsm_path, bbox=None)
    vegdsm, _transf, _crs = common.load_raster(veg_dsm_path, bbox=None)
    vegdsm2 = np.zeros(dsm.shape)
    azi = 45.0
    alt = 30.0
    scale = 1 / dsm_transf.a
    vegmax = vegdsm.max()
    amaxvalue = dsm.max() - dsm.min()
    amaxvalue = np.maximum(amaxvalue, vegmax)
    bush = np.zeros(dsm.shape)
    wall_hts, _transf, _crs = common.load_raster(wall_hts_path, bbox=None)
    wall_asp, _transf, _crs = common.load_raster(wall_aspect_path, bbox=None)

    return dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp


def test_shadowing():
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


# v38
# test_shadowing_wallheight_23: min=1.269s, max=1.574s, avg=1.379s
# test_shadowing_wallheight_25: min=0.338s, max=0.348s, avg=0.343s


def test_svf():
    repeats = 1

    dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp = make_test_arrays()

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


# v38
# svfForProcessing153: min=304.075s, max=304.075s, avg=304.075s
# calculate_svf_153: min=13.345s, max=13.345s, avg=13.345s
