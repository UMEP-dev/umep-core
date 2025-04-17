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
        f"test_shadowing_wallheight_23 (10 runs): min={min(times_py):.3f}s, max={max(times_py):.3f}s, avg={sum(times_py) / len(times_py):.3f}s"
    )

    def run_rust():
        shadowing.shadowingfunction_wallheight_25(
            dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp * np.pi / 180.0, None, None
        )

    times_rust = timeit.repeat(run_rust, number=1, repeat=repeats)
    print(
        f"test_shadowing_wallheight_23 (10 runs): min={min(times_rust):.3f}s, max={max(times_rust):.3f}s, avg={sum(times_rust) / len(times_rust):.3f}s"
    )

    vegsh, sh, vbshvegsh, wallsh, wallsun, wallshve, facesh, facesun = shadowingfunction_wallheight_23(
        dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp * np.pi / 180.0
    )

    result_rust = shadowing.shadowingfunction_wallheight_25(
        dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp * np.pi / 180.0, None, None
    )

    assert np.allclose(vegsh, result_rust.veg_shadow_map, atol=0.001)  # Updated property name
    assert np.allclose(sh, result_rust.bldg_shadow_map, atol=0.001)  # Updated property name
    assert np.allclose(vbshvegsh, result_rust.vbshvegsh, atol=0.001)
    assert np.allclose(wallsh, result_rust.wallsh, atol=0.001)
    assert np.allclose(wallsun, result_rust.wallsun, atol=0.001)
    assert np.allclose(wallshve, result_rust.wallshve, atol=0.001)
    assert np.allclose(facesh, result_rust.facesh, atol=0.001)
    assert np.allclose(facesun, result_rust.facesun, atol=0.001)


# test_shadowing_wallheight_23 (10 runs): min=1.039s, max=1.107s, avg=1.070s
# test_shadowing_wallheight_23 (10 runs): min=0.672s, max=0.710s, avg=0.689s


def test_shadowing():
    shadowing.test_shadowing()
