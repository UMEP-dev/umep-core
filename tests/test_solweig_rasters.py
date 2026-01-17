from pathlib import Path
from typing import Optional

import numpy as np

from umep.class_configs import SolweigConfig
from umep.functions.SOLWEIGpython.solweig_runner_core import SolweigRunCore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "demos/data/athens/configsolweig.ini"
PARAMS_PATH = PROJECT_ROOT / "demos/data/athens/parametersforsolweig.json"


def _prepare_temp_config(tmp_path: Path) -> Path:
    """Return a config file that writes outputs into a temp dir."""
    config = SolweigConfig()
    config.from_file(str(CONFIG_PATH))

    output_dir = tmp_path / "output"
    working_dir = tmp_path / "working"

    config.output_dir = str(output_dir)
    config.working_dir = str(working_dir)

    path_overrides = {
        "dsm_path": PROJECT_ROOT / "demos/data/athens/DSM.tif",
        "cdsm_path": PROJECT_ROOT / "temp/athens/CDSM.tif",
        "dem_path": PROJECT_ROOT / "demos/data/athens/DEM.tif",
        "wh_path": PROJECT_ROOT / "temp/athens/walls/wall_hts.tif",
        "wa_path": PROJECT_ROOT / "temp/athens/walls/wall_aspects.tif",
        "svf_path": PROJECT_ROOT / "temp/athens/svf/svfs.zip",
        "aniso_path": PROJECT_ROOT / "temp/athens/svf/shadowmats.npz",
        "epw_path": PROJECT_ROOT / "demos/data/athens/athens_2023.epw",
    }
    for attr, path in path_overrides.items():
        setattr(config, attr, str(path))

    tmp_config = tmp_path / "configsolweig.ini"
    config.to_file(str(tmp_config))
    return tmp_config


def _assert_float32(array: Optional[np.ndarray], name: str):
    if array is not None:
        assert array.dtype == np.float32, f"{name} dtype was {array.dtype}, expected float32"


def test_athens_runner_rasters_are_float32(tmp_path):
    tmp_config = _prepare_temp_config(tmp_path)
    runner = SolweigRunCore(str(tmp_config), str(PARAMS_PATH))
    raster_data = runner.raster_data

    _assert_float32(raster_data.dsm, "dsm")
    _assert_float32(raster_data.wallheight, "wallheight")
    _assert_float32(raster_data.wallaspect, "wallaspect")
    _assert_float32(raster_data.dem, "dem")
    _assert_float32(raster_data.cdsm, "cdsm")
    _assert_float32(raster_data.tdsm, "tdsm")
    _assert_float32(raster_data.bush, "bush")
    _assert_float32(raster_data.svfbuveg, "svfbuveg")
    _assert_float32(raster_data.buildings, "buildings")

    runner.test_hook()
