# %%
from pathlib import Path

from umep import (
    skyviewfactor_algorithm,
    wall_heightaspect_algorithm,
)
from umep.functions.SOLWEIGpython import Solweig_run, solweig_runner_core

# %%
bbox = [476070, 4203550, 477110, 4204330]
working_folder = "temp/goteborg"
pixel_resolution = 1  # metres
working_crs = 3007

working_path = Path(working_folder).absolute()
working_path.mkdir(parents=True, exist_ok=True)
working_path_str = str(working_path)

# input files for computing
dsm_path = "demos/data/Goteborg_SWEREF99_1200/DSM_KRbig.tif"
cdsm_path = "demos/data/Goteborg_SWEREF99_1200/CDSM_KRbig.tif"
config_path = "demos/data/Goteborg_SWEREF99_1200/configsolweig_alt.ini"
params_path = "demos/data/Goteborg_SWEREF99_1200/parametersforsolweig.json"

# %%
# wall info for SOLWEIG (height and aspect)
wall_heightaspect_algorithm.generate_wall_hts(
    dsm_path=dsm_path,
    bbox=None,
    out_dir=working_path_str + "/walls",
)

# %%
# skyview factor for SOLWEIG
skyviewfactor_algorithm.generate_svf(
    dsm_path=dsm_path,
    bbox=None,
    out_dir=working_path_str + "/svf",
    cdsm_path=cdsm_path,
    trans_veg_perc=3,
    trunk_ratio_perc=25,
)

# %%
SWC = solweig_runner_core.SolweigRunCore(
    config_path_str=config_path,
    params_json_path=params_path,
)
SWC.run()

# %%
# For comparison
Solweig_run.solweig_run("demos/data/Goteborg_SWEREF99_1200/config_solweig_old_fmt.ini", None)

# %%
