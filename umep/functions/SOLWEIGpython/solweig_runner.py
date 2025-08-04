# common imports


import json
import zipfile
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from ... import common


class SolweigConfig:
    def __init__(
        self,
    ):
        pass

    def set_params(
        self,
        output_dir: str,
        working_dir: str,
        dsm_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        met_path: Optional[str] = None,
        epw_path: Optional[str] = None,
        cdsm_path: Optional[str] = None,
        tdsm_path: Optional[str] = None,
        dem_path: Optional[str] = None,
        lc_path: Optional[str] = None,
        wh_path: Optional[str] = None,
        wa_path: Optional[str] = None,
        svf_path: Optional[str] = None,
        aniso_path: Optional[str] = None,
        poi_path: Optional[str] = None,
        poi_field: Optional[str] = None,
        wall_path: Optional[str] = None,
        woi_path: Optional[str] = None,
        woi_field: Optional[str] = None,
        only_global: bool = True,
        use_veg_dem: bool = True,
        conifer: bool = False,
        person_cylinder: bool = True,
        utc: bool = True,
        use_epw_file: bool = False,
        use_landcover: bool = True,
        use_dem_for_buildings: bool = False,
        use_aniso: bool = False,
        use_wall_scheme: bool = False,
        wall_type: Optional[str] = "Brick",
        output_tmrt: bool = True,
        output_kup: bool = True,
        output_kdown: bool = True,
        output_lup: bool = True,
        output_ldown: bool = True,
        output_sh: bool = True,
        save_buildings: bool = True,
        output_kdiff: bool = True,
        output_tree_planter: bool = True,
        wall_netcdf: bool = False,
    ):
        # TODO can we change names?
        self.params = None
        self.output_dir = output_dir
        self.working_dir = working_dir
        self.dsm_path = dsm_path
        self.start_date = start_date
        self.end_date = end_date
        self.met_path = met_path
        self.epw_path = epw_path
        self.cdsm_path = cdsm_path
        self.tdsm_path = tdsm_path
        self.dem_path = dem_path
        self.lc_path = lc_path
        self.wh_path = wh_path
        self.wa_path = wa_path
        self.svf_path = svf_path
        self.aniso_path = aniso_path
        self.poi_path = poi_path
        self.poi_field = poi_field
        self.wall_path = wall_path
        self.woi_path = woi_path
        self.woi_field = woi_field
        self.only_global = only_global
        self.use_veg_dem = use_veg_dem
        self.conifer = conifer
        self.person_cylinder = person_cylinder
        self.utc = utc
        self.use_epw_file = use_epw_file
        self.use_landcover = use_landcover
        self.use_dem_for_buildings = use_dem_for_buildings
        self.use_aniso = use_aniso
        self.use_wall_scheme = use_wall_scheme
        self.wall_type = wall_type
        self.output_tmrt = output_tmrt
        self.output_kup = output_kup
        self.output_kdown = output_kdown
        self.output_lup = output_lup
        self.output_ldown = output_ldown
        self.output_sh = output_sh
        self.save_buildings = save_buildings
        self.output_kdiff = output_kdiff
        self.output_tree_planter = output_tree_planter
        self.wall_netcdf = wall_netcdf

    def to_file(self, file_path: str):
        """Save configuration to a file."""
        with open(file_path, "w") as f:
            for key in type(self).__annotations__:
                value = getattr(self, key)
                if value is None:
                    value = ""  # Default to empty string if None
                if isinstance(self.__annotations__[key], bool):
                    f.write(f"{key}={int(value)}\n")
                else:
                    f.write(f"{key}={value}\n")

    def from_file(self, config_path_str: str):
        """Load configuration from a file."""
        config_path = common.check_path(config_path_str)
        with open(config_path) as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    if key in type(self).__annotations__:
                        if value.strip() == "":
                            value = None
                        if type(self).__annotations__[key] == bool:
                            setattr(self, key, value == "1" or value.lower() == "true")
                        else:
                            setattr(self, key, value)
                    else:
                        print(f"Unknown key in config: {key}")

    def validate(self):
        """Validate configuration parameters."""
        if not self.output_dir:
            raise ValueError("Output directory must be set.")
        if not self.working_dir:
            raise ValueError("Working directory must be set.")
        if not self.dsm_path:
            raise ValueError("DSM path must be set.")
        if self.params is None:
            raise ValueError("Parameters must be loaded.")
        # Add more validation as needed


class SolweigRun:
    def __init__(self, config: SolweigConfig, params_json_path: str, qgis_env: bool):
        self.config = config
        self.config.validate()
        self.feedback = None
        self.qgis_env = qgis_env

        with open(params_json_path) as f:
            params_dict = json.load(f)
            self.params = SimpleNamespace(**params_dict)

    def set_feedback(self, feedback: Any):
        """Set feedback object for progress updates."""
        self.feedback = feedback

    def load_poi_xys(self, trf_arr: list[int]) -> tuple[Any, Any]:
        """Load points of interest (POIs) from a file."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def run(self) -> None:
        """Run the SOLWEIG algorithm."""
        # Load DSM
        dsm_arr, dsm_trf_arr, dsm_crs_wkt, dsm_nd_val = common.load_raster(self.config.dsm_path, bbox=None)
        scale = 1 / dsm_trf_arr[1]
        left_x = dsm_trf_arr[0]
        top_y = dsm_trf_arr[3]
        lng, lat = common.xy_to_lnglat(dsm_crs_wkt, left_x, top_y)
        rows = dsm_arr.shape[0]
        cols = dsm_arr.shape[1]

        dsm_arr[dsm_arr == dsm_nd_val] = 0.0
        if dsm_arr.min() < 0:
            dsmraise = np.abs(dsm_arr.min())
            dsm_arr = dsm_arr + dsmraise
        else:
            dsmraise = 0

        alt = np.median(dsm_arr)
        if alt < 0:
            alt = 3

        # Vegetation
        transVeg = self.params.Tree_settings.Value.Transmissivity
        trunkratio = self.params.Tree_settings.Value.Trunk_ratio
        if self.config.use_veg_dem:
            vegdsm, _, _, _ = common.load_raster(self.config.cdsm_path, bbox=None)
            if self.config.tdsm_path:
                vegdsm2, _, _, _ = common.load_raster(self.config.tdsm_path, bbox=None)
            else:
                vegdsm2 = vegdsm * trunkratio
        else:
            vegdsm = 0
            vegdsm2 = 0

        # Land cover
        if self.config.use_landcover:
            lcgrid, _, _, _ = common.load_raster(self.config.lc_path, bbox=None)
        else:
            lcgrid = 0

        # DEM for buildings
        if self.config.use_dem_for_buildings:
            dem, _, _, dem_nd_val = common.load_raster(self.config.dem_path, bbox=None)
            dem[dem == dem_nd_val] = 0.0
            if dem.min() < 0:
                demraise = np.abs(dem.min())
                dem = dem + demraise
            else:
                demraise = 0

        # SVF
        with zipfile.ZipFile(self.config.svf_path, "r") as zip_ref:
            zip_ref.extractall(self.config.working_dir)

        svf, _, _, _ = common.load_raster(self.config.working_dir + "/svf.tif", bbox=None)
        svfN, _, _, _ = common.load_raster(self.config.working_dir + "/svfN.tif", bbox=None)
        svfS, _, _, _ = common.load_raster(self.config.working_dir + "/svfS.tif", bbox=None)
        svfE, _, _, _ = common.load_raster(self.config.working_dir + "/svfE.tif", bbox=None)
        svfW, _, _, _ = common.load_raster(self.config.working_dir + "/svfW.tif", bbox=None)

        if self.config.use_veg_dem:
            svfveg, _, _, _ = common.load_raster(self.config.working_dir + "/svfveg.tif", bbox=None)
            svfNveg, _, _, _ = common.load_raster(self.config.working_dir + "/svfNveg.tif", bbox=None)
            svfSveg, _, _, _ = common.load_raster(self.config.working_dir + "/svfSveg.tif", bbox=None)
            svfEveg, _, _, _ = common.load_raster(self.config.working_dir + "/svfEveg.tif", bbox=None)
            svfWveg, _, _, _ = common.load_raster(self.config.working_dir + "/svfWveg.tif", bbox=None)
            svfaveg, _, _, _ = common.load_raster(self.config.working_dir + "/svfaveg.tif", bbox=None)
            svfNaveg, _, _, _ = common.load_raster(self.config.working_dir + "/svfNaveg.tif", bbox=None)
            svfSaveg, _, _, _ = common.load_raster(self.config.working_dir + "/svfSaveg.tif", bbox=None)
            svfEaveg, _, _, _ = common.load_raster(self.config.working_dir + "/svfEaveg.tif", bbox=None)
            svfWaveg, _, _, _ = common.load_raster(self.config.working_dir + "/svfWaveg.tif", bbox=None)
        else:
            svfveg = np.ones((rows, cols))
            svfNveg = np.ones((rows, cols))
            svfSveg = np.ones((rows, cols))
            svfEveg = np.ones((rows, cols))
            svfWveg = np.ones((rows, cols))
            svfaveg = np.ones((rows, cols))
            svfNaveg = np.ones((rows, cols))
            svfSaveg = np.ones((rows, cols))
            svfEaveg = np.ones((rows, cols))
            svfWaveg = np.ones((rows, cols))

        tmp = svf + svfveg - 1.0
        tmp[tmp < 0.0] = 0.0
        svfalfa = np.arcsin(np.exp(np.log(1.0 - tmp) / 2.0))

        wallheight, _, _, _ = common.load_raster(self.config.wh_path, bbox=None)
        wallaspect, _, _, _ = common.load_raster(self.config.wa_path, bbox=None)

        # Metdata
        metdata = np.loadtxt(self.config.met_path, skiprows=1, delimiter=" ")
        location = {"longitude": lng, "latitude": lat, "altitude": alt}
        YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax = Solweig_2015a_metdata_noload(
            metdata, location, int(self.config.utc)
        )

        DOY, hours, minu, Ta, RH, radG, radD, radI, P, Ws = (
            metdata[:, 1],
            metdata[:, 2],
            metdata[:, 3],
            metdata[:, 11],
            metdata[:, 10],
            metdata[:, 14],
            metdata[:, 21],
            metdata[:, 22],
            metdata[:, 12],
            metdata[:, 9],
        )

        # POIs check
        poisxy = None
        if self.config.poi_path:
            self.load_pois(self.config.poi_path, self.config.poi_field)

            for k in range(poisxy.shape[0]):
                data_out = self.config.output_dir + "/POI_" + str(poiname[k]) + ".txt"
                np.savetxt(data_out, [], delimiter=" ", header=header, comments="")

            numformat = "%d %d %d %d %.5f " + "%.2f " * 36
            sensorheight = self.params.Wind_Height.Value.magl
            age = self.params.PET_settings.Value.Age
            mbody = self.params.PET_settings.Value.Weight
            ht = self.params.PET_settings.Value.Height
            clo = self.params.PET_settings.Value.clo
            activity = self.params.PET_settings.Value.Activity
            sex = self.params.PET_settings.Value.Sex

        # Posture settings
        if self.params.Tmrt_params.Value.posture == "Standing":
            posture = self.params.Posture.Standing.Value
            pos = 1
        else:
            posture = self.params.Posture.Sitting.Value
            pos = 0
        Fside, Fup, height, Fcyl = posture.Fside, posture.Fup, posture.height, posture.Fcyl

        # Radiative surface influence
        first = np.round(height)
        if first == 0.0:
            first = 1.0
        second = np.round(height * 20.0)

        # ... The rest of the logic from solweig_run should be refactored here ...
        # This includes the main calculation loop, output generation, etc.
        # All `configDict` should be replaced with `self.config`
        # All `param` should be replaced with `self.params`
        # All `standAlone` checks should be replaced with `self.qgis_env` checks

        # Main loop placeholder
        tmrtplot = np.zeros((rows, cols))
        if not self.qgis_env:
            from tqdm import tqdm

            progress = tqdm(total=Ta.__len__())

        for i in np.arange(0, Ta.__len__()):
            if self.feedback:
                self.feedback.setProgress(int(i * (100.0 / Ta.__len__())))
                if self.feedback.isCanceled():
                    break
            elif not self.qgis_env:
                progress.update(1)

            # ... All calculations from the original loop go here ...

        if not self.qgis_env:
            progress.close()

        # Final averaging and saving
        tmrtplot = tmrtplot / Ta.__len__()
        common.save_raster(
            self.config.output_dir + "/Tmrt_average.tif",
            tmrtplot,
            dsm_trf_arr,
            dsm_crs_wkt,
            dsm_nd_val,
        )
