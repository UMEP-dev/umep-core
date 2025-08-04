import json
import zipfile
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd

from ... import common


@dataclass
class SolweigConfig:
    """Configuration class for SOLWEIG parameters."""

    output_dir: Optional[str] = None
    working_dir: Optional[str] = None
    dsm_path: Optional[str] = None
    svf_path: Optional[str] = None
    wh_path: Optional[str] = None
    wa_path: Optional[str] = None
    epw_path: Optional[str] = None
    epw_start_date: Optional[str] = None
    epw_end_date: Optional[str] = None
    epw_hours: Optional[list[int]] = None
    met_path: Optional[str] = None
    cdsm_path: Optional[str] = None
    tdsm_path: Optional[str] = None
    dem_path: Optional[str] = None
    lc_path: Optional[str] = None
    aniso_path: Optional[str] = None
    poi_path: Optional[str] = None
    poi_field: Optional[str] = None
    wall_path: Optional[str] = None
    woi_path: Optional[str] = None
    woi_field: Optional[str] = None
    only_global: bool = True
    use_veg_dem: bool = True
    conifer: bool = False
    person_cylinder: bool = True
    utc: bool = True
    use_landcover: bool = True
    use_dem_for_buildings: bool = False
    use_aniso: bool = False
    use_wall_scheme: bool = False
    wall_type: Optional[str] = "Brick"
    output_tmrt: bool = True
    output_kup: bool = True
    output_kdown: bool = True
    output_lup: bool = True
    output_ldown: bool = True
    output_sh: bool = True
    save_buildings: bool = True
    output_kdiff: bool = True
    output_tree_planter: bool = True
    wall_netcdf: bool = False

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
        if (self.met_path is None and self.epw_path is None) or (self.met_path and self.epw_path):
            raise ValueError("Provide either MET or EPW weather file.")
        if self.epw_path is not None:
            if self.epw_start_date is None or self.epw_end_date is None:
                raise ValueError("EPW start and end dates must be provided if EPW path is set.")
            if self.epw_hours is None:
                self.epw_hours = list(range(24))  # Default to all hours if not specified
        # Add more validation as needed


@dataclass
class WeatherData:
    """Class to handle weather data loading and processing."""

    DOY: np.ndarray
    hours: np.ndarray
    minu: np.ndarray
    Ta: np.ndarray
    RH: np.ndarray
    radG: np.ndarray
    radD: np.ndarray
    radI: np.ndarray
    P: np.ndarray
    Ws: np.ndarray

    def to_array(self) -> np.ndarray:
        """Convert weather data to a structured numpy array."""
        return np.array(
            [
                self.DOY,
                self.hours,
                self.minu,
                self.Ta,
                self.RH,
                self.radG,
                self.radD,
                self.radI,
                self.P,
                self.Ws,
            ]
        ).T


class SolweigRun:
    """Class to run the SOLWEIG algorithm with given configuration."""

    def __init__(self, config: SolweigConfig, params_json_path: str, qgis_env: bool):
        """Initialize the SOLWEIG runner with configuration and parameters."""
        # Load configuration
        self.config = config
        self.config.validate()
        self.feedback = None
        self.qgis_env = qgis_env
        # Initialize POI data
        self.poi_names: list[Any] = []
        self.poi_pixel_xys: list[tuple[float]] = []
        self.poi_results = []
        # Load parameters from JSON file
        with open(params_json_path) as f:
            params_dict = json.load(f)
            self.params = SimpleNamespace(**params_dict)

    def set_feedback(self, feedback: Any):
        """Set feedback object for progress updates."""
        self.feedback = feedback

    def load_epw_weather(self) -> WeatherData:
        """Load weather data from an EPW file."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_met_weather(self) -> WeatherData:
        """Load weather data from a MET file."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_poi_data(self, trf_arr: list[int]) -> None:
        """Load point of interest (POI) data from a file."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def save_poi_results(self, crs_wkt: str) -> None:
        """Save results for points of interest (POIs) to files."""
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
            vegdsm = None
            vegdsm2 = None

        # Land cover
        if self.config.use_landcover:
            lcgrid, _, _, _ = common.load_raster(self.config.lc_path, bbox=None)
        else:
            lcgrid = None

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
            svfWveg, _, _, _ = common.load_raster(self.config.working_dir + "/svfWaveg.tif", bbox=None)
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

        # weather data
        if self.qgis_env:
            weather_data = self.load_epw_weather()
        else:
            weather_data = self.load_met_weather(header_rows=1, delim=" ")

        location = {"longitude": lng, "latitude": lat, "altitude": alt}
        YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax = Solweig_2015a_metdata_noload(
            weather_data.to_array(), location, int(self.config.utc)
        )

        # POIs check
        if self.config.poi_path:
            self.load_poi_data(dsm_trf_arr)

        # Posture settings
        if self.params.Tmrt_params.Value.posture == "Standing":
            posture = self.params.Posture.Standing.Value
        else:
            posture = self.params.Posture.Sitting.Value

        Fside, Fup, height, Fcyl = posture.Fside, posture.Fup, posture.height, posture.Fcyl

        # Radiative surface influence
        first = np.round(height)
        if first == 0.0:
            first = 1.0
        second = np.round(height * 20.0)

        if self.config.use_veg_dem:
            # Conifer or deciduous
            if self.config.conifer:
                leafon = np.ones((1, weather_data.DOY.shape[0]))
            else:
                leafon = np.zeros((1, weather_data.DOY.shape[0]))
                if self.params.Tree_settings.Value.First_day_leaf > self.params.Tree_settings.Value.Last_day_leaf:
                    leaf_bool = (self.params.Tree_settings.Value.First_day_leaf < weather_data.DOY) | (
                        self.params.Tree_settings.Value.Last_day_leaf > weather_data.DOY
                    )
                else:
                    leaf_bool = (self.params.Tree_settings.Value.First_day_leaf < weather_data.DOY) & (
                        self.params.Tree_settings.Value.Last_day_leaf > weather_data.DOY
                    )
                leafon[0, leaf_bool] = 1

            # % Vegetation transmittivity of shortwave radiation
            psi = leafon * transVeg
            psi[leafon == 0] = 0.5
            # amaxvalue
            vegmax = vegdsm.max()
            amaxvalue = dsm_arr.max() - dsm_arr.min()
            amaxvalue = np.maximum(amaxvalue, vegmax)

            # Elevation vegdsms if buildingDEM includes ground heights
            vegdsm = vegdsm + dsm_arr
            vegdsm[vegdsm == dsm_arr] = 0
            vegdsm2 = vegdsm2 + dsm_arr
            vegdsm2[vegdsm2 == dsm_arr] = 0

            # % Bush separation
            bush = np.logical_not(vegdsm2 * vegdsm) * vegdsm

            svfbuveg = svf - (1.0 - svfveg) * (1.0 - transVeg)  # % major bug fixed 20141203
        else:
            psi = leafon * 0.0 + 1.0
            svfbuveg = svf
            bush = np.zeros([rows, cols])
            amaxvalue = 0

        # Initialization of maps
        Knight = np.zeros((rows, cols))
        Tgmap1 = np.zeros((rows, cols))
        Tgmap1E = np.zeros((rows, cols))
        Tgmap1S = np.zeros((rows, cols))
        Tgmap1W = np.zeros((rows, cols))
        Tgmap1N = np.zeros((rows, cols))

        # Create building boolean raster from either land cover or height rasters
        if not self.config.use_dem_for_buildings:
            buildings = np.copy(lcgrid)
            buildings[buildings == 7] = 1
            buildings[buildings == 6] = 1
            buildings[buildings == 5] = 1
            buildings[buildings == 4] = 1
            buildings[buildings == 3] = 1
            buildings[buildings == 2] = 0
        else:
            buildings = dsm_arr - dem
            buildings[buildings < 2.0] = 1.0
            buildings[buildings >= 2.0] = 0.0

        if self.config.save_buildings:
            common.save_raster(
                self.config.output_dir + "/buildings.tif",
                buildings,
                dsm_trf_arr,
                dsm_crs_wkt,
                dsm_nd_val,
            )

        # Import shadow matrices (Anisotropic sky)
        if self.config.use_aniso:
            data = np.load(self.config.aniso_path)
            shmat = data["shadowmat"]
            vegshmat = data["vegshadowmat"]
            vbshvegshmat = data["vbshmat"]
            if self.config.use_veg_dem:
                diffsh = np.zeros((rows, cols, shmat.shape[2]))
                for i in range(0, shmat.shape[2]):
                    diffsh[:, :, i] = shmat[:, :, i] - (1 - vegshmat[:, :, i]) * (
                        1 - transVeg
                    )  # changes in psi not implemented yet
            else:
                diffsh = shmat

            # Estimate number of patches based on shadow matrices
            if shmat.shape[2] == 145:
                patch_option = 1  # patch_option = 1 # 145 patches
            elif shmat.shape[2] == 153:
                patch_option = 2  # patch_option = 2 # 153 patches
            elif shmat.shape[2] == 306:
                patch_option = 3  # patch_option = 3 # 306 patches
            elif shmat.shape[2] == 612:
                patch_option = 4  # patch_option = 4 # 612 patches

            # asvf to calculate sunlit and shaded patches
            asvf = np.arccos(np.sqrt(svf))

            # Empty array for steradians
            steradians = np.zeros(shmat.shape[2])
        else:
            # anisotropic_sky = 0
            diffsh = None
            shmat = None
            vegshmat = None
            vbshvegshmat = None
            asvf = None
            patch_option = 0
            steradians = 0

        # % Ts parameterisation maps
        if self.config.use_landcover:
            # Get land cover properties for Tg wave (land cover scheme based on Bogren et al. 2000, explained in Lindberg et al., 2008 and Lindberg, Onomura & Grimmond, 2016)
            [TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall] = Tgmaps_v1(
                lcgrid.copy(), self.params
            )
        else:
            TgK = Knight + self.params.Ts_deg.Value.Cobble_stone_2014a
            Tstart = Knight - self.params.Tstart.Value.Cobble_stone_2014a
            TmaxLST = self.params.TmaxLST.Value.Cobble_stone_2014a
            alb_grid = Knight + self.params.Albedo.Effective.Value.Cobble_stone_2014a
            emis_grid = Knight + self.params.Emissivity.Value.Cobble_stone_2014a
            TgK_wall = self.params.Ts_deg.Value.Walls
            Tstart_wall = self.params.Tstart.Value.Walls
            TmaxLST_wall = self.params.TmaxLST.Value.Walls

        # Import data for wall temperature parameterization
        if self.config.use_wall_scheme:
            wallData = np.load(self.config.wall_path)
            voxelMaps = wallData["voxelId"]
            voxelTable = wallData["voxelTable"]

            # Get wall type
            if self.qgis_env:
                wall_type = self.config.wall_type
            else:
                wall_type_standalone = {"Brick_wall": "100", "Concrete_wall": "101", "Wood_wall": "102"}
                wall_type = wall_type_standalone[self.config.wall_type]

            # Get heights and aspects of walls
            wall_hts, _, _, _, _ = common.load_raster(self.config.wh_path, bbox=None)
            wall_dirs, _, _, _, _ = common.load_raster(self.config.wa_path, bbox=None)

            # Calculate timeStep
            first_timestep = (
                pd.to_datetime(YYYY[0][0], format="%Y")
                + pd.to_timedelta(weather_data.DOY[0] - 1, unit="d")
                + pd.to_timedelta(weather_data.hours[0], unit="h")
                + pd.to_timedelta(weather_data.minu[0], unit="m")
            )
            second_timestep = (
                pd.to_datetime(YYYY[0][1], format="%Y")
                + pd.to_timedelta(weather_data.DOY[1] - 1, unit="d")
                + pd.to_timedelta(weather_data.hours[1], unit="h")
                + pd.to_timedelta(weather_data.minu[1], unit="m")
            )
            timeStep = (second_timestep - first_timestep).seconds

            # Load voxelTable as Pandas DataFrame
            voxelTable, wall_dirs = load_walls(
                voxelTable,
                self.params,
                wall_type,
                wall_dirs,
                weather_data.Ta[0],
                timeStep,
                alb_grid,
                self.config.use_landcover,
                lcgrid,
                dsm_arr,
            )

            # Use wall of interest
            if self.config.woi_path:
                # TODO: Implement WOI logic for the new structure
                pass

            # Create pandas datetime object for NetCDF output
            if self.config.wall_netcdf:
                met_for_xarray = (
                    pd.to_datetime(YYYY[0][:], format="%Y")
                    + pd.to_timedelta(weather_data.DOY - 1, unit="d")
                    + pd.to_timedelta(weather_data.hours, unit="h")
                    + pd.to_timedelta(weather_data.minu, unit="m")
                )
        else:
            voxelMaps = None
            voxelTable = None
            timeStep = 0
            wall_hts = np.ones((rows, cols)) * 10.0
            wall_dirs = np.ones((rows, cols)) * 10.0
