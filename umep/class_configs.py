import datetime
import zipfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Attempt to import pandas for DataFrame support
# If pandas is not available in QGIS... set it to None
try:
    import pandas as pd
except ImportError:
    pd = None

from . import common
from .functions import wallalgorithms as wa
from .functions.SOLWEIGpython.wall_surface_temperature import load_walls
from .util.SEBESOLWEIGCommonFiles import sun_position as sp


class TgMaps:
    """
    Get land cover properties for Tg wave (land cover scheme based on Bogren et al. 2000,
    explained in Lindberg et al., 2008 and Lindberg, Onomura & Grimmond, 2016)
    """

    TgK: np.ndarray
    Tstart: np.ndarray
    alb_grid: np.ndarray
    emis_grid: np.ndarray
    TgK_wall: float
    Tstart_wall: float
    # TODO: check if this should be a float - original conflict in use_landcover option
    TmaxLST: np.ndarray | float
    TmaxLST_wall: float
    # maps
    Knight: np.ndarray
    Tgmap1: np.ndarray
    Tgmap1E: np.ndarray
    Tgmap1S: np.ndarray
    Tgmap1W: np.ndarray
    Tgmap1N: np.ndarray
    TgOut1: np.ndarray

    def __init__(self, use_landcover: bool, lc_grid: np.ndarray | None, model_params, rows: int, cols: int):
        """
        This is a vectorized version that avoids looping over pixels.
        """
        # Initialization of maps
        self.Knight = np.zeros((rows, cols))
        self.Tgmap1 = np.zeros((rows, cols))
        self.Tgmap1E = np.zeros((rows, cols))
        self.Tgmap1S = np.zeros((rows, cols))
        self.Tgmap1W = np.zeros((rows, cols))
        self.Tgmap1N = np.zeros((rows, cols))
        self.TgOut1 = np.zeros((rows, cols))

        # Set up the Tg maps based on whether land cover is used
        if not use_landcover:
            self.TgK = self.Knight + model_params.Ts_deg.Value.Cobble_stone_2014a
            self.Tstart = self.Knight - model_params.Tstart.Value.Cobble_stone_2014a
            self.alb_grid = self.Knight + model_params.Albedo.Effective.Value.Cobble_stone_2014a
            self.emis_grid = self.Knight + model_params.Emissivity.Value.Cobble_stone_2014a
            self.TmaxLST = model_params.TmaxLST.Value.Cobble_stone_2014a  # Assuming this is a float
            self.TgK_wall = model_params.Ts_deg.Value.Walls
            self.Tstart_wall = model_params.Tstart.Value.Walls
            self.TmaxLST_wall = model_params.TmaxLST.Value.Walls
        else:
            # Copy land cover grid
            lc_grid = np.copy(lc_grid)
            # Sanitize
            lc_grid[lc_grid >= 100] = 2
            # Get unique land cover IDs and filter them
            unique_ids = np.unique(lc_grid)
            valid_ids = unique_ids[unique_ids <= 7].astype(int)
            # Initialize output grids by copying the original land cover grid
            self.TgK = np.copy(lc_grid)
            self.Tstart = np.copy(lc_grid)
            self.alb_grid = np.copy(lc_grid)
            self.emis_grid = np.copy(lc_grid)
            self.TmaxLST = np.copy(lc_grid)
            # Create mapping dictionaries from land cover ID to parameter values
            id_to_name = {i: getattr(model_params.Names.Value, str(i)) for i in valid_ids}
            name_to_tstart = {name: getattr(model_params.Tstart.Value, name) for name in id_to_name.values()}
            name_to_albedo = {name: getattr(model_params.Albedo.Effective.Value, name) for name in id_to_name.values()}
            name_to_emissivity = {name: getattr(model_params.Emissivity.Value, name) for name in id_to_name.values()}
            name_to_tmaxlst = {name: getattr(model_params.TmaxLST.Value, name) for name in id_to_name.values()}
            name_to_tsdeg = {name: getattr(model_params.Ts_deg.Value, name) for name in id_to_name.values()}
            # Perform replacements for each valid land cover ID
            for i in valid_ids:
                mask = lc_grid == i
                name = id_to_name[i]
                self.Tstart[mask] = name_to_tstart[name]
                self.alb_grid[mask] = name_to_albedo[name]
                self.emis_grid[mask] = name_to_emissivity[name]
                self.TmaxLST[mask] = name_to_tmaxlst[name]
                self.TgK[mask] = name_to_tsdeg[name]
            # Get wall-specific parameters
            self.TgK_wall = getattr(model_params.Ts_deg.Value, "Walls", None)
            self.Tstart_wall = getattr(model_params.Tstart.Value, "Walls", None)
            self.TmaxLST_wall = getattr(model_params.TmaxLST.Value, "Walls", None)


@dataclass
class SolweigConfig:
    """Configuration class for SOLWEIG parameters."""

    output_dir: Optional[str] = None
    working_dir: Optional[str] = None
    dsm_path: Optional[str] = None
    svf_path: Optional[str] = None
    wh_path: Optional[str] = None
    wa_path: Optional[str] = None
    use_epw_file: bool = False
    epw_path: Optional[str] = None
    epw_start_date: Optional[str | list[int]] = None
    epw_end_date: Optional[str | list[int]] = None
    epw_hours: Optional[str | list[int]] = None
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
        self.output_dir = str(common.check_path(self.output_dir, make_dir=True))
        if not self.working_dir:
            raise ValueError("Working directory must be set.")
        self.working_dir = str(common.check_path(self.working_dir, make_dir=True))
        if not self.dsm_path:
            raise ValueError("DSM path must be set.")
        self.utc = bool(self.utc)
        if (self.met_path is None and self.epw_path is None) or (self.met_path and self.epw_path):
            raise ValueError("Provide either MET or EPW weather file.")
        if self.epw_path is not None:
            if self.epw_start_date is None or self.epw_end_date is None:
                raise ValueError("EPW start and end dates must be provided if EPW path is set.")
            # year,month,day,hour
            # parse the start and end dates to lists
            try:
                self.epw_start_date = [int(x) for x in self.epw_start_date.split(",")]
                self.epw_end_date = [int(x) for x in self.epw_end_date.split(",")]
                if len(self.epw_start_date) != 4 or len(self.epw_end_date) != 4:
                    raise ValueError("EPW start and end dates must be in the format: year,month,day,hour")
            except ValueError as err:
                raise ValueError(f"Invalid EPW date format: {self.epw_start_date} or {self.epw_end_date}") from err
            if self.epw_hours is None:
                self.epw_hours = list(range(24))  # Default to all hours if not specified
            elif isinstance(self.epw_hours, str):
                self.epw_hours = [int(h) for h in self.epw_hours.split(",")]
            if not all(0 <= h < 24 for h in self.epw_hours):
                raise ValueError("EPW hours must be between 0 and 23.")
        if self.use_landcover and self.lc_path is None:
            raise ValueError("Land cover path must be set if use_landcover is True.")
        if self.use_dem_for_buildings and self.dem_path is None:
            raise ValueError("DEM path must be set if use_dem_for_buildings is True.")
        if not self.use_landcover and not self.use_dem_for_buildings:
            raise ValueError("Either use_landcover or use_dem_for_buildings must be True.")
        # Add more validation as needed


@dataclass
class EnvironData:
    """Class to handle weather data loading and processing."""

    YYYY: np.ndarray
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
    altitude: np.ndarray
    azimuth: np.ndarray
    zen: np.ndarray
    jday: np.ndarray
    leafon: np.ndarray
    psi: np.ndarray
    dectime: np.ndarray
    altmax: np.ndarray

    def __init__(
        self,
        model_configs: SolweigConfig,
        model_params,
        YYYY: np.ndarray,
        DOY: np.ndarray,
        hours: np.ndarray,
        minu: np.ndarray,
        Ta: np.ndarray,
        RH: np.ndarray,
        radG: np.ndarray,
        radD: np.ndarray,
        radI: np.ndarray,
        P: np.ndarray,
        Ws: np.ndarray,
        location: dict | None,
        UTC: bool = True,
    ):
        """
        This function is used to process the input meteorological file.
        It also calculates Sun position based on the time specified in the met-file
        """
        if location is None:
            raise ValueError("Location must be set before loading MET data.")
        # Initialize attributes
        self.YYYY = YYYY
        self.DOY = DOY
        self.hours = hours
        self.minu = minu
        self.Ta = Ta
        self.RH = RH
        self.radG = radG
        self.radD = radD
        self.radI = radI
        self.P = P
        self.Ws = Ws
        # Calculate remaining attributes
        data_len = len(self.YYYY)
        self.dectime = self.DOY + self.hours / 24 + self.minu / (60 * 24.0)
        if data_len == 1:
            halftimestepdec = 0
        else:
            halftimestepdec = (self.dectime[1] - self.dectime[0]) / 2.0
        time = {
            "sec": 0,
            "UTC": UTC,
        }
        sunmaximum = 0.0

        # initialize matrices
        self.altitude = np.empty(data_len)
        self.azimuth = np.empty(data_len)
        self.zen = np.empty(data_len)
        self.jday = np.empty(data_len)
        self.leafon = np.empty(data_len)
        self.psi = np.empty(data_len)
        self.altmax = np.empty(data_len)

        sunmax = dict()

        for i in range(data_len):
            YMD = datetime.datetime(int(self.YYYY[i]), 1, 1) + datetime.timedelta(int(self.DOY[i]) - 1)
            # Finding maximum altitude in 15 min intervals (20141027)
            if (i == 0) or (np.mod(self.dectime[i], np.floor(self.dectime[i])) == 0):
                fifteen = 0.0
                sunmaximum = -90.0
                sunmax["zenith"] = 90.0
                while sunmaximum <= 90.0 - sunmax["zenith"]:
                    sunmaximum = 90.0 - sunmax["zenith"]
                    fifteen = fifteen + 15.0 / 1440.0
                    HM = datetime.timedelta(days=(60 * 10) / 1440.0 + fifteen)
                    YMDHM = YMD + HM
                    time["year"] = YMDHM.year
                    time["month"] = YMDHM.month
                    time["day"] = YMDHM.day
                    time["hour"] = YMDHM.hour
                    time["min"] = YMDHM.minute
                    sunmax = sp.sun_position(time, location)
            self.altmax[i] = sunmaximum
            # Calculate sun position
            half = datetime.timedelta(days=halftimestepdec)
            H = datetime.timedelta(hours=int(self.hours[i]))
            M = datetime.timedelta(minutes=int(self.minu[i]))
            YMDHM = YMD + H + M - half
            time["year"] = YMDHM.year
            time["month"] = YMDHM.month
            time["day"] = YMDHM.day
            time["hour"] = YMDHM.hour
            time["min"] = YMDHM.minute
            sun = sp.sun_position(time, location)
            if (sun["zenith"] > 89.0) & (
                sun["zenith"] <= 90.0
            ):  # Hopefully fixes weird values in Perez et al. when altitude < 1.0, i.e. close to sunrise/sunset
                sun["zenith"] = 89.0
            self.altitude[i] = 90.0 - sun["zenith"]
            self.zen[i] = sun["zenith"] * (np.pi / 180.0)
            self.azimuth[i] = sun["azimuth"]
            # day of year and check for leap year
            # if calendar.isleap(time["year"]):
            #     dayspermonth = np.atleast_2d([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
            # else:
            #     dayspermonth = np.atleast_2d([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
            # jday[0, i] = np.sum(dayspermonth[0, 0:time['month']-1]) + time['day'] # bug when a new day 20191015
            doy = YMD.timetuple().tm_yday
            self.jday[i] = doy
            # Leaf on/off
            if model_configs.conifer:
                # Conifer trees are always leaf on
                self.leafon[i] = 1
            else:
                # Deciduous trees
                self.leafon[i] = 0
                # Check leaf on period
                if model_params.Tree_settings.Value.First_day_leaf > model_params.Tree_settings.Value.Last_day_leaf:
                    self.leafon[i] = int(
                        (model_params.Tree_settings.Value.First_day_leaf < doy)
                        | (model_params.Tree_settings.Value.Last_day_leaf > doy)
                    )
                else:
                    self.leafon[i] = int(
                        (model_params.Tree_settings.Value.First_day_leaf < doy)
                        & (model_params.Tree_settings.Value.Last_day_leaf > doy)
                    )
        # Calculate psi (transmissivity)
        self.psi = self.leafon * model_params.Tree_settings.Value.Transmissivity
        # TODO: check if this is correct
        self.psi[self.leafon == 0] = 0.5


class SvfData:
    """Class to handle SVF data loading and processing."""

    svf: np.ndarray
    svf_east: np.ndarray
    svf_south: np.ndarray
    svf_west: np.ndarray
    svf_north: np.ndarray
    svf_veg: np.ndarray
    svf_veg_east: np.ndarray
    svf_veg_south: np.ndarray
    svf_veg_west: np.ndarray
    svf_veg_north: np.ndarray
    svf_veg_blocks_bldg_sh: np.ndarray
    svf_veg_blocks_bldg_sh_east: np.ndarray
    svf_veg_blocks_bldg_sh_south: np.ndarray
    svf_veg_blocks_bldg_sh_west: np.ndarray
    svf_veg_blocks_bldg_sh_north: np.ndarray
    svfalfa: np.ndarray

    def __init__(self, model_configs: SolweigConfig):
        """
        Loads SVF and shadow matrix results from disk and returns a SVFResults dataclass instance.
        """
        svf_path_str = str(common.check_path(model_configs.svf_path, make_dir=False))
        in_path_str = str(common.check_path(model_configs.working_dir, make_dir=False))
        # Unzip
        with zipfile.ZipFile(svf_path_str, "r") as zip_ref:
            zip_ref.extractall(in_path_str)
        # Load SVF rasters
        self.svf, _, _, _ = common.load_raster(in_path_str + "/" + "svf.tif")
        self.svf_east, _, _, _ = common.load_raster(in_path_str + "/" + "svfE.tif")
        self.svf_south, _, _, _ = common.load_raster(in_path_str + "/" + "svfS.tif")
        self.svf_west, _, _, _ = common.load_raster(in_path_str + "/" + "svfW.tif")
        self.svf_north, _, _, _ = common.load_raster(in_path_str + "/" + "svfN.tif")
        if model_configs.use_veg_dem:
            self.svf_veg, _, _, _ = common.load_raster(in_path_str + "/" + "svfveg.tif")
            self.svf_veg_east, _, _, _ = common.load_raster(in_path_str + "/" + "svfEveg.tif")
            self.svf_veg_south, _, _, _ = common.load_raster(in_path_str + "/" + "svfSveg.tif")
            self.svf_veg_west, _, _, _ = common.load_raster(in_path_str + "/" + "svfWveg.tif")
            self.svf_veg_north, _, _, _ = common.load_raster(in_path_str + "/" + "svfNveg.tif")
            self.svf_veg_blocks_bldg_sh, _, _, _ = common.load_raster(in_path_str + "/" + "svfaveg.tif")
            self.svf_veg_blocks_bldg_sh_east, _, _, _ = common.load_raster(in_path_str + "/" + "svfEaveg.tif")
            self.svf_veg_blocks_bldg_sh_south, _, _, _ = common.load_raster(in_path_str + "/" + "svfSaveg.tif")
            self.svf_veg_blocks_bldg_sh_west, _, _, _ = common.load_raster(in_path_str + "/" + "svfWaveg.tif")
            self.svf_veg_blocks_bldg_sh_north, _, _, _ = common.load_raster(in_path_str + "/" + "svfNaveg.tif")
        else:
            self.svf_veg = np.ones_like(self.svf)
            self.svf_veg_east = np.ones_like(self.svf)
            self.svf_veg_south = np.ones_like(self.svf)
            self.svf_veg_west = np.ones_like(self.svf)
            self.svf_veg_north = np.ones_like(self.svf)
            self.svf_veg_blocks_bldg_sh = np.ones_like(self.svf)
            self.svf_veg_blocks_bldg_sh_east = np.ones_like(self.svf)
            self.svf_veg_blocks_bldg_sh_south = np.ones_like(self.svf)
            self.svf_veg_blocks_bldg_sh_west = np.ones_like(self.svf)
            self.svf_veg_blocks_bldg_sh_north = np.ones_like(self.svf)
        # Calculate SVF alpha
        tmp = self.svf + self.svf_veg - 1.0
        tmp[tmp < 0.0] = 0.0
        self.svfalfa = np.arcsin(np.exp(np.log(1.0 - tmp) / 2.0))


class ShadowMatrices:
    """ """

    use_aniso: bool
    shmat: np.ndarray | None
    diffsh: np.ndarray | None
    vegshmat: np.ndarray | None
    vbshvegshmat: np.ndarray | None
    asvf: np.ndarray | None
    patch_option: int
    steradians: int

    def __init__(
        self,
        model_configs: SolweigConfig,
        model_params,
        rows: int,
        cols: int,
        svf_data: SvfData,
    ):
        self.use_aniso = model_configs.use_aniso

        # Import shadow matrices (Anisotropic sky)
        if self.use_aniso:
            aniso_path_str = str(common.check_path(model_configs.aniso_path, make_dir=False))
            data = np.load(aniso_path_str)
            self.shmat = data["shadowmat"]
            self.vegshmat = data["vegshadowmat"]
            self.vbshvegshmat = data["vbshmat"]
            if model_configs.use_veg_dem:
                self.diffsh = np.zeros((rows, cols, self.shmat.shape[2]))
                for i in range(0, self.shmat.shape[2]):
                    self.diffsh[:, :, i] = self.shmat[:, :, i] - (1 - self.vegshmat[:, :, i]) * (
                        1 - model_params.Tree_settings.Value.Transmissivity
                    )  # changes in psi not implemented yet
            else:
                self.diffsh = self.shmat

            # Estimate number of patches based on shadow matrices
            if self.shmat.shape[2] == 145:
                self.patch_option = 1  # patch_option = 1 # 145 patches
            elif self.shmat.shape[2] == 153:
                self.patch_option = 2  # patch_option = 2 # 153 patches
            elif self.shmat.shape[2] == 306:
                self.patch_option = 3  # patch_option = 3 # 306 patches
            elif self.shmat.shape[2] == 612:
                self.patch_option = 4  # patch_option = 4 # 612 patches

            # asvf to calculate sunlit and shaded patches
            self.asvf = np.arccos(np.sqrt(svf_data.svf))

            # Empty array for steradians
            self.steradians = np.zeros(self.shmat.shape[2])
        else:
            # anisotropic_sky = 0
            self.diffsh = None
            self.shmat = None
            self.vegshmat = None
            self.vbshvegshmat = None
            self.asvf = None
            self.patch_option = 0
            self.steradians = 0


class WallsData:
    """Class to represent wall characteristics and configurations."""

    voxelMaps: np.ndarray | None
    voxelTable: np.ndarray | None
    timeStep: int
    walls_scheme: np.ndarray
    dirwalls_scheme: np.ndarray
    met_for_xarray: tuple[pd.DatetimeIndex] | None

    def __init__(
        self,
        model_configs: SolweigConfig,
        model_params,
        scale: float,
        rows: int,
        cols: int,
        weather_data: EnvironData,
        tg_maps: TgMaps,
        dsm_arr: np.ndarray,
        lcgrid: np.ndarray | None,
    ):
        """Initialize the WallScheme with necessary parameters."""
        if model_configs.use_wall_scheme:
            wall_path_str = str(common.check_path(model_configs.wall_path, make_dir=False))
            wallData = np.load(wall_path_str)
            #
            self.voxelMaps = wallData["voxelId"]
            self.voxelTable = wallData["voxelTable"]
            # Get wall type
            # TODO:
            # wall_type_standalone = {"Brick_wall": "100", "Concrete_wall": "101", "Wood_wall": "102"}
            wall_type = model_configs.wall_type
            # Get heights of walls including corners
            self.walls_scheme = wa.findwalls_sp(dsm_arr, 2, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]))
            # Get aspects of walls including corners
            self.dirwalls_scheme = wa.filter1Goodwin_as_aspect_v3(
                self.walls_scheme.copy(), scale, dsm_arr, None, 100.0 / 180.0
            )
            # Calculate timeStep
            first_timestep = (
                pd.to_datetime(weather_data.YYYY[0], format="%Y")
                + pd.to_timedelta(weather_data.DOY[0] - 1, unit="d")
                + pd.to_timedelta(weather_data.hours[0], unit="h")
                + pd.to_timedelta(weather_data.minu[0], unit="m")
            )
            second_timestep = (
                pd.to_datetime(weather_data.YYYY[1], format="%Y")
                + pd.to_timedelta(weather_data.DOY[1] - 1, unit="d")
                + pd.to_timedelta(weather_data.hours[1], unit="h")
                + pd.to_timedelta(weather_data.minu[1], unit="m")
            )
            self.timeStep = (second_timestep - first_timestep).seconds
            # Load voxelTable as Pandas DataFrame
            self.voxelTable, self.dirwalls_scheme = load_walls(
                self.voxelTable,
                model_params,
                wall_type,
                self.dirwalls_scheme,
                weather_data.Ta[0],
                self.timeStep,
                tg_maps.alb_grid,
                model_configs.use_landcover,
                lcgrid,
                dsm_arr,
            )
            # Create pandas datetime object for NetCDF output
            self.met_for_xarray = (
                pd.to_datetime(weather_data.YYYY, format="%Y")
                + pd.to_timedelta(weather_data.DOY - 1, unit="d")
                + pd.to_timedelta(weather_data.hours, unit="h")
                + pd.to_timedelta(weather_data.minu, unit="m")
            )
        else:
            self.voxelMaps = None
            self.voxelTable = None
            self.timeStep = 0
            self.walls_scheme = np.ones((rows, cols)) * 10.0
            self.dirwalls_scheme = np.ones((rows, cols)) * 10.0
            self.met_for_xarray = None
