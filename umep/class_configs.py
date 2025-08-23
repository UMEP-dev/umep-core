import datetime
import logging
import zipfile
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import scipy.ndimage as ndi

from .util.SEBESOLWEIGCommonFiles.clearnessindex_2013b import clearnessindex_2013b

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    epw_start_date: Optional[str] = None
    epw_end_date: Optional[str] = None
    epw_hours: Optional[str] = None
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
    plot_poi_patches: bool = False

    def to_file(self, file_path: str):
        """Save configuration to a file."""
        logger.info("Saving configuration to %s", file_path)
        with open(file_path, "w") as f:
            for key in type(self).__annotations__:
                value = getattr(self, key)
                if value is None:
                    value = ""  # Default to empty string if None
                if type(self).__annotations__[key] == bool:
                    f.write(f"{key}={int(value)}\n")
                else:
                    f.write(f"{key}={value}\n")

    def from_file(self, config_path_str: str):
        """Load configuration from a file."""
        config_path = common.check_path(config_path_str)
        logger.info("Loading configuration from %s", config_path)
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
                        logger.warning("Unknown key in config: %s", key)

    def validate(self):
        """Validate configuration parameters."""
        logger.info("Validating SOLWEIG configuration.")
        if not self.output_dir:
            logger.error("Output directory must be set.")
            raise ValueError("Output directory must be set.")
        self.output_dir = str(common.check_path(self.output_dir, make_dir=True))
        if not self.working_dir:
            logger.error("Working directory must be set.")
            raise ValueError("Working directory must be set.")
        self.working_dir = str(common.check_path(self.working_dir, make_dir=True))
        if not self.dsm_path:
            logger.error("DSM path must be set.")
            raise ValueError("DSM path must be set.")
        self.utc = bool(self.utc)
        if (self.met_path is None and self.epw_path is None) or (self.met_path and self.epw_path):
            logger.error("Provide either MET or EPW weather file.")
            raise ValueError("Provide either MET or EPW weather file.")
        if self.epw_path is not None:
            if self.epw_start_date is None or self.epw_end_date is None:
                logger.error("EPW start and end dates must be provided if EPW path is set.")
                raise ValueError("EPW start and end dates must be provided if EPW path is set.")
            try:
                if isinstance(self.epw_start_date, str):
                    self.epw_start_date = [int(x) for x in self.epw_start_date.split(",")]
                if isinstance(self.epw_end_date, str):
                    self.epw_end_date = [int(x) for x in self.epw_end_date.split(",")]
                if len(self.epw_start_date) != 4 or len(self.epw_end_date) != 4:
                    logger.error("EPW start and end dates must be in the format: year,month,day,hour")
                    raise ValueError("EPW start and end dates must be in the format: year,month,day,hour")
            except ValueError as err:
                logger.error("Invalid EPW date format: %s or %s", self.epw_start_date, self.epw_end_date)
                raise ValueError(f"Invalid EPW date format: {self.epw_start_date} or {self.epw_end_date}") from err
            if self.epw_hours is None:
                self.epw_hours = list(range(24))
            elif isinstance(self.epw_hours, str):
                self.epw_hours = [int(h) for h in self.epw_hours.split(",")]
            if not all(0 <= h < 24 for h in self.epw_hours):
                logger.error("EPW hours must be between 0 and 23.")
                raise ValueError("EPW hours must be between 0 and 23.")
        if self.use_landcover and self.lc_path is None:
            logger.error("Land cover path must be set if use_landcover is True.")
            raise ValueError("Land cover path must be set if use_landcover is True.")
        if self.use_dem_for_buildings and self.dem_path is None:
            logger.error("DEM path must be set if use_dem_for_buildings is True.")
            raise ValueError("DEM path must be set if use_dem_for_buildings is True.")
        if not self.use_landcover and not self.use_dem_for_buildings:
            logger.error("Either use_landcover or use_dem_for_buildings must be True.")
            raise ValueError("Either use_landcover or use_dem_for_buildings must be True.")
        if self.use_aniso and self.aniso_path is None:
            logger.error("Anisotropic sky path must be set if use_aniso is True.")
            raise ValueError("Anisotropic sky path must be set if use_aniso is True.")
        if self.use_wall_scheme and self.wall_path is None:
            logger.error("Wall scheme path must be set if use_wall_scheme is True.")
            raise ValueError("Wall scheme path must be set if use_wall_scheme is True.")
        if self.plot_poi_patches and (not self.use_aniso or not self.poi_path):
            logger.error("POI path and use_aniso must be set if plot_poi_patches is True.")
            raise ValueError("POI path and use_aniso must be set if plot_poi_patches is True.")


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
    Twater: np.ndarray
    CI: np.ndarray

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

        # initialize arrays
        self.altitude = np.empty(data_len)
        self.azimuth = np.empty(data_len)
        self.zen = np.empty(data_len)
        self.jday = np.empty(data_len)
        self.leafon = np.empty(data_len)
        self.psi = np.empty(data_len)
        self.altmax = np.empty(data_len)
        self.Twater = np.empty(data_len)
        self.CI = np.empty(data_len)

        sunmax = dict()

        # These variables lag across days until updated
        Twater = None
        CI = 1
        # Iterate over time steps and set vars
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
            # Check if the current time is the start of a new day
            if (self.dectime[i] - np.floor(self.dectime[i])) == 0 or (i == 0):
                # Find average temperature for the current day
                Twater = np.mean(self.Ta[self.jday == np.floor(self.dectime[i])])
            # Lags across hours until updated
            self.Twater[i] = Twater

            # Nocturnal cloudfraction from Offerle et al. 2003
            # Check for start of day
            if (self.dectime[i] - np.floor(self.dectime[i])) == 0:
                # Fallback
                CI = 1.0
                # Find all current day idxs
                daylines = np.where(np.floor(self.dectime) == self.dectime[i])
                # np.where returns a tuple, so check the first element
                if len(daylines[0]) > 1:
                    # Get the altitudes for day's idxs
                    alt_day = self.altitude[daylines[0]]
                    # Find all idxs with altitude greater than 1
                    alt2 = np.where(alt_day > 1)
                    # np.where returns a tuple, so check the first element
                    if len(alt2[0]) > 0:
                        # Take the first altitude greater than 1
                        rise = alt2[0][0]
                        # Calculate clearness index for the next time step after sunrise
                        [_, CI_candidate, _, _, _] = clearnessindex_2013b(
                            self.zen[i + rise + 1],
                            self.jday[i + rise + 1],
                            self.Ta[i + rise + 1],
                            self.RH[i + rise + 1] / 100.0,
                            self.radG[i + rise + 1],
                            location,
                            self.P[i + rise + 1],
                        )
                        if np.isfinite(CI_candidate) and CI_candidate <= 1.0:
                            CI = CI_candidate
            # Lags across hours until updated
            self.CI[i] = CI
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
        logger.info("Loading SVF data from %s", model_configs.svf_path)
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
            logger.info("Vegetation SVF data loaded.")
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
        logger.info("SVF data loaded and processed.")


def raster_preprocessing(
    dsm: np.ndarray,
    dem: np.ndarray | None,
    cdsm: np.ndarray | None,
    tdsm: np.ndarray | None,
    trunk_ratio: float,
    pix_size: float,
    amax_local_window_m: int = 100,
    amax_local_perc: float = 99.0,
):
    # amax
    if dem is None:
        amaxvalue = float(np.nanmax(dsm) - np.nanmin(dsm))
    else:
        # Calculate local maxima/minima ranges
        # Number of pixels to cover ~amax_local_window_m radius (use a square window)
        radius_pix = max(1, int(np.ceil(amax_local_window_m / pix_size)))
        window = 2 * radius_pix + 1
        try:
            local_min = ndi.minimum_filter(dsm, size=window, mode="nearest")
            local_range = dsm - local_min
            amaxvalue = float(np.nanpercentile(local_range, amax_local_perc))
            logger.info(
                f"amax {amaxvalue}m derived from {amax_local_window_m}m window and {amax_local_perc}th percentile."
            )
        except Exception:
            # Fallback to global range if filtering fails for any reason
            amaxvalue = float(np.nanmax(dsm) - np.nanmin(dsm))
            logger.warning(f"Failed to calculate local amax; using global range of {amaxvalue}m instead.")

    # CDSM is relative to flat surface without DEM
    if cdsm is None:
        cdsm = np.zeros_like(dsm)
    else:
        if np.nanmax(cdsm) > 50:
            logger.warning(
                f"CDSM max {np.nanmax(cdsm)} exceeds 50 m, vegetation heights to be relative to ground (no DEM)."
            )
        cdsm[np.isnan(cdsm)] = 0.0
        # TDSM is relative to flat surface without DEM
        if tdsm is None:
            logger.info("Tree trunk TDSM not provided; using trunk ratio for TDSM.")
            tdsm = cdsm * trunk_ratio
        if np.nanmax(tdsm) > 50:
            logger.warning(
                f"TDSM max {np.nanmax(tdsm)} exceeds 50m, vegetation heights to be relative to ground (no DEM)."
            )
        if np.nanmax(tdsm) > np.nanmax(cdsm):
            logger.warning("Found TDSM heights exceeding CDSM heights, check input rasters.")
        tdsm[np.isnan(tdsm)] = 0.0

        # Compare veg heights against DSM and update amax if necessary
        # Do before boosting to DEM / CDSM
        vegmax = np.nanmax(cdsm) - np.nanmin(cdsm)
        if vegmax > amaxvalue:
            logger.warning(f"Overriding amax {amaxvalue}m with veg max height of {vegmax}m.")
            amaxvalue = vegmax

        # Set vegetated pixels to DEM + CDSM otherwise DSM + CDSM
        if dem is not None:
            cdsm = np.where(~np.isnan(dem), dem + cdsm, np.nan)
            cdsm = np.where(cdsm - dem < 0.1, 0, cdsm)
            tdsm = np.where(~np.isnan(dem), dem + tdsm, np.nan)
            tdsm = np.where(tdsm - dem < 0.1, 0, tdsm)
        else:
            cdsm = np.where(~np.isnan(dsm), dsm + cdsm, np.nan)
            cdsm = np.where(cdsm - dsm < 0.1, 0, cdsm)
            tdsm = np.where(~np.isnan(dsm), dsm + tdsm, np.nan)
            tdsm = np.where(tdsm - dsm < 0.1, 0, tdsm)

    logger.info("Calculated max height for shadows: %.2fm", amaxvalue)
    if amaxvalue > 100:
        logger.warning("Max shadow height exceeds 100m, double-check the input rasters for anomalies.")

    return dsm, dem, cdsm, tdsm, amaxvalue


class RasterData:
    """Class to represent vegetation parameters."""

    amaxvalue: float
    dsm: np.ndarray
    crs_wkt: str
    trf_arr: np.ndarray
    nd_val: float
    scale: float
    rows: int
    cols: int
    wallheight: np.ndarray
    wallaspect: np.ndarray
    dem: Optional[np.ndarray]
    cdsm: Optional[np.ndarray]
    tdsm: Optional[np.ndarray]
    bush: np.ndarray
    svfbuveg: np.ndarray
    lcgrid: Optional[np.ndarray]
    buildings: Optional[np.ndarray]

    def __init__(self, model_configs: SolweigConfig, model_params, svf_data: SvfData):
        # Load DSM
        self.dsm, self.trf_arr, self.crs_wkt, self.nd_val = common.load_raster(model_configs.dsm_path, bbox=None)
        logger.info("DSM loaded from %s", model_configs.dsm_path)
        self.scale = 1 / self.trf_arr[1]
        self.rows = self.dsm.shape[0]
        self.cols = self.dsm.shape[1]
        # TODO: is this needed?
        # if self.dsm.min() < 0:
        #     dsmraise = np.abs(self.dsm.min())
        #     self.dsm = self.dsm + dsmraise
        # else:
        #     dsmraise = 0

        # WALLS
        # heights
        self.wallheight, wh_trf, wh_crs, _ = common.load_raster(model_configs.wh_path, bbox=None)
        if not self.wallheight.shape == self.dsm.shape:
            raise ValueError("Mismatching raster shapes for wall heights and DSM.")
        if not np.allclose(self.trf_arr, wh_trf):
            raise ValueError("Mismatching spatial transform for wall heights and DSM.")
        if not self.crs_wkt == wh_crs:
            raise ValueError("Mismatching CRS for wall heights and DSM.")
        logger.info("Wall heights loaded")
        # aspects
        self.wallaspect, wa_trf, wa_crs, _ = common.load_raster(model_configs.wa_path, bbox=None)
        if not self.wallaspect.shape == self.dsm.shape:
            raise ValueError("Mismatching raster shapes for wall aspects and DSM.")
        if not np.allclose(self.trf_arr, wa_trf):
            raise ValueError("Mismatching spatial transform for wall aspects and DSM.")
        if not self.crs_wkt == wa_crs:
            raise ValueError("Mismatching CRS for wall aspects and DSM.")
        logger.info("Wall aspects loaded")

        # DEM
        # TODO: Is DEM always provided?
        if model_configs.dem_path:
            dem_path_str = str(common.check_path(model_configs.dem_path))
            self.dem, dem_trf, dem_crs, dem_nd_val = common.load_raster(dem_path_str, bbox=None)
            if not self.dem.shape == self.dsm.shape:
                raise ValueError("Mismatching raster shapes for DEM and CDSM.")
            if dem_crs is not None and dem_crs != self.crs_wkt:
                raise ValueError("Mismatching CRS for DEM and CDSM.")
            if not np.allclose(self.trf_arr, dem_trf):
                raise ValueError("Mismatching spatial transform for DEM and CDSM.")
            logger.info("DEM loaded from %s", model_configs.dem_path)
            # dem[dem == dem_nd_val] = 0.0
            # TODO: Check if this is needed re DSM ramifications
            # if dem.min() < 0:
            #     demraise = np.abs(dem.min())
            #     dem = dem + demraise
        else:
            self.dem = None

        # Vegetation
        if model_configs.use_veg_dem:
            self.cdsm, vegdsm_trf, vegdsm_crs, _ = common.load_raster(model_configs.cdsm_path, bbox=None)
            if not self.cdsm.shape == self.dsm.shape:
                raise ValueError("Mismatching raster shapes for DSM and CDSM.")
            if vegdsm_crs is not None and vegdsm_crs != self.crs_wkt:
                raise ValueError("Mismatching CRS for DSM and CDSM.")
            if not np.allclose(self.trf_arr, vegdsm_trf):
                raise ValueError("Mismatching spatial transform for DSM and CDSM.")
            logger.info("Vegetation DSM loaded from %s", model_configs.cdsm_path)
            # Tree DSM
            if model_configs.tdsm_path:
                self.tdsm, vegdsm2_trf, vegdsm2_crs, _ = common.load_raster(model_configs.tdsm_path, bbox=None)
                if not self.tdsm.shape == self.dsm.shape:
                    raise ValueError("Mismatching raster shapes for DSM and CDSM.")
                if vegdsm2_crs is not None and vegdsm2_crs != self.crs_wkt:
                    raise ValueError("Mismatching CRS for DSM and CDSM.")
                if not np.allclose(self.trf_arr, vegdsm2_trf):
                    raise ValueError("Mismatching spatial transform for DSM and CDSM.")
                logger.info("Tree DSM loaded from %s", model_configs.tdsm_path)
            else:
                self.tdsm = None
        else:
            self.cdsm = None
            self.tdsm = None

        self.dsm, self.dem, self.cdsm, self.tdsm, self.amaxvalue = raster_preprocessing(  # type: ignore
            self.dsm,
            self.dem,
            self.cdsm,
            self.tdsm,
            model_params.Tree_settings.Value.Trunk_ratio,
            self.trf_arr[1],
        )

        # bushes etc
        if model_configs.use_veg_dem:
            self.bush = np.logical_not(self.tdsm * self.cdsm) * self.cdsm
            self.svfbuveg = svf_data.svf - (1.0 - svf_data.svf_veg) * (
                1.0 - model_params.Tree_settings.Value.Transmissivity
            )
        else:
            logger.info("Vegetation DEM not used; vegetation arrays set to None.")
            self.bush = np.zeros([self.rows, self.cols])
            self.svfbuveg = svf_data.svf

        common.save_raster(
            model_configs.output_dir + "/input-dsm.tif",
            self.dsm,
            self.trf_arr,
            self.crs_wkt,
            self.nd_val,
        )
        common.save_raster(
            model_configs.output_dir + "/input-cdsm.tif",
            self.cdsm,
            self.trf_arr,
            self.crs_wkt,
            self.nd_val,
        )
        common.save_raster(
            model_configs.output_dir + "/input-tdsm.tif",
            self.tdsm,
            self.trf_arr,
            self.crs_wkt,
            self.nd_val,
        )
        common.save_raster(
            model_configs.output_dir + "/input-svfbuveg.tif",
            self.svfbuveg,
            self.trf_arr,
            self.crs_wkt,
            self.nd_val,
        )
        common.save_raster(
            model_configs.output_dir + "/input-bush.tif",
            self.bush,
            self.trf_arr,
            self.crs_wkt,
            self.nd_val,
        )

        # Land cover
        if model_configs.use_landcover:
            lc_path_str = str(common.check_path(model_configs.lc_path))
            self.lcgrid, lc_trf, lc_crs, _ = common.load_raster(lc_path_str, bbox=None)
            if not self.lcgrid.shape == self.dsm.shape:
                raise ValueError("Mismatching raster shapes for land cover and DSM.")
            if lc_crs is not None and lc_crs != self.crs_wkt:
                raise ValueError("Mismatching CRS for land cover and DSM.")
            if not np.allclose(self.trf_arr, lc_trf):
                raise ValueError("Mismatching spatial transform for land cover and DSM.")
            logger.info("Land cover loaded from %s", model_configs.lc_path)
        else:
            self.lcgrid = None
            logger.info("Land cover not used; lcgrid set to None.")

        # Buildings from land cover option
        # TODO: Check intended logic here
        if not model_configs.use_dem_for_buildings and self.lcgrid is not None:
            # Create building boolean raster from either land cover if no DEM is used
            buildings = np.copy(self.lcgrid)
            buildings[buildings == 7] = 1
            buildings[buildings == 6] = 1
            buildings[buildings == 5] = 1
            buildings[buildings == 4] = 1
            buildings[buildings == 3] = 1
            buildings[buildings == 2] = 0
            self.buildings = buildings
            logger.info("Buildings raster created from land cover data.")
        elif model_configs.use_dem_for_buildings:
            buildings = np.where(~np.isnan(self.dem) & ~np.isnan(self.dsm), self.dsm - self.dem, 0.0)
            # TODO: Check intended logic here - 1 vs 0
            buildings[buildings < 2.0] = 1.0
            buildings[buildings >= 2.0] = 0.0
            self.buildings = buildings
            logger.info("Buildings raster created from DSM and DEM data.")
        else:
            self.buildings = None
            logger.info("Buildings raster not created.")
        # Save buildings raster if requested
        if self.buildings is not None and model_configs.save_buildings:
            common.save_raster(
                model_configs.output_dir + "/buildings.tif",
                self.buildings,
                self.trf_arr,
                self.crs_wkt,
                self.nd_val,
            )
            logger.info("Buildings raster saved to %s/buildings.tif", model_configs.output_dir)


class ShadowMatrices:
    """Shadow matrices and related anisotropic sky data."""

    use_aniso: bool
    shmat: Optional[np.ndarray]
    diffsh: Optional[np.ndarray]
    vegshmat: Optional[np.ndarray]
    vbshvegshmat: Optional[np.ndarray]
    asvf: Optional[np.ndarray]
    patch_option: int
    steradians: Union[int, np.ndarray]

    def __init__(
        self,
        model_configs: SolweigConfig,
        model_params,
        raster_data: RasterData,
        svf_data: SvfData,
    ):
        self.use_aniso = model_configs.use_aniso
        if self.use_aniso:
            logger.info("Loading anisotropic shadow matrices from %s", model_configs.aniso_path)
            aniso_path_str = str(common.check_path(model_configs.aniso_path, make_dir=False))
            data = np.load(aniso_path_str)
            self.shmat = data["shadowmat"]
            self.vegshmat = data["vegshadowmat"]
            self.vbshvegshmat = data["vbshmat"]
            if model_configs.use_veg_dem:
                self.diffsh = np.zeros((raster_data.rows, raster_data.cols, self.shmat.shape[2]))
                for i in range(0, self.shmat.shape[2]):
                    self.diffsh[:, :, i] = self.shmat[:, :, i] - (1 - self.vegshmat[:, :, i]) * (
                        1 - model_params.Tree_settings.Value.Transmissivity
                    )
                logger.info("Shadow matrices with vegetation loaded.")
            else:
                self.diffsh = self.shmat
                logger.info("Shadow matrices loaded (no vegetation).")

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
            # no anisotropic sky
            # downstream functions only access these if use_aniso is True
            # be aware that Solweig_2025a_calc expects an int not bool for use_aniso
            self.diffsh = None
            self.shmat = None
            self.vegshmat = None
            self.vbshvegshmat = None
            self.asvf = None
            self.patch_option = 0
            self.steradians = 0
            logger.info("Anisotropic sky not used; shadow matrices not loaded.")


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
    TmaxLST: Union[np.ndarray, float]
    TmaxLST_wall: float
    Knight: np.ndarray
    Tgmap1: np.ndarray
    Tgmap1E: np.ndarray
    Tgmap1S: np.ndarray
    Tgmap1W: np.ndarray
    Tgmap1N: np.ndarray
    TgOut1: np.ndarray

    def __init__(self, use_landcover: bool, model_params, raster_data: RasterData):
        """
        This is a vectorized version that avoids looping over pixels.
        """
        # Initialization of maps
        self.Knight = np.zeros((raster_data.rows, raster_data.cols))
        self.Tgmap1 = np.zeros((raster_data.rows, raster_data.cols))
        self.Tgmap1E = np.zeros((raster_data.rows, raster_data.cols))
        self.Tgmap1S = np.zeros((raster_data.rows, raster_data.cols))
        self.Tgmap1W = np.zeros((raster_data.rows, raster_data.cols))
        self.Tgmap1N = np.zeros((raster_data.rows, raster_data.cols))
        self.TgOut1 = np.zeros((raster_data.rows, raster_data.cols))

        # Set up the Tg maps based on whether land cover is used
        if use_landcover is False:
            self.TgK = self.Knight + model_params.Ts_deg.Value.Cobble_stone_2014a
            self.Tstart = self.Knight - model_params.Tstart.Value.Cobble_stone_2014a
            self.alb_grid = self.Knight + model_params.Albedo.Effective.Value.Cobble_stone_2014a
            self.emis_grid = self.Knight + model_params.Emissivity.Value.Cobble_stone_2014a
            self.TmaxLST = model_params.TmaxLST.Value.Cobble_stone_2014a  # Assuming this is a float
            self.TgK_wall = model_params.Ts_deg.Value.Walls
            self.Tstart_wall = model_params.Tstart.Value.Walls
            self.TmaxLST_wall = model_params.TmaxLST.Value.Walls
            logger.info("TgMaps initialized with default (no land cover) parameters.")
        else:
            if raster_data.lcgrid is None:
                raise ValueError("Land cover grid is not available.")
            # Copy land cover grid
            lc_grid = np.copy(raster_data.lcgrid)
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
            logger.info("TgMaps initialized using land cover grid.")


class WallsData:
    """Class to represent wall characteristics and configurations."""

    voxelMaps: Optional[np.ndarray]
    voxelTable: Optional[np.ndarray]
    timeStep: int
    walls_scheme: np.ndarray
    dirwalls_scheme: np.ndarray
    met_for_xarray: Optional[Tuple[Any]]

    def __init__(
        self,
        model_configs: SolweigConfig,
        model_params,
        raster_data: RasterData,
        weather_data: EnvironData,
        tg_maps: TgMaps,
    ):
        if model_configs.use_wall_scheme:
            logger.info("Loading wall scheme data from %s", model_configs.wall_path)
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
            self.walls_scheme = wa.findwalls_sp(raster_data.dsm, 2, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]))
            # Get aspects of walls including corners
            self.dirwalls_scheme = wa.filter1Goodwin_as_aspect_v3(
                self.walls_scheme.copy(), raster_data.scale, raster_data.dsm, None, 100.0 / 180.0
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
                raster_data.lcgrid,
                raster_data.dsm,
            )
            # Create pandas datetime object for NetCDF output
            self.met_for_xarray = (
                pd.to_datetime(weather_data.YYYY, format="%Y")
                + pd.to_timedelta(weather_data.DOY - 1, unit="d")
                + pd.to_timedelta(weather_data.hours, unit="h")
                + pd.to_timedelta(weather_data.minu, unit="m")
            )
            logger.info("Wall scheme data loaded and processed.")
        else:
            self.voxelMaps = None
            self.voxelTable = None
            self.timeStep = 0
            self.walls_scheme = np.ones((raster_data.rows, raster_data.cols)) * 10.0
            self.dirwalls_scheme = np.ones((raster_data.rows, raster_data.cols)) * 10.0
            self.met_for_xarray = None
            logger.info("Wall scheme not used; default wall data initialized.")
