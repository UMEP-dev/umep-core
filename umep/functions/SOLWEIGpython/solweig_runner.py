import json
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# handle for QGIS which does not have matplotlib by default?
try:
    from matplotlib import pyplot as plt

    PLT = True
except ImportError:
    PLT = False

from ... import common
from ...class_configs import EnvironData, ShadowMatrices, SolweigConfig, SvfData, TgMaps, Vegetation, WallsData
from ...util.SEBESOLWEIGCommonFiles.clearnessindex_2013b import clearnessindex_2013b
from . import PET_calculations
from . import Solweig_2025a_calc_forprocessing as so
from . import UTCI_calculations as utci
from .CirclePlotBar import PolarBarPlot
from .wallsAsNetCDF import walls_as_netcdf


def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d


class SolweigRun:
    """Class to run the SOLWEIG algorithm with given configuration."""

    config: SolweigConfig
    progress: Optional[Any]
    iters_total: Optional[int]
    iters_count: int = 0
    poi_names: List[Any] = []
    poi_pixel_xys: Optional[np.ndarray]
    poi_results = []
    woi_names: List[Any] = []
    woi_pixel_xys: Optional[np.ndarray]
    woi_results = []
    dsm_arr: np.ndarray
    dsm_trf_arr: np.ndarray
    dsm_crs_wkt: str
    dsm_nd_val: float
    scale: float
    rows: int
    cols: int
    location: Dict[str, float]
    svf_data: SvfData
    environ_data: EnvironData
    tg_maps: TgMaps
    shadow_mats: ShadowMatrices
    vegetation: Vegetation
    buildings: np.ndarray
    wallheight: np.ndarray
    wallaspect: np.ndarray
    walls_data: WallsData

    def __init__(self, config: SolweigConfig, params_json_path: str):
        """Initialize the SOLWEIG runner with configuration and parameters."""
        logger.info("Starting SOLWEIG setup")
        self.config = config
        self.config.validate()
        # Progress tracking settings
        self.progress = None
        self.iters_total = None
        self.iters_count = 0
        # Initialize POI data
        self.poi_names = []
        self.poi_pixel_xys = None
        self.poi_results = []
        # Initialize WOI data
        self.woi_names = []
        self.woi_pixel_xys = None
        self.woi_results = []
        # Load parameters from JSON file
        params_path = common.check_path(params_json_path)
        try:
            with open(params_path) as f:
                params_dict = json.load(f)
                self.params = dict_to_namespace(params_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load parameters from {params_json_path}: {e}")
        # Load DSM
        self.dsm_arr, self.dsm_trf_arr, self.dsm_crs_wkt, self.dsm_nd_val = common.load_raster(
            self.config.dsm_path, bbox=None
        )
        logger.info("DSM loaded from %s", self.config.dsm_path)
        self.scale = 1 / self.dsm_trf_arr[1]
        self.rows = self.dsm_arr.shape[0]
        self.cols = self.dsm_arr.shape[1]

        left_x = self.dsm_trf_arr[0]
        top_y = self.dsm_trf_arr[3]
        lng, lat = common.xy_to_lnglat(self.dsm_crs_wkt, left_x, top_y)
        alt = np.median(self.dsm_arr)
        if alt < 0:
            alt = 3
        self.location = {"longitude": lng, "latitude": lat, "altitude": alt}

        self.dsm_arr[self.dsm_arr == self.dsm_nd_val] = 0.0
        if self.dsm_arr.min() < 0:
            dsmraise = np.abs(self.dsm_arr.min())
            self.dsm_arr = self.dsm_arr + dsmraise
        else:
            dsmraise = 0

        # DEM
        # TODO: Is DEM always provided?
        if self.config.dem_path:
            dem_path_str = str(common.check_path(self.config.dem_path))
            dem, _, _, dem_nd_val = common.load_raster(dem_path_str, bbox=None)
            logger.info("DEM loaded from %s", self.config.dem_path)
            dem[dem == dem_nd_val] = 0.0
            # TODO: Check if this is needed re DSM ramifications
            if dem.min() < 0:
                demraise = np.abs(dem.min())
                dem = dem + demraise

        # Land cover
        if self.config.use_landcover:
            lc_path_str = str(common.check_path(self.config.lc_path))
            self.lcgrid, _, _, _ = common.load_raster(lc_path_str, bbox=None)
            logger.info("Land cover loaded from %s", self.config.lc_path)
        else:
            self.lcgrid = None

        # Buildings from land cover option
        # TODO: Check intended logic here
        if not self.config.use_dem_for_buildings and self.lcgrid is not None:
            # Create building boolean raster from either land cover if no DEM is used
            buildings = np.copy(self.lcgrid)
            buildings[buildings == 7] = 1
            buildings[buildings == 6] = 1
            buildings[buildings == 5] = 1
            buildings[buildings == 4] = 1
            buildings[buildings == 3] = 1
            buildings[buildings == 2] = 0
        elif self.config.use_dem_for_buildings:
            buildings = self.dsm_arr - dem
            buildings[buildings < 2.0] = 1.0
            buildings[buildings >= 2.0] = 0.0
        else:
            raise ValueError("No DEM or buildings data available.")
        self.buildings = buildings
        # Save buildings raster if requested
        if self.config.save_buildings:
            common.save_raster(
                self.config.output_dir + "/buildings.tif",
                buildings,
                self.dsm_trf_arr,
                self.dsm_crs_wkt,
                self.dsm_nd_val,
            )
            logger.info("Buildings raster saved to %s/buildings.tif", self.config.output_dir)

        # Load SVF data
        self.svf_data = SvfData(self.config)
        logger.info("SVF data loaded")

        self.vegetation = Vegetation(self.config, self.params, self.rows, self.cols, self.svf_data, self.dsm_arr)
        logger.info("Vegetation data initialized")

        # Load walls
        self.wallheight, _, _, _ = common.load_raster(self.config.wh_path, bbox=None)
        self.wallaspect, _, _, _ = common.load_raster(self.config.wa_path, bbox=None)
        logger.info("Wall rasters loaded")

        # weather data
        if self.config.use_epw_file:
            self.environ_data = self.load_epw_weather()
            logger.info("Weather data loaded from EPW file")
        else:
            self.environ_data = self.load_met_weather(header_rows=1, delim=" ")
            logger.info("Weather data loaded from MET file")

        # POIs check
        if self.config.poi_path:
            self.load_poi_data()
            logger.info("POI data loaded from %s", self.config.poi_path)

        # Import shadow matrices (Anisotropic sky)
        self.shadow_mats = ShadowMatrices(self.config, self.params, self.rows, self.cols, self.svf_data)
        logger.info("Shadow matrices initialized")

        # % Ts parameterisation maps
        self.tg_maps = TgMaps(self.config.use_landcover, self.lcgrid, self.params, self.rows, self.cols)
        logger.info("TgMaps initialized")

        # Import data for wall temperature parameterization
        # Use wall of interest
        if self.config.woi_path:
            self.load_woi_data()
            logger.info("WOI data loaded from %s", self.config.woi_path)
        self.walls_data = WallsData(
            self.config,
            self.params,
            self.scale,
            self.rows,
            self.cols,
            self.environ_data,
            self.tg_maps,
            self.dsm_arr,
            self.lcgrid,
        )
        logger.info("WallsData initialized")

    def test_hook(self) -> None:
        """Test hook for testing loaded init state."""
        pass

    def prep_progress(self, num: int) -> None:
        """Prepare progress for environment."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def iter_progress(self) -> bool:
        """Iterate progress ."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_epw_weather(self) -> EnvironData:
        """Load weather data from an EPW file."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_met_weather(self, header_rows: int = 1, delim: str = " ") -> EnvironData:
        """Load weather data from a MET file."""
        met_path_str = str(common.check_path(self.config.met_path))
        met_data = np.loadtxt(met_path_str, skiprows=header_rows, delimiter=delim)
        return EnvironData(
            self.config,
            self.params,
            YYYY=met_data[:, 0],
            DOY=met_data[:, 1],
            hours=met_data[:, 2],
            minu=met_data[:, 3],
            Ta=met_data[:, 11],
            RH=met_data[:, 10],
            radG=met_data[:, 14],
            radD=met_data[:, 21],
            radI=met_data[:, 22],
            P=met_data[:, 12],
            Ws=met_data[:, 9],
            location=self.location,
            UTC=self.config.utc,
        )

    def load_poi_data(self) -> None:
        """Load point of interest (POI) data from a file."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def save_poi_results(self) -> None:
        """Save results for points of interest (POIs) to files."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_woi_data(self) -> None:
        """Load wall of interest (WOI) data from a file."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def save_woi_results(self) -> None:
        """Save results for walls of interest (WOIs) to files."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def hemispheric_image(self):
        """
        Calculate patch characteristics for points of interest (POIs).
        This method is vectorized for efficiency as it processes all POIs simultaneously.
        """
        n_patches = self.shadow_mats.shmat.shape[2]
        n_pois = self.poi_pixel_xys.shape[0]
        patch_characteristics = np.zeros((n_patches, n_pois))

        # Get POI indices as integer arrays
        poi_y = self.poi_pixel_xys[:, 2].astype(int)
        poi_x = self.poi_pixel_xys[:, 1].astype(int)

        for idy in range(n_patches):
            # Precompute masks for this patch
            temp_sky = (self.shadow_mats.shmat[:, :, idy] == 1) & (self.shadow_mats.vegshmat[:, :, idy] == 1)
            temp_vegsh = (self.shadow_mats.vegshmat[:, :, idy] == 0) | (self.shadow_mats.vbshvegshmat[:, :, idy] == 0)
            temp_vbsh = (1 - self.shadow_mats.shmat[:, :, idy]) * self.shadow_mats.vbshvegshmat[:, :, idy]
            temp_sh = temp_vbsh == 1

            if self.config.use_wall_scheme:
                temp_sh_w = temp_sh * self.walls_data.voxelMaps[:, :, idy]
                temp_sh_roof = temp_sh * (self.walls_data.voxelMaps[:, :, idy] == 0)
            else:
                temp_sh_w = None
                temp_sh_roof = None

            # Gather mask values for all POIs at once
            sky_vals = temp_sky[poi_y, poi_x]
            veg_vals = temp_vegsh[poi_y, poi_x]
            sh_vals = temp_sh[poi_y, poi_x]

            if self.config.use_wall_scheme:
                sh_w_vals = temp_sh_w[poi_y, poi_x]
                sh_roof_vals = temp_sh_roof[poi_y, poi_x]

            # Assign patch characteristics in vectorized way
            patch_characteristics[idy, sky_vals] = 1.8
            patch_characteristics[idy, ~sky_vals & veg_vals] = 2.5
            if self.config.use_wall_scheme:
                patch_characteristics[idy, ~sky_vals & ~veg_vals & sh_vals & sh_w_vals] = 4.5
                patch_characteristics[idy, ~sky_vals & ~veg_vals & sh_vals & ~sh_w_vals & sh_roof_vals] = 4.5
            else:
                patch_characteristics[idy, ~sky_vals & ~veg_vals & sh_vals] = 4.5

        return patch_characteristics

    def calc_solweig(
        self,
        iter: int,
        elvis: float,
        CI: float,
        Twater: float,
        first: float,
        second: float,
        firstdaytime: float,
        timeadd: float,
        timestepdec: float,
        posture,
    ):
        """
        Calculate SOLWEIG results for a given iteration.
        Separated from the main run method so that it can be overridden by subclasses.
        Over time we can simplify the function signature by passing consolidated classes to solweig calc methods.
        """
        return so.Solweig_2025a_calc(
            iter,
            self.dsm_arr,
            self.scale,
            self.rows,
            self.cols,
            self.svf_data.svf,
            self.svf_data.svf_north,
            self.svf_data.svf_west,
            self.svf_data.svf_east,
            self.svf_data.svf_south,
            self.svf_data.svf_veg,
            self.svf_data.svf_veg_north,
            self.svf_data.svf_veg_east,
            self.svf_data.svf_veg_south,
            self.svf_data.svf_veg_west,
            self.svf_data.svf_veg_blocks_bldg_sh,
            self.svf_data.svf_veg_blocks_bldg_sh_east,
            self.svf_data.svf_veg_blocks_bldg_sh_south,
            self.svf_data.svf_veg_blocks_bldg_sh_west,
            self.svf_data.svf_veg_blocks_bldg_sh_north,
            self.vegetation.vegdsm,
            self.vegetation.vegdsm2,
            self.params.Albedo.Effective.Value.Walls,
            self.params.Tmrt_params.Value.absK,
            self.params.Tmrt_params.Value.absL,
            self.params.Emissivity.Value.Walls,
            posture.Fside,
            posture.Fup,
            posture.Fcyl,
            self.environ_data.altitude[iter],
            self.environ_data.azimuth[iter],
            self.environ_data.zen[iter],
            self.environ_data.jday[iter],
            self.config.use_veg_dem,
            self.config.only_global,
            self.buildings,
            self.location,
            self.environ_data.psi[iter],
            self.config.use_landcover,
            self.lcgrid,
            self.environ_data.dectime[iter],
            self.environ_data.altmax[iter],
            self.wallaspect,
            self.wallheight,
            int(self.config.person_cylinder),  # expects int though should work either way
            elvis,
            self.environ_data.Ta[iter],
            self.environ_data.RH[iter],
            self.environ_data.radG[iter],
            self.environ_data.radD[iter],
            self.environ_data.radI[iter],
            self.environ_data.P[iter],
            self.vegetation.amaxvalue,
            self.vegetation.bush,
            Twater,
            self.tg_maps.TgK,
            self.tg_maps.Tstart,
            self.tg_maps.alb_grid,
            self.tg_maps.emis_grid,
            self.tg_maps.TgK_wall,
            self.tg_maps.Tstart_wall,
            self.tg_maps.TmaxLST,
            self.tg_maps.TmaxLST_wall,
            first,
            second,
            self.svf_data.svfalfa,
            self.vegetation.svfbuveg,
            firstdaytime,
            timeadd,
            timestepdec,
            self.tg_maps.Tgmap1,
            self.tg_maps.Tgmap1E,
            self.tg_maps.Tgmap1S,
            self.tg_maps.Tgmap1W,
            self.tg_maps.Tgmap1N,
            CI,
            self.tg_maps.TgOut1,
            self.shadow_mats.diffsh,
            self.shadow_mats.shmat,
            self.shadow_mats.vegshmat,
            self.shadow_mats.vbshvegshmat,
            int(self.config.use_aniso),  # expects int though should work either way
            self.shadow_mats.asvf,
            self.shadow_mats.patch_option,
            self.walls_data.voxelMaps,
            self.walls_data.voxelTable,
            self.environ_data.Ws[iter],
            self.config.use_wall_scheme,
            self.walls_data.timeStep,
            self.shadow_mats.steradians,
            self.walls_data.walls_scheme,
            self.walls_data.dirwalls_scheme,
        )

    def run(self) -> None:
        # Posture settings
        if self.params.Tmrt_params.Value.posture == "Standing":
            posture = self.params.Posture.Standing.Value
        else:
            posture = self.params.Posture.Sitting.Value
        # Radiative surface influence
        first = np.round(posture.height)
        if first == 0.0:
            first = 1.0
        second = np.round(posture.height * 20.0)
        # Save hemispheric image
        if self.config.use_aniso and self.poi_pixel_xys is not None:
            patch_characteristics = self.hemispheric_image()
            logger.info("Hemispheric image calculated for POIs")
        # Initialisation of time related variables
        if self.environ_data.Ta.__len__() == 1:
            timestepdec = 0
        else:
            timestepdec = self.environ_data.dectime[1] - self.environ_data.dectime[0]
        timeadd = 0.0
        firstdaytime = 1.0
        # Initiate array for I0 values plotting
        if np.unique(self.environ_data.DOY).shape[0] > 1:
            unique_days = np.unique(self.environ_data.DOY)
            first_unique_day = self.environ_data.DOY[unique_days[0] == self.environ_data.DOY]
            I0_array = np.zeros_like(first_unique_day)
        else:
            first_unique_day = self.environ_data.DOY.copy()
            I0_array = np.zeros_like(self.environ_data.DOY)
        # For Tmrt plot
        tmrtplot = np.zeros((self.rows, self.cols))
        # Number of iterations
        num = len(self.environ_data.Ta)
        # Prepare progress tracking
        self.prep_progress(num)
        logger.info("Progress tracking prepared for %d iterations", num)
        # TODO: confirm intent of water temperature handling
        # Assuming it should be initialized to NaN outside the loop so that it can be updated at the start of each day
        Twater = np.nan
        CI = 1.0
        elvis = 0.0

        for i in range(num):
            proceed = self.iter_progress()
            if not proceed:
                break

            # Daily water body temperature - only if land cover is used
            if self.config.use_landcover:  # noqa: SIM102
                # Check if the current time is the start of a new day
                if (self.environ_data.dectime[i] - np.floor(self.environ_data.dectime[i])) == 0 or (i == 0):
                    # Find average temperature for the current day
                    Twater = np.mean(
                        self.environ_data.Ta[self.environ_data.jday == np.floor(self.environ_data.dectime[i])]
                    )

            # Nocturnal cloudfraction from Offerle et al. 2003
            # Check for start of day
            if (self.environ_data.dectime[i] - np.floor(self.environ_data.dectime[i])) == 0:
                # Find all current day idxs
                daylines = np.where(np.floor(self.environ_data.dectime) == self.environ_data.dectime[i])
                # np.where returns a tuple, so check the first element
                if len(daylines[0]) > 1:
                    # Get the altitudes for day's idxs
                    alt_day = self.environ_data.altitude[daylines[0]]
                    # Find all idxs with altitude greater than 1
                    alt2 = np.where(alt_day > 1)
                    # np.where returns a tuple, so check the first element
                    if len(alt2[0]) > 0:
                        # Take the first altitude greater than 1
                        rise = alt2[0][0]
                        # Calculate clearness index for the next time step after sunrise
                        [_, CI, _, _, _] = clearnessindex_2013b(
                            self.environ_data.zen[i + rise + 1],
                            self.environ_data.jday[i + rise + 1],
                            self.environ_data.Ta[i + rise + 1],
                            self.environ_data.RH[i + rise + 1] / 100.0,
                            self.environ_data.radG[i + rise + 1],
                            self.location,
                            self.environ_data.P[i + rise + 1],
                        )
                        if (CI > 1.0) or (~np.isfinite(CI)):
                            CI = 1.0
                    else:
                        CI = 1.0
                else:
                    CI = 1.0
            # Run the SOLWEIG calculations
            (
                Tmrt,
                Kdown,
                Kup,
                Ldown,
                Lup,
                Tg,
                ea,
                esky,
                I0,
                CI,
                shadow,
                firstdaytime,
                timestepdec,
                timeadd,
                self.tg_maps.Tgmap1,
                self.tg_maps.Tgmap1E,
                self.tg_maps.Tgmap1S,
                self.tg_maps.Tgmap1W,
                self.tg_maps.Tgmap1N,
                Keast,
                Ksouth,
                Kwest,
                Knorth,
                Least,
                Lsouth,
                Lwest,
                Lnorth,
                KsideI,
                self.tg_maps.TgOut1,
                TgOut,
                radIout,
                radDout,
                Lside,
                Lsky_patch_characteristics,
                CI_Tg,
                CI_TgG,
                KsideD,
                dRad,
                Kside,
                self.shadow_mats.steradians,
                voxelTable,
            ) = self.calc_solweig(
                i,
                elvis,
                CI,
                Twater,
                first,
                second,
                firstdaytime,
                timeadd,
                timestepdec,
                posture,
            )
            # Save I0 for I0 vs. Kdown output plot to check if UTC is off
            if i < first_unique_day.shape[0]:
                I0_array[i] = I0
            elif i == first_unique_day.shape[0] and PLT is True:
                # Output I0 vs. Kglobal plot
                radG_for_plot = self.environ_data.radG[first_unique_day[0] == self.environ_data.DOY]
                dectime_for_plot = self.environ_data.dectime[first_unique_day[0] == self.environ_data.DOY]
                fig, ax = plt.subplots()
                ax.plot(dectime_for_plot, I0_array, label="I0")
                ax.plot(dectime_for_plot, radG_for_plot, label="Kglobal")
                ax.set_ylabel("Shortwave radiation [$Wm^{-2}$]")
                ax.set_xlabel("Decimal time")
                ax.set_title("UTC" + str(self.config.utc))
                ax.legend()
                fig.savefig(self.config.output_dir + "/metCheck.png", dpi=150)

            tmrtplot = tmrtplot + Tmrt

            if self.environ_data.altitude[i] > 0:
                w = "D"
            else:
                w = "N"

            if self.environ_data.hours[i] < 10:
                XH = "0"
            else:
                XH = ""

            if self.environ_data.minu[i] < 10:
                XM = "0"
            else:
                XM = ""

            if self.poi_pixel_xys is not None:
                for n in range(0, self.poi_pixel_xys.shape[0]):
                    idx, row_idx, col_idx = self.poi_pixel_xys[n]
                    row_idx = int(row_idx)
                    col_idx = int(col_idx)
                    result_row = {
                        "poi_idx": idx,
                        "col_idx": col_idx,
                        "row_idx": row_idx,
                        "yyyy": self.environ_data.YYYY[i],
                        "id": self.environ_data.jday[i],
                        "it": self.environ_data.hours[i],
                        "imin": self.environ_data.minu[i],
                        "dectime": self.environ_data.dectime[i],
                        "altitude": self.environ_data.altitude[i],
                        "azimuth": self.environ_data.azimuth[i],
                        "kdir": radIout,
                        "kdiff": radDout,
                        "kglobal": self.environ_data.radG[i],
                        "kdown": Kdown[row_idx, col_idx],
                        "kup": Kup[row_idx, col_idx],
                        "keast": Keast[row_idx, col_idx],
                        "ksouth": Ksouth[row_idx, col_idx],
                        "kwest": Kwest[row_idx, col_idx],
                        "knorth": Knorth[row_idx, col_idx],
                        "ldown": Ldown[row_idx, col_idx],
                        "lup": Lup[row_idx, col_idx],
                        "least": Least[row_idx, col_idx],
                        "lsouth": Lsouth[row_idx, col_idx],
                        "lwest": Lwest[row_idx, col_idx],
                        "lnorth": Lnorth[row_idx, col_idx],
                        "Ta": self.environ_data.Ta[i],
                        "Tg": TgOut[row_idx, col_idx],
                        "RH": self.environ_data.RH[i],
                        "Esky": esky,
                        "Tmrt": Tmrt[row_idx, col_idx],
                        "I0": I0,
                        "CI": CI,
                        "Shadow": shadow[row_idx, col_idx],
                        "SVF_b": self.svf_data.svf[row_idx, col_idx],
                        "SVF_bv": self.vegetation.svfbuveg[row_idx, col_idx],
                        "KsideI": KsideI[row_idx, col_idx],
                    }
                    # Recalculating wind speed based on powerlaw
                    WsPET = (1.1 / self.params.Wind_Height.Value.magl) ** 0.2 * self.environ_data.Ws[i]
                    WsUTCI = (10.0 / self.params.Wind_Height.Value.magl) ** 0.2 * self.environ_data.Ws[i]
                    resultPET = PET_calculations._PET(
                        self.environ_data.Ta[i],
                        self.environ_data.RH[i],
                        Tmrt[row_idx, col_idx],
                        WsPET,
                        self.params.PET_settings.Value.Weight,
                        self.params.PET_settings.Value.Age,
                        self.params.PET_settings.Value.Height,
                        self.params.PET_settings.Value.Activity,
                        self.params.PET_settings.Value.clo,
                        self.params.PET_settings.Value.Sex,
                    )
                    result_row["PET"] = resultPET
                    resultUTCI = utci.utci_calculator(
                        self.environ_data.Ta[i], self.environ_data.RH[i], Tmrt[row_idx, col_idx], WsUTCI
                    )
                    result_row["UTCI"] = resultUTCI
                    result_row["CI_Tg"] = CI_Tg
                    result_row["CI_TgG"] = CI_TgG
                    result_row["KsideD"] = KsideD[row_idx, col_idx]
                    result_row["Lside"] = Lside[row_idx, col_idx]
                    result_row["diffDown"] = dRad[row_idx, col_idx]
                    result_row["Kside"] = Kside[row_idx, col_idx]
                    self.poi_results.append(result_row)

            if self.config.use_wall_scheme and self.woi_pixel_xys is not None:
                for n in range(0, self.woi_pixel_xys.shape[0]):
                    idx, row_idx, col_idx = self.woi_pixel_xys[n]
                    row_idx = int(row_idx)
                    col_idx = int(col_idx)

                    temp_wall = voxelTable.loc[
                        ((voxelTable["ypos"] == row_idx) & (voxelTable["xpos"] == col_idx)), "wallTemperature"
                    ].to_numpy()
                    K_in = voxelTable.loc[
                        ((voxelTable["ypos"] == row_idx) & (voxelTable["xpos"] == col_idx)), "K_in"
                    ].to_numpy()
                    L_in = voxelTable.loc[
                        ((voxelTable["ypos"] == row_idx) & (voxelTable["xpos"] == col_idx)), "L_in"
                    ].to_numpy()
                    wallShade = voxelTable.loc[
                        ((voxelTable["ypos"] == row_idx) & (voxelTable["xpos"] == col_idx)), "wallShade"
                    ].to_numpy()

                    result_row = {
                        "woi_idx": idx,
                        "woi_name": self.woi_names[idx],
                        "yyyy": self.environ_data.YYYY[i],
                        "id": self.environ_data.jday[i],
                        "it": self.environ_data.hours[i],
                        "imin": self.environ_data.minu[i],
                        "dectime": self.environ_data.dectime[i],
                        "Ta": self.environ_data.Ta[i],
                        "SVF": self.svf_data.svf[row_idx, col_idx],
                        "Ts": temp_wall,
                        "Kin": K_in,
                        "Lin": L_in,
                        "shade": wallShade,
                        "pixel_x": col_idx,
                        "pixel_y": row_idx,
                    }
                    self.woi_results.append(result_row)

                if self.config.wall_netcdf:
                    netcdf_output = self.config.output_dir + "/walls.nc"
                    walls_as_netcdf(
                        voxelTable,
                        self.rows,
                        self.cols,
                        self.walls_data.met_for_xarray,
                        i,
                        self.dsm_arr,
                        self.config.dsm_path,
                        netcdf_output,
                    )

            time_code = (
                str(int(self.environ_data.YYYY[i]))
                + "_"
                + str(int(self.environ_data.DOY[i]))
                + "_"
                + XH
                + str(int(self.environ_data.hours[i]))
                + XM
                + str(int(self.environ_data.minu[i]))
                + w
            )

            if self.config.output_tmrt:
                common.save_raster(
                    self.config.output_dir + "/Tmrt_" + time_code + ".tif",
                    Tmrt,
                    self.dsm_trf_arr,
                    self.dsm_crs_wkt,
                    self.dsm_nd_val,
                )
            if self.config.output_kup:
                common.save_raster(
                    self.config.output_dir + "/Kup_" + time_code + ".tif",
                    Kup,
                    self.dsm_trf_arr,
                    self.dsm_crs_wkt,
                    self.dsm_nd_val,
                )
            if self.config.output_kdown:
                common.save_raster(
                    self.config.output_dir + "/Kdown_" + time_code + ".tif",
                    Kdown,
                    self.dsm_trf_arr,
                    self.dsm_crs_wkt,
                    self.dsm_nd_val,
                )
            if self.config.output_lup:
                common.save_raster(
                    self.config.output_dir + "/Lup_" + time_code + ".tif",
                    Lup,
                    self.dsm_trf_arr,
                    self.dsm_crs_wkt,
                    self.dsm_nd_val,
                )
            if self.config.output_ldown:
                common.save_raster(
                    self.config.output_dir + "/Ldown_" + time_code + ".tif",
                    Ldown,
                    self.dsm_trf_arr,
                    self.dsm_crs_wkt,
                    self.dsm_nd_val,
                )
            if self.config.output_sh:
                common.save_raster(
                    self.config.output_dir + "/Shadow_" + time_code + ".tif",
                    shadow,
                    self.dsm_trf_arr,
                    self.dsm_crs_wkt,
                    self.dsm_nd_val,
                )
            if self.config.output_kdiff:
                common.save_raster(
                    self.config.output_dir + "/Kdiff_" + time_code + ".tif",
                    dRad,
                    self.dsm_trf_arr,
                    self.dsm_crs_wkt,
                    self.dsm_nd_val,
                )

            # Sky view image of patches
            if (
                i == 0
                and PLT is True
                and self.config.plot_poi_patches
                and self.config.use_aniso
                and self.poi_pixel_xys is not None
            ):
                for k in range(self.poi_pixel_xys.shape[0]):
                    Lsky_patch_characteristics[:, 2] = patch_characteristics[:, k]
                    skyviewimage_out = self.config.output_dir + "/POI_" + str(self.poi_names[k]) + ".png"
                    PolarBarPlot(
                        Lsky_patch_characteristics,
                        self.environ_data.altitude[i],
                        self.environ_data.azimuth[i],
                        "Hemisphere partitioning",
                        skyviewimage_out,
                        0,
                        5,
                        0,
                    )

        # Save POI results
        if self.poi_results:
            self.save_poi_results(self.dsm_trf_arr, self.dsm_crs_wkt)

        # Save WOI results
        if self.woi_results:
            self.save_woi_results(self.dsm_trf_arr, self.dsm_crs_wkt)

        # Save Tree Planter results
        if self.config.output_tree_planter:
            pos = 1 if self.params.Tmrt_params.Value.posture == "Standing" else 0

            settingsHeader = [
                "UTC",
                "posture",
                "onlyglobal",
                "landcover",
                "anisotropic",
                "cylinder",
                "albedo_walls",
                "albedo_ground",
                "emissivity_walls",
                "emissivity_ground",
                "absK",
                "absL",
                "elevation",
                "patch_option",
            ]
            settingsFmt = (
                "%i",
                "%i",
                "%i",
                "%i",
                "%i",
                "%i",
                "%1.2f",
                "%1.2f",
                "%1.2f",
                "%1.2f",
                "%1.2f",
                "%1.2f",
                "%1.2f",
                "%i",
            )
            settingsData = np.array(
                [
                    [
                        int(self.config.utc),
                        pos,
                        self.config.only_global,
                        self.config.use_landcover,
                        self.config.use_aniso,
                        self.config.person_cylinder,
                        self.params.Albedo.Effective.Value.Walls,
                        self.params.Albedo.Effective.Value.Cobble_stone_2014a,
                        self.params.Emissivity.Value.Walls,
                        self.params.Emissivity.Value.Cobble_stone_2014a,
                        self.params.Tmrt_params.Value.absK,
                        self.params.Tmrt_params.Value.absL,
                        self.location["altitude"],
                        self.shadow_mats.patch_option,
                    ]
                ]
            )
            np.savetxt(
                self.config.output_dir + "/treeplantersettings.txt",
                settingsData,
                fmt=settingsFmt,
                header=", ".join(settingsHeader),
                delimiter=" ",
            )

        # Save average Tmrt raster
        tmrtplot = tmrtplot / self.iters_total
        common.save_raster(
            self.config.output_dir + "/Tmrt_average.tif",
            tmrtplot,
            self.dsm_trf_arr,
            self.dsm_crs_wkt,
            self.dsm_nd_val,
        )
