import json
import logging
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

from ... import common
from ...class_configs import (
    EnvironData,
    RasterData,
    ShadowMatrices,
    SolweigConfig,
    SvfData,
    TgMaps,
    WallsData,
)
from ...tile_manager import TileManager
from . import PET_calculations
from . import Solweig_2025a_calc_forprocessing as so
from . import UTCI_calculations as utci
from .CirclePlotBar import PolarBarPlot
from .wallsAsNetCDF import walls_as_netcdf

try:
    from matplotlib import pyplot as plt

    PLT = True
except ImportError:
    PLT = False


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    raster_data: RasterData
    location: Dict[str, float]
    svf_data: SvfData
    environ_data: EnvironData
    tg_maps: TgMaps
    shadow_mats: ShadowMatrices
    walls_data: WallsData

    def __init__(
        self,
        config: SolweigConfig,
        params_json_path: str,
        amax_local_window_m: int = 100,
        amax_local_perc: float = 99.9,
        use_tiled_loading: bool = False,
        tile_size: int = 1000,
    ):
        """Initialize the SOLWEIG runner with configuration and parameters."""
        logger.info("Starting SOLWEIG setup")
        self.config = config
        self.use_tiled_loading = use_tiled_loading
        self.tile_size = tile_size
        self.config.validate()
        # Progress tracking settings
        self.progress = None
        self.iters_total = None
        self.iters_count = 0
        self.proceed = True
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

        # Initialize SVF and Raster data
        # Check if tiled loading is enabled
        if self.use_tiled_loading:
            logger.info("Using tiled loading for raster data")

            # Get metadata from DSM to initialize TileManager and Location
            dsm_meta = common.get_raster_metadata(self.config.dsm_path)
            rows = dsm_meta["rows"]
            cols = dsm_meta["cols"]
            # Transform is always in GDAL format [c, a, b, f, d, e]
            pixel_size = dsm_meta["transform"][1]  # transform[1] is pixel width

            # Check if svf.tif exists, if not unzip
            # This is critical because tiled loading expects files to be present
            svf_file = Path(self.config.working_dir) / "svf.tif"
            if not svf_file.exists():
                logger.info("Unzipping SVF files for tiled access...")
                with zipfile.ZipFile(self.config.svf_path, "r") as zip_ref:
                    zip_ref.extractall(self.config.working_dir)

            # Initialize TileManager
            self.tile_manager = TileManager(
                rows=rows,
                cols=cols,
                tile_size=self.tile_size,
                pixel_size=pixel_size,
                buffer_dist=150.0,  # Fixed buffer
            )

            # Location data from metadata
            # Transform is always in GDAL format [c, a, b, f, d, e]
            left_x = dsm_meta["transform"][0]  # c (xoff)
            top_y = dsm_meta["transform"][3]  # f (yoff)
            lng, lat = common.xy_to_lnglat(dsm_meta["crs"], left_x, top_y)

            # Altitude approximation (we don't have full DSM loaded)
            # Read a small window from center
            center_r, center_c = rows // 2, cols // 2
            center_val = common.read_raster_window(
                self.config.dsm_path, (slice(center_r, center_r + 1), slice(center_c, center_c + 1))
            )
            alt = float(center_val[0, 0])
            if alt < 0:
                alt = 3

            self.location = {"longitude": lng, "latitude": lat, "altitude": alt}

            # Store metadata for later use
            self.rows = rows
            self.cols = cols
            # Transform is already in GDAL format [c, a, b, f, d, e]
            self.transform = dsm_meta["transform"]
            self.crs = dsm_meta["crs"]

            # We do NOT instantiate RasterData/SvfData here.
            self.raster_data = None
            self.svf_data = None
            self.shadow_mats = None
            self.tg_maps = None
            self.walls_data = None

            # Store params for tiled instantiation
            self.amax_local_window_m = amax_local_window_m
            self.amax_local_perc = amax_local_perc

        else:
            logger.info("Using eager loading for raster data")
            self.svf_data = SvfData(self.config)
            self.raster_data = RasterData(
                self.config,
                self.params,
                self.svf_data,
                amax_local_window_m,
                amax_local_perc,
            )
            # Location data
            left_x = self.raster_data.trf_arr[0]
            top_y = self.raster_data.trf_arr[3]
            lng, lat = common.xy_to_lnglat(self.raster_data.crs_wkt, left_x, top_y)
            alt = float(np.nanmedian(self.raster_data.dsm))
            if alt < 0:
                alt = 3
            self.location = {"longitude": lng, "latitude": lat, "altitude": alt}

            self.rows = self.raster_data.rows
            self.cols = self.raster_data.cols
            self.transform = self.raster_data.trf_arr
            self.crs = self.raster_data.crs_wkt

        # weather data
        if self.config.use_epw_file:
            self.environ_data = self.load_epw_weather()
            logger.info("Weather data loaded from EPW file")
        else:
            self.environ_data = self.load_met_weather(header_rows=1, delim=" ")
            logger.info("Weather data loaded from MET file")

        if self.config.poi_path:
            self.load_poi_data()
            logger.info("POI data loaded from %s", self.config.poi_path)

        if self.config.woi_path:
            self.load_woi_data()
            logger.info("WOI data loaded from %s", self.config.woi_path)

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
        met_data = np.loadtxt(met_path_str, skiprows=header_rows, delimiter=delim, dtype=np.float32)
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
        patch_characteristics = np.zeros((n_patches, n_pois), dtype=np.float32)

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
            self.raster_data.dsm,
            self.raster_data.scale,
            self.raster_data.rows,
            self.raster_data.cols,
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
            self.raster_data.cdsm,
            self.raster_data.tdsm,
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
            self.raster_data.buildings,
            self.location,
            self.environ_data.psi[iter],
            self.config.use_landcover,
            self.raster_data.lcgrid,
            self.environ_data.dectime[iter],
            self.environ_data.altmax[iter],
            self.raster_data.wallaspect,
            self.raster_data.wallheight,
            int(self.config.person_cylinder),  # expects int though should work either way
            elvis,
            self.environ_data.Ta[iter],
            self.environ_data.RH[iter],
            self.environ_data.radG[iter],
            self.environ_data.radD[iter],
            self.environ_data.radI[iter],
            self.environ_data.P[iter],
            self.raster_data.amaxvalue,
            self.raster_data.bush,
            self.environ_data.Twater[iter],
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
            self.raster_data.svfbuveg,
            firstdaytime,
            timeadd,
            timestepdec,
            self.tg_maps.Tgmap1,
            self.tg_maps.Tgmap1E,
            self.tg_maps.Tgmap1S,
            self.tg_maps.Tgmap1W,
            self.tg_maps.Tgmap1N,
            self.environ_data.CI[iter],
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
        """Run the SOLWEIG model."""
        if self.use_tiled_loading:
            self.run_tiled()
        else:
            self.run_standard()

    def run_standard(self) -> None:
        # Initialize execution-specific data structures (same pattern as run_tiled)
        logger.info("Initializing data for standard execution...")

        # Import shadow matrices (Anisotropic sky)
        self.shadow_mats = ShadowMatrices(self.config, self.params, self.svf_data)
        logger.info("Shadow matrices initialized")

        # Ts parameterisation maps
        self.tg_maps = TgMaps(
            self.config.use_landcover,
            self.params,
            self.raster_data,
        )
        logger.info("TgMaps initialized")

        self.walls_data = WallsData(
            self.config,
            self.params,
            self.raster_data,
            self.environ_data,
            self.tg_maps,
        )
        logger.info("WallsData initialized")

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
            I0_array = np.zeros_like(first_unique_day, dtype=np.float32)
        else:
            first_unique_day = self.environ_data.DOY.copy()
            I0_array = np.zeros_like(self.environ_data.DOY, dtype=np.float32)
        # For Tmrt plot
        tmrt_agg = np.zeros((self.raster_data.rows, self.raster_data.cols), dtype=np.float32)
        # Number of iterations
        num = len(self.environ_data.Ta)
        # Prepare progress tracking
        self.prep_progress(num)
        logger.info("Progress tracking prepared for %d iterations", num)
        elvis = 0.0
        #
        for i in range(num):
            self.proceed = self.iter_progress()
            if not self.proceed:
                break
            self.iters_count += 1
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
                first,
                second,
                firstdaytime,
                timeadd,
                timestepdec,
                posture,
            )

            # Aggregate Tmrt
            # Guard against NaN and Inf - replace non-finite with avg if available
            if (~np.isfinite(Tmrt)).any() and self.iters_count > 1:
                logger.warning("Tmrt contains non-finite values, replacing with preceding average.")
                tmrt_avg = tmrt_agg / self.iters_count
                tmrt_agg = np.where(np.isfinite(Tmrt), tmrt_agg + Tmrt, tmrt_avg)
            elif (~np.isfinite(tmrt_agg)).any():
                raise ValueError("Tmrt aggregation contains non-finite values.")
            else:
                tmrt_agg = tmrt_agg + Tmrt

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
                        "SVF_bv": self.raster_data.svfbuveg[row_idx, col_idx],
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
                        self.raster_data.rows,
                        self.raster_data.cols,
                        self.walls_data.met_for_xarray,
                        i,
                        self.raster_data.dsm,
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
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                    coerce_f64_to_f32=True,
                )
            if self.config.output_kup:
                common.save_raster(
                    self.config.output_dir + "/Kup_" + time_code + ".tif",
                    Kup,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                    coerce_f64_to_f32=True,
                )
            if self.config.output_kdown:
                common.save_raster(
                    self.config.output_dir + "/Kdown_" + time_code + ".tif",
                    Kdown,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                    coerce_f64_to_f32=True,
                )
            if self.config.output_lup:
                common.save_raster(
                    self.config.output_dir + "/Lup_" + time_code + ".tif",
                    Lup,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                    coerce_f64_to_f32=True,
                )
            if self.config.output_ldown:
                common.save_raster(
                    self.config.output_dir + "/Ldown_" + time_code + ".tif",
                    Ldown,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                    coerce_f64_to_f32=True,
                )
            if self.config.output_sh:
                common.save_raster(
                    self.config.output_dir + "/Shadow_" + time_code + ".tif",
                    shadow,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                    coerce_f64_to_f32=True,
                )
            if self.config.output_kdiff:
                common.save_raster(
                    self.config.output_dir + "/Kdiff_" + time_code + ".tif",
                    dRad,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                    coerce_f64_to_f32=True,
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

        # Abort if loop was broken
        if not self.proceed:
            return

        # Save POI results
        if self.poi_results:
            self.save_poi_results()

        # Save WOI results
        if self.woi_results:
            self.save_woi_results()

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
                ],
                dtype=np.float32,
            )
            np.savetxt(
                self.config.output_dir + "/treeplantersettings.txt",
                settingsData,
                fmt=settingsFmt,
                header=", ".join(settingsHeader),
                delimiter=" ",
            )

        # Save average Tmrt raster
        if self.iters_count > 0:
            tmrt_avg = tmrt_agg / self.iters_count
            common.save_raster(
                self.config.output_dir + "/Tmrt_average.tif",
                tmrt_avg,
                self.raster_data.trf_arr,
                self.raster_data.crs_wkt,
                self.raster_data.nd_val,
                coerce_f64_to_f32=True,
            )

    def run_tiled(self) -> None:
        """Run SOLWEIG with tiled loading (Tile -> Timestep loop)."""
        logger.info("Starting tiled execution")

        # Posture settings (same as standard)
        if self.params.Tmrt_params.Value.posture == "Standing":
            posture = self.params.Posture.Standing.Value
        else:
            posture = self.params.Posture.Sitting.Value

        first = np.round(posture.height)
        if first == 0.0:
            first = 1.0
        second = np.round(posture.height * 20.0)

        # Time variables
        num = len(self.environ_data.Ta)
        if num == 0:
            logger.error("No timesteps to process")
            return

        # Validate environ_data arrays have consistent length
        if not all(
            len(arr) == num
            for arr in [self.environ_data.YYYY, self.environ_data.DOY, self.environ_data.hours, self.environ_data.minu]
        ):
            logger.error("Inconsistent lengths in environ_data arrays")
            return

        if self.environ_data.Ta.__len__() == 1:
            timestepdec = 0
        else:
            timestepdec = self.environ_data.dectime[1] - self.environ_data.dectime[0]

        # Prepare output files
        logger.info("Initializing output rasters...")
        time_codes = []
        for i in range(num):
            # Generate time code
            if self.environ_data.altitude[i] > 0:
                w = "D"
            else:
                w = "N"
            XH = "0" if self.environ_data.hours[i] < 10 else ""
            XM = "0" if self.environ_data.minu[i] < 10 else ""

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
            time_codes.append(time_code)

            # Create empty rasters for enabled outputs
            outputs = [
                ("output_tmrt", "Tmrt"),
                ("output_kup", "Kup"),
                ("output_kdown", "Kdown"),
                ("output_lup", "Lup"),
                ("output_ldown", "Ldown"),
                ("output_sh", "Shadow"),
                ("output_kdiff", "Kdiff"),
            ]

            for cfg_attr, prefix in outputs:
                if getattr(self.config, cfg_attr):
                    out_path = str(self.config.output_dir) + "/" + prefix + "_" + time_code + ".tif"
                    common.create_empty_raster(
                        out_path,
                        self.rows,
                        self.cols,
                        self.transform,
                        str(self.crs) if self.crs else "",
                        nodata=-9999.0,
                    )

        # Create average Tmrt raster
        common.create_empty_raster(
            str(self.config.output_dir) + "/Tmrt_average.tif",
            self.rows,
            self.cols,
            self.transform,
            str(self.crs) if self.crs else "",
            nodata=-9999.0,
        )

        # Prepare progress
        self.prep_progress(self.tile_manager.total_tiles * num)

        # 1. Initialize State for all tiles
        logger.info("Initializing state for all tiles...")
        tile_states = []
        tmrt_agg_tiles = []  # Store aggregation per tile

        tiles_list = list(self.tile_manager.get_tiles())
        if len(tiles_list) == 0:
            logger.error("No tiles generated by TileManager")
            return

        logger.info(f"Initializing {len(tiles_list)} tiles...")

        for tile in tiles_list:
            # Load minimal data for TgMaps (lcgrid)
            if self.config.use_landcover:
                lcgrid = common.read_raster_window(self.config.lc_path, tile.full_slice)
            else:
                lcgrid = None

            # Mock RasterData for TgMaps initialization
            mock_rd = SimpleNamespace(
                rows=tile.full_shape[0],
                cols=tile.full_shape[1],
                lcgrid=lcgrid,
            )

            tg_maps = TgMaps(self.config.use_landcover, self.params, mock_rd, quiet=True)
            tile_states.append(tg_maps)
            tmrt_agg_tiles.append(np.zeros(tile.core_shape, dtype=np.float32))

        # Reset time variables
        elvis = 0.0
        firstdaytime = 1.0
        timeadd = 0.0

        # Cache for steradians (constant across tiles and timesteps)
        cached_steradians = None

        # 2. Iterate Timesteps
        for i in range(num):
            logger.info(f"Processing timestep {i + 1}/{num}")

            # Capture current time state for this timestep to ensure all tiles use the same time
            current_firstdaytime = firstdaytime
            current_timeadd = timeadd
            current_timestepdec = timestepdec

            # Variables to hold the next state (will be set by tiles)
            next_firstdaytime = None
            next_timeadd = None
            next_timestepdec = None  # 3. Iterate Tiles
            for tile_idx, tile in enumerate(self.tile_manager.get_tiles()):
                self.proceed = self.iter_progress()
                if not self.proceed:
                    break

                # Load Tile Data (Expensive IO)
                self.svf_data = SvfData(self.config, tile_spec=tile)
                self.raster_data = RasterData(
                    self.config,
                    self.params,
                    self.svf_data,
                    self.amax_local_window_m,
                    self.amax_local_perc,
                    tile_spec=tile,
                )

                # Restore State
                self.tg_maps = tile_states[tile_idx]

                # Initialize other components for this tile
                # ShadowMatrices and WallsData are stateless (reloaded/recalc per step)
                self.shadow_mats = ShadowMatrices(self.config, self.params, self.svf_data, tile_spec=tile)

                # Restore cached steradians if available
                if cached_steradians is not None:
                    self.shadow_mats.steradians = cached_steradians

                self.walls_data = WallsData(
                    self.config,
                    self.params,
                    self.raster_data,
                    self.environ_data,
                    self.tg_maps,
                    tile_spec=tile,
                )

                # Run Calculation
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
                    res_firstdaytime,
                    res_timestepdec,
                    res_timeadd,
                    Tgmap1_new,
                    Tgmap1E_new,
                    Tgmap1S_new,
                    Tgmap1W_new,
                    Tgmap1N_new,
                    Keast,
                    Ksouth,
                    Kwest,
                    Knorth,
                    Least,
                    Lsouth,
                    Lwest,
                    Lnorth,
                    KsideI,
                    TgOut1_new,
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
                    steradians_new,
                    voxelTable,
                ) = self.calc_solweig(
                    i,
                    elvis,
                    first,
                    second,
                    current_firstdaytime,
                    current_timeadd,
                    current_timestepdec,
                    posture,
                )

                # Explicitly update tile state with new thermal arrays
                tile_states[tile_idx].Tgmap1 = Tgmap1_new
                tile_states[tile_idx].Tgmap1E = Tgmap1E_new
                tile_states[tile_idx].Tgmap1S = Tgmap1S_new
                tile_states[tile_idx].Tgmap1W = Tgmap1W_new
                tile_states[tile_idx].Tgmap1N = Tgmap1N_new
                tile_states[tile_idx].TgOut1 = TgOut1_new

                # Update steradians cache and shadow_mats
                if steradians_new is not None:
                    self.shadow_mats.steradians = steradians_new

                # Capture the next state (idempotent across tiles)
                next_firstdaytime = res_firstdaytime
                next_timestepdec = res_timestepdec
                next_timeadd = res_timeadd

                # Cache steradians if calculated (only happens on first iteration)
                if cached_steradians is None and steradians_new is not None and np.any(steradians_new):
                    cached_steradians = steradians_new

                # Process POIs that intersect this tile's write window
                if self.poi_pixel_xys is not None:
                    write_win = tile.write_window

                    # Track POIs processed for validation (only on first timestep, first tile)
                    if i == 0 and tile_idx == 0:
                        self._poi_processed_flags = np.zeros(self.poi_pixel_xys.shape[0], dtype=bool)

                    for n in range(self.poi_pixel_xys.shape[0]):
                        idx, row_global, col_global = self.poi_pixel_xys[n]
                        row_global = int(row_global)
                        col_global = int(col_global)

                        # Check if POI is in this tile's write window
                        if (
                            write_win.row_off <= row_global < write_win.row_off + write_win.height
                            and write_win.col_off <= col_global < write_win.col_off + write_win.width
                        ):
                            # Convert global to tile-local coordinates
                            row_tile = row_global - tile.read_window.row_off
                            col_tile = col_global - tile.read_window.col_off

                            # Build result row
                            result_row = {
                                "poi_idx": idx,
                                "poi_name": self.poi_names[idx],
                                "col_idx": col_global,
                                "row_idx": row_global,
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
                                "kdown": Kdown[row_tile, col_tile],
                                "kup": Kup[row_tile, col_tile],
                                "keast": Keast[row_tile, col_tile],
                                "ksouth": Ksouth[row_tile, col_tile],
                                "kwest": Kwest[row_tile, col_tile],
                                "knorth": Knorth[row_tile, col_tile],
                                "ldown": Ldown[row_tile, col_tile],
                                "lup": Lup[row_tile, col_tile],
                                "least": Least[row_tile, col_tile],
                                "lsouth": Lsouth[row_tile, col_tile],
                                "lwest": Lwest[row_tile, col_tile],
                                "lnorth": Lnorth[row_tile, col_tile],
                                "Ta": self.environ_data.Ta[i],
                                "Tg": TgOut[row_tile, col_tile],
                                "RH": self.environ_data.RH[i],
                                "Esky": esky,
                                "Tmrt": Tmrt[row_tile, col_tile],
                                "I0": I0,
                                "CI": CI,
                                "Shadow": shadow[row_tile, col_tile],
                                "SVF_b": self.svf_data.svf[row_tile, col_tile],
                                "SVF_bv": self.raster_data.svfbuveg[row_tile, col_tile],
                                "KsideI": KsideI[row_tile, col_tile],
                            }

                            # Calculate PET and UTCI
                            WsPET = (1.1 / self.params.Wind_Height.Value.magl) ** 0.2 * self.environ_data.Ws[i]
                            WsUTCI = (10.0 / self.params.Wind_Height.Value.magl) ** 0.2 * self.environ_data.Ws[i]

                            resultPET = PET_calculations._PET(
                                self.environ_data.Ta[i],
                                self.environ_data.RH[i],
                                Tmrt[row_tile, col_tile],
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
                                self.environ_data.Ta[i], self.environ_data.RH[i], Tmrt[row_tile, col_tile], WsUTCI
                            )
                            result_row["UTCI"] = resultUTCI
                            result_row["CI_Tg"] = CI_Tg
                            result_row["CI_TgG"] = CI_TgG
                            result_row["KsideD"] = KsideD[row_tile, col_tile]
                            result_row["Lside"] = Lside[row_tile, col_tile]
                            result_row["diffDown"] = dRad[row_tile, col_tile]
                            result_row["Kside"] = Kside[row_tile, col_tile]

                            self.poi_results.append(result_row)

                            # Mark POI as processed (first timestep only)
                            if i == 0 and hasattr(self, "_poi_processed_flags"):
                                self._poi_processed_flags[n] = True

                # Process WOIs that intersect this tile's write window
                if self.config.use_wall_scheme and self.woi_pixel_xys is not None:
                    write_win = tile.write_window

                    # Track WOIs processed for validation (only on first timestep, first tile)
                    if i == 0 and tile_idx == 0:
                        self._woi_processed_flags = np.zeros(self.woi_pixel_xys.shape[0], dtype=bool)

                    for n in range(self.woi_pixel_xys.shape[0]):
                        idx, row_global, col_global = self.woi_pixel_xys[n]
                        row_global = int(row_global)
                        col_global = int(col_global)

                        # Check if WOI is in this tile's write window
                        if (
                            write_win.row_off <= row_global < write_win.row_off + write_win.height
                            and write_win.col_off <= col_global < write_win.col_off + write_win.width
                        ):
                            # Convert global to tile-local coordinates
                            row_tile = row_global - tile.read_window.row_off
                            col_tile = col_global - tile.read_window.col_off

                            # Extract wall data from voxelTable (uses tile-local coords)
                            temp_wall = voxelTable.loc[
                                ((voxelTable["ypos"] == row_tile) & (voxelTable["xpos"] == col_tile)), "wallTemperature"
                            ].to_numpy()
                            K_in = voxelTable.loc[
                                ((voxelTable["ypos"] == row_tile) & (voxelTable["xpos"] == col_tile)), "K_in"
                            ].to_numpy()
                            L_in = voxelTable.loc[
                                ((voxelTable["ypos"] == row_tile) & (voxelTable["xpos"] == col_tile)), "L_in"
                            ].to_numpy()
                            wallShade = voxelTable.loc[
                                ((voxelTable["ypos"] == row_tile) & (voxelTable["xpos"] == col_tile)), "wallShade"
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
                                "SVF": self.svf_data.svf[row_tile, col_tile],
                                "Ts": temp_wall,
                                "Kin": K_in,
                                "Lin": L_in,
                                "shade": wallShade,
                                "pixel_x": col_global,
                                "pixel_y": row_global,
                            }
                            self.woi_results.append(result_row)

                            # Mark WOI as processed (first timestep only)
                            if i == 0 and hasattr(self, "_woi_processed_flags"):
                                self._woi_processed_flags[n] = True

                # Crop results to core (remove buffer)
                core_slice = tile.core_slice()
                Tmrt_core = Tmrt[core_slice]

                # Write outputs
                time_code = time_codes[i]

                if self.config.output_tmrt:
                    common.write_raster_window(
                        str(self.config.output_dir) + "/Tmrt_" + time_code + ".tif",
                        Tmrt_core,
                        tile.write_window.to_slices(),
                    )
                if self.config.output_kup:
                    common.write_raster_window(
                        str(self.config.output_dir) + "/Kup_" + time_code + ".tif",
                        Kup[core_slice],
                        tile.write_window.to_slices(),
                    )
                if self.config.output_kdown:
                    common.write_raster_window(
                        str(self.config.output_dir) + "/Kdown_" + time_code + ".tif",
                        Kdown[core_slice],
                        tile.write_window.to_slices(),
                    )
                if self.config.output_lup:
                    common.write_raster_window(
                        str(self.config.output_dir) + "/Lup_" + time_code + ".tif",
                        Lup[core_slice],
                        tile.write_window.to_slices(),
                    )
                if self.config.output_ldown:
                    common.write_raster_window(
                        str(self.config.output_dir) + "/Ldown_" + time_code + ".tif",
                        Ldown[core_slice],
                        tile.write_window.to_slices(),
                    )
                if self.config.output_sh:
                    common.write_raster_window(
                        str(self.config.output_dir) + "/Shadow_" + time_code + ".tif",
                        shadow[core_slice],
                        tile.write_window.to_slices(),
                    )
                if self.config.output_kdiff:
                    common.write_raster_window(
                        str(self.config.output_dir) + "/Kdiff_" + time_code + ".tif",
                        dRad[core_slice],
                        tile.write_window.to_slices(),
                    )

                # Aggregate Tmrt (handle NaN and inf values safely)
                # Check for non-finite values and warn if found
                if (~np.isfinite(Tmrt_core)).any():
                    n_invalid = (~np.isfinite(Tmrt_core)).sum()
                    logger.warning(
                        f"Timestep {i + 1}, Tile {tile_idx + 1}: {n_invalid} non-finite Tmrt values detected"
                    )

                tmrt_core_safe = np.nan_to_num(Tmrt_core, nan=0.0, posinf=0.0, neginf=0.0)
                tmrt_agg_tiles[tile_idx] += tmrt_core_safe

                # Clean up tile data to free memory
                self.svf_data = None
                self.raster_data = None
                self.shadow_mats = None
                self.walls_data = None

            if not self.proceed:
                break

            # Update state for next timestep
            if next_firstdaytime is not None:
                firstdaytime = next_firstdaytime
                timestepdec = next_timestepdec
                timeadd = next_timeadd
            else:
                # This should not happen unless all tiles failed/were skipped
                logger.warning(f"No time state updates from timestep {i + 1}")
                if i < num - 1:  # Not the last timestep
                    logger.error("Missing time state updates before final timestep - results may be incorrect")
                if i < num - 1:  # Not the last timestep
                    logger.error("Missing time state updates before final timestep - results may be incorrect")

        # Abort if loop was broken
        if not self.proceed:
            return

        # Validate POI/WOI processing completeness
        if hasattr(self, "_poi_processed_flags") and self.poi_pixel_xys is not None:
            unprocessed_pois = np.where(~self._poi_processed_flags)[0]
            if len(unprocessed_pois) > 0:
                poi_list = unprocessed_pois.tolist()[:10]
                logger.warning(
                    f"{len(unprocessed_pois)} POIs were outside all tile boundaries and not processed: indices {poi_list}"
                )

        if hasattr(self, "_woi_processed_flags") and self.woi_pixel_xys is not None:
            unprocessed_wois = np.where(~self._woi_processed_flags)[0]
            if len(unprocessed_wois) > 0:
                woi_list = unprocessed_wois.tolist()[:10]
                logger.warning(
                    f"{len(unprocessed_wois)} WOIs were outside all tile boundaries and not processed: indices {woi_list}"
                )

        # Save POI results
        if self.poi_results:
            self.save_poi_results()

        # Save WOI results
        if self.woi_results:
            self.save_woi_results()

        # Save Tree Planter results (if needed)
        if self.config.output_tree_planter:
            logger.info("Generating tree planter settings file...")
            # We need shadow_mats for patch_option, but it was cleaned up
            # Recreate it temporarily from the last tile
            tiles_for_output = list(self.tile_manager.get_tiles())
            if len(tiles_for_output) == 0:
                logger.error("Cannot generate tree planter settings: no tiles available")
            else:
                last_tile = tiles_for_output[-1]
            svf_data_temp = SvfData(self.config, tile_spec=last_tile)
            shadow_mats_temp = ShadowMatrices(self.config, self.params, svf_data_temp, tile_spec=last_tile)

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
                        shadow_mats_temp.patch_option,
                    ]
                ],
                dtype=np.float32,
            )
            np.savetxt(
                self.config.output_dir + "/treeplantersettings.txt",
                settingsData,
                fmt=settingsFmt,
                header=", ".join(settingsHeader),
                delimiter=" ",
            )

        # Write average Tmrt for all tiles
        if num > 0:
            for tile_idx, tile in enumerate(self.tile_manager.get_tiles()):
                tmrt_avg_tile = tmrt_agg_tiles[tile_idx] / num
                common.write_raster_window(
                    str(self.config.output_dir) + "/Tmrt_average.tif",
                    tmrt_avg_tile,
                    tile.write_window.to_slices(),
                )
