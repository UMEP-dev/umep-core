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
from ...class_configs import EnvironData, RasterData, ShadowMatrices, SolweigConfig, SvfData, TgMaps, WallsData
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
    raster_data: RasterData
    location: Dict[str, float]
    svf_data: SvfData
    environ_data: EnvironData
    tg_maps: TgMaps
    shadow_mats: ShadowMatrices
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
        self.svf_data = SvfData(self.config)
        self.raster_data = RasterData(self.config, self.params, self.svf_data)
        # Location data
        left_x = self.raster_data.trf_arr[0]
        top_y = self.raster_data.trf_arr[3]
        lng, lat = common.xy_to_lnglat(self.raster_data.crs_wkt, left_x, top_y)
        alt = np.median(self.raster_data.dsm)
        if alt < 0:
            alt = 3
        self.location = {"longitude": lng, "latitude": lat, "altitude": alt}
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
        self.shadow_mats = ShadowMatrices(self.config, self.params, self.raster_data, self.svf_data)
        logger.info("Shadow matrices initialized")
        # % Ts parameterisation maps
        self.tg_maps = TgMaps(
            self.config.use_landcover,
            self.params,
            self.raster_data,
        )
        logger.info("TgMaps initialized")
        # Import data for wall temperature parameterization
        # Use wall of interest
        if self.config.woi_path:
            self.load_woi_data()
            logger.info("WOI data loaded from %s", self.config.woi_path)
        self.walls_data = WallsData(
            self.config,
            self.params,
            self.raster_data,
            self.environ_data,
            self.tg_maps,
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
        tmrt_agg = np.zeros((self.raster_data.rows, self.raster_data.cols))
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
                )
            if self.config.output_kup:
                common.save_raster(
                    self.config.output_dir + "/Kup_" + time_code + ".tif",
                    Kup,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                )
            if self.config.output_kdown:
                common.save_raster(
                    self.config.output_dir + "/Kdown_" + time_code + ".tif",
                    Kdown,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                )
            if self.config.output_lup:
                common.save_raster(
                    self.config.output_dir + "/Lup_" + time_code + ".tif",
                    Lup,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                )
            if self.config.output_ldown:
                common.save_raster(
                    self.config.output_dir + "/Ldown_" + time_code + ".tif",
                    Ldown,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                )
            if self.config.output_sh:
                common.save_raster(
                    self.config.output_dir + "/Shadow_" + time_code + ".tif",
                    shadow,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                )
            if self.config.output_kdiff:
                common.save_raster(
                    self.config.output_dir + "/Kdiff_" + time_code + ".tif",
                    dRad,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
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
        if self.iters_count > 0:
            tmrt_avg = tmrt_agg / self.iters_count
            common.save_raster(
                self.config.output_dir + "/Tmrt_average.tif",
                tmrt_avg,
                self.raster_data.trf_arr,
                self.raster_data.crs_wkt,
                self.raster_data.nd_val,
            )
