import json
from types import SimpleNamespace
from typing import Any

import numpy as np

from ... import common
from ...class_configs import EnvironData, ShadowMatrices, SolweigConfig, SvfData, TgMaps, WallsData
from ...util.SEBESOLWEIGCommonFiles.clearnessindex_2013b import clearnessindex_2013b
from . import PET_calculations
from . import Solweig_2025a_calc_forprocessing as so
from . import UTCI_calculations as utci
from .CirclePlotBar import PolarBarPlot
from .patch_characteristics import hemispheric_image
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

    def __init__(self, config: SolweigConfig, params_json_path: str):
        """Initialize the SOLWEIG runner with configuration and parameters."""
        self.config = config
        self.config.validate()
        # Progress tracking settings
        self.progress = None
        self.iters_total: int | None = None
        self.iters_count: int = 0
        # Initialize POI data
        self.poi_names: list[Any] = []
        self.poi_pixel_xys: np.ndarray | None = None
        self.poi_results = []
        # Initialize WOI data
        self.woi_names: list[Any] = []
        self.woi_pixel_xys: np.ndarray | None = None
        self.woi_results = []
        # Load parameters from JSON file
        params_path = common.check_path(params_json_path)
        try:
            with open(params_path) as f:
                params_dict = json.load(f)
                self.params = dict_to_namespace(params_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load parameters from {params_json_path}: {e}")
        # Prepare core data
        self.dsm_arr: np.ndarray | None = None
        self.scale: float | None = None
        self.rows: int | None = None
        self.cols: int | None = None
        self.location: dict[str, float] | None = None

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

    def load_poi_data(self, trf_arr: list[float]) -> None:
        """Load point of interest (POI) data from a file."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def save_poi_results(self, trf_arr: list[float], crs_wkt: str) -> None:
        """Save results for points of interest (POIs) to files."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_woi_data(self, trf_arr: list[float]) -> None:
        """Load wall of interest (WOI) data from a file."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def save_woi_results(self, trf_arr: list[float], crs_wkt: str) -> None:
        """Save results for walls of interest (WOIs) to files."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def calc_solweig(
        self,
        iter: int,
        buildings: np.ndarray,
        vegdsm: np.ndarray,
        vegdsm2: np.ndarray,
        svfbuveg: np.ndarray,
        bush: np.ndarray,
        lcgrid: np.ndarray,
        wallaspect: np.ndarray,
        wallheight: np.ndarray,
        elvis: float,
        CI: float,
        amaxvalue: float,
        Twater: float,
        first: float,
        second: float,
        firstdaytime: float,
        timeadd: float,
        timestepdec: float,
        posture,
        svf_data: SvfData,
        environ_data: EnvironData,
        tg_maps: TgMaps,
        shadow_mats: ShadowMatrices,
        walls_data: WallsData,
    ):
        """
        Calculate SOLWEIG results for a given iteration.
        Separated from the main run method so that it can be overridden by subclasses.
        Over time we can simplify the function signature by passing consolidated parameters to solweig calc methods.
        """
        return so.Solweig_2025a_calc(
            iter,
            self.dsm_arr,
            self.scale,
            self.rows,
            self.cols,
            svf_data.svf,
            svf_data.svf_north,
            svf_data.svf_west,
            svf_data.svf_east,
            svf_data.svf_south,
            svf_data.svf_veg,
            svf_data.svf_veg_north,
            svf_data.svf_veg_east,
            svf_data.svf_veg_south,
            svf_data.svf_veg_west,
            svf_data.svf_veg_blocks_bldg_sh,
            svf_data.svf_veg_blocks_bldg_sh_east,
            svf_data.svf_veg_blocks_bldg_sh_south,
            svf_data.svf_veg_blocks_bldg_sh_west,
            svf_data.svf_veg_blocks_bldg_sh_north,
            vegdsm,
            vegdsm2,
            self.params.Albedo.Effective.Value.Walls,
            self.params.Tmrt_params.Value.absK,
            self.params.Tmrt_params.Value.absL,
            self.params.Emissivity.Value.Walls,
            posture.Fside,
            posture.Fup,
            posture.Fcyl,
            environ_data.altitude[iter],
            environ_data.azimuth[iter],
            environ_data.zen[iter],
            environ_data.jday[iter],
            self.config.use_veg_dem,
            self.config.only_global,
            buildings,
            self.location,
            environ_data.psi[iter],
            self.config.use_landcover,
            lcgrid,
            environ_data.dectime[iter],
            environ_data.altmax[iter],
            wallaspect,
            wallheight,
            self.config.person_cylinder,
            elvis,
            environ_data.Ta[iter],
            environ_data.RH[iter],
            environ_data.radG[iter],
            environ_data.radD[iter],
            environ_data.radI[iter],
            environ_data.P[iter],
            amaxvalue,
            bush,
            Twater,
            tg_maps.TgK,
            tg_maps.Tstart,
            tg_maps.alb_grid,
            tg_maps.emis_grid,
            tg_maps.TgK_wall,
            tg_maps.Tstart_wall,
            tg_maps.TmaxLST,
            tg_maps.TmaxLST_wall,
            first,
            second,
            svf_data.svfalfa,
            svfbuveg,
            firstdaytime,
            timeadd,
            timestepdec,
            tg_maps.Tgmap1,
            tg_maps.Tgmap1E,
            tg_maps.Tgmap1S,
            tg_maps.Tgmap1W,
            tg_maps.Tgmap1N,
            CI,
            tg_maps.TgOut1,
            shadow_mats.diffsh,
            shadow_mats.shmat,
            shadow_mats.vegshmat,
            shadow_mats.vbshvegshmat,
            self.config.use_aniso,
            shadow_mats.asvf,
            shadow_mats.patch_option,
            walls_data.voxelMaps,
            walls_data.voxelTable,
            environ_data.Ws[iter],
            self.config.use_wall_scheme,
            walls_data.timeStep,
            shadow_mats.steradians,
            walls_data.walls_scheme,
            walls_data.dirwalls_scheme,
        )

    def run(self) -> None:
        """Run the SOLWEIG algorithm."""
        # Load DSM
        self.dsm_arr, dsm_trf_arr, dsm_crs_wkt, dsm_nd_val = common.load_raster(self.config.dsm_path, bbox=None)
        self.scale = 1 / dsm_trf_arr[1]
        self.rows = self.dsm_arr.shape[0]
        self.cols = self.dsm_arr.shape[1]

        left_x = dsm_trf_arr[0]
        top_y = dsm_trf_arr[3]
        lng, lat = common.xy_to_lnglat(dsm_crs_wkt, left_x, top_y)
        alt = np.median(self.dsm_arr)
        if alt < 0:
            alt = 3
        self.location = {"longitude": lng, "latitude": lat, "altitude": alt}

        self.dsm_arr[self.dsm_arr == dsm_nd_val] = 0.0
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
            dem[dem == dem_nd_val] = 0.0
            # TODO: Check if this is needed re DSM ramifications
            if dem.min() < 0:
                demraise = np.abs(dem.min())
                dem = dem + demraise

        # Land cover
        if self.config.use_landcover:
            lc_path_str = str(common.check_path(self.config.lc_path))
            lcgrid, _, _, _ = common.load_raster(lc_path_str, bbox=None)
        else:
            lcgrid = None

        # Buildings from land cover option
        # TODO: Check intended logic here
        if not self.config.use_dem_for_buildings and lcgrid is not None:
            # Create building boolean raster from either land cover if no DEM is used
            buildings = np.copy(lcgrid)
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
        # Save buildings raster if requested
        if self.config.save_buildings:
            common.save_raster(
                self.config.output_dir + "/buildings.tif",
                buildings,
                dsm_trf_arr,
                dsm_crs_wkt,
                dsm_nd_val,
            )

        # Vegetation
        if self.config.use_veg_dem:
            vegdsm, _, _, _ = common.load_raster(self.config.cdsm_path, bbox=None)
            if self.config.tdsm_path:
                vegdsm2, _, _, _ = common.load_raster(self.config.tdsm_path, bbox=None)
            else:
                vegdsm2 = vegdsm * self.params.Tree_settings.Value.Trunk_ratio
        else:
            vegdsm = None
            vegdsm2 = None

        # Load SVF data
        svf_data = SvfData(self.config)

        if self.config.use_veg_dem:
            # amaxvalue
            vegmax = vegdsm.max()
            amaxvalue = self.dsm_arr.max() - self.dsm_arr.min()
            amaxvalue = np.maximum(amaxvalue, vegmax)
            # Elevation vegdsms if buildingDEM includes ground heights
            vegdsm = vegdsm + self.dsm_arr
            vegdsm[vegdsm == self.dsm_arr] = 0
            vegdsm2 = vegdsm2 + self.dsm_arr
            vegdsm2[vegdsm2 == self.dsm_arr] = 0
            # % Bush separation
            bush = np.logical_not(vegdsm2 * vegdsm) * vegdsm
            svfbuveg = svf_data.svf - (1.0 - svf_data.svf_veg) * (
                1.0 - self.params.Tree_settings.Value.Transmissivity
            )  # % major bug fixed 20141203
        else:
            svfbuveg = svf_data.svf
            bush = np.zeros([self.rows, self.cols])
            amaxvalue = 0

        # Load walls
        wallheight, _, _, _ = common.load_raster(self.config.wh_path, bbox=None)
        wallaspect, _, _, _ = common.load_raster(self.config.wa_path, bbox=None)

        # weather data
        if self.config.use_epw_file:
            environ_data = self.load_epw_weather()
        else:
            environ_data = self.load_met_weather(header_rows=1, delim=" ")

        # POIs check
        if self.config.poi_path:
            self.load_poi_data(dsm_trf_arr)

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

        # Import shadow matrices (Anisotropic sky)
        shadow_mats = ShadowMatrices(self.config, self.params, self.rows, self.cols, svf_data)

        # % Ts parameterisation maps
        tg_maps = TgMaps(self.config.use_landcover, lcgrid, self.params, self.rows, self.cols)

        # Import data for wall temperature parameterization
        # Use wall of interest
        if self.config.woi_path:
            self.load_woi_data(dsm_trf_arr)
        walls_data = WallsData(
            self.config,
            self.params,
            self.scale,
            self.rows,
            self.cols,
            environ_data,
            tg_maps,
            self.dsm_arr,
            lcgrid,
        )

        # Initialisation of time related variables
        if environ_data.Ta.__len__() == 1:
            timestepdec = 0
        else:
            timestepdec = environ_data.dectime[1] - environ_data.dectime[0]
        timeadd = 0.0
        firstdaytime = 1.0

        # Save hemispheric image
        if self.config.use_aniso and self.poi_pixel_xys is not None:
            patch_characteristics = hemispheric_image(
                self.poi_pixel_xys,
                shadow_mats.shmat,
                shadow_mats.vegshmat,
                shadow_mats.vbshvegshmat,
                walls_data.voxelMaps,
                walls_data.walls_scheme,
            )

        # Initiate array for I0 values plotting
        if np.unique(environ_data.DOY).shape[0] > 1:
            unique_days = np.unique(environ_data.DOY)
            first_unique_day = environ_data.DOY[unique_days[0] == environ_data.DOY]
            I0_array = np.zeros_like(first_unique_day)
        else:
            first_unique_day = environ_data.DOY.copy()
            I0_array = np.zeros_like(environ_data.DOY)
        # For Tmrt plot
        tmrtplot = np.zeros((self.rows, self.cols))
        # Number of iterations
        num = len(environ_data.Ta)
        # Prepare progress tracking
        self.prep_progress(num)
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
                if (environ_data.dectime[i] - np.floor(environ_data.dectime[i])) == 0 or (i == 0):
                    # Find average temperature for the current day
                    Twater = np.mean(environ_data.Ta[environ_data.jday == np.floor(environ_data.dectime[i])])

            # Nocturnal cloudfraction from Offerle et al. 2003
            # Check for start of day
            if (environ_data.dectime[i] - np.floor(environ_data.dectime[i])) == 0:
                # Find all current day idxs
                daylines = np.where(np.floor(environ_data.dectime) == environ_data.dectime[i])
                # np.where returns a tuple, so check the first element
                if len(daylines[0]) > 1:
                    # Get the altitudes for day's idxs
                    alt_day = environ_data.altitude[daylines[0]]
                    # Find all idxs with altitude greater than 1
                    alt2 = np.where(alt_day > 1)
                    # np.where returns a tuple, so check the first element
                    if len(alt2[0]) > 0:
                        # Take the first altitude greater than 1
                        rise = alt2[0][0]
                        # Calculate clearness index for the next time step after sunrise
                        [_, CI, _, _, _] = clearnessindex_2013b(
                            environ_data.zen[i + rise + 1],
                            environ_data.jday[i + rise + 1],
                            environ_data.Ta[i + rise + 1],
                            environ_data.RH[i + rise + 1] / 100.0,
                            environ_data.radG[i + rise + 1],
                            self.location,
                            environ_data.P[i + rise + 1],
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
                tg_maps.Tgmap1,
                tg_maps.Tgmap1E,
                tg_maps.Tgmap1S,
                tg_maps.Tgmap1W,
                tg_maps.Tgmap1N,
                Keast,
                Ksouth,
                Kwest,
                Knorth,
                Least,
                Lsouth,
                Lwest,
                Lnorth,
                KsideI,
                tg_maps.TgOut1,
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
                shadow_mats.steradians,
                voxelTable,
            ) = self.calc_solweig(
                i,
                buildings,
                vegdsm,
                vegdsm2,
                svfbuveg,
                bush,
                lcgrid,
                wallaspect,
                wallheight,
                elvis,
                CI,
                amaxvalue,
                Twater,
                first,
                second,
                firstdaytime,
                timeadd,
                timestepdec,
                posture,
                svf_data,
                environ_data,
                tg_maps,
                shadow_mats,
                walls_data,
            )
            # Save I0 for I0 vs. Kdown output plot to check if UTC is off
            if i < first_unique_day.shape[0]:
                I0_array[i] = I0
            elif i == first_unique_day.shape[0]:
                # Output I0 vs. Kglobal plot
                radG_for_plot = environ_data.radG[first_unique_day[0] == environ_data.DOY]
                dectime_for_plot = environ_data.dectime[first_unique_day[0] == environ_data.DOY]
                fig, ax = plt.subplots()
                ax.plot(dectime_for_plot, I0_array, label="I0")
                ax.plot(dectime_for_plot, radG_for_plot, label="Kglobal")
                ax.set_ylabel("Shortwave radiation [$Wm^{-2}$]")
                ax.set_xlabel("Decimal time")
                ax.set_title("UTC" + str(self.config.utc))
                ax.legend()
                fig.savefig(self.config.output_dir + "/metCheck.png", dpi=150)

            tmrtplot = tmrtplot + Tmrt

            if environ_data.altitude[i] > 0:
                w = "D"
            else:
                w = "N"

            if environ_data.hours[i] < 10:
                XH = "0"
            else:
                XH = ""

            if environ_data.minu[i] < 10:
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
                        "yyyy": environ_data.YYYY[i],
                        "id": environ_data.jday[i],
                        "it": environ_data.hours[i],
                        "imin": environ_data.minu[i],
                        "dectime": environ_data.dectime[i],
                        "altitude": environ_data.altitude[i],
                        "azimuth": environ_data.azimuth[i],
                        "kdir": radIout,
                        "kdiff": radDout,
                        "kglobal": environ_data.radG[i],
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
                        "Ta": environ_data.Ta[i],
                        "Tg": TgOut[row_idx, col_idx],
                        "RH": environ_data.RH[i],
                        "Esky": esky,
                        "Tmrt": Tmrt[row_idx, col_idx],
                        "I0": I0,
                        "CI": CI,
                        "Shadow": shadow[row_idx, col_idx],
                        "SVF_b": svf_data.svf[row_idx, col_idx],
                        "SVF_bv": svfbuveg[row_idx, col_idx],
                        "KsideI": KsideI[row_idx, col_idx],
                    }
                    # Recalculating wind speed based on powerlaw
                    WsPET = (1.1 / self.params.Wind_Height.Value.magl) ** 0.2 * environ_data.Ws[i]
                    WsUTCI = (10.0 / self.params.Wind_Height.Value.magl) ** 0.2 * environ_data.Ws[i]
                    resultPET = PET_calculations._PET(
                        environ_data.Ta[i],
                        environ_data.RH[i],
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
                        environ_data.Ta[i], environ_data.RH[i], Tmrt[row_idx, col_idx], WsUTCI
                    )
                    result_row["UTCI"] = resultUTCI
                    result_row["CI_Tg"] = CI_Tg
                    result_row["CI_TgG"] = CI_TgG
                    result_row["KsideD"] = KsideD[row_idx, col_idx]
                    result_row["Lside"] = Lside[row_idx, col_idx]
                    result_row["diffDown"] = dRad[row_idx, col_idx]
                    result_row["Kside"] = Kside[row_idx, col_idx]
                    self.poi_results.append(result_row)

            if self.woi_pixel_xys is not None:
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
                        "yyyy": environ_data.YYYY[i],
                        "id": environ_data.jday[i],
                        "it": environ_data.hours[i],
                        "imin": environ_data.minu[i],
                        "dectime": environ_data.dectime[i],
                        "Ta": environ_data.Ta[i],
                        "SVF": svf_data.svf[row_idx, col_idx],
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
                        walls_data.met_for_xarray,
                        i,
                        self.dsm_arr,
                        self.config.dsm_path,
                        netcdf_output,
                    )

            time_code = (
                str(int(environ_data.YYYY[i]))
                + "_"
                + str(int(environ_data.DOY[i]))
                + "_"
                + XH
                + str(int(environ_data.hours[i]))
                + XM
                + str(int(environ_data.minu[i]))
                + w
            )

            if self.config.output_tmrt:
                common.save_raster(
                    self.config.output_dir + "/Tmrt_" + time_code + ".tif",
                    Tmrt,
                    dsm_trf_arr,
                    dsm_crs_wkt,
                    dsm_nd_val,
                )
            if self.config.output_kup:
                common.save_raster(
                    self.config.output_dir + "/Kup_" + time_code + ".tif",
                    Kup,
                    dsm_trf_arr,
                    dsm_crs_wkt,
                    dsm_nd_val,
                )
            if self.config.output_kdown:
                common.save_raster(
                    self.config.output_dir + "/Kdown_" + time_code + ".tif",
                    Kdown,
                    dsm_trf_arr,
                    dsm_crs_wkt,
                    dsm_nd_val,
                )
            if self.config.output_lup:
                common.save_raster(
                    self.config.output_dir + "/Lup_" + time_code + ".tif",
                    Lup,
                    dsm_trf_arr,
                    dsm_crs_wkt,
                    dsm_nd_val,
                )
            if self.config.output_ldown:
                common.save_raster(
                    self.config.output_dir + "/Ldown_" + time_code + ".tif",
                    Ldown,
                    dsm_trf_arr,
                    dsm_crs_wkt,
                    dsm_nd_val,
                )
            if self.config.output_sh:
                common.save_raster(
                    self.config.output_dir + "/Shadow_" + time_code + ".tif",
                    shadow,
                    dsm_trf_arr,
                    dsm_crs_wkt,
                    dsm_nd_val,
                )
            if self.config.output_kdiff:
                common.save_raster(
                    self.config.output_dir + "/Kdiff_" + time_code + ".tif",
                    dRad,
                    dsm_trf_arr,
                    dsm_crs_wkt,
                    dsm_nd_val,
                )

            # Sky view image of patches
            if (self.config.aniso_path) and (i == 0) and (self.poi_pixel_xys is not None):
                for k in range(self.poi_pixel_xys.shape[0]):
                    Lsky_patch_characteristics[:, 2] = patch_characteristics[:, k]
                    skyviewimage_out = self.config.output_dir + "/POI_" + str(self.poi_names[k]) + ".png"
                    PolarBarPlot(
                        Lsky_patch_characteristics,
                        environ_data.altitude[i],
                        environ_data.azimuth[i],
                        "Hemisphere partitioning",
                        skyviewimage_out,
                        0,
                        5,
                        0,
                    )

        # Save POI results
        if self.poi_results:
            self.save_poi_results(dsm_trf_arr, dsm_crs_wkt)

        # Save WOI results
        if self.woi_results:
            self.save_woi_results(dsm_trf_arr, dsm_crs_wkt)

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
                        shadow_mats.patch_option,
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
            dsm_trf_arr,
            dsm_crs_wkt,
            dsm_nd_val,
        )
