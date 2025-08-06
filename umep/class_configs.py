from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import common


class SvfData:
    """Class to handle SVF data loading and processing."""

    def __init__(self, in_path_str: str, use_cdsm: bool = False):
        """
        Loads SVF and shadow matrix results from disk and returns a SVFResults dataclass instance.
        """
        in_path_str = str(common.check_path(in_path_str, make_dir=False))

        # Load SVF rasters
        self.svf, _, _, _ = common.load_raster(in_path_str + "/" + "svf.tif")
        self.svf_east, _, _, _ = common.load_raster(in_path_str + "/" + "svfE.tif")
        self.svf_south, _, _, _ = common.load_raster(in_path_str + "/" + "svfS.tif")
        self.svf_west, _, _, _ = common.load_raster(in_path_str + "/" + "svfW.tif")
        self.svf_north, _, _, _ = common.load_raster(in_path_str + "/" + "svfN.tif")
        if use_cdsm:
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
        if (self.met_path is None and self.epw_path is None) or (self.met_path and self.epw_path):
            raise ValueError("Provide either MET or EPW weather file.")
        if self.epw_path is not None:
            if self.epw_start_date is None or self.epw_end_date is None:
                raise ValueError("EPW start and end dates must be provided if EPW path is set.")
            # year,month,day,hour
            # parse the start and end dates to lists
            try:
                start_date = [int(x) for x in self.epw_start_date.split(",")]
                end_date = [int(x) for x in self.epw_end_date.split(",")]
                if len(start_date) != 4 or len(end_date) != 4:
                    raise ValueError("EPW start and end dates must be in the format: year,month,day,hour")
            except ValueError as err:
                raise ValueError(f"Invalid EPW date format: {self.epw_start_date} or {self.epw_end_date}") from err
            if self.epw_hours is None:
                self.epw_hours = list(range(24))  # Default to all hours if not specified
            elif isinstance(self.epw_hours, str):
                self.epw_hours = [int(h) for h in self.epw_hours.split(",")]
            if not all(0 <= h < 24 for h in self.epw_hours):
                raise ValueError("EPW hours must be between 0 and 23.")
        # Add more validation as needed
