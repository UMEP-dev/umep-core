from dataclasses import dataclass
from typing import Optional

import numpy as np

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
