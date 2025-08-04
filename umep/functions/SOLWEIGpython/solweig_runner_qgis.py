from typing import Any

import numpy as np

from ...functions.SOLWEIGpython.wallOfInterest import pointOfInterest
from .solweig_runner import SolweigConfig, SolweigRun, WeatherData


class SolweigRunQgis(SolweigRun):
    """Run SOLWEIG in QGIS environment."""

    def __init__(self, config_path_str: str, feedback: Any):
        """ """
        config = SolweigConfig()
        config.from_file(config_path_str)
        super().__init__(config, qgis_env=True)
        self.set_feedback(feedback)

    def load_poi_data(self, trf_arr: list[int]) -> tuple[Any, Any]:
        """Load points of interest (POIs) from a file."""
        scale = 1 / trf_arr[1]
        poi_names, poi_xys = pointOfInterest(self.config.poi_file, self.config.poi_field, scale, trf_arr)
        for poi_name, poi_xy in zip(poi_names, poi_xys):
            self.poi_names.append(poi_name)
            self.poi_pixel_xys.append((poi_xy[0], poi_xy[1], poi_xy[2]))

    def save_poi_results(self, crs_wkt: str) -> None:
        for k in range(len(self.poi_pixel_xys)):
            data_out = self.config.output_dir + "/POI_" + str(self.poi_names[k]) + ".txt"
            np.savetxt(data_out, self.poi_pixel_xys[k], delimiter=" ", header=header, comments="")

    def load_met_weather(self, header_rows: int = 1, delim: str = " ") -> WeatherData:
        """Load weather data from a MET file."""
        met_data = np.loadtxt(self.config.met_path, skiprows=header_rows, delimiter=delim)
        return WeatherData(
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
        )
