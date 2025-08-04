from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pvlib.iotools import read_epw
from rasterio.transform import Affine, rowcol

from .solweig_runner import SolweigConfig, SolweigRun, WeatherData


class SolweigRunStandalone(SolweigRun):
    """Run SOLWEIG in standalone mode without QGIS."""

    def __init__(self, config: SolweigConfig):
        super().__init__(config, qgis_env=False)

    def run(self):
        print("Running SOLWEIG in standalone mode.")

    def load_poi_data(self, trf_arr: list[int]) -> tuple[Any, Any]:
        """Load points of interest (POIs) from a file."""
        pois_gdf = gpd.read_file(self.config.poi_file)
        trf = Affine.from_gdal(*trf_arr)
        for n, (idx, row) in enumerate(pois_gdf.iterrows()):
            self.poi_names.append(idx)
            y, x = rowcol(trf, row["geometry"].centroid.x, row["geometry"].centroid.y)
            self.poi_pixel_xys.append((n, x, y))

    def save_poi_results(self, crs_wkt: str) -> None:
        poi_df = gpd.GeoDataFrame(self.poi_results, geometry="geometry", crs=crs_wkt)
        # Create a datetime column for multi-index
        poi_df["snapshot"] = pd.to_datetime(
            poi_df["yyyy"].astype(int).astype(str)
            + "-"
            + poi_df["id"].astype(int).astype(str).str.zfill(3)
            + " "
            + poi_df["it"].astype(int).astype(str).str.zfill(2)
            + ":"
            + poi_df["imin"].astype(int).astype(str).str.zfill(2),
            format="%Y-%j %H:%M",
        )
        # GPD doesn't handle multi-index
        poi_df.to_file(self.config.output_dir + "/POI.gpkg", driver="GPKG")

    def load_epw_weather(self) -> WeatherData:
        """Load weather data from an EPW file."""
        epw_df, epw_info = read_epw(self.config.epw_path)
        # Filter by date range
        filtered_df = epw_df.loc[self.config.epw_start_date : self.config.epw_end_date]
        # Filter by hours
        filtered_df = filtered_df[filtered_df.index.hour.isin(self.config.epw_hours)]
        # raise if empty
        if len(filtered_df) == 0:
            raise ValueError("No EPW dates intersect start and end dates and / or hours.")
        umep_df = pd.DataFrame(
            {
                "iy": filtered_df.index.year,
                "id": filtered_df.index.dayofyear,
                "it": filtered_df.index.hour,
                "imin": filtered_df.index.minute,
                "Q": -999,
                "QH": -999,
                "QE": -999,
                "Qs": -999,
                "Qf": -999,
                "Wind": filtered_df["wind_speed"],
                "RH": filtered_df["relative_humidity"],
                "Tair": filtered_df["temp_air"],
                "pres": filtered_df["atmospheric_pressure"].astype(float),  # Pascal, ensure float
                "rain": -999,
                "Kdown": filtered_df["ghi"],
                "snow": filtered_df["snow_depth"],
                "ldown": filtered_df["ghi_infrared"],
                "fcld": filtered_df["total_sky_cover"],
                "wuh": filtered_df["precipitable_water"],
                "xsmd": -999,
                "lai_hr": -999,
                "Kdiff": filtered_df["dhi"],
                "Kdir": filtered_df["dni"],
                "Wdir": filtered_df["wind_direction"],
            }
        )
        # Check for negative Kdown values
        umep_df_filt = umep_df[(umep_df["Kdown"] < 0) & (umep_df["Kdown"] > 1300)]
        if len(umep_df_filt):
            raise ValueError(
                "Error: Kdown - beyond what is expected",
            )

        # use -999 for NaN to mesh with UMEP
        umep_df = umep_df.fillna(-999)

        return WeatherData(
            DOY=umep_df["id"].to_numpy(),
            hours=umep_df["it"].to_numpy(),
            minu=umep_df["imin"].to_numpy(),
            Ta=umep_df["Tair"].to_numpy(),
            RH=umep_df["RH"].to_numpy(),
            radG=umep_df["Kdown"].to_numpy(),
            radD=umep_df["ldown"].to_numpy(),
            radI=umep_df["Kdiff"].to_numpy(),
            P=umep_df["pres"].to_numpy() / 100.0  # convert from Pa to hPa,
            Ws=umep_df["Wind"].to_numpy(),
        )
