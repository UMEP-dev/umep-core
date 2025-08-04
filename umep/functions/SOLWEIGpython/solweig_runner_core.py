from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.transform import Affine, rowcol

from .solweig_runner import SolweigConfig, SolweigRun


class SolweigRunStandalone(SolweigRun):
    """Run SOLWEIG in standalone mode without QGIS."""

    def __init__(self, config: SolweigConfig):
        super().__init__(config, qgis_env=False)

    def run(self):
        print("Running SOLWEIG in standalone mode.")

    def load_poi_xys(self, trf_arr: list[int]) -> tuple[Any, Any]:
        pois_gdf = gpd.read_file(self.config.poi_file)
        numfeat = pois_gdf.shape[0]
        pois_xy = np.zeros((numfeat, 3)) - 999
        if self.config.poi_field is None:
            pois_name = pois_gdf.index.to_list()
        else:
            pois_name = pois_gdf[self.config.poi_field].to_list()
        trf = Affine.from_gdal(*trf_arr)
        for idx, row in pois_gdf.iterrows():
            y, x = rowcol(trf, row["geometry"].centroid.x, row["geometry"].centroid.y)
            pois_xy[idx, 0] = idx
            pois_xy[idx, 1] = x
            pois_xy[idx, 2] = y

        return pois_xy, pois_name
