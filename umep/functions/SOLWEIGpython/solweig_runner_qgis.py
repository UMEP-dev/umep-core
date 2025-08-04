from typing import Any

from ...functions.SOLWEIGpython.wallOfInterest import pointOfInterest
from .solweig_runner import SolweigConfig, SolweigRun


class SolweigRunQgis(SolweigRun):
    """Run SOLWEIG in QGIS environment."""

    def __init__(self, config_path_str: str, feedback: Any):
        """ """
        config = SolweigConfig()
        config.from_file(config_path_str)
        super().__init__(config, qgis_env=True)
        self.set_feedback(feedback)

    def load_poi_xys(self, trf_arr: list[int]) -> tuple[Any, Any]:
        scale = 1 / trf_arr[1]
        return pointOfInterest(self.config.poi_file, self.config.poi_field, scale, trf_arr)
