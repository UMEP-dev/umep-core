import numpy as np
import pyproj
import pytest

from umep.common import load_raster, save_raster, xy_to_lnglat


def _make_gt(width, height, pixel_size=1):
    # top-left origin at (0, height) so bounds are [0, 0, width, height]
    return [0.0, float(pixel_size), 0.0, float(height), 0.0, -float(pixel_size)]


def test_save_and_load_raster_roundtrip(tmp_path):
    out = tmp_path / "out_dir" / "test.tif"
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    trf = _make_gt(5, 5)
    crs_wkt = pyproj.CRS.from_epsg(4326).to_wkt()

    # save and load
    save_raster(str(out), data, trf, crs_wkt, no_data_val=-9999)
    rast, trf_out, crs_out, nodata = load_raster(str(out))

    np.testing.assert_array_equal(rast, data)
    assert isinstance(trf_out, list) and len(trf_out) == 6
    assert np.allclose(trf_out, trf)
    assert nodata == -9999 or nodata is None
    # parsed CRS should map to EPSG:4326
    parsed_crs = pyproj.CRS.from_wkt(crs_out) if crs_out is not None else None
    assert parsed_crs is not None and parsed_crs.to_epsg() == 4326


def test_load_raster_with_bbox(tmp_path):
    out = tmp_path / "bbox_dir" / "bbox.tif"
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    trf = _make_gt(10, 10)
    crs_wkt = pyproj.CRS.from_epsg(4326).to_wkt()
    save_raster(str(out), data, trf, crs_wkt, -9999)

    # bbox in spatial coords: [minx, miny, maxx, maxy]
    # For our geotransform bounds = [0,0,10,10]
    bbox = [2, 2, 5, 5]
    rast_crop, trf_crop, crs_crop, nd = load_raster(str(out), bbox=bbox)

    # Expected slice computed from implementation mapping
    assert rast_crop.shape == (3, 3)
    expected = data[5:8, 2:5]  # as per transform -> yoff = 10-5 =5, xoff =2
    np.testing.assert_array_equal(rast_crop, expected)
    assert isinstance(trf_crop, list) and len(trf_crop) == 6
    expected_trf = [bbox[0], trf[1], 0.0, bbox[3], 0.0, trf[5]]
    assert np.allclose(trf_crop, expected_trf)
    assert crs_crop is not None and pyproj.CRS.from_wkt(crs_crop) == pyproj.CRS.from_wkt(crs_wkt)
    assert nd == -9999


def test_xy_to_lnglat_scalar_and_array():
    # WGS84 should be identity
    crs_wkt = pyproj.CRS.from_epsg(4326).to_wkt()
    x, y = 10.0, 20.0
    lon, lat = xy_to_lnglat(crs_wkt, x, y)
    assert lon == pytest.approx(10.0)
    assert lat == pytest.approx(20.0)

    # array case
    xa = np.array([0.0, 30.0])
    ya = np.array([0.0, -15.0])
    lons, lats = xy_to_lnglat(crs_wkt, xa, ya)
    assert np.array_equal(lons, np.array([0.0, 30.0]))
    assert np.array_equal(lats, np.array([0.0, -15.0]))
