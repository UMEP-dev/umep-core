import numpy as np
import pyproj
import pytest

from umep.common import load_raster, save_raster, shrink_bbox_to_pixel_grid, xy_to_lnglat


def _make_gt(width, height, pixel_size=1):
    # top-left origin at (0, height) so bounds are [0, 0, width, height]
    return [0.0, float(pixel_size), 0.0, float(height), 0.0, -float(pixel_size)]


def test_save_and_load_raster_roundtrip(tmp_path):
    out = tmp_path / "out_dir" / "test.tif"
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    trf = _make_gt(5, 5)
    crs_wkt = pyproj.CRS.from_epsg(4326).to_wkt()

    # save and load
    save_raster(str(out), data, trf, crs_wkt, no_data_val=-9999, coerce_f64_to_f32=True)
    rast, trf_out, crs_out, nodata = load_raster(str(out), coerce_f64_to_f32=True)

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
    save_raster(str(out), data, trf, crs_wkt, -9999, coerce_f64_to_f32=True)

    # bbox in spatial coords: [minx, miny, maxx, maxy]
    # For our geotransform bounds = [0,0,10,10]
    bbox = [2, 2, 5, 5]
    rast_crop, trf_crop, crs_crop, nd = load_raster(str(out), bbox=bbox, coerce_f64_to_f32=True)

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


def test_shrink_bbox_to_pixel_grid():
    """Test bbox snapping to pixel grid by shrinking to nearest whole pixels."""
    # Grid with origin at (0, 0), 1m pixels
    origin_x, origin_y = 0.0, 0.0
    pixel_width, pixel_height = 1.0, 1.0

    # Bbox that aligns exactly with grid
    bbox = (2.0, 3.0, 5.0, 7.0)
    result = shrink_bbox_to_pixel_grid(bbox, origin_x, origin_y, pixel_width, pixel_height)
    assert result == (2.0, 3.0, 5.0, 7.0), "Aligned bbox should not change"

    # Bbox that needs inward snapping
    bbox = (2.3, 3.7, 5.9, 7.2)
    result = shrink_bbox_to_pixel_grid(bbox, origin_x, origin_y, pixel_width, pixel_height)
    # Should snap to: minx=ceil(2.3)=3, miny=ceil(3.7)=4, maxx=floor(5.9)=5, maxy=floor(7.2)=7
    assert result == (3.0, 4.0, 5.0, 7.0), "Bbox should shrink to nearest whole pixels"

    # Grid with non-zero origin
    origin_x, origin_y = 10.0, 20.0
    bbox = (12.5, 23.2, 15.8, 26.9)
    result = shrink_bbox_to_pixel_grid(bbox, origin_x, origin_y, pixel_width, pixel_height)
    # Relative to origin: (2.5, 3.2, 5.8, 6.9) -> ceil/floor -> (3, 4, 5, 6) -> absolute (13, 24, 15, 26)
    assert result == (13.0, 24.0, 15.0, 26.0), "Non-zero origin should be handled correctly"

    # Negative pixel height (north-up raster)
    origin_x, origin_y = 0.0, 100.0
    pixel_width, pixel_height = 1.0, -1.0
    bbox = (2.3, 93.1, 5.7, 96.8)
    result = shrink_bbox_to_pixel_grid(bbox, origin_x, origin_y, pixel_width, pixel_height)
    # For y-axis with negative pixel size:
    # miny=93.1 -> origin_y - y = 100 - 93.1 = 6.9 -> floor(6.9) = 6 -> y = 100 - 6 = 94
    # maxy=96.8 -> origin_y - y = 100 - 96.8 = 3.2 -> ceil(3.2) = 4 -> y = 100 - 4 = 96
    # Expected: (3.0, 94.0, 5.0, 96.0)
    assert result == (3.0, 94.0, 5.0, 96.0), "Negative pixel height should work correctly"

    # Sub-pixel bbox that would collapse
    bbox = (2.1, 3.2, 2.8, 3.7)
    with pytest.raises(ValueError, match="collapsed"):
        shrink_bbox_to_pixel_grid(bbox, origin_x, origin_y, pixel_width, pixel_height)

    # Invalid bbox (min >= max)
    with pytest.raises(ValueError, match="invalid"):
        shrink_bbox_to_pixel_grid((5.0, 3.0, 2.0, 7.0), origin_x, origin_y, pixel_width, pixel_height)
