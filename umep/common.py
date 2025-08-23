import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pyproj
    import rasterio
    from rasterio.features import rasterize
    from rasterio.mask import mask
    from rasterio.transform import Affine, from_origin
    from shapely import geometry

    GDAL_ENV = False
    logger.info("Using rasterio for raster operations.")

except:
    from osgeo import gdal

    GDAL_ENV = True
    logger.info("Using GDAL for raster operations.")


def rasterise_gdf(gdf, geom_col, ht_col, bbox=None, pixel_size: int = 1):
    # Define raster parameters
    if bbox is not None:
        # Unpack bbox values
        minx, miny, maxx, maxy = bbox
    else:
        # Use the total bounds of the GeoDataFrame
        minx, miny, maxx, maxy = gdf.total_bounds
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)
    transform = from_origin(minx, maxy, pixel_size, pixel_size)
    # Create a blank array for the raster
    raster = np.zeros((height, width), dtype=np.float32)
    # Burn geometries into the raster
    shapes = ((geom, value) for geom, value in zip(gdf[geom_col], gdf[ht_col], strict=True))
    raster = rasterize(shapes, out_shape=raster.shape, transform=transform, fill=0, dtype=np.float32)

    return raster, transform


def check_path(path_str: str | Path, make_dir: bool = False) -> Path:
    # Ensure path exists
    path = Path(path_str).absolute()
    if not path.parent.exists():
        if make_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            raise OSError(f"Parent directory {path} does not exist. Set make_dir=True to create it.")
    if not path.exists() and not path.suffix:
        if make_dir:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise OSError(f"Path {path} does not exist. Set make_dir=True to create it.")
    return path


def save_raster(
    out_path_str: str, data_arr: np.ndarray, trf_arr: list[float], crs_wkt: str, no_data_val: float = -9999
):
    attempts = 2
    while attempts > 0:
        attempts -= 1
        try:
            # Save raster using GDAL or rasterio
            out_path = check_path(out_path_str, make_dir=True)
            height, width = data_arr.shape
            if GDAL_ENV is False:
                trf = Affine.from_gdal(*trf_arr)
                crs = None
                if crs_wkt:
                    crs = pyproj.CRS(crs_wkt)
                with rasterio.open(
                    out_path,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=1,
                    dtype=data_arr.dtype,
                    crs=crs,
                    transform=trf,
                    nodata=no_data_val,
                ) as dst:
                    dst.write(data_arr, 1)
            else:
                # Map numpy dtype to GDAL type
                dtype_map = {
                    np.dtype("uint8"): gdal.GDT_Byte,
                    np.dtype("int16"): gdal.GDT_Int16,
                    np.dtype("uint16"): gdal.GDT_UInt16,
                    np.dtype("int32"): gdal.GDT_Int32,
                    np.dtype("uint32"): gdal.GDT_UInt32,
                    np.dtype("float32"): gdal.GDT_Float32,
                    np.dtype("float64"): gdal.GDT_Float64,
                }
                gdal_dtype = dtype_map.get(data_arr.dtype, gdal.GDT_Float32)
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.Create(str(out_path), width, height, 1, gdal_dtype)
                # trf is a list: [top left x, w-e pixel size, rotation, top left y, rotation, n-s pixel size]
                ds.SetGeoTransform(tuple(trf_arr))
                # GetProjection returns WKT (string)
                if crs_wkt:
                    ds.SetProjection(crs_wkt)
                band = ds.GetRasterBand(1)
                band.WriteArray(data_arr, 0, 0)
                band.SetNoDataValue(no_data_val)
                ds.FlushCache()
                ds = None
        except Exception as e:
            print(f"Error saving raster, attempts left {attempts}: {e}")
            if attempts == 0:
                raise e


def load_raster(
    path_str: str, bbox: list[int] | None = None, band: int = 0
) -> tuple[np.ndarray, list[float], str | None, float | None]:
    # Load raster, optionally crop to bbox
    path = check_path(path_str, make_dir=False)
    if not path.exists():
        raise FileNotFoundError(f"Raster file {path} does not exist.")
    if GDAL_ENV is False:
        with rasterio.open(path) as dataset:
            crs_wkt = dataset.crs.to_wkt() if dataset.crs is not None else None
            dataset_bounds = dataset.bounds
            no_data_val = dataset.nodata
            if bbox is not None:
                # Create bbox geometry for masking
                bbox_geom = geometry.box(*bbox)
                if not (
                    dataset_bounds.left <= bbox[0] <= dataset_bounds.right
                    and dataset_bounds.left <= bbox[2] <= dataset_bounds.right
                    and dataset_bounds.bottom <= bbox[1] <= dataset_bounds.top
                    and dataset_bounds.bottom <= bbox[3] <= dataset_bounds.top
                ):
                    raise ValueError("Bounding box is not fully contained within the raster dataset bounds")
                rast, trf = mask(dataset, [bbox_geom], crop=True)
            else:
                rast = dataset.read()
                trf = dataset.transform
            # Convert rasterio Affine to GDAL-style list
            trf_arr = [trf.c, trf.a, trf.b, trf.f, trf.d, trf.e]
            # rast shape: (bands, rows, cols)
            if rast.ndim == 3:
                if band < 0 or band >= rast.shape[0]:
                    raise IndexError(f"Requested band {band} out of range; raster has {rast.shape[0]} band(s)")
                rast_arr = rast[band].astype(float)
            else:
                rast_arr = rast.astype(float)
    else:
        dataset = gdal.Open(str(path))
        if dataset is None:
            raise FileNotFoundError(f"Could not open {path}")
        trf = dataset.GetGeoTransform()
        # GetProjection returns WKT string (or empty string)
        crs_wkt = dataset.GetProjection() or None
        rb = dataset.GetRasterBand(band + 1)
        if rb is None:
            dataset = None
            raise IndexError(f"Requested band {band} out of range in GDAL dataset")
        rast_arr = rb.ReadAsArray().astype(float)
        no_data_val = rb.GetNoDataValue()
        if bbox is not None:
            min_x, min_y, max_x, max_y = bbox
            xoff = int((min_x - trf[0]) / trf[1])
            yoff = int((trf[3] - max_y) / abs(trf[5]))
            xsize = int((max_x - min_x) / trf[1])
            ysize = int((max_y - min_y) / abs(trf[5]))
            # guard offsets/sizes
            if xoff < 0 or yoff < 0 or xsize <= 0 or ysize <= 0:
                dataset = None
                raise ValueError("Computed window from bbox is out of raster bounds or invalid")
            rast_arr = rast_arr[yoff : yoff + ysize, xoff : xoff + xsize]
            trf_arr = [min_x, trf[1], 0, max_y, 0, trf[5]]
        else:
            trf_arr = [trf[0], trf[1], 0, trf[3], 0, trf[5]]
        dataset = None  # ensure dataset closed
    # Handle no-data (support NaN)
    if no_data_val is not None and not np.isnan(no_data_val):
        logger.info(f"No-data value is {no_data_val}, replacing with NaN")
        rast_arr[rast_arr == no_data_val] = np.nan
    if rast_arr.size == 0:
        raise ValueError("Raster array is empty after loading/cropping")
    if rast_arr.min() < 0:
        raise ValueError("Raster contains negative values")
    return rast_arr, trf_arr, crs_wkt, no_data_val


def xy_to_lnglat(crs_wkt: str | None, x, y):
    """Convert x, y coordinates to longitude and latitude.

    Accepts scalar or array-like x/y. If crs_wkt is None the inputs are
    assumed already to be lon/lat and are returned unchanged.
    """
    if crs_wkt is None:
        logger.info("No CRS provided, assuming coordinates are already in WGS84 (lon/lat).")
        return x, y

    try:
        if GDAL_ENV is False:
            source_crs = pyproj.CRS(crs_wkt)
            target_crs = pyproj.CRS(4326)  # WGS84
            transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
            lng, lat = transformer.transform(x, y)
        else:
            old_cs = gdal.osr.SpatialReference()
            old_cs.ImportFromWkt(crs_wkt)
            new_cs = gdal.osr.SpatialReference()
            new_cs.ImportFromEPSG(4326)
            transform = gdal.osr.CoordinateTransformation(old_cs, new_cs)
            out = transform.TransformPoint(float(x), float(y))
            lng, lat = out[0], out[1]

        return lng, lat

    except Exception:
        logger.exception("Failed to transform coordinates")
        raise
