"""Diagnostic tool to check umep installation and dependencies."""

import sys


def parse_version(v: str) -> tuple:
    """Parse version string to comparable tuple."""
    try:
        return tuple(int(x) for x in v.split(".")[:2])
    except (ValueError, AttributeError):
        return (0, 0)


def check_gdal_compatibility(rasterio_version: str, gdal_version: str) -> tuple[bool, str]:
    """
    Check if rasterio and GDAL versions are compatible.

    Rasterio wheels bundle a specific GDAL version. If osgeo.gdal is also installed
    with a different GDAL version, DLL conflicts can occur on Windows.

    Rather than maintaining a static compatibility matrix, we check:
    1. If both rasterio and osgeo.gdal are present, their GDAL versions should match
    2. The bundled GDAL version is reported for informational purposes
    """
    # For now, just report what we found - the mismatch detection happens elsewhere
    return True, f"rasterio {rasterio_version} bundles GDAL {gdal_version}"


def check_environment():
    """Check the umep environment and report any issues."""
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version}")
    print()

    issues = []
    rasterio_version = None
    rasterio_gdal_version = None
    osgeo_gdal_version = None

    # Check for conflicting GDAL installations
    print("Checking dependencies...")
    print("-" * 40)

    # Check rasterio
    try:
        import rasterio

        rasterio_version = rasterio.__version__
        rasterio_gdal_version = rasterio.gdal_version()
        print(f"✓ rasterio {rasterio_version}")
        print(f"  Bundled GDAL: {rasterio_gdal_version}")
    except ImportError as e:
        print(f"✗ rasterio: {e}")
        issues.append("rasterio")
    except Exception as e:
        print(f"✗ rasterio (DLL/loading error): {e}")
        issues.append("rasterio-dll")

    # Check if osgeo is also installed (potential conflict)
    try:
        from osgeo import gdal

        osgeo_gdal_version_raw = gdal.VersionInfo("VERSION_NUM")
        # Convert from format like "3100200" to "3.10.2"
        v = int(osgeo_gdal_version_raw)
        osgeo_gdal_version = f"{v // 1000000}.{(v // 10000) % 100}.{(v // 100) % 100}"
        print(f"! osgeo.gdal installed (version {osgeo_gdal_version})")

        # Check for version mismatch
        if rasterio_gdal_version and osgeo_gdal_version:
            rio_gdal_tuple = parse_version(rasterio_gdal_version)
            osgeo_gdal_tuple = parse_version(osgeo_gdal_version)

            if rio_gdal_tuple != osgeo_gdal_tuple:
                print(f"  WARNING: GDAL version mismatch!")
                print(f"    rasterio bundled: {rasterio_gdal_version}")
                print(f"    osgeo.gdal:       {osgeo_gdal_version}")
                print("  This WILL cause DLL conflicts on Windows.")
                if sys.platform == "win32":
                    issues.append("gdal-mismatch")
            else:
                print(f"  GDAL versions match ({rasterio_gdal_version}) - OK")

    except ImportError:
        print("  (osgeo.gdal not installed - this is fine)")
    except Exception as e:
        print(f"! osgeo.gdal installed but failed to load: {e}")
        if sys.platform == "win32":
            issues.append("osgeo-dll")

    # Check rasterio/GDAL compatibility
    if rasterio_version and rasterio_gdal_version:
        compatible, msg = check_gdal_compatibility(rasterio_version, rasterio_gdal_version)
        if not compatible:
            print(f"  WARNING: {msg}")
            issues.append("version-mismatch")
        else:
            print(f"  {msg}")

    # Check other key dependencies
    print()
    print("Other dependencies:")
    for pkg, import_name in [
        ("pyproj", "pyproj"),
        ("shapely", "shapely"),
        ("geopandas", "geopandas"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("xarray", "xarray"),
        ("rioxarray", "rioxarray"),
    ]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"✓ {pkg} {ver}")

            # Extra checks for specific packages
            if pkg == "pyproj":
                try:
                    import pyproj
                    proj_ver = pyproj.proj_version_str
                    print(f"  PROJ version: {proj_ver}")
                except Exception:
                    pass
            elif pkg == "shapely":
                try:
                    import shapely
                    geos_ver = shapely.geos_version_string
                    print(f"  GEOS version: {geos_ver}")
                except Exception:
                    pass
        except ImportError as e:
            print(f"✗ {pkg}: {e}")
            issues.append(pkg)

    print("-" * 40)

    # Check if running in OSGeo4W
    if sys.platform == "win32":
        import os

        path = os.environ.get("PATH", "")
        if "OSGeo4W" in path or "QGIS" in path:
            print()
            print("WARNING: OSGeo4W/QGIS detected in PATH")
            print("This can cause DLL conflicts with pip-installed packages.")
            print("Recommendation: Use a separate virtual environment:")
            print("  uv venv && uv pip install umep")
            issues.append("osgeo4w-path")

    print()
    if issues:
        print(f"Issues found: {', '.join(issues)}")
        print()
        print("Troubleshooting:")
        if any(i in issues for i in ["rasterio-dll", "gdal-mismatch", "osgeo4w-path", "osgeo-dll"]):
            print("• DLL/GDAL conflicts detected. Create a clean virtual environment:")
            print("    uv venv --python 3.12")
            print("    uv pip install umep")
            print()
            print("• Or use conda-forge for consistent binaries:")
            print("    conda create -n umep -c conda-forge python=3.12 rasterio geopandas")
            print("    conda activate umep && pip install umep")
        if "version-mismatch" in issues:
            print()
            print("• Version incompatibility detected. Try upgrading:")
            print("    pip install --upgrade rasterio")
        return 1
    else:
        print("All checks passed!")
        return 0


def main():
    """Entry point for umep-doctor command."""
    sys.exit(check_environment())


if __name__ == "__main__":
    main()