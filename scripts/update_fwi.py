#!/usr/bin/env python3

"""
IR-FWI
Build Iran-wide FWI web product from Copernicus GWIS / ECMWF WMS.

Process:
1. Read IRAN.geojson
2. Request ECMWF FWI from GWIS WMS
3. Download WMS tiles with several fallbacks
4. Assemble tiles into a georeferenced EPSG:4326 GeoTIFF
5. Clip exactly to IRAN.geojson
6. Convert clipped raster to transparent PNG
7. Calculate exact image bounds from the clipped raster
8. Write metadata JSON
"""

from pathlib import Path
import io
import json
import math
import time
from datetime import datetime, timezone

import requests
import numpy as np

from PIL import Image

import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.transform import from_bounds, array_bounds
from rasterio.warp import reproject, Resampling


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BOUNDARY_FILE = BASE_DIR / "IRAN.geojson"

WEB_DIR = BASE_DIR / "web"

PNG_FILE = WEB_DIR / "fwi_iran_latest.png"
JSON_FILE = WEB_DIR / "fwi_iran_latest.json"

TMP_DIR = BASE_DIR / "tmp_fwi"
TMP_DIR.mkdir(parents=True, exist_ok=True)

WMS_URL = "https://maps.effis.emergency.copernicus.eu/gwis"

LAYER_NAME = "ecmwf.fwi"

TARGET_CRS = "EPSG:4326"

# Forecast date.
# The workflow is intended to run daily.
FORECAST_DATE = (
    datetime.now(timezone.utc).date()
)

# Number of tiles.
# 3 x 3 is used to avoid very large WMS requests.
NX = 3
NY = 3

# Main tile pixel size is calculated from geographic dimensions.
MAX_TILE_WIDTH = 800
MAX_TILE_HEIGHT = 800

# HTTP settings
REQUEST_TIMEOUT = 180

MAX_ATTEMPTS_PER_METHOD = 3

SLEEP_BETWEEN_ATTEMPTS = 3

# Slight expansion used only when the original WMS BBOX fails.
# The returned image is subsequently reprojected/cropped back
# to the exact requested BBOX.
EXPAND_DEGREES = 0.15


# ============================================================
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": (
            "IR-FWI-GitHubActions/1.0 "
            "(Copernicus GWIS ECMWF FWI)"
        ),
        "Accept": (
            "image/png,image/jpeg,image/*;q=0.8,*/*;q=0.5"
        ),
        "Accept-Encoding": "identity",
        "Connection": "close",
    }
)


# ============================================================
# HELPERS
# ============================================================

def log(message):
    print(message, flush=True)


def validate_image(content):
    """
    Validate that HTTP response actually contains a readable image.
    """
    if not content:
        raise ValueError("Empty response")

    if len(content) < 100:
        raise ValueError(
            f"Response too small: {len(content)} bytes"
        )

    try:
        image = Image.open(io.BytesIO(content))
        image.load()
        return image
    except Exception as exc:
        raise ValueError(
            f"Returned content is not a valid image: {exc}"
        )


def get_boundary():
    """
    Read Iran boundary and return GeoDataFrame in EPSG:4326.
    """

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Boundary file not found: {BOUNDARY_FILE}"
        )

    gdf = gpd.read_file(BOUNDARY_FILE)

    if gdf.empty:
        raise RuntimeError("Iran boundary is empty.")

    if gdf.crs is None:
        raise RuntimeError(
            "IRAN.geojson has no CRS."
        )

    gdf = gdf.to_crs(TARGET_CRS)

    return gdf


def calculate_bbox(gdf):
    """
    Return:
        west, south, east, north
    """

    minx, miny, maxx, maxy = gdf.total_bounds

    return (
        float(minx),
        float(miny),
        float(maxx),
        float(maxy),
    )


def calculate_tile_size(
    west,
    south,
    east,
    north,
    nx,
    ny,
):
    """
    Calculate pixel dimensions while preserving geographic aspect ratio.
    """

    tile_width_geo = (east - west) / nx
    tile_height_geo = (north - south) / ny

    ratio = tile_width_geo / tile_height_geo

    width = MAX_TILE_WIDTH
    height = max(
        1,
        int(round(width / ratio))
    )

    if height > MAX_TILE_HEIGHT:
        height = MAX_TILE_HEIGHT
        width = max(
            1,
            int(round(height * ratio))
        )

    return width, height


def wms_request(
    bbox,
    width,
    height,
    forecast_date,
    version="1.1.1",
    image_format="image/png",
    transparent=True,
):
    """
    Request one WMS image.

    Returns PIL.Image.
    """

    west, south, east, north = bbox

    if version == "1.1.1":

        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "LAYERS": LAYER_NAME,
            "STYLES": "",
            "FORMAT": image_format,
            "TRANSPARENT": (
                "TRUE" if transparent else "FALSE"
            ),
            "SRS": "EPSG:4326",
            "BBOX": (
                f"{west},{south},{east},{north}"
            ),
            "WIDTH": str(width),
            "HEIGHT": str(height),
            "TIME": forecast_date.isoformat(),
        }

    elif version == "1.3.0":

        # EPSG:4326 in WMS 1.3.0 uses latitude/longitude
        # axis order.
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": LAYER_NAME,
            "STYLES": "",
            "FORMAT": image_format,
            "TRANSPARENT": (
                "TRUE" if transparent else "FALSE"
            ),
            "CRS": "EPSG:4326",
            "BBOX": (
                f"{south},{west},{north},{east}"
            ),
            "WIDTH": str(width),
            "HEIGHT": str(height),
            "TIME": forecast_date.isoformat(),
        }

    else:
        raise ValueError(
            f"Unsupported WMS version: {version}"
        )

    last_error = None

    for attempt in range(
        1,
        MAX_ATTEMPTS_PER_METHOD + 1
    ):

        try:

            log(
                f"      HTTP attempt "
                f"{attempt}/{MAX_ATTEMPTS_PER_METHOD} "
                f"version={version} "
                f"format={image_format}"
            )

            response = SESSION.get(
                WMS_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
                stream=True,
            )

            response.raise_for_status()

            content = response.content

            image = validate_image(content)

            log(
                f"      OK "
                f"{len(content)} bytes "
                f"{image.width}x{image.height}"
            )

            return image

        except Exception as exc:

            last_error = exc

            log(
                f"      FAILED: {repr(exc)}"
            )

            try:
                response.close()
            except Exception:
                pass

            if attempt < MAX_ATTEMPTS_PER_METHOD:
                time.sleep(
                    SLEEP_BETWEEN_ATTEMPTS
                )

    raise RuntimeError(
        f"WMS request failed: {last_error}"
    )


def image_to_rgba_array(image):
    """
    Convert PIL image to RGBA numpy array.
    """

    return np.asarray(
        image.convert("RGBA"),
        dtype=np.uint8,
    )


def fetch_tile(
    bbox,
    width,
    height,
    forecast_date,
    tile_name,
):
    """
    Download one tile using multiple fallback methods.

    Returns:
        PIL.Image
        actual_bbox_used
    """

    west, south, east, north = bbox

    log("")
    log(
        f"  {tile_name}"
    )
    log(
        f"    BBOX = "
        f"({west}, {south}, {east}, {north})"
    )
    log(
        f"    SIZE = "
        f"{width} x {height}"
    )

    # --------------------------------------------------------
    # METHOD 1
    # WMS 1.1.1 PNG
    # --------------------------------------------------------

    methods = [
        {
            "name": "WMS 1.1.1 PNG",
            "version": "1.1.1",
            "format": "image/png",
            "transparent": True,
        },
        {
            "name": "WMS 1.1.1 JPEG",
            "version": "1.1.1",
            "format": "image/jpeg",
            "transparent": False,
        },
        {
            "name": "WMS 1.3.0 PNG",
            "version": "1.3.0",
            "format": "image/png",
            "transparent": True,
        },
    ]

    for method in methods:

        log(
            f"    Trying {method['name']}"
        )

        try:

            image = wms_request(
                bbox=bbox,
                width=width,
                height=height,
                forecast_date=forecast_date,
                version=method["version"],
                image_format=method["format"],
                transparent=method["transparent"],
            )

            return image.convert("RGBA"), bbox

        except Exception as exc:

            log(
                f"    {method['name']} failed:"
            )
            log(
                f"      {repr(exc)}"
            )

    # --------------------------------------------------------
    # METHOD 4
    # EXPANDED BBOX
    # --------------------------------------------------------

    log("")
    log(
        "    Original BBOX failed."
    )
    log(
        "    Trying expanded BBOX fallback..."
    )

    expanded_bbox = (
        west - EXPAND_DEGREES,
        south - EXPAND_DEGREES,
        east + EXPAND_DEGREES,
        north + EXPAND_DEGREES,
    )

    expanded_width = width + 40
    expanded_height = height + 40

    try:

        image = wms_request(
            bbox=expanded_bbox,
            width=expanded_width,
            height=expanded_height,
            forecast_date=forecast_date,
            version="1.1.1",
            image_format="image/jpeg",
            transparent=False,
        )

        log(
            "    Expanded BBOX succeeded."
        )

        return (
            image.convert("RGBA"),
            expanded_bbox,
        )

    except Exception as exc:

        log(
            "    Expanded BBOX failed:"
        )
        log(
            f"      {repr(exc)}"
        )

    raise RuntimeError(
        f"All WMS fallback methods failed for {tile_name}"
    )


def normalize_tile_to_requested_bbox(
    image,
    actual_bbox,
    requested_bbox,
    width,
    height,
):
    """
    If the server returned an expanded BBOX, resample it
    back to the exact requested BBOX.

    If actual_bbox == requested_bbox, simply resize if needed.
    """

    actual_west, actual_south, actual_east, actual_north = (
        actual_bbox
    )

    req_west, req_south, req_east, req_north = (
        requested_bbox
    )

    source = image.convert("RGBA")

    # Exact BBOX.
    if (
        abs(actual_west - req_west) < 1e-12
        and abs(actual_south - req_south) < 1e-12
        and abs(actual_east - req_east) < 1e-12
        and abs(actual_north - req_north) < 1e-12
    ):

        if source.size != (width, height):
            source = source.resize(
                (width, height),
                Image.Resampling.BILINEAR,
            )

        return source

    # --------------------------------------------------------
    # Expanded BBOX:
    # georeference source then warp to exact target.
    # --------------------------------------------------------

    src_array = np.asarray(
        source,
        dtype=np.uint8,
    )

    src_height, src_width = (
        src_array.shape[0],
        src_array.shape[1],
    )

    src_transform = from_bounds(
        actual_west,
        actual_south,
        actual_east,
        actual_north,
        src_width,
        src_height,
    )

    dst_transform = from_bounds(
        req_west,
        req_south,
        req_east,
        req_north,
        width,
        height,
    )

    dst_array = np.zeros(
        (4, height, width),
        dtype=np.uint8,
    )

    for band in range(4):

        reproject(
            source=src_array[:, :, band],
            destination=dst_array[band],
            src_transform=src_transform,
            src_crs=TARGET_CRS,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
        )

    output = np.moveaxis(
        dst_array,
        0,
        2,
    )

    return Image.fromarray(
        output,
        mode="RGBA",
    )


def assemble_tiles(
    tile_images,
    tile_bboxes,
    full_bbox,
    nx,
    ny,
):
    """
    Assemble downloaded tiles into one RGBA image.
    """

    west, south, east, north = full_bbox

    full_width = sum(
        tile_images[(0, x)].width
        for x in range(nx)
    )

    full_height = sum(
        tile_images[(y, 0)].height
        for y in range(ny)
    )

    canvas = Image.new(
        "RGBA",
        (full_width, full_height),
        (0, 0, 0, 0),
    )

    # All tiles have identical dimensions.
    tile_width = tile_images[(0, 0)].width
    tile_height = tile_images[(0, 0)].height

    for y in range(ny):

        for x in range(nx):

            image = tile_images[(y, x)]

            px = x * tile_width
            py = y * tile_height

            canvas.alpha_composite(
                image,
                (px, py),
            )

    return canvas, full_width, full_height


def save_geotiff(
    image,
    bbox,
    output_file,
):
    """
    Save RGBA image as EPSG:4326 GeoTIFF.
    """

    west, south, east, north = bbox

    rgba = np.asarray(
        image.convert("RGBA"),
        dtype=np.uint8,
    )

    height, width, _ = rgba.shape

    transform = from_bounds(
        west,
        south,
        east,
        north,
        width,
        height,
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": "uint8",
        "crs": TARGET_CRS,
        "transform": transform,
        "compress": "deflate",
        "predictor": 2,
    }

    with rasterio.open(
        output_file,
        "w",
        **profile,
    ) as dst:

        for i in range(4):

            dst.write(
                rgba[:, :, i + 0],
                i + 1,
            )


def create_geotiff_from_image(
    image,
    bbox,
    output_file,
):
    """
    Correctly save image bands to GeoTIFF.
    """

    west, south, east, north = bbox

    rgba = np.asarray(
        image.convert("RGBA"),
        dtype=np.uint8,
    )

    height = rgba.shape[0]
    width = rgba.shape[1]

    transform = from_bounds(
        west,
        south,
        east,
        north,
        width,
        height,
    )

    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=4,
        dtype="uint8",
        crs=TARGET_CRS,
        transform=transform,
        compress="deflate",
    ) as dst:

        for band in range(4):

            dst.write(
                rgba[:, :, band + 0],
                band + 1,
            )


def clip_to_iran(
    source_tif,
    boundary_gdf,
    clipped_tif,
):
    """
    Clip raster exactly to Iran boundary.
    """

    geometries = [
        geom.__geo_interface__
        for geom in boundary_gdf.geometry
        if geom is not None
    ]

    if not geometries:
        raise RuntimeError(
            "No valid Iran geometry found."
        )

    with rasterio.open(source_tif) as src:

        if src.crs is None:
            raise RuntimeError(
                "Source raster has no CRS."
            )

        boundary = boundary_gdf.to_crs(
            src.crs
        )

        geometry_list = [
            geom.__geo_interface__
            for geom in boundary.geometry
            if geom is not None
        ]

        clipped, clipped_transform = mask(
            src,
            geometry_list,
            crop=True,
            filled=True,
            nodata=0,
        )

        profile = src.profile.copy()

        profile.update(
            {
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": clipped_transform,
                "compress": "deflate",
            }
        )

        with rasterio.open(
            clipped_tif,
            "w",
            **profile,
        ) as dst:

            dst.write(clipped)

    return clipped_transform


def save_png_from_clipped(
    clipped_tif,
):
    """
    Convert clipped RGBA GeoTIFF to PNG.
    """

    with rasterio.open(clipped_tif) as src:

        data = src.read()

        if data.shape[0] < 4:
            raise RuntimeError(
                "Expected 4-band RGBA raster."
            )

        rgba = np.moveaxis(
            data[:4],
            0,
            2,
        )

        image = Image.fromarray(
            rgba.astype(np.uint8),
            mode="RGBA",
        )

        image.save(
            PNG_FILE,
            format="PNG",
            optimize=True,
        )

        bounds = array_bounds(
            src.height,
            src.width,
            src.transform,
        )

        west = float(bounds[0])
        south = float(bounds[1])
        east = float(bounds[2])
        north = float(bounds[3])

        return {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
            "width": int(src.width),
            "height": int(src.height),
            "crs": str(src.crs),
        }


# ============================================================
# MAIN
# ============================================================

def main():

    log("")
    log("=" * 70)
    log("IR-FWI - Iran ECMWF FWI Builder")
    log("=" * 70)
    log("")

    WEB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Boundary
    # --------------------------------------------------------

    log("Loading Iran boundary...")

    boundary = get_boundary()

    west, south, east, north = calculate_bbox(
        boundary
    )

    log(
        f"Iran boundary:"
    )

    log(
        f"  west  = {west}"
    )

    log(
        f"  south = {south}"
    )

    log(
        f"  east  = {east}"
    )

    log(
        f"  north = {north}"
    )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    log("")
    log(
        f"FWI date: {FORECAST_DATE}"
    )

    # --------------------------------------------------------
    # Tiles
    # --------------------------------------------------------

    tile_width, tile_height = calculate_tile_size(
        west,
        south,
        east,
        north,
        NX,
        NY,
    )

    log("")
    log(
        f"Tile grid: {NX} x {NY}"
    )

    log(
        f"Tile size: {tile_width} x {tile_height}"
    )

    # --------------------------------------------------------
    # Download tiles
    # --------------------------------------------------------

    tile_images = {}
    tile_bboxes = {}

    for y in range(NY):

        for x in range(NX):

            tile_west = (
                west
                + (east - west) * x / NX
            )

            tile_east = (
                west
                + (east - west) * (x + 1) / NX
            )

            # y=0 is north tile.
            tile_north = (
                north
                - (north - south) * y / NY
            )

            tile_south = (
                north
                - (north - south) * (y + 1) / NY
            )

            requested_bbox = (
                tile_west,
                tile_south,
                tile_east,
                tile_north,
            )

            tile_name = (
                f"Tile {y + 1}/{NY}, "
                f"column {x + 1}/{NX}"
            )

            image, actual_bbox = fetch_tile(
                bbox=requested_bbox,
                width=tile_width,
                height=tile_height,
                forecast_date=FORECAST_DATE,
                tile_name=tile_name,
            )

            image = normalize_tile_to_requested_bbox(
                image=image,
                actual_bbox=actual_bbox,
                requested_bbox=requested_bbox,
                width=tile_width,
                height=tile_height,
            )

            tile_images[(y, x)] = image
            tile_bboxes[(y, x)] = requested_bbox

    # --------------------------------------------------------
    # Assemble
    # --------------------------------------------------------

    log("")
    log("=" * 70)
    log("Assembling WMS tiles...")
    log("=" * 70)

    full_image, full_width, full_height = assemble_tiles(
        tile_images=tile_images,
        tile_bboxes=tile_bboxes,
        full_bbox=(
            west,
            south,
            east,
            north,
        ),
        nx=NX,
        ny=NY,
    )

    log(
        f"Assembled image: "
        f"{full_width} x {full_height}"
    )

    # --------------------------------------------------------
    # Temporary GeoTIFF
    # --------------------------------------------------------

    assembled_tif = (
        TMP_DIR / "fwi_iran_assembled.tif"
    )

    log("")
    log(
        "Writing georeferenced GeoTIFF..."
    )

    create_geotiff_from_image(
        image=full_image,
        bbox=(
            west,
            south,
            east,
            north,
        ),
        output_file=assembled_tif,
    )

    log(
        f"Created: {assembled_tif}"
    )

    # --------------------------------------------------------
    # Exact Iran clip
    # --------------------------------------------------------

    clipped_tif = (
        TMP_DIR / "fwi_iran_clipped.tif"
    )

    log("")
    log("=" * 70)
    log("Clipping raster to IRAN.geojson...")
    log("=" * 70)

    clip_to_iran(
        source_tif=assembled_tif,
        boundary_gdf=boundary,
        clipped_tif=clipped_tif,
    )

    log(
        f"Clipped raster: {clipped_tif}"
    )

    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    log("")
    log(
        "Creating final PNG..."
    )

    image_info = save_png_from_clipped(
        clipped_tif
    )

    log(
        f"Final PNG: {PNG_FILE}"
    )

    log(
        f"Final image size: "
        f"{image_info['width']} x "
        f"{image_info['height']}"
    )

    log(
        "Final image bounds:"
    )

    log(
        f"  west  = {image_info['west']}"
    )

    log(
        f"  south = {image_info['south']}"
    )

    log(
        f"  east  = {image_info['east']}"
    )

    log(
        f"  north = {image_info['north']}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "source": "Copernicus GWIS / ECMWF",
        "layer": LAYER_NAME,
        "date": FORECAST_DATE.isoformat(),
        "crs": TARGET_CRS,

        "image": {
            "file": PNG_FILE.name,
            "width": image_info["width"],
            "height": image_info["height"],
        },

        "wms_bbox": {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        },

        "image_bounds": {
            "west": image_info["west"],
            "south": image_info["south"],
            "east": image_info["east"],
            "north": image_info["north"],
        },

        "iran_boundary": {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
            "file": "IRAN.geojson",
        },

        "georeferencing": {
            "method": (
                "WMS tiles -> EPSG:4326 GeoTIFF "
                "-> exact IRAN.geojson mask/crop "
                "-> PNG"
            ),
            "clip": "IRAN.geojson",
            "image_bounds_source": (
                "rasterio.transform.array_bounds"
            ),
        },

        "tile_grid": {
            "columns": NX,
            "rows": NY,
            "tile_width": tile_width,
            "tile_height": tile_height,
        },

        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }

    with open(
        JSON_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    log("")
    log(
        f"Metadata: {JSON_FILE}"
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not PNG_FILE.exists():
        raise RuntimeError(
            "Final PNG was not created."
        )

    if not JSON_FILE.exists():
        raise RuntimeError(
            "Final JSON was not created."
        )

    png_size = PNG_FILE.stat().st_size
    json_size = JSON_FILE.stat().st_size

    if png_size < 1000:
        raise RuntimeError(
            f"Final PNG is suspiciously small: "
            f"{png_size} bytes"
        )

    if json_size < 100:
        raise RuntimeError(
            f"Final JSON is suspiciously small: "
            f"{json_size} bytes"
        )

    log("")
    log("=" * 70)
    log("BUILD SUCCESSFUL")
    log("=" * 70)
    log("")
    log(
        f"PNG : {PNG_FILE}"
    )
    log(
        f"JSON: {JSON_FILE}"
    )
    log("")


if __name__ == "__main__":
    main()
