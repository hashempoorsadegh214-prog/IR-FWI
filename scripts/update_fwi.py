#!/usr/bin/env python3

import io
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
from PIL import Image
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.transform import from_bounds
from shapely.geometry import mapping
from shapely.ops import unary_union


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BOUNDARY_FILE = ROOT / "IRAN.geojson"
WEB_DIR = ROOT / "web"

OUTPUT_IMAGE = WEB_DIR / "fwi_iran_latest.png"
OUTPUT_JSON = WEB_DIR / "fwi_iran_latest.json"
TEMP_GEOTIFF = WEB_DIR / "fwi_iran_working.tif"


# ============================================================
# Copernicus GWIS / ECMWF
# ============================================================

WMS_URL = "https://maps.effis.emergency.copernicus.eu/gwis"
LAYER_NAME = "ecmwf.fwi"


# ============================================================
# Image settings
# ============================================================

IMAGE_WIDTH = 1600

TILES_X = 2
TILES_Y = 2

TIMEOUT = 180
MAX_DOWNLOAD_ATTEMPTS = 5
RETRY_WAIT_SECONDS = 8


# ============================================================
# Boundary
# ============================================================

def load_iran_boundary():
    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Iran boundary not found: {BOUNDARY_FILE}"
        )

    gdf = gpd.read_file(BOUNDARY_FILE)

    if gdf.empty:
        raise RuntimeError("IRAN.geojson is empty.")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")

    geometry = unary_union(gdf.geometry)

    if geometry.is_empty:
        raise RuntimeError("Iran boundary geometry is empty.")

    return geometry


def get_bbox(geometry):
    """
    Create a small WMS request margin around Iran.
    """

    minx, miny, maxx, maxy = geometry.bounds

    width = maxx - minx
    height = maxy - miny

    margin_x = width * 0.01
    margin_y = height * 0.01

    return (
        minx - margin_x,
        miny - margin_y,
        maxx + margin_x,
        maxy + margin_y,
    )


# ============================================================
# Image dimensions
# ============================================================

def get_image_dimensions(bbox):
    west, south, east, north = bbox

    geographic_width = east - west
    geographic_height = north - south

    aspect_ratio = geographic_width / geographic_height

    width = IMAGE_WIDTH
    height = round(width / aspect_ratio)

    return width, height


# ============================================================
# Tile calculation
# ============================================================

def get_tile_bbox(bbox, tile_x, tile_y):
    west, south, east, north = bbox

    total_width = east - west
    total_height = north - south

    tile_width = total_width / TILES_X
    tile_height = total_height / TILES_Y

    tile_west = west + tile_x * tile_width
    tile_east = west + (tile_x + 1) * tile_width

    # Row 0 is the northern tile.
    tile_north = north - tile_y * tile_height
    tile_south = north - (tile_y + 1) * tile_height

    return (
        tile_west,
        tile_south,
        tile_east,
        tile_north,
    )


def get_tile_dimensions(full_width, full_height, tile_x, tile_y):
    base_width = full_width // TILES_X
    base_height = full_height // TILES_Y

    if tile_x == TILES_X - 1:
        tile_width = full_width - base_width * (TILES_X - 1)
    else:
        tile_width = base_width

    if tile_y == TILES_Y - 1:
        tile_height = full_height - base_height * (TILES_Y - 1)
    else:
        tile_height = base_height

    return tile_width, tile_height


# ============================================================
# WMS download
# ============================================================

def download_wms_tile(
    bbox,
    width,
    height,
    target_date,
):
    west, south, east, north = bbox

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": LAYER_NAME,
        "STYLES": "",
        "FORMAT": "image/png",
        "TRANSPARENT": "TRUE",
        "SRS": "EPSG:4326",
        "BBOX": f"{west},{south},{east},{north}",
        "WIDTH": width,
        "HEIGHT": height,
        "TIME": target_date,
    }

    last_error = None

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):

        try:
            print(
                f"Downloading WMS tile "
                f"{width}x{height}, "
                f"attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS}"
            )

            response = requests.get(
                WMS_URL,
                params=params,
                timeout=TIMEOUT,
                headers={
                    "User-Agent": "IR-FWI-GitHub-Actions/1.0"
                },
            )

            response.raise_for_status()

            image = Image.open(
                io.BytesIO(response.content)
            ).convert("RGBA")

            if image.size != (width, height):
                raise RuntimeError(
                    f"Unexpected image size: "
                    f"{image.size}; expected {(width, height)}"
                )

            return image

        except Exception as exc:
            last_error = exc

            print(
                f"WMS tile download failed: {exc}"
            )

            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError(
        f"Failed to download WMS tile after "
        f"{MAX_DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    )


# ============================================================
# Download and assemble WMS image
# ============================================================

def download_fwi(
    bbox,
    width,
    height,
    target_date,
):
    print("WMS request bbox:")
    print(bbox)

    print("Final image size:")
    print(width, height)

    canvas = Image.new(
        "RGBA",
        (width, height),
        (0, 0, 0, 0),
    )

    west, south, east, north = bbox

    for tile_y in range(TILES_Y):

        for tile_x in range(TILES_X):

            tile_bbox = get_tile_bbox(
                bbox,
                tile_x,
                tile_y,
            )

            tile_width, tile_height = get_tile_dimensions(
                width,
                height,
                tile_x,
                tile_y,
            )

            print(
                f"Tile {tile_y + 1}/{TILES_Y}, "
                f"{tile_x + 1}/{TILES_X}"
            )

            print(
                "Tile bbox:",
                tile_bbox
            )

            tile = download_wms_tile(
                tile_bbox,
                tile_width,
                tile_height,
                target_date,
            )

            pixel_x = sum(
                get_tile_dimensions(
                    width,
                    height,
                    x,
                    tile_y,
                )[0]
                for x in range(tile_x)
            )

            pixel_y = sum(
                get_tile_dimensions(
                    width,
                    height,
                    tile_x,
                    y,
                )[1]
                for y in range(tile_y)
            )

            canvas.paste(
                tile,
                (pixel_x, pixel_y),
            )

    return canvas


# ============================================================
# GeoTIFF creation
# ============================================================

def create_georeferenced_geotiff(
    image,
    bbox,
):
    west, south, east, north = bbox

    width, height = image.size

    transform = from_bounds(
        west,
        south,
        east,
        north,
        width,
        height,
    )

    rgba = np.array(image)

    with rasterio.open(
        TEMP_GEOTIFF,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=4,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        compress="deflate",
    ) as dst:

        for band in range(4):
            dst.write(
                rgba[:, :, band],
                band + 1,
            )

    print(
        "Georeferenced GeoTIFF created:",
        TEMP_GEOTIFF
    )


# ============================================================
# Clip exactly to Iran geometry
# ============================================================

def clip_geotiff_to_iran(geometry):
    with rasterio.open(TEMP_GEOTIFF) as src:

        clipped, clipped_transform = mask(
            src,
            [mapping(geometry)],
            crop=True,
            filled=True,
            nodata=0,
        )

        clipped_height = clipped.shape[1]
        clipped_width = clipped.shape[2]

        profile = src.profile.copy()

        profile.update(
            {
                "height": clipped_height,
                "width": clipped_width,
                "transform": clipped_transform,
                "nodata": 0,
                "compress": "deflate",
            }
        )

        image_bounds = rasterio.transform.array_bounds(
            clipped_height,
            clipped_width,
            clipped_transform,
        )

        print(
            "Clipped raster bounds:"
        )
        print(image_bounds)

        return (
            clipped,
            profile,
            image_bounds,
        )


# ============================================================
# Save clipped PNG
# ============================================================

def save_web_png(
    clipped,
    profile,
):
    height = clipped.shape[1]
    width = clipped.shape[2]

    rgba = np.transpose(
        clipped,
        (1, 2, 0),
    )

    rgba = np.asarray(
        rgba,
        dtype=np.uint8,
    )

    image = Image.fromarray(
        rgba,
        mode="RGBA",
    )

    image.save(
        OUTPUT_IMAGE,
        format="PNG",
        optimize=True,
    )

    print(
        "Web PNG created:",
        OUTPUT_IMAGE
    )

    return width, height


# ============================================================
# Main
# ============================================================

def main():

    WEB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("IR-FWI UPDATE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load Iran boundary
    # --------------------------------------------------------

    iran_geometry = load_iran_boundary()

    iran_west, iran_south, iran_east, iran_north = (
        iran_geometry.bounds
    )

    print("Iran boundary:")
    print(
        iran_west,
        iran_south,
        iran_east,
        iran_north,
    )

    # --------------------------------------------------------
    # WMS bbox
    # --------------------------------------------------------

    wms_bbox = get_bbox(
        iran_geometry
    )

    print("WMS bbox:")
    print(wms_bbox)

    # --------------------------------------------------------
    # Target forecast date
    # --------------------------------------------------------

    now_utc = datetime.now(
        timezone.utc
    )

    target_date = (
        now_utc + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    print(
        "Target FWI date:",
        target_date
    )

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    width, height = get_image_dimensions(
        wms_bbox
    )

    print(
        "Geographic aspect ratio:",
        (wms_bbox[2] - wms_bbox[0])
        / (wms_bbox[3] - wms_bbox[1])
    )

    print(
        "Image aspect ratio:",
        width / height
    )

    # --------------------------------------------------------
    # Download WMS
    # --------------------------------------------------------

    image = download_fwi(
        wms_bbox,
        width,
        height,
        target_date,
    )

    # --------------------------------------------------------
    # Create georeferenced GeoTIFF
    # --------------------------------------------------------

    create_georeferenced_geotiff(
        image,
        wms_bbox,
    )

    # --------------------------------------------------------
    # Clip using actual Iran geometry
    # --------------------------------------------------------

    (
        clipped,
        profile,
        image_bounds,
    ) = clip_geotiff_to_iran(
        iran_geometry
    )

    # --------------------------------------------------------
    # Save PNG
    # --------------------------------------------------------

    clipped_width, clipped_height = save_web_png(
        clipped,
        profile,
    )

    # --------------------------------------------------------
    # Remove temporary GeoTIFF
    # --------------------------------------------------------

    if TEMP_GEOTIFF.exists():
        TEMP_GEOTIFF.unlink()

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    image_west = image_bounds[0]
    image_south = image_bounds[1]
    image_east = image_bounds[2]
    image_north = image_bounds[3]

    metadata = {
        "source": "Copernicus GWIS / ECMWF",
        "layer": LAYER_NAME,
        "date": target_date,

        "crs": "EPSG:4326",

        "image": {
            "file": OUTPUT_IMAGE.name,
            "width": clipped_width,
            "height": clipped_height,
        },

        # Original WMS request extent
        "wms_bbox": {
            "west": wms_bbox[0],
            "south": wms_bbox[1],
            "east": wms_bbox[2],
            "north": wms_bbox[3],
        },

        # Actual geographic extent of the
        # clipped raster pixels.
        "image_bounds": {
            "west": image_west,
            "south": image_south,
            "east": image_east,
            "north": image_north,
        },

        # Exact source boundary extent
        "iran_boundary": {
            "west": iran_west,
            "south": iran_south,
            "east": iran_east,
            "north": iran_north,
            "file": "IRAN.geojson",
        },

        "georeferencing": {
            "method": (
                "EPSG:4326 georeferenced GeoTIFF "
                "followed by exact geographic crop "
                "using IRAN.geojson"
            ),
            "clip": "IRAN.geojson",
            "crop": True,
        },
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "Metadata created:",
        OUTPUT_JSON
    )

    print("=" * 70)
    print("IR-FWI UPDATE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
