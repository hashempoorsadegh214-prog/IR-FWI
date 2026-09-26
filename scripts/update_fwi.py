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

# Smaller tiles = much more reliable WMS downloads
TILES_X = 3
TILES_Y = 3

TIMEOUT = 180

MAX_DOWNLOAD_ATTEMPTS = 6

RETRY_WAIT_SECONDS = 10


# ============================================================
# Boundary
# ============================================================

def load_iran_boundary():

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Iran boundary not found: {BOUNDARY_FILE}"
        )

    gdf = gpd.read_file(
        BOUNDARY_FILE
    )

    if gdf.empty:
        raise RuntimeError(
            "IRAN.geojson is empty."
        )

    if gdf.crs is None:
        gdf = gdf.set_crs(
            "EPSG:4326"
        )

    gdf = gdf.to_crs(
        "EPSG:4326"
    )

    geometry = unary_union(
        gdf.geometry
    )

    if geometry.is_empty:
        raise RuntimeError(
            "Iran boundary geometry is empty."
        )

    return geometry


def get_bbox(geometry):

    minx, miny, maxx, maxy = (
        geometry.bounds
    )

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

    geographic_width = (
        east - west
    )

    geographic_height = (
        north - south
    )

    aspect_ratio = (
        geographic_width /
        geographic_height
    )

    width = IMAGE_WIDTH

    height = round(
        width / aspect_ratio
    )

    return width, height


# ============================================================
# Tile bbox
# ============================================================

def get_tile_bbox(
    bbox,
    tile_x,
    tile_y,
):

    west, south, east, north = bbox

    total_width = east - west
    total_height = north - south

    tile_width = (
        total_width /
        TILES_X
    )

    tile_height = (
        total_height /
        TILES_Y
    )

    tile_west = (
        west +
        tile_x * tile_width
    )

    tile_east = (
        west +
        (tile_x + 1) *
        tile_width
    )

    # Tile row 0 = north
    tile_north = (
        north -
        tile_y * tile_height
    )

    tile_south = (
        north -
        (tile_y + 1) *
        tile_height
    )

    return (
        tile_west,
        tile_south,
        tile_east,
        tile_north,
    )


# ============================================================
# Tile dimensions
# ============================================================

def get_tile_dimensions(
    full_width,
    full_height,
    tile_x,
    tile_y,
):

    base_width = (
        full_width //
        TILES_X
    )

    base_height = (
        full_height //
        TILES_Y
    )

    if tile_x == TILES_X - 1:

        tile_width = (
            full_width -
            base_width *
            (TILES_X - 1)
        )

    else:

        tile_width = base_width


    if tile_y == TILES_Y - 1:

        tile_height = (
            full_height -
            base_height *
            (TILES_Y - 1)
        )

    else:

        tile_height = base_height


    return (
        tile_width,
        tile_height,
    )


# ============================================================
# WMS Tile Download
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

        "BBOX":
            f"{west},{south},{east},{north}",

        "WIDTH": width,

        "HEIGHT": height,

        "TIME": target_date,
    }


    headers = {

        "User-Agent":
            "Mozilla/5.0 "
            "(compatible; IR-FWI/1.0)",

        "Accept":
            "image/png,image/*;q=0.9,*/*;q=0.8",

        "Accept-Encoding":
            "identity",

        "Connection":
            "close",
    }


    last_error = None


    for attempt in range(
        1,
        MAX_DOWNLOAD_ATTEMPTS + 1
    ):

        try:

            print(
                f"Downloading WMS tile "
                f"{width}x{height}, "
                f"attempt "
                f"{attempt}/"
                f"{MAX_DOWNLOAD_ATTEMPTS}"
            )


            response = requests.get(

                WMS_URL,

                params=params,

                headers=headers,

                timeout=TIMEOUT,

                stream=True,
            )


            response.raise_for_status()


            content_length = (
                response.headers.get(
                    "Content-Length"
                )
            )


            data = bytearray()


            for chunk in response.iter_content(
                chunk_size=16384
            ):

                if chunk:

                    data.extend(
                        chunk
                    )


            response.close()


            if not data:

                raise RuntimeError(
                    "WMS returned "
                    "an empty response."
                )


            if content_length:

                expected = int(
                    content_length
                )

                received = len(data)

                print(
                    f"Received "
                    f"{received} / "
                    f"{expected} bytes"
                )

                if received != expected:

                    raise RuntimeError(
                        "Incomplete WMS response: "
                        f"{received} / "
                        f"{expected} bytes"
                    )


            image = Image.open(
                io.BytesIO(
                    bytes(data)
                )
            ).convert("RGBA")


            if image.size != (
                width,
                height
            ):

                raise RuntimeError(
                    "Unexpected WMS image size: "
                    f"{image.size}; "
                    f"expected "
                    f"{(width, height)}"
                )


            print(
                "Tile downloaded successfully."
            )


            return image


        except Exception as exc:

            last_error = exc

            print(
                "WMS tile download failed:",
                repr(exc)
            )


            if attempt < (
                MAX_DOWNLOAD_ATTEMPTS
            ):

                wait_time = (
                    RETRY_WAIT_SECONDS *
                    attempt
                )

                print(
                    f"Waiting "
                    f"{wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )


    raise RuntimeError(

        "Failed to download WMS tile "
        f"after "
        f"{MAX_DOWNLOAD_ATTEMPTS} "
        f"attempts: "
        f"{last_error}"

    )


# ============================================================
# Download all WMS tiles
# ============================================================

def download_fwi(
    bbox,
    width,
    height,
    target_date,
):

    print(
        "WMS request bbox:"
    )

    print(bbox)


    print(
        "Final image size:"
    )

    print(
        width,
        height
    )


    canvas = Image.new(

        "RGBA",

        (
            width,
            height
        ),

        (
            0,
            0,
            0,
            0
        ),
    )


    for tile_y in range(
        TILES_Y
    ):

        for tile_x in range(
            TILES_X
        ):

            tile_bbox = get_tile_bbox(

                bbox,

                tile_x,

                tile_y,
            )


            tile_width, tile_height = (
                get_tile_dimensions(

                    width,

                    height,

                    tile_x,

                    tile_y,
                )
            )


            print(
                "------------------------------------------------"
            )

            print(
                f"Tile "
                f"{tile_y + 1}/"
                f"{TILES_Y}, "
                f"{tile_x + 1}/"
                f"{TILES_X}"
            )


            print(
                "Tile bbox:",
                tile_bbox
            )


            print(
                "Tile size:",
                tile_width,
                tile_height
            )


            tile = download_wms_tile(

                tile_bbox,

                tile_width,

                tile_height,

                target_date,
            )


            pixel_x = 0

            for x in range(
                tile_x
            ):

                previous_width, _ = (
                    get_tile_dimensions(

                        width,

                        height,

                        x,

                        tile_y,
                    )
                )

                pixel_x += (
                    previous_width
                )


            pixel_y = 0

            for y in range(
                tile_y
            ):

                _, previous_height = (
                    get_tile_dimensions(

                        width,

                        height,

                        tile_x,

                        y,
                    )
                )

                pixel_y += (
                    previous_height
                )


            canvas.paste(

                tile,

                (
                    pixel_x,
                    pixel_y
                ),
            )


    return canvas


# ============================================================
# Create georeferenced GeoTIFF
# ============================================================

def create_georeferenced_geotiff(
    image,
    bbox,
):

    west, south, east, north = bbox

    width, height = (
        image.size
    )


    transform = from_bounds(

        west,

        south,

        east,

        north,

        width,

        height,
    )


    rgba = np.asarray(
        image,
        dtype=np.uint8
    )


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

        nodata=0,

        compress="deflate",

    ) as dst:

        for band in range(4):

            dst.write(

                rgba[:, :, band],

                band + 1
            )


    print(
        "Georeferenced GeoTIFF created:"
    )

    print(
        TEMP_GEOTIFF
    )


# ============================================================
# Clip exactly to Iran
# ============================================================

def clip_geotiff_to_iran(
    geometry
):

    with rasterio.open(
        TEMP_GEOTIFF
    ) as src:

        clipped, clipped_transform = (
            mask(

                src,

                [mapping(geometry)],

                crop=True,

                filled=True,

                nodata=0,
            )
        )


        clipped_height = (
            clipped.shape[1]
        )

        clipped_width = (
            clipped.shape[2]
        )


        profile = (
            src.profile.copy()
        )


        profile.update({

            "height":
                clipped_height,

            "width":
                clipped_width,

            "transform":
                clipped_transform,

            "nodata":
                0,

            "compress":
                "deflate",

        })


        image_bounds = (
            rasterio.transform.array_bounds(

                clipped_height,

                clipped_width,

                clipped_transform,
            )
        )


        print(
            "Clipped raster bounds:"
        )

        print(
            image_bounds
        )


        print(
            "Clipped raster size:"
        )

        print(
            clipped_width,
            clipped_height
        )


        return (

            clipped,

            profile,

            image_bounds,

        )


# ============================================================
# Save PNG
# ============================================================

def save_web_png(
    clipped,
    profile
):

    rgba = np.transpose(

        clipped,

        (
            1,
            2,
            0
        )
    )


    rgba = np.asarray(

        rgba,

        dtype=np.uint8
    )


    image = Image.fromarray(

        rgba,

        mode="RGBA"
    )


    image.save(

        OUTPUT_IMAGE,

        format="PNG",

        optimize=True
    )


    print(
        "Web PNG created:"
    )

    print(
        OUTPUT_IMAGE
    )


    return (

        image.width,

        image.height

    )


# ============================================================
# Main
# ============================================================

def main():

    WEB_DIR.mkdir(

        parents=True,

        exist_ok=True
    )


    print(
        "============================================================"
    )

    print(
        "IR-FWI UPDATE"
    )

    print(
        "============================================================"
    )


    # --------------------------------------------------------
    # Iran boundary
    # --------------------------------------------------------

    iran_geometry = (
        load_iran_boundary()
    )


    (
        iran_west,
        iran_south,
        iran_east,
        iran_north,
    ) = iran_geometry.bounds


    print(
        "Iran boundary:"
    )

    print(

        iran_west,

        iran_south,

        iran_east,

        iran_north

    )


    # --------------------------------------------------------
    # WMS bbox
    # --------------------------------------------------------

    wms_bbox = get_bbox(
        iran_geometry
    )


    print(
        "WMS bbox:"
    )

    print(
        wms_bbox
    )


    # --------------------------------------------------------
    # Forecast date
    # --------------------------------------------------------

    now_utc = datetime.now(
        timezone.utc
    )


    target_date = (

        now_utc +

        timedelta(days=1)

    ).strftime(
        "%Y-%m-%d"
    )


    print(
        "Target FWI date:",
        target_date
    )


    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    width, height = (
        get_image_dimensions(
            wms_bbox
        )
    )


    print(
        "Geographic aspect ratio:",
        (
            wms_bbox[2] -
            wms_bbox[0]
        ) /
        (
            wms_bbox[3] -
            wms_bbox[1]
        )
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

        wms_bbox

    )


    # --------------------------------------------------------
    # Clip to Iran
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

    clipped_width, clipped_height = (
        save_web_png(

            clipped,

            profile

        )
    )


    # --------------------------------------------------------
    # Remove temporary GeoTIFF
    # --------------------------------------------------------

    if TEMP_GEOTIFF.exists():

        TEMP_GEOTIFF.unlink()


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "source":
            "Copernicus GWIS / ECMWF",

        "layer":
            LAYER_NAME,

        "date":
            target_date,

        "crs":
            "EPSG:4326",


        "image": {

            "file":
                OUTPUT_IMAGE.name,

            "width":
                clipped_width,

            "height":
                clipped_height,

        },


        "wms_bbox": {

            "west":
                wms_bbox[0],

            "south":
                wms_bbox[1],

            "east":
                wms_bbox[2],

            "north":
                wms_bbox[3],

        },


        "image_bounds": {

            "west":
                image_bounds[0],

            "south":
                image_bounds[1],

            "east":
                image_bounds[2],

            "north":
                image_bounds[3],

        },


        "iran_boundary": {

            "west":
                iran_west,

            "south":
                iran_south,

            "east":
                iran_east,

            "north":
                iran_north,

            "file":
                "IRAN.geojson",

        },


        "georeferencing": {

            "method":
                "EPSG:4326 georeferenced GeoTIFF followed by exact geographic crop using IRAN.geojson",

            "clip":
                "IRAN.geojson",

            "crop":
                True,

        },

    }


    with open(

        OUTPUT_JSON,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            metadata,

            f,

            ensure_ascii=False,

            indent=2

        )


    print(
        "Metadata created:"
    )

    print(
        OUTPUT_JSON
    )


    print(
        "============================================================"
    )

    print(
        "IR-FWI UPDATE COMPLETED"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":

    main()
