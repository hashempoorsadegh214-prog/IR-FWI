#!/usr/bin/env python3

import io
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from PIL import Image
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
# Copernicus GWIS / ECMWF FWI
# ============================================================

WMS_URL = "https://maps.effis.emergency.copernicus.eu/gwis"

LAYER_NAME = "ecmwf.fwi"

IMAGE_WIDTH = 1600

TILES_X = 2
TILES_Y = 2

TIMEOUT = 180

MAX_DOWNLOAD_ATTEMPTS = 5

RETRY_WAIT_SECONDS = 8


# ============================================================
# Load Iran boundary
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
            "IRAN.geojson contains no geometry."
        )

    if gdf.crs is None:

        print(
            "Boundary CRS is undefined. "
            "Assuming EPSG:4326."
        )

        gdf = gdf.set_crs(
            "EPSG:4326"
        )

    else:

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


# ============================================================
# Calculate geographic BBOX
# ============================================================

def get_bbox(geometry):

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
# Calculate image dimensions
# ============================================================

def get_image_dimensions(bbox):

    minx, miny, maxx, maxy = bbox

    geographic_width = maxx - minx
    geographic_height = maxy - miny

    if geographic_width <= 0:
        raise RuntimeError(
            "Invalid BBOX width."
        )

    if geographic_height <= 0:
        raise RuntimeError(
            "Invalid BBOX height."
        )

    image_height = round(
        IMAGE_WIDTH
        * geographic_height
        / geographic_width
    )

    if image_height <= 0:
        raise RuntimeError(
            "Calculated image height is invalid."
        )

    return (
        IMAGE_WIDTH,
        image_height,
    )


# ============================================================
# Calculate tile BBOX
# ============================================================

def get_tile_bbox(
    bbox,
    tile_x,
    tile_y,
    tiles_x,
    tiles_y,
):

    minx, miny, maxx, maxy = bbox

    geographic_width = maxx - minx
    geographic_height = maxy - miny

    tile_width = (
        geographic_width / tiles_x
    )

    tile_height = (
        geographic_height / tiles_y
    )

    tile_minx = (
        minx + tile_x * tile_width
    )

    tile_maxx = (
        minx + (tile_x + 1) * tile_width
    )

    # Row 0 = north
    tile_maxy = (
        maxy - tile_y * tile_height
    )

    tile_miny = (
        maxy - (tile_y + 1) * tile_height
    )

    return (
        tile_minx,
        tile_miny,
        tile_maxx,
        tile_maxy,
    )


# ============================================================
# Calculate tile dimensions
# ============================================================

def get_tile_dimensions(
    final_width,
    final_height,
    tiles_x,
    tiles_y,
    tile_x,
    tile_y,
):

    base_width = (
        final_width // tiles_x
    )

    base_height = (
        final_height // tiles_y
    )

    if tile_x == tiles_x - 1:

        tile_width = (
            final_width
            - base_width * (tiles_x - 1)
        )

    else:

        tile_width = base_width

    if tile_y == tiles_y - 1:

        tile_height = (
            final_height
            - base_height * (tiles_y - 1)
        )

    else:

        tile_height = base_height

    return (
        tile_width,
        tile_height,
    )


# ============================================================
# Download one WMS tile
# ============================================================

def download_wms_tile(
    tile_bbox,
    target_date,
    tile_width,
    tile_height,
    tile_number,
    total_tiles,
):

    minx, miny, maxx, maxy = tile_bbox

    params = {

        "SERVICE": "WMS",

        "VERSION": "1.1.1",

        "REQUEST": "GetMap",

        "LAYERS": LAYER_NAME,

        "STYLES": "",

        "SRS": "EPSG:4326",

        "BBOX":
            f"{minx},{miny},{maxx},{maxy}",

        "WIDTH":
            tile_width,

        "HEIGHT":
            tile_height,

        "FORMAT":
            "image/png",

        "TRANSPARENT":
            "TRUE",

        "TIME":
            target_date,
    }

    headers = {

        "User-Agent":
            "IR-FWI/2.0 (GitHub Actions)",

        "Accept":
            "image/png,image/*;q=0.9,*/*;q=0.8",

        "Accept-Encoding":
            "identity",

        "Cache-Control":
            "no-cache",

        "Pragma":
            "no-cache",
    }

    print()
    print(
        f"Tile {tile_number}/{total_tiles}"
    )

    print(
        "BBOX:",
        params["BBOX"]
    )

    print(
        "Size:",
        tile_width,
        "x",
        tile_height
    )

    last_error = None

    for attempt in range(
        1,
        MAX_DOWNLOAD_ATTEMPTS + 1,
    ):

        response = None

        try:

            print(
                f"Tile download attempt "
                f"{attempt}/{MAX_DOWNLOAD_ATTEMPTS}"
            )

            response = requests.get(
                WMS_URL,
                params=params,
                headers=headers,
                timeout=TIMEOUT,
                stream=True,
            )

            response.raise_for_status()

            content_type = (
                response.headers.get(
                    "Content-Type",
                    ""
                )
            )

            print(
                "HTTP status:",
                response.status_code
            )

            print(
                "Content-Type:",
                content_type
            )

            if "image" not in content_type.lower():

                text = response.text[:1000]

                raise RuntimeError(
                    "GWIS did not return an image.\n"
                    f"Response:\n{text}"
                )

            chunks = []

            total_bytes = 0

            for chunk in response.iter_content(
                chunk_size=32 * 1024
            ):

                if chunk:

                    chunks.append(chunk)

                    total_bytes += len(chunk)

            content = b"".join(chunks)

            print(
                "Downloaded:",
                total_bytes,
                "bytes"
            )

            if total_bytes == 0:

                raise RuntimeError(
                    "GWIS returned an empty image."
                )

            image = Image.open(
                io.BytesIO(content)
            ).convert("RGBA")

            print(
                "Received image size:",
                image.size
            )

            if image.size != (
                tile_width,
                tile_height,
            ):

                raise RuntimeError(
                    "Returned tile dimensions do not "
                    "match requested dimensions. "
                    f"Expected "
                    f"{tile_width}x{tile_height}, "
                    f"received "
                    f"{image.width}x{image.height}."
                )

            return image

        except Exception as exc:

            last_error = exc

            print(
                "Tile download failed:"
            )

            print(
                repr(exc)
            )

            if attempt < MAX_DOWNLOAD_ATTEMPTS:

                print(
                    f"Waiting "
                    f"{RETRY_WAIT_SECONDS} "
                    f"seconds before retry..."
                )

                time.sleep(
                    RETRY_WAIT_SECONDS
                )

        finally:

            if response is not None:

                try:
                    response.close()
                except Exception:
                    pass

    raise RuntimeError(
        f"Unable to download WMS tile "
        f"{tile_number}/{total_tiles} "
        f"after "
        f"{MAX_DOWNLOAD_ATTEMPTS} attempts."
    ) from last_error


# ============================================================
# Download all WMS tiles and assemble them
# ============================================================

def download_fwi(
    bbox,
    target_date,
    image_width,
    image_height,
):

    print()
    print(
        "Downloading ECMWF FWI..."
    )

    print(
        "URL:",
        WMS_URL
    )

    print(
        "Layer:",
        LAYER_NAME
    )

    print(
        "Date:",
        target_date
    )

    print(
        "Final image:",
        image_width,
        "x",
        image_height
    )

    print(
        "WMS tiles:",
        f"{TILES_X} x {TILES_Y}"
    )

    total_tiles = (
        TILES_X * TILES_Y
    )

    final_image = Image.new(
        "RGBA",
        (
            image_width,
            image_height,
        ),
        (0, 0, 0, 0),
    )

    for tile_y in range(
        TILES_Y
    ):

        for tile_x in range(
            TILES_X
        ):

            tile_width, tile_height = (
                get_tile_dimensions(
                    final_width=image_width,
                    final_height=image_height,
                    tiles_x=TILES_X,
                    tiles_y=TILES_Y,
                    tile_x=tile_x,
                    tile_y=tile_y,
                )
            )

            tile_bbox = get_tile_bbox(
                bbox=bbox,
                tile_x=tile_x,
                tile_y=tile_y,
                tiles_x=TILES_X,
                tiles_y=TILES_Y,
            )

            tile_number = (
                tile_y * TILES_X
                + tile_x
                + 1
            )

            tile = download_wms_tile(
                tile_bbox=tile_bbox,
                target_date=target_date,
                tile_width=tile_width,
                tile_height=tile_height,
                tile_number=tile_number,
                total_tiles=total_tiles,
            )

            x_offset = (
                tile_x
                * (image_width // TILES_X)
            )

            y_offset = (
                tile_y
                * (image_height // TILES_Y)
            )

            final_image.paste(
                tile,
                (
                    x_offset,
                    y_offset,
                )
            )

    print()
    print(
        "All WMS tiles downloaded successfully."
    )

    print(
        "Final assembled image size:",
        final_image.size
    )

    return final_image


# ============================================================
# Convert assembled PNG to georeferenced GeoTIFF
# ============================================================

def create_georeferenced_geotiff(
    image,
    bbox,
    output_path,
):

    print()
    print(
        "Creating georeferenced GeoTIFF..."
    )

    width = image.width
    height = image.height

    transform = from_bounds(
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
        width,
        height,
    )

    rgba = np.asarray(
        image,
        dtype=np.uint8,
    )

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=4,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        compress="deflate",
        predictor=2,
        tiled=True,
    ) as dst:

        dst.write(
            rgba[:, :, 0],
            1,
        )

        dst.write(
            rgba[:, :, 1],
            2,
        )

        dst.write(
            rgba[:, :, 2],
            3,
        )

        dst.write(
            rgba[:, :, 3],
            4,
        )

    print(
        "GeoTIFF created:",
        output_path
    )

    print(
        "CRS: EPSG:4326"
    )

    print(
        "Transform:",
        transform
    )


# ============================================================
# Clip georeferenced raster to Iran boundary
# ============================================================

def clip_geotiff_to_iran(
    geotiff_path,
    geometry,
):

    print()
    print(
        "Clipping georeferenced raster "
        "to IRAN.geojson..."
    )

    with rasterio.open(
        geotiff_path,
    ) as src:

        print(
            "Source CRS:",
            src.crs
        )

        print(
            "Source bounds:",
            src.bounds
        )

        print(
            "Source size:",
            src.width,
            "x",
            src.height
        )

        if src.crs is None:

            raise RuntimeError(
                "GeoTIFF has no CRS."
            )

        clipped, clipped_transform = mask(
            src,
            [mapping(geometry)],
            crop=False,
            filled=True,
            nodata=0,
        )

        profile = src.profile.copy()

        profile.update(
            transform=clipped_transform,
            width=clipped.shape[2],
            height=clipped.shape[1],
            count=4,
            dtype="uint8",
            compress="deflate",
            predictor=2,
        )

        print(
            "Clip completed."
        )

        return clipped, profile


# ============================================================
# Convert clipped raster to web PNG
# ============================================================

def save_web_png(
    clipped,
    output_path,
):

    print()
    print(
        "Creating final web PNG..."
    )

    if clipped.shape[0] != 4:

        raise RuntimeError(
            "Expected a 4-band RGBA raster."
        )

    rgba = np.transpose(
        clipped,
        (1, 2, 0),
    )

    image = Image.fromarray(
        rgba,
        mode="RGBA",
    )

    image.save(
        output_path,
        format="PNG",
        optimize=True,
    )

    print(
        "Saved:",
        output_path
    )

    print(
        "PNG size:",
        image.width,
        "x",
        image.height
    )

    return image


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "IR-FWI - Iran ECMWF FWI"
    )

    print("=" * 70)

    WEB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load boundary
    # --------------------------------------------------------

    iran_geometry = (
        load_iran_boundary()
    )

    print(
        "Iran boundary loaded."
    )

    minx, miny, maxx, maxy = (
        iran_geometry.bounds
    )

    print()
    print(
        "Original Iran bounds:"
    )

    print(
        "West :",
        minx
    )

    print(
        "South:",
        miny
    )

    print(
        "East :",
        maxx
    )

    print(
        "North:",
        maxy
    )

    # --------------------------------------------------------
    # BBOX
    # --------------------------------------------------------

    bbox = get_bbox(
        iran_geometry
    )

    print()
    print(
        "WMS BBOX:"
    )

    print(
        "West :",
        bbox[0]
    )

    print(
        "South:",
        bbox[1]
    )

    print(
        "East :",
        bbox[2]
    )

    print(
        "North:",
        bbox[3]
    )

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    image_width, image_height = (
        get_image_dimensions(
            bbox
        )
    )

    print()
    print(
        "Calculated image dimensions:"
    )

    print(
        "Width :",
        image_width
    )

    print(
        "Height:",
        image_height
    )

    geographic_aspect_ratio = (
        (bbox[2] - bbox[0])
        /
        (bbox[3] - bbox[1])
    )

    image_aspect_ratio = (
        image_width
        /
        image_height
    )

    print()
    print(
        "Geographic aspect ratio:",
        geographic_aspect_ratio
    )

    print(
        "Image aspect ratio:",
        image_aspect_ratio
    )

    # --------------------------------------------------------
    # Forecast date
    # --------------------------------------------------------

    today = (
        datetime
        .now(timezone.utc)
        .date()
    )

    target_date = (
        today + timedelta(days=1)
    )

    target_date_str = (
        target_date.isoformat()
    )

    print()
    print(
        "Forecast date:",
        target_date_str
    )

    # --------------------------------------------------------
    # Download WMS
    # --------------------------------------------------------

    image = download_fwi(
        bbox=bbox,
        target_date=target_date_str,
        image_width=image_width,
        image_height=image_height,
    )

    # --------------------------------------------------------
    # Create georeferenced GeoTIFF
    # --------------------------------------------------------

    create_georeferenced_geotiff(
        image=image,
        bbox=bbox,
        output_path=TEMP_GEOTIFF,
    )

    # --------------------------------------------------------
    # Clip using actual geographic geometry
    # --------------------------------------------------------

    clipped, profile = (
        clip_geotiff_to_iran(
            geotiff_path=TEMP_GEOTIFF,
            geometry=iran_geometry,
        )
    )

    # --------------------------------------------------------
    # Save PNG
    # --------------------------------------------------------

    final_image = save_web_png(
        clipped=clipped,
        output_path=OUTPUT_IMAGE,
    )

    # --------------------------------------------------------
    # Remove temporary GeoTIFF
    # --------------------------------------------------------

    try:

        TEMP_GEOTIFF.unlink()

        print(
            "Temporary GeoTIFF removed."
        )

    except FileNotFoundError:

        pass

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "source":
            "Copernicus GWIS / ECMWF",

        "layer":
            LAYER_NAME,

        "date":
            target_date_str,

        "crs":
            "EPSG:4326",

        "image": {

            "file":
                OUTPUT_IMAGE.name,

            "width":
                final_image.width,

            "height":
                final_image.height,

        },

        "bbox": {

            "west":
                bbox[0],

            "south":
                bbox[1],

            "east":
                bbox[2],

            "north":
                bbox[3],

        },

        "iran_boundary": {

            "west":
                minx,

            "south":
                miny,

            "east":
                maxx,

            "north":
                maxy,

            "file":
                "IRAN.geojson",

        },

        "georeferencing": {

            "method":
                "EPSG:4326 georeferenced GeoTIFF "
                "followed by geographic mask",

            "clip":
                "IRAN.geojson",

        },

    }

    OUTPUT_JSON.write_text(

        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),

        encoding="utf-8",
    )

    print()
    print(
        "Saved:",
        OUTPUT_JSON
    )

    print()
    print("=" * 70)

    print(
        "DONE"
    )

    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
