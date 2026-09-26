#!/usr/bin/env python3

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import requests
from PIL import Image, ImageDraw
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


# ============================================================
# Copernicus GWIS / ECMWF FWI
# ============================================================

WMS_URL = "https://maps.effis.emergency.copernicus.eu/gwis"

LAYER_NAME = "ecmwf.fwi"

# تصویر با عرض ثابت ساخته می‌شود.
# ارتفاع بعداً بر اساس نسبت واقعی BBOX محاسبه می‌شود.
IMAGE_WIDTH = 2200

TIMEOUT = 180


# ============================================================
# Load Iran boundary
# ============================================================

def load_iran_boundary():

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Iran boundary not found: {BOUNDARY_FILE}"
        )

    gdf = gpd.read_file(BOUNDARY_FILE)

    if gdf.empty:
        raise RuntimeError(
            "IRAN.geojson contains no geometry."
        )

    if gdf.crs is None:

        print(
            "Boundary CRS is undefined. "
            "Assuming EPSG:4326."
        )

        gdf = gdf.set_crs("EPSG:4326")

    else:

        gdf = gdf.to_crs("EPSG:4326")

    geometry = unary_union(gdf.geometry)

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

    # فقط برای درخواست WMS مقدار کمی حاشیه اضافه می‌کنیم.
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

    # نسبت تصویر باید دقیقاً با نسبت جغرافیایی BBOX
    # یکسان باشد.
    image_height = round(
        IMAGE_WIDTH *
        geographic_height /
        geographic_width
    )

    if image_height <= 0:
        raise RuntimeError(
            "Calculated image height is invalid."
        )

    return IMAGE_WIDTH, image_height


# ============================================================
# Convert geographic geometry to pixel mask
# ============================================================

def geometry_to_pixel_mask(
    geometry,
    bbox,
    width,
    height,
):

    minx, miny, maxx, maxy = bbox

    mask = Image.new(
        "L",
        (width, height),
        0,
    )

    draw = ImageDraw.Draw(mask)

    def to_pixel(x, y):

        px = (
            (x - minx)
            /
            (maxx - minx)
            *
            (width - 1)
        )

        py = (
            (maxy - y)
            /
            (maxy - miny)
            *
            (height - 1)
        )

        return (
            int(round(px)),
            int(round(py)),
        )

    def draw_polygon(coords):

        pixels = [
            to_pixel(x, y)
            for x, y in coords
        ]

        if len(pixels) >= 3:

            draw.polygon(
                pixels,
                fill=255,
            )

    geojson = mapping(geometry)

    # --------------------------------------------------------
    # Polygon
    # --------------------------------------------------------

    if geojson["type"] == "Polygon":

        coordinates = geojson["coordinates"]

        # Exterior
        draw_polygon(
            coordinates[0]
        )

        # Holes
        for hole in coordinates[1:]:

            pixels = [
                to_pixel(x, y)
                for x, y in hole
            ]

            if len(pixels) >= 3:

                draw.polygon(
                    pixels,
                    fill=0,
                )

    # --------------------------------------------------------
    # MultiPolygon
    # --------------------------------------------------------

    elif geojson["type"] == "MultiPolygon":

        for polygon in geojson["coordinates"]:

            # Exterior
            draw_polygon(
                polygon[0]
            )

            # Holes
            for hole in polygon[1:]:

                pixels = [
                    to_pixel(x, y)
                    for x, y in hole
                ]

                if len(pixels) >= 3:

                    draw.polygon(
                        pixels,
                        fill=0,
                    )

    else:

        raise RuntimeError(
            "Unsupported boundary geometry type: "
            f"{geojson['type']}"
        )

    return mask


# ============================================================
# Clip image exactly to Iran boundary
# ============================================================

def clip_image_to_boundary(
    image,
    geometry,
    bbox,
):

    image = image.convert("RGBA")

    mask = geometry_to_pixel_mask(
        geometry=geometry,
        bbox=bbox,
        width=image.width,
        height=image.height,
    )

    alpha = image.getchannel("A")

    # فقط قسمت داخل مرز ایران باقی می‌ماند.
    combined_alpha = Image.composite(
        alpha,
        Image.new(
            "L",
            image.size,
            0,
        ),
        mask,
    )

    image.putalpha(
        combined_alpha
    )

    return image


# ============================================================
# Download FWI from GWIS
# ============================================================

def download_fwi(
    bbox,
    target_date,
    image_width,
    image_height,
):

    minx, miny, maxx, maxy = bbox

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
            image_width,

        "HEIGHT":
            image_height,

        "FORMAT":
            "image/png",

        "TRANSPARENT":
            "TRUE",

        "TIME":
            target_date,
    }

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
        "BBOX:",
        params["BBOX"]
    )

    print(
        "Image width:",
        image_width
    )

    print(
        "Image height:",
        image_height
    )

    response = requests.get(
        WMS_URL,
        params=params,
        timeout=TIMEOUT,
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

    print(
        "Downloaded:",
        len(response.content),
        "bytes"
    )

    if "image" not in content_type.lower():

        print(
            response.text[:1000]
        )

        raise RuntimeError(
            "GWIS did not return an image."
        )

    return Image.open(
        io.BytesIO(
            response.content
        )
    ).convert("RGBA")


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
    # Load Iran boundary
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
    # WMS BBOX
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
    # Calculate image dimensions
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

    print()
    print(
        "Geographic aspect ratio:",
        (
            bbox[2] - bbox[0]
        )
        /
        (
            bbox[3] - bbox[1]
        )
    )

    print(
        "Image aspect ratio:",
        image_width / image_height
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
        today +
        timedelta(days=1)
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
    # Download FWI
    # --------------------------------------------------------

    image = download_fwi(
        bbox=bbox,
        target_date=target_date_str,
        image_width=image_width,
        image_height=image_height,
    )

    print(
        "Downloaded image size:",
        image.size
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if image.size != (
        image_width,
        image_height,
    ):

        raise RuntimeError(
            "Returned image dimensions do not "
            "match requested dimensions."
        )

    # --------------------------------------------------------
    # Clip exactly to Iran boundary
    # --------------------------------------------------------

    print(
        "Clipping image to IRAN.geojson..."
    )

    clipped = (
        clip_image_to_boundary(
            image=image,
            geometry=iran_geometry,
            bbox=bbox,
        )
    )

    print(
        "Clipping completed."
    )

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    clipped.save(
        OUTPUT_IMAGE,
        format="PNG",
        optimize=True,
    )

    print(
        "Saved:",
        OUTPUT_IMAGE
    )

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
                clipped.width,

            "height":
                clipped.height,

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

    }

    OUTPUT_JSON.write_text(

        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),

        encoding="utf-8",
    )

    print(
        "Saved:",
        OUTPUT_JSON
    )

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
