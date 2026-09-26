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

IMAGE_WIDTH = 2200
IMAGE_HEIGHT = 1600

TIMEOUT = 180


# ============================================================
# Helpers
# ============================================================

def load_iran_boundary():
    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Iran boundary not found: {BOUNDARY_FILE}"
        )

    gdf = gpd.read_file(BOUNDARY_FILE)

    if gdf.empty:
        raise RuntimeError("IRAN.geojson contains no geometry.")

    if gdf.crs is None:
        print("Boundary CRS is undefined. Assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")

    else:
        gdf = gdf.to_crs("EPSG:4326")

    geometry = unary_union(gdf.geometry)

    if geometry.is_empty:
        raise RuntimeError("Iran boundary geometry is empty.")

    return geometry


def get_bbox(geometry):
    minx, miny, maxx, maxy = geometry.bounds

    width = maxx - minx
    height = maxy - miny

    # Small margin only for WMS request.
    margin_x = width * 0.01
    margin_y = height * 0.01

    return (
        minx - margin_x,
        miny - margin_y,
        maxx + margin_x,
        maxy + margin_y,
    )


def geometry_to_pixel_mask(geometry, bbox, width, height):
    minx, miny, maxx, maxy = bbox

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    def to_pixel(x, y):
        px = (x - minx) / (maxx - minx) * (width - 1)
        py = (maxy - y) / (maxy - miny) * (height - 1)

        return int(round(px)), int(round(py))

    def draw_polygon(coords):
        pixels = [to_pixel(x, y) for x, y in coords]
        if len(pixels) >= 3:
            draw.polygon(pixels, fill=255)

    geojson = mapping(geometry)

    if geojson["type"] == "Polygon":

        coordinates = geojson["coordinates"]

        # Exterior
        draw_polygon(coordinates[0])

        # Holes
        for hole in coordinates[1:]:
            pixels = [to_pixel(x, y) for x, y in hole]
            if len(pixels) >= 3:
                draw.polygon(pixels, fill=0)

    elif geojson["type"] == "MultiPolygon":

        for polygon in geojson["coordinates"]:

            # Exterior
            draw_polygon(polygon[0])

            # Holes
            for hole in polygon[1:]:
                pixels = [to_pixel(x, y) for x, y in hole]
                if len(pixels) >= 3:
                    draw.polygon(pixels, fill=0)

    else:
        raise RuntimeError(
            f"Unsupported boundary geometry type: {geojson['type']}"
        )

    return mask


def clip_image_to_boundary(image, geometry, bbox):
    image = image.convert("RGBA")

    mask = geometry_to_pixel_mask(
        geometry,
        bbox,
        image.width,
        image.height,
    )

    alpha = image.getchannel("A")

    # Keep pixels only inside Iran.
    combined_alpha = Image.composite(
        alpha,
        Image.new("L", image.size, 0),
        mask,
    )

    image.putalpha(combined_alpha)

    return image


# ============================================================
# Download FWI
# ============================================================

def download_fwi(bbox, target_date):
    minx, miny, maxx, maxy = bbox

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": LAYER_NAME,
        "STYLES": "",
        "SRS": "EPSG:4326",
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": IMAGE_WIDTH,
        "HEIGHT": IMAGE_HEIGHT,
        "FORMAT": "image/png",
        "TRANSPARENT": "TRUE",
        "TIME": target_date,
    }

    print("Downloading ECMWF FWI...")
    print("URL:", WMS_URL)
    print("Layer:", LAYER_NAME)
    print("Date:", target_date)
    print("BBOX:", params["BBOX"])

    response = requests.get(
        WMS_URL,
        params=params,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    print("HTTP status:", response.status_code)
    print("Content-Type:", content_type)
    print("Downloaded:", len(response.content), "bytes")

    if "image" not in content_type.lower():
        print(response.text[:1000])
        raise RuntimeError(
            "GWIS did not return an image."
        )

    return Image.open(io.BytesIO(response.content)).convert("RGBA")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("IR-FWI - Iran ECMWF FWI")
    print("=" * 70)

    WEB_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load Iran boundary
    # --------------------------------------------------------

    iran_geometry = load_iran_boundary()

    print("Iran boundary loaded.")

    minx, miny, maxx, maxy = iran_geometry.bounds

    print("Original Iran bounds:")
    print("West :", minx)
    print("South:", miny)
    print("East :", maxx)
    print("North:", maxy)

    # --------------------------------------------------------
    # WMS request bbox
    # --------------------------------------------------------

    bbox = get_bbox(iran_geometry)

    print()
    print("WMS BBOX:")
    print("West :", bbox[0])
    print("South:", bbox[1])
    print("East :", bbox[2])
    print("North:", bbox[3])

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    today = datetime.now(timezone.utc).date()

    target_date = today + timedelta(days=1)

    target_date_str = target_date.isoformat()

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    image = download_fwi(
        bbox=bbox,
        target_date=target_date_str,
    )

    print("Image size:", image.size)

    # --------------------------------------------------------
    # Clip exactly to Iran boundary
    # --------------------------------------------------------

    print("Clipping image to IRAN.geojson...")

    clipped = clip_image_to_boundary(
        image=image,
        geometry=iran_geometry,
        bbox=bbox,
    )

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    clipped.save(
        OUTPUT_IMAGE,
        format="PNG",
        optimize=True,
    )

    print("Saved:", OUTPUT_IMAGE)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "source": "Copernicus GWIS / ECMWF",
        "layer": LAYER_NAME,
        "date": target_date_str,
        "crs": "EPSG:4326",

        "image": {
            "file": OUTPUT_IMAGE.name,
            "width": clipped.width,
            "height": clipped.height,
        },

        "bbox": {
            "west": bbox[0],
            "south": bbox[1],
            "east": bbox[2],
            "north": bbox[3],
        },

        "iran_boundary": {
            "west": minx,
            "south": miny,
            "east": maxx,
            "north": maxy,
            "file": "IRAN.geojson",
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

    print("Saved:", OUTPUT_JSON)

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
