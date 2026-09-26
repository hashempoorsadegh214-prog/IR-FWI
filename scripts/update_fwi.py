#!/usr/bin/env python3
import json
import os
from datetime import datetime, timedelta, timezone
from io import BytesIO

import geopandas as gpd
import requests
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOUNDARY_FILE = os.path.join(ROOT, "IRAN.geojson")
WEB_DIR = os.path.join(ROOT, "web")

WMS_URL = "https://maps.effis.emergency.copernicus.eu/gwis"
WMS_LAYER = "ecmwf.fwi"

WIDTH = 2200
HEIGHT = 1600
TIMEOUT = 180

IRAN_TZ = timezone(timedelta(hours=3, minutes=30))
NOW = datetime.now(IRAN_TZ)
TARGET = NOW.date() + timedelta(days=1)
TARGET_DATE = TARGET.strftime("%Y-%m-%d")


def ring_pixels(ring, west, south, east, north, width, height):
    pixels = []
    for lon, lat, *_ in ring.coords:
        x = (lon - west) / (east - west) * (width - 1)
        y = (north - lat) / (north - south) * (height - 1)
        pixels.append((round(x), round(y)))
    return pixels


def make_mask(geometry, west, south, east, north, width, height):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if geometry.geom_type == "Polygon":
        polygons = [geometry]
    elif geometry.geom_type == "MultiPolygon":
        polygons = list(geometry.geoms)
    else:
        polygons = []

    for polygon in polygons:
        draw.polygon(
            ring_pixels(
                polygon.exterior,
                west, south, east, north,
                width, height
            ),
            fill=255,
        )

        for hole in polygon.interiors:
            draw.polygon(
                ring_pixels(
                    hole,
                    west, south, east, north,
                    width, height
                ),
                fill=0,
            )

    return mask


def main():
    if not os.path.exists(BOUNDARY_FILE):
        raise FileNotFoundError(
            f"Missing boundary file: {BOUNDARY_FILE}"
        )

    os.makedirs(WEB_DIR, exist_ok=True)

    iran = gpd.read_file(BOUNDARY_FILE)

    if iran.empty:
        raise RuntimeError("IRAN.geojson is empty.")

    if iran.crs is None:
        iran = iran.set_crs("EPSG:4326")
    else:
        iran = iran.to_crs("EPSG:4326")

    geometry = iran.geometry.union_all()

    west, south, east, north = geometry.bounds

    # Tiny margin around the actual uploaded Iran boundary.
    margin_x = (east - west) * 0.01
    margin_y = (north - south) * 0.01

    west -= margin_x
    east += margin_x
    south -= margin_y
    north += margin_y

    bbox = f"{west},{south},{east},{north}"

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": WMS_LAYER,
        "STYLES": "",
        "SRS": "EPSG:4326",
        "BBOX": bbox,
        "WIDTH": str(WIDTH),
        "HEIGHT": str(HEIGHT),
        "FORMAT": "image/png",
        "TRANSPARENT": "TRUE",
        "TIME": TARGET_DATE,
    }

    print("================================")
    print("IRAN FWI")
    print("================================")
    print("Forecast:", TARGET_DATE)
    print("Boundary:", BOUNDARY_FILE)
    print("Layer:", WMS_LAYER)
    print("BBOX:", bbox)

    response = requests.get(
        WMS_URL,
        params=params,
        timeout=TIMEOUT,
        headers={"User-Agent": "IR-FWI/1.0"},
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "image" not in content_type.lower():
        raise RuntimeError(
            "GWIS did not return an image. "
            f"Content-Type={content_type}"
        )

    image = Image.open(BytesIO(response.content)).convert("RGBA")

    # Exact clipping to the uploaded Iran boundary.
    mask = make_mask(
        geometry,
        west, south, east, north,
        image.width, image.height
    )
    image.putalpha(mask)

    image_path = os.path.join(WEB_DIR, "fwi_iran_latest.png")
    meta_path = os.path.join(WEB_DIR, "fwi_iran_latest.json")

    image.save(image_path, optimize=True)

    metadata = {
        "forecast_gregorian": TARGET_DATE,
        "generated_at_iran": NOW.strftime("%Y-%m-%d %H:%M"),
        "source": "Copernicus EFFIS / ECMWF GWIS",
        "layer": WMS_LAYER,
        "crs": "EPSG:4326",
        "region": "Iran",
        "boundary": "IRAN.geojson",
        "bbox": [west, south, east, north],
        "image": {
            "url": "fwi_iran_latest.png",
            "width": image.width,
            "height": image.height
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Saved:", image_path)
    print("Saved:", meta_path)
    print("================================")


if __name__ == "__main__":
    main()
