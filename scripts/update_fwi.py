#!/usr/bin/env python3

"""
IR-FWI
Diagnostic build of Iran ECMWF FWI from Copernicus GWIS WMS.

This version keeps the existing WMS -> GeoTIFF -> Iran clip workflow,
but adds detailed georeferencing diagnostics to GitHub Actions logs.

Outputs:
    web/fwi_iran_latest.png
    web/fwi_iran_latest.json

Temporary:
    tmp_fwi/fwi_iran_assembled.tif
    tmp_fwi/fwi_iran_clipped.tif
"""

from pathlib import Path
import io
import json
import time
from datetime import datetime, timezone

import numpy as np
import requests

from PIL import Image

import geopandas as gpd

import rasterio
from rasterio.mask import mask
from rasterio.transform import from_bounds, array_bounds
from rasterio.warp import reproject, Resampling


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BOUNDARY_FILE = BASE_DIR / "IRAN.geojson"

WEB_DIR = BASE_DIR / "web"

PNG_FILE = WEB_DIR / "fwi_iran_latest.png"
JSON_FILE = WEB_DIR / "fwi_iran_latest.json"

TMP_DIR = BASE_DIR / "tmp_fwi"

TMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# WMS
# ============================================================

WMS_URL = (
    "https://maps.effis.emergency.copernicus.eu/gwis"
)

LAYER_NAME = "ecmwf.fwi"

TARGET_CRS = "EPSG:4326"

FORECAST_DATE = datetime.now(
    timezone.utc
).date()


# ============================================================
# TILE CONFIG
# ============================================================

NX = 3
NY = 3

TILE_WIDTH = 800
TILE_HEIGHT = 611

REQUEST_TIMEOUT = 180

MAX_ATTEMPTS = 3

SLEEP_SECONDS = 3

EXPAND_DEGREES = 0.15


# ============================================================
# HTTP
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": (
            "IR-FWI-GitHubActions/1.0"
        ),
        "Accept": (
            "image/png,image/jpeg,image/*,"
            "q=0.8,*/*;q=0.5"
        ),
        "Accept-Encoding": "identity",
        "Connection": "close",
    }
)


# ============================================================
# LOG
# ============================================================

def log(message=""):
    print(message, flush=True)


# ============================================================
# BOUNDARY
# ============================================================

def load_boundary():

    if not BOUNDARY_FILE.exists():

        raise FileNotFoundError(
            f"Boundary not found: {BOUNDARY_FILE}"
        )

    gdf = gpd.read_file(
        BOUNDARY_FILE
    )

    if gdf.empty:

        raise RuntimeError(
            "IRAN.geojson is empty."
        )

    if gdf.crs is None:

        raise RuntimeError(
            "IRAN.geojson has no CRS."
        )

    gdf = gdf.to_crs(
        TARGET_CRS
    )

    return gdf


# ============================================================
# BOUNDARY DIAGNOSTIC
# ============================================================

def boundary_diagnostic(gdf):

    minx, miny, maxx, maxy = (
        gdf.total_bounds
    )

    log("")
    log("=" * 70)
    log("BOUNDARY DIAGNOSTIC")
    log("=" * 70)

    log(
        f"CRS: {gdf.crs}"
    )

    log(
        f"West : {minx:.12f}"
    )

    log(
        f"South: {miny:.12f}"
    )

    log(
        f"East : {maxx:.12f}"
    )

    log(
        f"North: {maxy:.12f}"
    )

    log(
        f"Width : {maxx - minx:.12f} degrees"
    )

    log(
        f"Height: {maxy - miny:.12f} degrees"
    )

    return (
        float(minx),
        float(miny),
        float(maxx),
        float(maxy),
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(content):

    if not content:

        raise RuntimeError(
            "Empty WMS response."
        )

    if len(content) < 100:

        raise RuntimeError(
            f"WMS response too small: "
            f"{len(content)} bytes"
        )

    try:

        image = Image.open(
            io.BytesIO(content)
        )

        image.load()

        return image

    except Exception as exc:

        raise RuntimeError(
            f"Invalid image response: {exc}"
        )


# ============================================================
# WMS REQUEST
# ============================================================

def request_wms(
    bbox,
    width,
    height,
    version="1.1.1",
    image_format="image/png",
    transparent=True,
):

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
                "TRUE"
                if transparent
                else "FALSE"
            ),
            "SRS": "EPSG:4326",
            "BBOX": (
                f"{west},{south},"
                f"{east},{north}"
            ),
            "WIDTH": str(width),
            "HEIGHT": str(height),
            "TIME": (
                FORECAST_DATE.isoformat()
            ),
        }

    else:

        # WMS 1.3.0 / EPSG:4326 axis order:
        # latitude, longitude

        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": LAYER_NAME,
            "STYLES": "",
            "FORMAT": image_format,
            "TRANSPARENT": (
                "TRUE"
                if transparent
                else "FALSE"
            ),
            "CRS": "EPSG:4326",
            "BBOX": (
                f"{south},{west},"
                f"{north},{east}"
            ),
            "WIDTH": str(width),
            "HEIGHT": str(height),
            "TIME": (
                FORECAST_DATE.isoformat()
            ),
        }

    last_error = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1
    ):

        try:

            log(
                f"      Attempt "
                f"{attempt}/{MAX_ATTEMPTS} "
                f"{version} "
                f"{image_format}"
            )

            response = SESSION.get(
                WMS_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
                stream=True,
            )

            response.raise_for_status()

            content = response.content

            image = validate_image(
                content
            )

            log(
                f"      Success: "
                f"{len(content)} bytes "
                f"{image.width}x{image.height}"
            )

            try:
                response.close()
            except Exception:
                pass

            return image

        except Exception as exc:

            last_error = exc

            log(
                f"      Failed: {repr(exc)}"
            )

            try:
                response.close()
            except Exception:
                pass

            if attempt < MAX_ATTEMPTS:

                time.sleep(
                    SLEEP_SECONDS
                )

    raise RuntimeError(
        f"WMS request failed: "
        f"{last_error}"
    )


# ============================================================
# FETCH TILE
# ============================================================

def fetch_tile(
    bbox,
    width,
    height,
    tile_name,
):

    log("")
    log(
        f"  {tile_name}"
    )

    log(
        "    Requested BBOX:"
    )

    log(
        f"      west  = {bbox[0]:.12f}"
    )

    log(
        f"      south = {bbox[1]:.12f}"
    )

    log(
        f"      east  = {bbox[2]:.12f}"
    )

    log(
        f"      north = {bbox[3]:.12f}"
    )

    methods = [

        (
            "WMS 1.1.1 PNG",
            "1.1.1",
            "image/png",
            True,
        ),

        (
            "WMS 1.1.1 JPEG",
            "1.1.1",
            "image/jpeg",
            False,
        ),

        (
            "WMS 1.3.0 PNG",
            "1.3.0",
            "image/png",
            True,
        ),
    ]

    for (
        method_name,
        version,
        image_format,
        transparent,
    ) in methods:

        log(
            f"    Trying {method_name}"
        )

        try:

            image = request_wms(
                bbox=bbox,
                width=width,
                height=height,
                version=version,
                image_format=image_format,
                transparent=transparent,
            )

            return (
                image.convert("RGBA"),
                bbox,
            )

        except Exception as exc:

            log(
                f"    {method_name} failed:"
            )

            log(
                f"      {repr(exc)}"
            )

    # --------------------------------------------------------
    # Expanded BBOX fallback
    # --------------------------------------------------------

    west, south, east, north = bbox

    expanded_bbox = (
        west - EXPAND_DEGREES,
        south - EXPAND_DEGREES,
        east + EXPAND_DEGREES,
        north + EXPAND_DEGREES,
    )

    log("")
    log(
        "    Trying expanded BBOX..."
    )

    try:

        image = request_wms(
            bbox=expanded_bbox,
            width=width + 40,
            height=height + 40,
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
            f"    Expanded BBOX failed: "
            f"{repr(exc)}"
        )

    raise RuntimeError(
        f"All WMS methods failed for "
        f"{tile_name}"
    )


# ============================================================
# NORMALIZE TILE
# ============================================================

def normalize_tile(
    image,
    actual_bbox,
    requested_bbox,
    width,
    height,
):

    aw, ass, ae, an = actual_bbox

    rw, rs, re, rn = requested_bbox

    exact = (
        abs(aw - rw) < 1e-12
        and abs(ass - rs) < 1e-12
        and abs(ae - re) < 1e-12
        and abs(an - rn) < 1e-12
    )

    if exact:

        if image.size != (
            width,
            height
        ):

            image = image.resize(
                (width, height),
                Image.Resampling.BILINEAR,
            )

        return image

    # --------------------------------------------------------
    # Expanded source -> exact requested grid
    # --------------------------------------------------------

    source = np.asarray(
        image.convert("RGBA"),
        dtype=np.uint8,
    )

    src_height = source.shape[0]
    src_width = source.shape[1]

    src_transform = from_bounds(
        aw,
        ass,
        ae,
        an,
        src_width,
        src_height,
    )

    dst_transform = from_bounds(
        rw,
        rs,
        re,
        rn,
        width,
        height,
    )

    destination = np.zeros(
        (
            4,
            height,
            width,
        ),
        dtype=np.uint8,
    )

    for band in range(4):

        reproject(
            source=source[:, :, band],
            destination=destination[band],
            src_transform=src_transform,
            src_crs=TARGET_CRS,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
        )

    result = np.moveaxis(
        destination,
        0,
        2,
    )

    return Image.fromarray(
        result,
        "RGBA",
    )


# ============================================================
# ASSEMBLE
# ============================================================

def assemble_tiles(
    tiles,
    nx,
    ny,
    tile_width,
    tile_height,
):

    full_width = (
        nx * tile_width
    )

    full_height = (
        ny * tile_height
    )

    canvas = Image.new(
        "RGBA",
        (
            full_width,
            full_height,
        ),
        (0, 0, 0, 0),
    )

    for y in range(ny):

        for x in range(nx):

            image = tiles[
                (y, x)
            ]

            px = (
                x * tile_width
            )

            py = (
                y * tile_height
            )

            canvas.alpha_composite(
                image,
                (px, py),
            )

    return canvas


# ============================================================
# CREATE GEOTIFF
# ============================================================

def save_rgba_geotiff(
    image,
    bbox,
    output_file,
):

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
                rgba[:, :, band],
                band + 1,
            )


# ============================================================
# RASTER DIAGNOSTIC
# ============================================================

def raster_diagnostic(
    tif_file,
    label,
):

    log("")
    log("=" * 70)
    log(
        f"RASTER DIAGNOSTIC: {label}"
    )
    log("=" * 70)

    with rasterio.open(
        tif_file
    ) as src:

        log(
            f"CRS: {src.crs}"
        )

        log(
            f"Width : {src.width}"
        )

        log(
            f"Height: {src.height}"
        )

        log(
            f"Transform:"
        )

        log(
            f"  {src.transform}"
        )

        bounds = src.bounds

        log(
            "Bounds:"
        )

        log(
            f"  west  = {bounds.left:.12f}"
        )

        log(
            f"  south = {bounds.bottom:.12f}"
        )

        log(
            f"  east  = {bounds.right:.12f}"
        )

        log(
            f"  north = {bounds.top:.12f}"
        )

        calculated = array_bounds(
            src.height,
            src.width,
            src.transform,
        )

        log(
            "array_bounds:"
        )

        log(
            f"  west  = {calculated[0]:.12f}"
        )

        log(
            f"  south = {calculated[1]:.12f}"
        )

        log(
            f"  east  = {calculated[2]:.12f}"
        )

        log(
            f"  north = {calculated[3]:.12f}"
        )

        # ----------------------------------------------------
        # Four raster corner coordinates
        # ----------------------------------------------------

        log("")
        log(
            "Four raster corners:"
        )

        nw = src.transform * (
            0,
            0,
        )

        ne = src.transform * (
            src.width,
            0,
        )

        sw = src.transform * (
            0,
            src.height,
        )

        se = src.transform * (
            src.width,
            src.height,
        )

        log(
            f"  NW = "
            f"lon {nw[0]:.12f}, "
            f"lat {nw[1]:.12f}"
        )

        log(
            f"  NE = "
            f"lon {ne[0]:.12f}, "
            f"lat {ne[1]:.12f}"
        )

        log(
            f"  SW = "
            f"lon {sw[0]:.12f}, "
            f"lat {sw[1]:.12f}"
        )

        log(
            f"  SE = "
            f"lon {se[0]:.12f}, "
            f"lat {se[1]:.12f}"
        )


# ============================================================
# CLIP
# ============================================================

def clip_to_boundary(
    source_file,
    boundary,
    output_file,
):

    geometries = [
        geom.__geo_interface__
        for geom in boundary.geometry
        if geom is not None
    ]

    if not geometries:

        raise RuntimeError(
            "No valid geometry found."
        )

    with rasterio.open(
        source_file
    ) as src:

        boundary_reprojected = (
            boundary.to_crs(src.crs)
        )

        geometries = [
            geom.__geo_interface__
            for geom
            in boundary_reprojected.geometry
            if geom is not None
        ]

        clipped, transform = mask(
            src,
            geometries,
            crop=True,
            filled=True,
            nodata=0,
        )

        profile = src.profile.copy()

        profile.update(
            {
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": transform,
                "compress": "deflate",
            }
        )

        with rasterio.open(
            output_file,
            "w",
            **profile,
        ) as dst:

            dst.write(
                clipped
            )


# ============================================================
# PNG
# ============================================================

def save_png(
    clipped_tif
):

    with rasterio.open(
        clipped_tif
    ) as src:

        data = src.read()

        if data.shape[0] < 4:

            raise RuntimeError(
                "Clipped raster is not RGBA."
            )

        rgba = np.moveaxis(
            data[:4],
            0,
            2,
        )

        image = Image.fromarray(
            rgba.astype(np.uint8),
            "RGBA",
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

        return {
            "west": float(bounds[0]),
            "south": float(bounds[1]),
            "east": float(bounds[2]),
            "north": float(bounds[3]),
            "width": int(src.width),
            "height": int(src.height),
            "crs": str(src.crs),
        }


# ============================================================
# COMPARE
# ============================================================

def compare_bounds(
    boundary_bbox,
    raster_bbox,
):

    bw, bs, be, bn = boundary_bbox

    rw, rs, re, rn = raster_bbox

    log("")
    log("=" * 70)
    log("BOUNDARY vs RASTER")
    log("=" * 70)

    log(
        "BOUNDARY:"
    )

    log(
        f"  W={bw:.12f}"
    )

    log(
        f"  S={bs:.12f}"
    )

    log(
        f"  E={be:.12f}"
    )

    log(
        f"  N={bn:.12f}"
    )

    log(
        "RASTER:"
    )

    log(
        f"  W={rw:.12f}"
    )

    log(
        f"  S={rs:.12f}"
    )

    log(
        f"  E={re:.12f}"
    )

    log(
        f"  N={rn:.12f}"
    )

    log(
        "DIFFERENCE:"
    )

    log(
        f"  west  = {rw - bw:.12f}"
    )

    log(
        f"  south = {rs - bs:.12f}"
    )

    log(
        f"  east  = {re - be:.12f}"
    )

    log(
        f"  north = {rn - bn:.12f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log("")
    log("=" * 70)
    log("IR-FWI DIAGNOSTIC BUILD")
    log("=" * 70)

    WEB_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Boundary
    # --------------------------------------------------------

    boundary = load_boundary()

    boundary_bbox = boundary_diagnostic(
        boundary
    )

    west, south, east, north = (
        boundary_bbox
    )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    log("")
    log(
        f"FWI date: {FORECAST_DATE}"
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    log("")
    log("=" * 70)
    log("WMS GRID")
    log("=" * 70)

    log(
        f"Columns: {NX}"
    )

    log(
        f"Rows: {NY}"
    )

    log(
        f"Tile width: {TILE_WIDTH}"
    )

    log(
        f"Tile height: {TILE_HEIGHT}"
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    tiles = {}

    for y in range(NY):

        for x in range(NX):

            tile_west = (
                west
                + (
                    east - west
                ) * x / NX
            )

            tile_east = (
                west
                + (
                    east - west
                ) * (x + 1) / NX
            )

            tile_north = (
                north
                - (
                    north - south
                ) * y / NY
            )

            tile_south = (
                north
                - (
                    north - south
                ) * (y + 1) / NY
            )

            bbox = (
                tile_west,
                tile_south,
                tile_east,
                tile_north,
            )

            name = (
                f"Tile row={y + 1}/{NY}, "
                f"col={x + 1}/{NX}"
            )

            image, actual_bbox = fetch_tile(
                bbox=bbox,
                width=TILE_WIDTH,
                height=TILE_HEIGHT,
                tile_name=name,
            )

            image = normalize_tile(
                image=image,
                actual_bbox=actual_bbox,
                requested_bbox=bbox,
                width=TILE_WIDTH,
                height=TILE_HEIGHT,
            )

            tiles[
                (y, x)
            ] = image

    # --------------------------------------------------------
    # Assemble
    # --------------------------------------------------------

    log("")
    log("=" * 70)
    log("ASSEMBLY")
    log("=" * 70)

    assembled = assemble_tiles(
        tiles=tiles,
        nx=NX,
        ny=NY,
        tile_width=TILE_WIDTH,
        tile_height=TILE_HEIGHT,
    )

    assembled_tif = (
        TMP_DIR
        / "fwi_iran_assembled.tif"
    )

    save_rgba_geotiff(
        image=assembled,
        bbox=boundary_bbox,
        output_file=assembled_tif,
    )

    log(
        f"Created: {assembled_tif}"
    )

    raster_diagnostic(
        assembled_tif,
        "ASSEMBLED BEFORE CLIP"
    )

    # --------------------------------------------------------
    # Clip
    # --------------------------------------------------------

    clipped_tif = (
        TMP_DIR
        / "fwi_iran_clipped.tif"
    )

    log("")
    log("=" * 70)
    log("EXACT IRAN CLIP")
    log("=" * 70)

    clip_to_boundary(
        source_file=assembled_tif,
        boundary=boundary,
        output_file=clipped_tif,
    )

    log(
        f"Created: {clipped_tif}"
    )

    raster_diagnostic(
        clipped_tif,
        "AFTER IRAN GEOJSON CLIP"
    )

    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    image_info = save_png(
        clipped_tif
    )

    log("")
    log("=" * 70)
    log("FINAL PNG")
    log("=" * 70)

    log(
        f"File: {PNG_FILE}"
    )

    log(
        f"Width : {image_info['width']}"
    )

    log(
        f"Height: {image_info['height']}"
    )

    log(
        f"West  : {image_info['west']:.12f}"
    )

    log(
        f"South : {image_info['south']:.12f}"
    )

    log(
        f"East  : {image_info['east']:.12f}"
    )

    log(
        f"North : {image_info['north']:.12f}"
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    compare_bounds(
        boundary_bbox=boundary_bbox,
        raster_bbox=(
            image_info["west"],
            image_info["south"],
            image_info["east"],
            image_info["north"],
        ),
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "source": (
            "Copernicus GWIS / ECMWF"
        ),

        "layer": LAYER_NAME,

        "date": (
            FORECAST_DATE.isoformat()
        ),

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
                "GWIS WMS -> EPSG:4326 "
                "GeoTIFF -> exact "
                "IRAN.geojson clip"
            ),
            "clip": "IRAN.geojson",
            "diagnostic": True,
        },

        "tile_grid": {
            "columns": NX,
            "rows": NY,
            "tile_width": TILE_WIDTH,
            "tile_height": TILE_HEIGHT,
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

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not PNG_FILE.exists():

        raise RuntimeError(
            "PNG was not created."
        )

    if not JSON_FILE.exists():

        raise RuntimeError(
            "JSON was not created."
        )

    log("")
    log("=" * 70)
    log("DIAGNOSTIC BUILD SUCCESSFUL")
    log("=" * 70)

    log(
        f"PNG : {PNG_FILE}"
    )

    log(
        f"JSON: {JSON_FILE}"
    )

    log("")
    log(
        "IMPORTANT:"
    )

    log(
        "The next step is to inspect the "
        "BOUNDARY vs RASTER diagnostic "
        "and the four raster corners."
    )

    log("")


if __name__ == "__main__":
    main()
