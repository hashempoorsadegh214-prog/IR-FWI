#!/usr/bin/env python3

"""
IR-FWI
Diagnostic internal-boundary overlay test.

Purpose
-------
This version tests whether the Iran boundary and the FWI raster
occupy the same geographic grid.

Outputs
-------
web/fwi_iran_latest.png
web/fwi_iran_latest.json
web/fwi_iran_diagnostic_overlay.png

Temporary files
---------------
tmp_fwi/fwi_iran_assembled.tif
tmp_fwi/fwi_iran_clipped.tif
tmp_fwi/fwi_iran_boundary_overlay.tif

The diagnostic overlay is generated in the SAME raster grid as
the assembled FWI raster.

This is important:
we do not draw the boundary as a separate Leaflet layer.
Instead, we rasterize IRAN.geojson directly onto the FWI grid.

Therefore:

If the boundary line is correctly located over the FWI image
inside this diagnostic PNG, the WMS georeferencing is internally
consistent.

If the boundary line is visibly shifted relative to the FWI
pattern, the problem is upstream of Leaflet.
"""


from pathlib import Path
import io
import json
import time
from datetime import datetime, timezone

import numpy as np
import requests

from PIL import Image, ImageDraw, ImageFont

import geopandas as gpd

import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
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

DIAGNOSTIC_FILE = (
    WEB_DIR
    / "fwi_iran_diagnostic_overlay.png"
)

TMP_DIR = BASE_DIR / "tmp_fwi"

TMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

WEB_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# WMS CONFIGURATION
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
# TILE CONFIGURATION
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
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": (
            "IR-FWI-GitHubActions/1.0"
        ),
        "Accept": (
            "image/png,image/jpeg,image/*;"
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
    print(
        message,
        flush=True
    )


# ============================================================
# LOAD IRAN BOUNDARY
# ============================================================

def load_boundary():

    if not BOUNDARY_FILE.exists():

        raise FileNotFoundError(
            f"Boundary not found: "
            f"{BOUNDARY_FILE}"
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
# BOUNDARY BBOX
# ============================================================

def get_boundary_bbox(gdf):

    minx, miny, maxx, maxy = (
        gdf.total_bounds
    )

    bbox = (
        float(minx),
        float(miny),
        float(maxx),
        float(maxy),
    )

    log("")
    log("=" * 70)
    log("IRAN BOUNDARY")
    log("=" * 70)

    log(
        f"CRS   : {gdf.crs}"
    )

    log(
        f"West  : {bbox[0]:.12f}"
    )

    log(
        f"South : {bbox[1]:.12f}"
    )

    log(
        f"East  : {bbox[2]:.12f}"
    )

    log(
        f"North : {bbox[3]:.12f}"
    )

    log(
        f"Width : "
        f"{bbox[2] - bbox[0]:.12f}"
    )

    log(
        f"Height: "
        f"{bbox[3] - bbox[1]:.12f}"
    )

    return bbox


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(
    content
):

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
            f"Invalid WMS image: {exc}"
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

    elif version == "1.3.0":

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

    else:

        raise ValueError(
            f"Unsupported WMS version: "
            f"{version}"
        )

    last_error = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1
    ):

        response = None

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
                f"{image.width}x"
                f"{image.height}"
            )

            return image

        except Exception as exc:

            last_error = exc

            log(
                f"      Failed: "
                f"{repr(exc)}"
            )

            if attempt < MAX_ATTEMPTS:

                time.sleep(
                    SLEEP_SECONDS
                )

        finally:

            if response is not None:

                try:
                    response.close()
                except Exception:
                    pass

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
        "    BBOX:"
    )

    log(
        f"      W={bbox[0]:.12f}"
    )

    log(
        f"      S={bbox[1]:.12f}"
    )

    log(
        f"      E={bbox[2]:.12f}"
    )

    log(
        f"      N={bbox[3]:.12f}"
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
                f"    Failed: "
                f"{repr(exc)}"
            )

    # --------------------------------------------------------
    # EXPANDED BBOX
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
        f"All WMS methods failed: "
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
                (
                    width,
                    height
                ),
                Image.Resampling.BILINEAR,
            )

        return image

    # --------------------------------------------------------
    # Expanded image -> exact requested grid
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
            width
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
# ASSEMBLE TILES
# ============================================================

def assemble_tiles(
    tiles
):

    full_width = (
        NX * TILE_WIDTH
    )

    full_height = (
        NY * TILE_HEIGHT
    )

    canvas = Image.new(
        "RGBA",
        (
            full_width,
            full_height,
        ),
        (0, 0, 0, 0),
    )

    for y in range(NY):

        for x in range(NX):

            image = tiles[
                (y, x)
            ]

            px = (
                x * TILE_WIDTH
            )

            py = (
                y * TILE_HEIGHT
            )

            canvas.alpha_composite(
                image,
                (
                    px,
                    py
                ),
            )

    return canvas


# ============================================================
# SAVE GEOTIFF
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
        f"RASTER: {label}"
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
            f"Transform: {src.transform}"
        )

        b = src.bounds

        log(
            f"West  : {b.left:.12f}"
        )

        log(
            f"South : {b.bottom:.12f}"
        )

        log(
            f"East  : {b.right:.12f}"
        )

        log(
            f"North : {b.top:.12f}"
        )

        # Four corners

        nw = src.transform * (
            0,
            0
        )

        ne = src.transform * (
            src.width,
            0
        )

        sw = src.transform * (
            0,
            src.height
        )

        se = src.transform * (
            src.width,
            src.height
        )

        log("")
        log(
            "Raster corners:"
        )

        log(
            f"NW = "
            f"{nw[0]:.12f}, "
            f"{nw[1]:.12f}"
        )

        log(
            f"NE = "
            f"{ne[0]:.12f}, "
            f"{ne[1]:.12f}"
        )

        log(
            f"SW = "
            f"{sw[0]:.12f}, "
            f"{sw[1]:.12f}"
        )

        log(
            f"SE = "
            f"{se[0]:.12f}, "
            f"{se[1]:.12f}"
        )


# ============================================================
# CREATE INTERNAL BOUNDARY OVERLAY
# ============================================================

def create_internal_boundary_overlay(
    raster_file,
    boundary,
    output_png,
):
    """
    Rasterize IRAN.geojson onto the EXACT grid of the FWI raster.

    This is the key diagnostic.

    The boundary is not transformed to another image.
    It is burned into the exact FWI raster grid.
    """

    log("")
    log("=" * 70)
    log("INTERNAL BOUNDARY OVERLAY")
    log("=" * 70)

    with rasterio.open(
        raster_file
    ) as src:

        raster_width = src.width

        raster_height = src.height

        raster_transform = src.transform

        raster_crs = src.crs

        log(
            f"Raster CRS: {raster_crs}"
        )

        log(
            f"Raster size: "
            f"{raster_width} x "
            f"{raster_height}"
        )

        # ----------------------------------------------------
        # Reproject boundary to raster CRS
        # ----------------------------------------------------

        boundary_raster_crs = (
            boundary.to_crs(
                raster_crs
            )
        )

        # ----------------------------------------------------
        # Rasterize boundary
        # ----------------------------------------------------

        shapes = [
            (
                geom,
                1
            )
            for geom
            in boundary_raster_crs.geometry
            if geom is not None
        ]

        boundary_mask = rasterize(
            shapes=shapes,
            out_shape=(
                raster_height,
                raster_width,
            ),
            transform=raster_transform,
            fill=0,
            all_touched=False,
            dtype="uint8",
        )

        # ----------------------------------------------------
        # Read FWI image
        # ----------------------------------------------------

        data = src.read()

        if data.shape[0] >= 4:

            rgba = np.moveaxis(
                data[:4],
                0,
                2,
            ).astype(
                np.uint8
            )

        elif data.shape[0] == 3:

            rgb = np.moveaxis(
                data[:3],
                0,
                2,
            ).astype(
                np.uint8
            )

            alpha = np.full(
                (
                    raster_height,
                    raster_width,
                    1
                ),
                255,
                dtype=np.uint8,
            )

            rgba = np.concatenate(
                (
                    rgb,
                    alpha
                ),
                axis=2,
            )

        else:

            raise RuntimeError(
                "Raster has unsupported band count."
            )

        # ----------------------------------------------------
        # Create diagnostic image
        # ----------------------------------------------------

        diagnostic = Image.fromarray(
            rgba,
            "RGBA",
        )

        draw = ImageDraw.Draw(
            diagnostic
        )

        # ----------------------------------------------------
        # Boundary pixels
        # ----------------------------------------------------

        rows, cols = np.where(
            boundary_mask == 1
        )

        log(
            f"Boundary rasterized pixels: "
            f"{len(rows)}"
        )

        if len(rows) == 0:

            raise RuntimeError(
                "Boundary rasterization produced "
                "zero pixels."
            )

        # ----------------------------------------------------
        # Draw boundary
        # ----------------------------------------------------

        # We deliberately use a bright white line.
        #
        # The diagnostic image is only for testing.
        # It is not used as the production FWI layer.

        pixel_set = set(
            zip(
                cols.tolist(),
                rows.tolist(),
            )
        )

        # Draw each boundary pixel.
        #
        # A small 3x3 neighborhood makes the line
        # visible after GitHub PNG compression/display.

        for x, y in pixel_set:

            for dy in (
                -1,
                0,
                1,
            ):

                for dx in (
                    -1,
                    0,
                    1,
                ):

                    xx = x + dx
                    yy = y + dy

                    if (
                        0 <= xx < raster_width
                        and
                        0 <= yy < raster_height
                    ):

                        diagnostic.putpixel(
                            (
                                xx,
                                yy
                            ),
                            (
                                255,
                                255,
                                255,
                                255
                            ),
                        )

        # ----------------------------------------------------
        # Draw bounding-box corners
        # ----------------------------------------------------

        bbox = boundary.total_bounds

        minx, miny, maxx, maxy = (
            bbox
        )

        corner_coordinates = [
            (
                "NW",
                minx,
                maxy,
            ),
            (
                "NE",
                maxx,
                maxy,
            ),
            (
                "SW",
                minx,
                miny,
            ),
            (
                "SE",
                maxx,
                miny,
            ),
        ]

        log("")
        log(
            "Boundary bbox corner pixels:"
        )

        for name, lon, lat in (
            corner_coordinates
        ):

            col, row = ~raster_transform * (
                lon,
                lat,
            )

            log(
                f"{name}: "
                f"lon={lon:.12f}, "
                f"lat={lat:.12f}, "
                f"pixel_col={col:.3f}, "
                f"pixel_row={row:.3f}"
            )

        # ----------------------------------------------------
        # Draw small corner markers
        # ----------------------------------------------------

        for name, lon, lat in (
            corner_coordinates
        ):

            col, row = (
                ~raster_transform
                * (
                    lon,
                    lat,
                )
            )

            x = int(
                round(col)
            )

            y = int(
                round(row)
            )

            radius = 8

            draw.ellipse(
                (
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                ),
                outline=(
                    255,
                    255,
                    255,
                    255
                ),
                width=3,
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        diagnostic.save(
            output_png,
            format="PNG",
            optimize=True,
        )

        log("")
        log(
            f"Diagnostic overlay saved:"
        )

        log(
            f"  {output_png}"
        )

        return {
            "width": raster_width,
            "height": raster_height,
            "boundary_pixels": int(
                len(rows)
            ),
        }


# ============================================================
# CLIP TO IRAN
# ============================================================

def clip_to_iran(
    source_file,
    boundary,
    output_file,
):

    log("")
    log("=" * 70)
    log("CLIPPING TO IRAN")
    log("=" * 70)

    with rasterio.open(
        source_file
    ) as src:

        boundary_raster_crs = (
            boundary.to_crs(
                src.crs
            )
        )

        geometries = [
            geom.__geo_interface__
            for geom
            in boundary_raster_crs.geometry
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

    log(
        f"Clipped raster:"
    )

    log(
        f"  {output_file}"
    )


# ============================================================
# SAVE FINAL PNG
# ============================================================

def save_final_png(
    clipped_file
):

    with rasterio.open(
        clipped_file
    ) as src:

        data = src.read()

        if data.shape[0] < 4:

            raise RuntimeError(
                "Expected RGBA raster."
            )

        rgba = np.moveaxis(
            data[:4],
            0,
            2,
        ).astype(
            np.uint8
        )

        image = Image.fromarray(
            rgba,
            "RGBA"
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
            "west": float(
                bounds[0]
            ),
            "south": float(
                bounds[1]
            ),
            "east": float(
                bounds[2]
            ),
            "north": float(
                bounds[3]
            ),
            "width": int(
                src.width
            ),
            "height": int(
                src.height
            ),
            "crs": str(
                src.crs
            ),
        }


# ============================================================
# MAIN
# ============================================================

def main():

    log("")
    log("=" * 70)
    log("IR-FWI INTERNAL OVERLAY DIAGNOSTIC")
    log("=" * 70)
    log("")

    # --------------------------------------------------------
    # Load boundary
    # --------------------------------------------------------

    boundary = load_boundary()

    boundary_bbox = (
        get_boundary_bbox(
            boundary
        )
    )

    west, south, east, north = (
        boundary_bbox
    )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    log("")
    log(
        f"FWI date: "
        f"{FORECAST_DATE}"
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

    log(
        f"Final width: "
        f"{NX * TILE_WIDTH}"
    )

    log(
        f"Final height: "
        f"{NY * TILE_HEIGHT}"
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

            requested_bbox = (
                tile_west,
                tile_south,
                tile_east,
                tile_north,
            )

            tile_name = (
                f"Tile "
                f"row={y + 1}/{NY}, "
                f"col={x + 1}/{NX}"
            )

            image, actual_bbox = (
                fetch_tile(
                    bbox=requested_bbox,
                    width=TILE_WIDTH,
                    height=TILE_HEIGHT,
                    tile_name=tile_name,
                )
            )

            image = normalize_tile(
                image=image,
                actual_bbox=actual_bbox,
                requested_bbox=requested_bbox,
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
    log("ASSEMBLING")
    log("=" * 70)

    assembled = assemble_tiles(
        tiles
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

    raster_diagnostic(
        assembled_tif,
        "ASSEMBLED FWI"
    )

    # --------------------------------------------------------
    # INTERNAL OVERLAY
    # --------------------------------------------------------

    overlay_info = (
        create_internal_boundary_overlay(
            raster_file=assembled_tif,
            boundary=boundary,
            output_png=DIAGNOSTIC_FILE,
        )
    )

    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    clipped_tif = (
        TMP_DIR
        / "fwi_iran_clipped.tif"
    )

    clip_to_iran(
        source_file=assembled_tif,
        boundary=boundary,
        output_file=clipped_tif,
    )

    raster_diagnostic(
        clipped_tif,
        "CLIPPED FWI"
    )

    # --------------------------------------------------------
    # Final PNG
    # --------------------------------------------------------

    image_info = (
        save_final_png(
            clipped_tif
        )
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
            "width": image_info[
                "width"
            ],
            "height": image_info[
                "height"
            ],
        },

        "diagnostic_overlay": {
            "file": (
                DIAGNOSTIC_FILE.name
            ),
            "description": (
                "IRAN.geojson rasterized "
                "directly onto the assembled "
                "FWI raster grid"
            ),
            "width": overlay_info[
                "width"
            ],
            "height": overlay_info[
                "height"
            ],
            "boundary_pixels": (
                overlay_info[
                    "boundary_pixels"
                ]
            ),
        },

        "wms_bbox": {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        },

        "image_bounds": {
            "west": image_info[
                "west"
            ],
            "south": image_info[
                "south"
            ],
            "east": image_info[
                "east"
            ],
            "north": image_info[
                "north"
            ],
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
                "WMS tiles -> EPSG:4326 "
                "GeoTIFF -> internal "
                "IRAN.geojson raster overlay "
                "-> exact clip"
            ),
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
    # Final checks
    # --------------------------------------------------------

    if not PNG_FILE.exists():

        raise RuntimeError(
            "Final PNG missing."
        )

    if not JSON_FILE.exists():

        raise RuntimeError(
            "Final JSON missing."
        )

    if not DIAGNOSTIC_FILE.exists():

        raise RuntimeError(
            "Diagnostic overlay missing."
        )

    log("")
    log("=" * 70)
    log("DIAGNOSTIC BUILD SUCCESSFUL")
    log("=" * 70)

    log("")
    log(
        f"FWI PNG:"
    )

    log(
        f"  {PNG_FILE}"
    )

    log(
        f"Metadata:"
    )

    log(
        f"  {JSON_FILE}"
    )

    log(
        f"Internal overlay:"
    )

    log(
        f"  {DIAGNOSTIC_FILE}"
    )

    log("")
    log(
        "The diagnostic overlay is the key output."
    )

    log(
        "It contains the FWI raster and the "
        "IRAN.geojson boundary on the SAME pixel grid."
    )

    log("")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
