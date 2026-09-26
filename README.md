# IR-FWI

Iran-wide FWI web GIS.

The national boundary is the uploaded `IRAN.geojson`.

Workflow:
1. Read the exact Iran boundary.
2. Calculate the bounding box from that geometry.
3. Request ECMWF FWI from Copernicus GWIS.
4. Clip the returned image exactly to `IRAN.geojson`.
5. Publish the map through GitHub Pages.

Main files:
- `IRAN.geojson`
- `scripts/update_fwi.py`
- `web/index.html`
- `.github/workflows/update_fwi.yml`
