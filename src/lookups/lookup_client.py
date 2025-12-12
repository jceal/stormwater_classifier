# Looks up lot and MS4 information for a parsed project location

import difflib
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path


BORO_TO_CODE = {
    "Manhattan": "MN",
    "Bronx": "BX",
    "Brooklyn": "BK",
    "Queens": "QN",
    "Staten Island": "SI",
}


class PlutoLookupClient:
    """
    Performs address lookup using local MapPLUTO and MS4 data.
    """

    def __init__(self, data_dir: Path):
        # Load MapPLUTO data
        pluto_path = data_dir / "ms4/MapPLUTO25v3.gdb"

        self.pluto = gpd.read_file(pluto_path, layer="MapPLUTO_25v3_clipped")
        # Make sure coordinates are in latitude/longitude
        if self.pluto.crs is None or self.pluto.crs.to_epsg() != 4326:
            self.pluto = self.pluto.to_crs("EPSG:4326")

        # Build address lists for fuzzy matching
        self.address_list_by_boro = {
            b: self.pluto[self.pluto["Borough"] == b]["Address"].astype(str).tolist()
            for b in self.pluto["Borough"].unique()
        }

        # Load MS4 drainage area polygons
        ms4_path = data_dir / "ms4/MS4OpenData.gdb"
        self.ms4 = gpd.read_file(ms4_path, layer="MS4DRAINAGEAREAS")  # Already correct layer
        if self.ms4.crs is None or self.ms4.crs.to_epsg() != 4326:
            self.ms4 = self.ms4.to_crs("EPSG:4326")

    # Main lookup method
    def lookup_location_features(self, address_obj):
        """
        address_obj comes from parse_description()
        Should contain normalized .address and .borough fields.
        """

        if not address_obj or not address_obj.street_address or not address_obj.borough:
            return {
                "in_ms4_area": False,
                "pollutants_of_concern": [],
                "lot_area_sf": None,
            }

        street_address = address_obj.street_address
        borough_name = address_obj.borough

        bcode = BORO_TO_CODE.get(borough_name)
        if bcode is None:
            return {
                "in_ms4_area": False,
                "pollutants_of_concern": [],
                "lot_area_sf": None,
            }

        # Try exact address match first
        match = self._exact_address_match(street_address, bcode)

        # Fall back to fuzzy matching if needed
        if match is None:
            match = self._fuzzy_address_match(street_address, bcode)

        if match is None:
            # Could not resolve address
            return {
                "in_ms4_area": False,
                "pollutants_of_concern": [],
                "lot_area_sf": None,
            }

        # Use the parcel centroid for checking MS4
        centroid = match.geometry.centroid
        pt = Point(centroid.x, centroid.y)

        # Check whether the parcel is inside an MS4 area
        in_ms4 = self.ms4.contains(pt).any()

        pollutants = self._get_ms4_attributes(pt)

        lot_area = match.get("LotArea", None)
        full_site = None
        if getattr(address_obj, "disturbed_area_sf", None) == "FULL_SITE":
            full_site = lot_area

        return {
            "in_ms4_area": in_ms4,
            "pollutants_of_concern": pollutants,
            "lot_area_sf": lot_area,
            "full_site_disturbed_sf": full_site,
        }

    def _exact_address_match(self, street_addr: str, bcode: str):
        df = self.pluto[
            (self.pluto["Borough"] == bcode)
            & (self.pluto["Address"].astype(str).str.contains(street_addr, case=False, na=False))
        ]
        return df.iloc[0] if len(df) > 0 else None

    def _fuzzy_address_match(self, street_addr: str, bcode: str):
        choices = self.address_list_by_boro.get(bcode, [])
        best = difflib.get_close_matches(street_addr, choices, n=1, cutoff=0.6)
        if not best:
            return None

        match_addr = best[0]
        df = self.pluto[
            (self.pluto["Borough"] == bcode)
            & (self.pluto["Address"] == match_addr)
        ]
        return df.iloc[0] if len(df) > 0 else None


    def _get_ms4_attributes(self, pt: Point):
        matches = self.ms4[self.ms4.contains(pt)]
        if matches.empty:
            return []

        row = matches.iloc[0]

        pollutants = []
        if str(row.get("FLOATABLES", "")).upper() == "YES":
            pollutants.append("floatables")
        if str(row.get("PATHOGENS", "")).upper() == "YES":
            pollutants.append("pathogens")
        if str(row.get("NITROGEN", "")).upper() == "YES":
            pollutants.append("nitrogen")
        if str(row.get("PHOSPHORUS", "")).upper() == "YES":
            pollutants.append("phosphorus")

        return pollutants