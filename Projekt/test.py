import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS
from pykml import parser

# Load cc_data
cc_data = pd.read_csv(r'Projekt\data\MC2\cc_data.csv', encoding='cp1252')

# Load the .prj file (projection information)
with open(r'Projekt\data\MC2\Geospatial\Abila.prj', 'r') as prj_file:
    prj_content = prj_file.read()
    crs = CRS.from_wkt(prj_content)
    print("CRS:", crs)

# Load the .kml file (geospatial data)
kml_file = r'Projekt\data\MC2\Geospatial\Abila.kml'


# Read the KML file using pykml
with open(kml_file, 'r', encoding="cp1252") as f:
   root = parser.parse(f).getroot()
   
places = []
for place in root.Document.Folder.Placemark:
    data = {item.get("name"): item.text for item in
            place.ExtendedData.SchemaData.SimpleData}
    # Check if the place has a Polygon element
    if hasattr(place, 'Polygon') and hasattr(place.Polygon, 'outerBoundaryIs'):
        coords = place.Polygon.outerBoundaryIs.LinearRing.coordinates.text.strip()
        data["Coordinates"] = coords
        places.append(data)
    else:
        print(f"Skipping Placemark without Polygon: {place.name if hasattr(place, 'name') else 'Unnamed'}")

df = pd.DataFrame(places)
