# save as plot_cities.py

# Requirements:
#   pip install geopandas matplotlib shapely pandas

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from shapely.geometry import Point
import pandas as pd
import os
import pathlib
import json

# === CONFIG ===
CITIES_CSV_PATH = "data/cities_with_coordinates.csv"
CANADA_MAP_PATH = "data/lcsd000a24a_e/lcsd000a24a_e.shp"  # shapefile for Canada
OUT_PNG = "cities_map.png"

MARKER = "o"
MARKER_SIZE = 30
MARKER_EDGEWIDTH = 0.5
MAP_FILL = "#ECC781"
TEXT_COLOR = "#755845"
POINT_COLOR = "#755845"
ZOOM_PADDING = 0.5  # degrees of padding around city bounds
LINK_COLOR = "#755845"
LINK_WIDTH = 1.5
LINK_STYLE = "-"

# === FUNCTIONS ===


def load_fonts_from_static():
    """Load Source Sans 3 fonts from the static folder."""
    static_dir = pathlib.Path(__file__).parent / "static"
    if not static_dir.exists():
        print(f"Warning: static folder not found at {static_dir}")
        return None

    # Load all font files from static folder
    font_files = list(static_dir.glob("*.ttf"))
    if not font_files:
        print(f"Warning: No font files found in {static_dir}")
        return None

    # Add all fonts to matplotlib's font manager
    for font_file in font_files:
        try:
            font_manager.fontManager.addfont(str(font_file))
        except Exception as e:
            print(f"Warning: Could not load font {font_file}: {e}")

    # Return the static directory path for use with FontProperties
    return static_dir


def load_canada_map(path):
    """Load Canada map shapefile and filter to Ontario and Quebec."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Map file not found: {path}")

    gdf = gpd.read_file(path)

    if gdf.crs is None:
        print("Warning: input map has no CRS. Assuming EPSG:4326 (lon/lat).")
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    # Filter to only Ontario and Quebec
    if "PRNAME" in gdf.columns:
        gdf = gdf[gdf["PRNAME"].str.contains("Ontario|Quebec", case=False, na=False)]

    return gdf


def load_cities_from_csv(csv_path, min_population=50000):
    """Load cities from CSV and convert to GeoDataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter out cities with missing coordinates
    df = df[df["Latitude"].notna() & df["Longitude"].notna()]

    # Filter to cities with population > min_population
    df = df[df["Population"] > min_population]

    # Create geometry column
    df["geometry"] = df.apply(lambda r: Point(r["Longitude"], r["Latitude"]), axis=1)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    return gdf


def calculate_bounds(cities_gdf, padding_meters=50000):
    """Calculate bounding box for cities with padding (in meters for projected CRS)."""
    bounds = cities_gdf.total_bounds  # [minx, miny, maxx, maxy]
    return {
        "minx": bounds[0] - padding_meters,
        "miny": bounds[1] - padding_meters,
        "maxx": bounds[2] + padding_meters,
        "maxy": bounds[3] + padding_meters,
    }


def plot_map(
    canada_gdf,
    cities_gdf,
    out_png=OUT_PNG,
    figsize=(12, 10),
    show=True,
    city_links=None,
    selected_cities=None,
):
    # --- Reproject to Canada Lambert Conformal Conic ---
    target_crs = "EPSG:3978"
    canada_gdf = canada_gdf.to_crs(target_crs)
    cities_gdf = cities_gdf.to_crs(target_crs)

    # --- Choose which cities to display ---
    if selected_cities is not None:
        # Use list of cities given by the caller
        cities_to_plot = cities_gdf[cities_gdf["City_Name"].isin(selected_cities)].copy()
    elif city_links:
        # only show cities that appear in the links
        linked_cities = {c for pair in city_links for c in pair}
        cities_to_plot = cities_gdf[cities_gdf["City_Name"].isin(linked_cities)].copy()
    else:
        # show all cities
        cities_to_plot = cities_gdf

    # Calculate bounds for zooming (after projection), based on the selected cities
    bounds = calculate_bounds(cities_to_plot, padding_meters=50000)

    # Load fonts from static folder
    static_dir = load_fonts_from_static()
    if static_dir:
        # Use FontProperties with the Bold font file for bold text
        bold_font_file = static_dir / "SourceSans3-Bold.ttf"
        if bold_font_file.exists():
            font_prop = FontProperties(fname=str(bold_font_file), size=7)
        else:
            # Fallback to regular font
            regular_font_file = static_dir / "SourceSans3-Regular.ttf"
            if regular_font_file.exists():
                font_prop = FontProperties(fname=str(regular_font_file), size=7)
            else:
                font_prop = FontProperties(
                    family="Source Sans 3", weight="bold", size=7
                )
    else:
        # Fallback if fonts can't be loaded
        font_prop = FontProperties(family="Source Sans 3", weight="bold", size=7)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot base map (clipped to bounds)
    canada_gdf.plot(ax=ax, color=MAP_FILL, edgecolor="#FFF8E8", linewidth=0.3, zorder=1)

    # Draw lines between city pairs if provided
    if city_links:
        # Create a dictionary mapping city names to their coordinates
        city_coords = {}
        for _, row in cities_to_plot.iterrows():
            city_coords[row["City_Name"]] = (row.geometry.x, row.geometry.y)

        # Draw lines for each city pair
        for city1, city2 in city_links:
            if city1 in city_coords and city2 in city_coords:
                x1, y1 = city_coords[city1]
                x2, y2 = city_coords[city2]
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=LINK_COLOR,
                    linewidth=LINK_WIDTH,
                    linestyle=LINK_STYLE,
                    zorder=2,
                    alpha=0.6,
                )
            else:
                missing = [c for c in [city1, city2] if c not in city_coords]
                print(
                    f"Warning: Could not find city coordinates for: {', '.join(missing)}"
                )

    # Plot city markers
    cities_to_plot.plot(
        ax=ax,
        color=POINT_COLOR,
        marker=MARKER,
        markersize=MARKER_SIZE,
        edgecolor="white",
        linewidth=MARKER_EDGEWIDTH,
        zorder=3,
    )

    # Add city labels for all cities
    for _, row in cities_to_plot.iterrows():
        x, y = row.geometry.x, row.geometry.y
        ax.annotate(
            row["City_Name"],
            xy=(x, y),
            xytext=(4, 2),
            textcoords="offset points",
            fontproperties=font_prop,
            color=TEXT_COLOR,
            ha="left",
            va="bottom",
            zorder=4,
        )

    # Set axis limits to zoom in on cities
    ax.set_xlim(bounds["minx"], bounds["maxx"])
    ax.set_ylim(bounds["miny"], bounds["maxy"])

    # Set aspect ratio correctly
    ax.set_aspect("equal", adjustable="box")

    # Remove axes and ticks
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, transparent=True, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def main():
    # City pairs to connect with lines
    city_links = [
        ("Québec", "Trois-Rivières"),
        ("Gatineau", "Trois-Rivières"),
        ("Gatineau", "Whitby"),
        ("Toronto", "Whitby"),
    ]
    with open("solution.json") as f:
        sol = json.load(f)

    selected_cities = sol["stations"]
    city_links = [tuple(e) for e in sol["edges"]]

    print("Loading cities from CSV...")
    cities_gdf = load_cities_from_csv(CITIES_CSV_PATH)
    print(f"Loaded {len(cities_gdf)} cities with coordinates")

    print("Loading Canada map...")
    canada_gdf = load_canada_map(CANADA_MAP_PATH)
    print(f"Loaded map with {len(canada_gdf)} features")

    print("Plotting map...")
    plot_map(canada_gdf,
        cities_gdf,
        city_links=city_links,
        selected_cities=selected_cities,)
    print(f"✅ Saved map to {OUT_PNG}")


if __name__ == "__main__":
    main()
