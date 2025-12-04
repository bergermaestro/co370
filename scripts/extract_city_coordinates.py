import pandas as pd
import geopandas as gpd
import re
from pathlib import Path

# Paths
csv_path = Path("data/1710015501-eng.csv")
shapefile_path = Path("data/lcsd000a24a_e/lcsd000a24a_e.shp")
output_path = Path("data/cities_with_coordinates.csv")

print("Loading CSV file...")
# Read the CSV file - need to find where footnotes start
# Read the file and extract only the data portion
with open(csv_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
    # Find where "Footnotes:" appears
    data_end = len(lines)
    for i, line in enumerate(lines):
        if 'Footnotes:' in line:
            data_end = i
            break
    
    # Extract only the data portion (skip first 8 lines of metadata, keep header and data)
    data_lines = lines[8:data_end]  # Skip metadata (lines 0-7), keep header and data
    
    # Write to a temporary string and read as CSV
    from io import StringIO
    data_text = ''.join(data_lines)
    df_csv = pd.read_csv(StringIO(data_text), header=0)
    
    # Remove the sub-header row (where Geography is NaN and 2024 is "Persons")
    df_csv = df_csv[df_csv["Geography"].notna() & (df_csv["2024"] != "Persons")]

# Clean up the data - remove rows that are totals/aggregates
# Remove rows that are just province names or aggregates
df_csv = df_csv[
    ~df_csv["Geography"].str.contains("Census subdivisions with", case=False, na=False)
]
# Remove rows that are just province names (exact match: "Quebec" or "Ontario")
df_csv = df_csv[~(df_csv["Geography"].str.strip() == "Quebec")]
df_csv = df_csv[~(df_csv["Geography"].str.strip() == "Ontario")]

# Clean population column - remove commas and convert to numeric
df_csv["2024"] = df_csv["2024"].astype(str).str.replace(",", "", regex=False)
df_csv["2024"] = pd.to_numeric(df_csv["2024"], errors="coerce")
df_csv = df_csv[df_csv["2024"].notna()]

# Rename columns
df_csv.columns = ["Geography", "Population"]
df_csv = df_csv.reset_index(drop=True)

print(f"Found {len(df_csv)} cities in CSV")

# Extract city name and province from "City (Type), Province" format
def parse_geography(geo_str):
    """Parse geography string to extract city name and province"""
    if pd.isna(geo_str):
        return None, None
    
    geo_str = str(geo_str).strip()
    
    # Pattern: "City Name (Type), Province"
    # Extract city name (everything before the first comma)
    parts = geo_str.split(", ")
    if len(parts) < 2:
        return None, None
    
    province = parts[-1]
    city_part = ", ".join(parts[:-1])
    
    # Remove type in parentheses like "(V)", "(CY)", etc.
    city_name = re.sub(r'\s*\([^)]+\)\s*$', '', city_part).strip()
    
    return city_name, province

# Apply parsing
parsed = df_csv["Geography"].apply(
    lambda x: pd.Series(parse_geography(x), index=["City_Name", "Province"])
)
df_csv = pd.concat([df_csv, parsed], axis=1)

df_csv = df_csv[df_csv["City_Name"].notna()]
print(f"Parsed {len(df_csv)} valid cities")

print("\nLoading shapefile...")
# Load shapefile
gdf = gpd.read_file(shapefile_path)

print(f"Shapefile has {len(gdf)} features")
print(f"Shapefile columns: {list(gdf.columns)}")

# Check what the name column might be called in the shapefile
# Common names: CSDNAME, CSDUID, NAME, CSDTYPE, etc.
print("\nFirst few rows of shapefile:")
print(gdf.head())

# Use CSDNAME for city names (known from inspection)
name_col = "CSDNAME"
prov_col = "PRNAME"

print(f"\nUsing column '{name_col}' for city names")
print(f"Using column '{prov_col}' for province names")

# Helper function to check if province matches (handles bilingual names like "Quebec / Québec")
def province_matches(shapefile_province, csv_province):
    """Check if provinces match, handling bilingual format"""
    if pd.isna(shapefile_province) or not csv_province:
        return False
    shapefile_prov_str = str(shapefile_province).lower()
    csv_prov_str = str(csv_province).lower()
    # Check if CSV province name is in shapefile province (handles "Quebec / Québec" format)
    return csv_prov_str in shapefile_prov_str

# Clean city names in shapefile for matching
gdf["City_Name_Clean"] = gdf[name_col].astype(str).str.strip()

# Also check province column
if prov_col not in gdf.columns:
    # Fallback to finding province column
    for col in ["PRNAME", "PROV", "PROVINCE", "PRNAME_EN"]:
        if col in gdf.columns:
            prov_col = col
            break

# Convert geometry to WGS84 (EPSG:4326) if needed for lat/lon
if gdf.crs and gdf.crs != "EPSG:4326":
    print(f"\nConverting from {gdf.crs} to WGS84 (EPSG:4326)...")
    gdf = gdf.to_crs("EPSG:4326")

# Calculate centroids (latitude and longitude)
print("\nCalculating centroids...")
gdf["geometry_centroid"] = gdf.geometry.centroid
gdf["longitude"] = gdf["geometry_centroid"].x
gdf["latitude"] = gdf["geometry_centroid"].y

# Function to match city names (fuzzy matching)
def find_matching_row(city_name, province, gdf_copy):
    """Try to find matching row in shapefile"""
    # Clean city name for matching
    city_clean = re.sub(r'[^\w\s]', '', city_name).strip().lower()
    
    # First, try exact match (cleaned names match exactly)
    exact_matches = []
    for idx, row in gdf_copy.iterrows():
        shapefile_name_clean = re.sub(r'[^\w\s]', '', str(row["City_Name_Clean"])).strip().lower()
        if shapefile_name_clean == city_clean:
            # Check province if available
            if prov_col in gdf_copy.columns and province:
                if province_matches(row.get(prov_col, ""), province):
                    exact_matches.append((idx, row))
                else:
                    # Still add even if province doesn't match (in case of missing data)
                    exact_matches.append((idx, row))
            else:
                exact_matches.append((idx, row))
    
    if len(exact_matches) == 1:
        return exact_matches[0][0]
    elif len(exact_matches) > 1:
        # If multiple exact matches, return first one (shouldn't happen often)
        return exact_matches[0][0]
    
    # Second, try match where shapefile name starts with city name or vice versa
    # (handles cases like "Montreal" matching "Montreal" not "Montreal-Est")
    start_matches = []
    for idx, row in gdf_copy.iterrows():
        shapefile_name_clean = re.sub(r'[^\w\s]', '', str(row["City_Name_Clean"])).strip().lower()
        # Check if city name equals shapefile name or shapefile name starts with city name (for exact matches)
        # Or if city name starts with shapefile name
        if (shapefile_name_clean == city_clean or 
            shapefile_name_clean.startswith(city_clean + ' ') or
            city_clean.startswith(shapefile_name_clean + ' ')):
            # Check province if available
            if prov_col in gdf_copy.columns and province:
                if province_matches(row.get(prov_col, ""), province):
                    start_matches.append((idx, row))
                else:
                    # Still add even if province doesn't match
                    start_matches.append((idx, row))
            else:
                start_matches.append((idx, row))
    
    if len(start_matches) == 1:
        return start_matches[0][0]
    elif len(start_matches) > 1:
        # Prefer exact match over partial match
        for idx, row in start_matches:
            shapefile_name_clean = re.sub(r'[^\w\s]', '', str(row["City_Name_Clean"])).strip().lower()
            if shapefile_name_clean == city_clean:
                return idx
        # If no exact match, return first one
        return start_matches[0][0]
    
    # Third, try contains match (as fallback)
    contains_matches = []
    for idx, row in gdf_copy.iterrows():
        shapefile_name_clean = re.sub(r'[^\w\s]', '', str(row["City_Name_Clean"])).strip().lower()
        if city_clean in shapefile_name_clean or shapefile_name_clean in city_clean:
            # Check province if available
            if prov_col in gdf_copy.columns and province:
                if province_matches(row.get(prov_col, ""), province):
                    contains_matches.append((idx, row))
                else:
                    # Still add even if province doesn't match
                    contains_matches.append((idx, row))
            else:
                contains_matches.append((idx, row))
    
    if len(contains_matches) > 0:
        # Return first match (least preferred method)
        return contains_matches[0][0]
    
    return None

print("\nMatching cities...")
# Match cities from CSV to shapefile
matched_data = []

for idx, row in df_csv.iterrows():
    city_name = row["City_Name"]
    province = row["Province"]
    population = row["Population"]
    
    # Find matching row in shapefile
    match_idx = find_matching_row(city_name, province, gdf)
    
    if match_idx is not None:
        match_row = gdf.loc[match_idx]
        matched_data.append({
            "City_Name": city_name,
            "Province": province,
            "Population": population,
            "Latitude": match_row["latitude"],
            "Longitude": match_row["longitude"],
            "Shapefile_Name": match_row[name_col]
        })
    else:
        # Keep city even if no match found (lat/lon will be NaN)
        matched_data.append({
            "City_Name": city_name,
            "Province": province,
            "Population": population,
            "Latitude": None,
            "Longitude": None,
            "Shapefile_Name": None
        })
    
    if (idx + 1) % 50 == 0:
        print(f"Processed {idx + 1}/{len(df_csv)} cities...")

# Create output dataframe
df_output = pd.DataFrame(matched_data)

# Count matches
matched_count = df_output["Latitude"].notna().sum()
print(f"\nMatched {matched_count}/{len(df_output)} cities ({matched_count/len(df_output)*100:.1f}%)")

# Save to CSV
df_output.to_csv(output_path, index=False)
print(f"\nSaved results to: {output_path}")

# Show preview
print("\nPreview of results:")
print(df_output.head(20))

# Show unmatched cities
unmatched = df_output[df_output["Latitude"].isna()]
if len(unmatched) > 0:
    print(f"\n{len(unmatched)} cities could not be matched:")
    print(unmatched[["City_Name", "Province", "Population"]].head(20))

