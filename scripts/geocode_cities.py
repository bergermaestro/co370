import pandas as pd
from geopy.geocoders import Nominatim
import time
import re
from io import StringIO

# Paths
csv_path = "data/1710015501-eng.csv"
output_path = "data/cities_with_coordinates.csv"

print("Loading CSV file...")
# Read the CSV file - extract data portion only
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
    data_text = ''.join(data_lines)
    df_csv = pd.read_csv(StringIO(data_text), header=0)
    
    # Remove the sub-header row (where Geography is NaN and 2024 is "Persons")
    df_csv = df_csv[df_csv["Geography"].notna() & (df_csv["2024"] != "Persons")]

# Clean up the data - remove rows that are totals/aggregates
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

# -------------------------------------
# Geocode each city using Nominatim
# -------------------------------------

geolocator = Nominatim(user_agent="city_geocoder")

latitudes = []
longitudes = []

print("\nFetching coordinates using Nominatim (this will take several minutes)...")
print("Rate limiting: 1 second delay between requests")

for idx, row in df_csv.iterrows():
    city_name = row["City_Name"]
    province = row["Province"]
    
    # Create query: "City Name, Province, Canada"
    query = f"{city_name}, {province}, Canada"
    
    try:
        loc = geolocator.geocode(query, timeout=10)
        if loc:
            latitudes.append(loc.latitude)
            longitudes.append(loc.longitude)
            if (idx + 1) % 50 == 0:
                print(f"Geocoded {idx + 1}/{len(df_csv)} cities... ({city_name})")
        else:
            latitudes.append(None)
            longitudes.append(None)
            print(f"Warning: Could not geocode {query}")
    except Exception as e:
        print(f"Error geocoding {query}: {e}")
        latitudes.append(None)
        longitudes.append(None)
    
    # Rate limiting - Nominatim requires 1 second delay between requests
    time.sleep(1)

df_csv["Latitude"] = latitudes
df_csv["Longitude"] = longitudes

# -------------------------------------
# Create output dataframe
# -------------------------------------

df_output = pd.DataFrame({
    "City_Name": df_csv["City_Name"],
    "Province": df_csv["Province"],
    "Population": df_csv["Population"],
    "Latitude": df_csv["Latitude"],
    "Longitude": df_csv["Longitude"]
})

# Count successful geocodes
matched_count = df_output["Latitude"].notna().sum()
print(f"\nSuccessfully geocoded {matched_count}/{len(df_output)} cities ({matched_count/len(df_output)*100:.1f}%)")

# Save to CSV
df_output.to_csv(output_path, index=False)
print(f"\nSaved results to: {output_path}")

# Show preview
print("\nPreview of results:")
print(df_output.head(20))

# Show cities that couldn't be geocoded
unmatched = df_output[df_output["Latitude"].isna()]
if len(unmatched) > 0:
    print(f"\n{len(unmatched)} cities could not be geocoded:")
    print(unmatched[["City_Name", "Province", "Population"]].head(20))

