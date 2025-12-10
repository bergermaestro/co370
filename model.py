# gurobi_hsr_model.py
import math
import itertools
import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
import json


CSV_PATH = "data/cities_with_coordinates.csv"
POP_THRESHOLD = 50000  # keep only cities with population >= 50,000
MAX_EDGE_DIST_KM = (
    350  # only consider connecting cities within this straight-line distance
)
EARTH_R = 6371.0  # Earth radius in km (used to convert lat/lon -> chord)

# Costs & fares (change to experiment)
COST_PER_KM = 51_634_834.0  # CAD per kilometre (construction)
# STATION_COST_SMALL = 50_000_000.0  # CAD for small station (population < 1,000,000)
# STATION_COST_LARGE = 200_000_000.0  # CAD for large station (population >= 1,000,000)
FARE = 100.0  # CAD per passenger per trip
OP_COST_PER_PASSENGER = 5.0  # CAD per passenger operating cost per trip (monthly basis)
ALPHA = 0.2  # baseline monthly ridership fraction of population (0.5%)
R = 35  # catchment radius in km

SOURCE_NAME = "Toronto"  # must match substring in City_Name column
SINK_NAME = "Québec"  # or "Québec", match substring


def chord_distance_km(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """
    Straight-line chord distance on sphere (approx Euclidean distance between two points on Earth's surface).
    This is a close straight-line approximation and returns km.
    """
    # convert to radians
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)
    # central angle
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    central = 2 * math.asin(min(1.0, math.sqrt(a)))
    # chord length (straight line through sphere) approximated by arc length (R * central)
    # For "Euclidean on sphere surface", R*central is arc distance (great-circle). This is appropriate
    # as the straight-line approximation for the rail corridor on Earth's surface.
    return EARTH_R * central


df = pd.read_csv(CSV_PATH)
cols = {c.lower(): c for c in df.columns}


def col(name):
    key = name.lower()
    return cols.get(key, name)  # return original if not found


cn_col = col("City_Name")
prov_col = col("Province")
pop_col = col("Population")
lat_col = col("Latitude")
lon_col = col("Longitude")

# Basic cleaning
df = df[[cn_col, prov_col, pop_col, lat_col, lon_col]].copy()
df.columns = ["City", "Province", "Population", "Latitude", "Longitude"]

# Ensure numeric
df["Population"] = pd.to_numeric(
    df["Population"].astype(str).str.replace(",", ""), errors="coerce"
)
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

# Filter by population threshold
df = df[df["Population"].notnull() & (df["Population"] >= POP_THRESHOLD)].reset_index(
    drop=True
)
print(f"Kept {len(df)} cities with pop >= {POP_THRESHOLD}")


# find source and sink indices (best-effort by substring)
def find_city_index(substring):
    substring_lower = substring.lower()
    for i, name in enumerate(df["City"]):
        if substring_lower in str(name).lower():
            return i
    return None


s_idx = find_city_index(SOURCE_NAME)
t_idx = find_city_index(SINK_NAME)
if s_idx is None or t_idx is None:
    raise ValueError(
        f"Could not locate source/sink by substring: source='{SOURCE_NAME}', sink='{SINK_NAME}'. "
        "Edit SOURCE_NAME / SINK_NAME to match City names in your CSV."
    )

source = df.loc[s_idx, "City"]
sink = df.loc[t_idx, "City"]
print("Source:", source, "Sink:", sink)

# ---------------------
# Build candidate edges (undirected i<j)
# ---------------------
nodes = list(df["City"])
n = len(nodes)
coords = {nodes[i]: (df.loc[i, "Latitude"], df.loc[i, "Longitude"]) for i in range(n)}
pop = {nodes[i]: int(df.loc[i, "Population"]) for i in range(n)}
longitude = {c: coords[c][1] for c in nodes}  # coords[c] = (lat, lon)

# Longitude of each city c
longitude = {nodes[i]: float(df.loc[i, "Longitude"]) for i in range(n)}

# compute pairwise distances and keep edges <= MAX_EDGE_DIST_KM
edges = []
dist = {}
for i, j in itertools.combinations(range(n), 2):
    city_i = nodes[i]
    city_j = nodes[j]
    lat_i, lon_i = coords[city_i]
    lat_j, lon_j = coords[city_j]
    d = chord_distance_km(lat_i, lon_i, lat_j, lon_j)
    if d <= MAX_EDGE_DIST_KM:
        edges.append((city_i, city_j))
        dist[(city_i, city_j)] = d
        dist[(city_j, city_i)] = d  # convenience

print(
    f"Candidate undirected edges kept (distance <= {MAX_EDGE_DIST_KM} km): {len(edges)}"
)

# ---------------------
# Precompute parameters for objective
# ---------------------
# Station cost per city (variable)



y_c1 = {
    c: (1 if pop[c] < 250_000 else 0) for c in nodes
}  # binary variable for calculating station cost
y_c2 = {c: (1 if (pop[c] >= 250_000 and pop[c] <= 750_000) else 0) for c in nodes}
y_c3 = {c: (1 if pop[c] > 750_000 else 0) for c in nodes}

station_cost = {
    c: 33_307_342 * y_c1[c] + 86_285_660 * y_c2[c] + 148_453_987 * y_c3[c]
    for c in nodes
}

# Expected monthly ridership (baseline)
# r_c = ALPHA * population * x_c
# Revenue per passenger: fare_per_km * AVG_TRIP_KM
fare_per_passenger = FARE
op_cost_per_passenger = OP_COST_PER_PASSENGER

# ---------------------
# Build Gurobi model
# ---------------------
m = Model("hsr_path_design")

# Decision variables
x = m.addVars(nodes, vtype=GRB.BINARY, name="x")  # station built at city c
e = m.addVars(edges, vtype=GRB.BINARY, name="e")  # undirected edge selected (i<j)

# a_{c,u} only if dist(c,u) <= R (assignment of city u served by station at c)
assign_pairs = [
    (c, u)
    for c in nodes
    for u in nodes
    if (c == u) or ((c, u) in dist and dist[(c, u)] <= R)
]

# print(f"Assign pairs: {assign_pairs}")

a = m.addVars(assign_pairs, vtype=GRB.BINARY, name="a")

# ridership variable per station c: monthly ridership r_c
r = m.addVars(nodes, vtype=GRB.CONTINUOUS, lb=0.0, name="r")
m_rev = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="m_rev")
capex = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="capex")
P = m.addVar(vtype=GRB.CONTINUOUS, name="P")

# we will create directed flow variables for each ordered pair where edge exists (two directions)
arc_list = []
for i, j in edges:
    arc_list.append((i, j))
    arc_list.append((j, i))
f = m.addVars(arc_list, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="f")  # unit flow

# ---------------------
# Constraints
# ---------------------

# Source and sink stations must be built
m.addConstr(x[source] == 1, name="source_station")
m.addConstr(x[sink] == 1, name="sink_station")

# 1) If an edge is selected, both endpoints must have stations
for i, j in edges:
    m.addConstr(e[(i, j)] <= x[i], name=f"edge_uses_station_{i}_{j}_i")
    m.addConstr(e[(i, j)] <= x[j], name=f"edge_uses_station_{i}_{j}_j")


# create a map city to edges
incident_map = {c: [] for c in nodes}
for i, j in edges:
    incident_map[i].append((i, j))
    incident_map[j].append((i, j))

# 3) Flow conservation: send 1 unit from source to sink (directed flow on arcs)
for c in nodes:
    inflow = quicksum(f[(i, c)] for i in nodes if (i, c) in f)
    outflow = quicksum(f[(c, j)] for j in nodes if (c, j) in f)

    # Flow conservation
    if c == source:
        m.addConstr(outflow - inflow == 1.0, name="flow_source")
    elif c == sink:
        m.addConstr(inflow - outflow == 1.0, name="flow_sink")
    else:
        m.addConstr(outflow - inflow == 0.0, name=f"flow_conserv_{c}")

# 4) Flow only on selected edges: for each directed arc (i->j), flow <= e[(min,max)]
for i, j in arc_list:
    und = (i, j) if (i, j) in e else (j, i)
    # map to undirected key (i,j) sorted to the e key
    key = (i, j) if (i, j) in e else (j, i)
    m.addConstr(f[(i, j)] <= e[key], name=f"flow_edge_link_{i}_{j}")

# 6) Degree constraints to enforce a simple path
for c in nodes:
    deg = quicksum(e[eij] for eij in incident_map[c])
    if c == source or c == sink:
        m.addConstr(deg == 1, name=f"deg_1_{c}")
    else:
        m.addConstr(deg == 2 * x[c], name=f"deg_internal_{c}")

# 7) Number of selected edges = number of selected stations – 1
# this is what we had in the proposal, it doesn't forbid one valid path + a loop disconnected from the path

m.addConstr(quicksum(e[eij] for eij in edges) == quicksum(x[c] for c in nodes) - 1)

# 8) Connectivity via longitude:
# Every selected city (except the source) must be connected by at least one
# selected edge to a city lying strictly to its west (smaller longitude).
# Since Toronto is the westernmost allowed city, this forces every selected city to be in the same
# connected component as Toronto, and rules out disconnected loops.

for c in nodes:
    if c == source:
        continue  # Toronto is the western anchor; no constraint needed

    # Collect edges from c to cities that are strictly west of c
    edges_to_west = []
    for i, j in incident_map[c]:
        other = j if i == c else i
        if longitude[other] < longitude[c]:
            edges_to_west.append(e[(i, j)])

    if edges_to_west:
        # If c is selected (x[c] = 1), at least one westward edge must be selected.
        m.addConstr(
            quicksum(edges_to_west) >= x[c],
            name=f"connect_west_{c}",
        )
    else:
        # If there is no candidate westward neighbour in the graph, c can never be selected.
        m.addConstr(
            x[c] == 0,
            name=f"cannot_select_{c}_no_west_neighbor",
        )

# Each city can only be assigned to at most one station
for u in nodes:
    assigns = [a[(c, u)] for (c, u2) in assign_pairs if u2 == u]
    if assigns:
        m.addConstr(quicksum(assigns) <= 1, name=f"assign_once_{u}")

# A city can only be served by a city with a station
for c, u in assign_pairs:
    m.addConstr(a[(c, u)] <= x[c], name=f"assign_requires_station_{c}_{u}")

# Monthly ridership per station: r_c = sum_{u} alpha * pop_u * a_{c u}
for c in nodes:
    assigns_for_c = [(c, u) for (c2, u) in assign_pairs if c2 == c]
    if assigns_for_c:
        m.addConstr(
            r[c]
            == quicksum(
                (ALPHA * pop[u] if c == u else ALPHA * 0.8 * pop[u]) * a[(c, u)]
                for (c, u) in assigns_for_c
            ),
            name=f"ridership_{c}",
        )
    else:
        m.addConstr(r[c] == 0, name=f"ridership_zero_{c}")

# Cities West of Toronto and East of Quebec City may not be chosen
toronto_long = longitude[source]
quebec_long = longitude[sink]

for c in nodes:
    if c != source:
        m.addConstr(longitude[c] * x[c] >= toronto_long * x[c])
    if c != sink:
        m.addConstr(longitude[c] * x[c] <= quebec_long * x[c])


# 8) Population constraint
for c in nodes:
    m.addConstr(y_c1[c] * pop[c] <= 249_999, name=f"y1_upper_{c}")

    m.addConstr(250_000 * y_c2[c] <= pop[c] * y_c2[c], name=f"y2_lower_{c}")

    m.addConstr(pop[c] * y_c2[c] <= 750_000 * y_c2[c], name=f"y2_upper_{c}")

    m.addConstr(750_001 * y_c3[c] <= pop[c] * y_c3[c], name=f"y3_lower_{c}")

    m.addConstr(
        y_c1[c] + y_c2[c] + y_c3[c] == 1,
        name=f"city_has_one_size{c}",
    )

# Monthly revenue: m_rev = sum_c fare * r_c
m.addConstr(m_rev == quicksum(FARE * r[c] for c in nodes), name="m_rev_def")

# Capital cost: capex = sum_{(i,j) in A} cost_per_km * d_ij * e_{i j} + sum_c station_cost_c * x_c

# ---------------------
# Objective
# ---------------------

station_cost_expr = quicksum(station_cost[c] * x[c] for c in nodes)
track_cost_expr = quicksum(COST_PER_KM * dist[(i, j)] * e[(i, j)] for (i, j) in edges)
capex = track_cost_expr + station_cost_expr

m.setObjective(480 * m_rev - capex, GRB.MAXIMIZE)


# ---------------------
# Solve
# ---------------------
m.params.OutputFlag = 1
m.optimize()

m.optimize()

total_r = 0.0
print("\nSelected cities and their ridership:")
for c in nodes:
    if x[c].X > 0.5:   # city is selected
        rc = r[c].X
        total_r += rc
        print(f"{c}: r = {rc}")

print(f"\nTotal r over selected cities = {total_r}")

# print("\n=== All a[c,u] values ===")

# for (c, u) in a.keys():
#     if a[c, u].X > 0.5:     # only non-zero will be printed
#         print(f"a[{c}, {u}] = 1")

# ---------------------
# Extract solution
# ---------------------
if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
    chosen_stations = [c for c in nodes if x[c].X > 0.5]
    chosen_edges = [eij for eij in edges if e[eij].X > 0.5]
    print("Stations chosen:", chosen_stations)
    print("Edges chosen:", chosen_edges)
    print("Objective value:", m.ObjVal)
    solution = {
        "stations": chosen_stations,
        "edges": [list(eij) for eij in chosen_edges],
    }
    with open("solution.json", "w") as f:
        json.dump(solution, f)
else:
    print("No optimal solution found. Status:", m.Status)
