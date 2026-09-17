# Geospatial-Data-Analysis-
# ============================================================
#              GEOSPATIAL DATA ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

# Keep your CSV file in the same folder as this program.
# Change the name if your downloaded dataset has another name.

file_name = "regional_sales.csv"

df = pd.read_csv(file_name)

print("=" * 60)
print("             GEOSPATIAL DATA ANALYSIS")
print("=" * 60)

print("\nFirst 5 records:")
print(df.head())

# ============================================================
# STEP 2: BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nNumber of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# ============================================================
# STEP 3: CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nCleaned column names:")
print(df.columns.tolist())

# ============================================================
# STEP 4: REMOVE DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate records:", duplicate_count)

df = df.drop_duplicates()

print("Duplicate records removed.")

# ============================================================
# STEP 5: CLEAN LOCATION IDENTIFIERS
# ============================================================

# Remove extra spaces from text columns

for column in df.select_dtypes(
    include="object"
).columns:

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
    )

# ------------------------------------------------------------
# Clean State
# ------------------------------------------------------------

if "state" in df.columns:

    df["state"] = (
        df["state"]
        .str.title()
    )

# ------------------------------------------------------------
# Clean City
# ------------------------------------------------------------

if "city" in df.columns:

    df["city"] = (
        df["city"]
        .str.title()
    )

# ------------------------------------------------------------
# Clean Postal Code
# ------------------------------------------------------------

if "postal_code" in df.columns:

    df["postal_code"] = (
        df["postal_code"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

# ============================================================
# STEP 6: CONVERT NUMERIC COLUMNS
# ============================================================

if "revenue" in df.columns:

    df["revenue"] = pd.to_numeric(
        df["revenue"],
        errors="coerce"
    )

if "users" in df.columns:

    df["users"] = pd.to_numeric(
        df["users"],
        errors="coerce"
    )

if "latitude" in df.columns:

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

if "longitude" in df.columns:

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

# ============================================================
# STEP 7: HANDLE MISSING VALUES
# ============================================================

if "revenue" in df.columns:

    df["revenue"] = df["revenue"].fillna(0)

if "users" in df.columns:

    df["users"] = df["users"].fillna(0)

# Remove rows without location

location_columns = []

if "latitude" in df.columns:
    location_columns.append("latitude")

if "longitude" in df.columns:
    location_columns.append("longitude")

if len(location_columns) > 0:

    df = df.dropna(
        subset=location_columns
    )

print("\nData cleaning completed.")

# ============================================================
# STEP 8: DISPLAY CLEAN DATA
# ============================================================

print("\n" + "=" * 60)
print("CLEANED DATA")
print("=" * 60)

print(df.head())

# ============================================================
# STEP 9: AGGREGATE REVENUE BY STATE
# ============================================================

print("\n" + "=" * 60)
print("REVENUE BY STATE")
print("=" * 60)

if "state" in df.columns:

    state_revenue = (
        df.groupby("state")
        ["revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(state_revenue)

# ============================================================
# STEP 10: AGGREGATE USERS BY STATE
# ============================================================

print("\n" + "=" * 60)
print("USERS BY STATE")
print("=" * 60)

if "state" in df.columns:

    state_users = (
        df.groupby("state")
        ["users"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(state_users)

# ============================================================
# STEP 11: CALCULATE AVERAGE REVENUE PER USER
# ============================================================

print("\n" + "=" * 60)
print("REVENUE PER USER")
print("=" * 60)

state_summary = pd.DataFrame()

if "state" in df.columns:

    state_summary = (
        df.groupby("state")
        .agg(
            total_revenue=("revenue", "sum"),
            total_users=("users", "sum")
        )
    )

    state_summary[
        "revenue_per_user"
    ] = np.where(
        state_summary["total_users"] > 0,
        state_summary["total_revenue"] /
        state_summary["total_users"],
        0
    )

    print(state_summary)

# ============================================================
# STEP 12: CALCULATE USER DENSITY
# ============================================================

print("\n" + "=" * 60)
print("USER DENSITY")
print("=" * 60)

# If area information exists, calculate users per area.

if (
    "state" in df.columns
    and "area_sq_km" in df.columns
):

    df["area_sq_km"] = pd.to_numeric(
        df["area_sq_km"],
        errors="coerce"
    )

    df["area_sq_km"] = (
        df["area_sq_km"]
        .fillna(1)
    )

    state_density = (
        df.groupby("state")
        .agg(
            users=("users", "sum"),
            area_sq_km=("area_sq_km", "mean")
        )
    )

    state_density[
        "user_density"
    ] = (
        state_density["users"] /
        state_density["area_sq_km"]
    )

    print(state_density)

else:

    print(
        "area_sq_km column not available."
    )

# ============================================================
# STEP 13: IDENTIFY UNDER-SERVED REGIONS
# ============================================================

print("\n" + "=" * 60)
print("UNDER-SERVED REGION ANALYSIS")
print("=" * 60)

if len(state_summary) > 0:

    # Calculate average revenue and users

    average_revenue = (
        state_summary["total_revenue"]
        .mean()
    )

    average_users = (
        state_summary["total_users"]
        .mean()
    )

    # A region is considered potentially underserved
    # when it has relatively high users but lower revenue.

    state_summary["potential_underserved"] = (
        (state_summary["total_users"] > average_users)
        &
        (state_summary["total_revenue"] < average_revenue)
    )

    underserved = state_summary[
        state_summary["potential_underserved"]
    ].copy()

    underserved = underserved.sort_values(
        by="total_users",
        ascending=False
    )

    print(
        "\nPotential underserved regions:"
    )

    print(underserved)

# ============================================================
# STEP 14: TOP 3 HIGH-POTENTIAL REGIONS
# ============================================================

print("\n" + "=" * 60)
print("TOP 3 HIGH-POTENTIAL UNDERSERVED REGIONS")
print("=" * 60)

if len(underserved) > 0:

    top_3 = underserved.head(3)

    print(top_3)

    top_3.to_csv(
        "top_3_underserved_regions.csv"
    )

else:

    print(
        "No underserved regions found using "
        "the selected criteria."
    )

# ============================================================
# STEP 15: STATE SUMMARY TABLE
# ============================================================

if len(state_summary) > 0:

    state_summary = state_summary.sort_values(
        by="total_revenue",
        ascending=False
    )

    state_summary.to_csv(
        "regional_summary.csv"
    )

# ============================================================
# STEP 16: BAR CHART - REVENUE BY STATE
# ============================================================

if "state" in df.columns:

    plt.figure(
        figsize=(10, 6)
    )

    state_revenue.plot(
        kind="bar"
    )

    plt.title(
        "Total Revenue by State"
    )

    plt.xlabel(
        "State"
    )

    plt.ylabel(
        "Total Revenue"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        "01_revenue_by_state.png"
    )

    plt.show()

# ============================================================
# STEP 17: BAR CHART - USERS BY STATE
# ============================================================

if "state" in df.columns:

    plt.figure(
        figsize=(10, 6)
    )

    state_users.plot(
        kind="bar"
    )

    plt.title(
        "Total Users by State"
    )

    plt.xlabel(
        "State"
    )

    plt.ylabel(
        "Number of Users"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        "02_users_by_state.png"
    )

    plt.show()

# ============================================================
# STEP 18: SCATTER PLOT
# ============================================================

if len(state_summary) > 0:

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        state_summary["total_users"],
        state_summary["total_revenue"]
    )

    # Add state names

    for state in state_summary.index:

        plt.annotate(
            state,
            (
                state_summary.loc[
                    state,
                    "total_users"
                ],

                state_summary.loc[
                    state,
                    "total_revenue"
                ]
            )
        )

    plt.title(
        "Users vs Revenue by State"
    )

    plt.xlabel(
        "Total Users"
    )

    plt.ylabel(
        "Total Revenue"
    )

    plt.tight_layout()

    plt.savefig(
        "03_users_vs_revenue.png"
    )

    plt.show()

# ============================================================
# STEP 19: CREATE INTERACTIVE FOLIUM MAP
# ============================================================

print("\n" + "=" * 60)
print("CREATING INTERACTIVE MAP")
print("=" * 60)

# Center of India

india_map = folium.Map(
    location=[
        20.5937,
        78.9629
    ],
    zoom_start=5
)

# Create marker cluster

marker_cluster = MarkerCluster().add_to(
    india_map
)

# ------------------------------------------------------------
# Add each location to map
# ------------------------------------------------------------

for index, row in df.iterrows():

    latitude = row["latitude"]
    longitude = row["longitude"]

    # Get information safely

    if "city" in df.columns:
        city = row["city"]
    else:
        city = "Unknown"

    if "state" in df.columns:
        state = row["state"]
    else:
        state = "Unknown"

    if "revenue" in df.columns:
        revenue = row["revenue"]
    else:
        revenue = 0

    if "users" in df.columns:
        users = row["users"]
    else:
        users = 0

    popup_text = f"""
    <b>City:</b> {city}<br>
    <b>State:</b> {state}<br>
    <b>Revenue:</b> ₹{revenue:,.2f}<br>
    <b>Users:</b> {users:,.0f}
    """

    folium.Marker(
        location=[
            latitude,
            longitude
        ],

        popup=folium.Popup(
            popup_text,
            max_width=300
        ),

        tooltip=city
    ).add_to(
        marker_cluster
    )

# ============================================================
# STEP 20: HIGHLIGHT TOP 3 UNDERSERVED REGIONS
# ============================================================

if len(underserved) > 0:

    print(
        "\nAdding top underserved regions to map..."
    )

    for state in top_3.index:

        state_data = df[
            df["state"] == state
        ]

        if len(state_data) > 0:

            avg_latitude = (
                state_data["latitude"]
                .mean()
            )

            avg_longitude = (
                state_data["longitude"]
                .mean()
            )

            revenue = (
                state_summary.loc[
                    state,
                    "total_revenue"
                ]
            )

            users = (
                state_summary.loc[
                    state,
                    "total_users"
                ]
            )

            popup_text = f"""
            <b>HIGH-POTENTIAL UNDERSERVED REGION</b><br>
            <b>State:</b> {state}<br>
            <b>Total Users:</b> {users:,.0f}<br>
            <b>Total Revenue:</b> ₹{revenue:,.2f}
            """

            folium.Marker(
                location=[
                    avg_latitude,
                    avg_longitude
                ],

                popup=folium.Popup(
                    popup_text,
                    max_width=300
                ),

                tooltip=(
                    "High Potential: " +
                    state
                ),

                icon=folium.Icon(
                    icon="star"
                )
            ).add_to(
                india_map
            )

# ============================================================
# STEP 21: SAVE INTERACTIVE MAP
# ============================================================

india_map.save(
    "geospatial_analysis_map.html"
)

print(
    "\nInteractive map saved as:"
)

print(
    "geospatial_analysis_map.html"
)

# ============================================================
# STEP 22: SAVE CLEANED DATA
# ============================================================

df.to_csv(
    "cleaned_regional_sales.csv",
    index=False
)

# ============================================================
# STEP 23: CREATE TEXT REPORT
# ============================================================

report = open(
    "geospatial_analysis_report.txt",
    "w"
)

report.write(
    "====================================================\n"
)

report.write(
    "             GEOSPATIAL DATA ANALYSIS\n"
)

report.write(
    "====================================================\n\n"
)

report.write(
    "Total records: {}\n".format(
        len(df)
    )
)

if len(state_summary) > 0:

    report.write(
        "\nREGIONAL SUMMARY\n"
    )

    report.write(
        "----------------------------------------------------\n"
    )

    report.write(
        state_summary.to_string()
    )

    report.write("\n\n")

if len(underserved) > 0:

    report.write(
        "TOP 3 HIGH-POTENTIAL UNDERSERVED REGIONS\n"
    )

    report.write(
        "----------------------------------------------------\n"
    )

    report.write(
        top_3.to_string()
    )

    report.write("\n\n")

report.write(
    "BUSINESS INTERPRETATION\n"
)

report.write(
    "----------------------------------------------------\n"
)

report.write("""
The analysis combines revenue and user information
with geographic location data.

Regions with relatively high user counts but lower
revenue are identified as potential underserved
markets.

These regions may provide opportunities for business
expansion, improved distribution, marketing campaigns
or additional services.

The interactive map allows geographic patterns to be
examined visually.

Further business research should be conducted before
making expansion decisions.
""")

report.close()

# ============================================================
# STEP 24: FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("             ANALYSIS COMPLETED")
print("=" * 60)

print("""
Files generated:

1. cleaned_regional_sales.csv
2. regional_summary.csv
3. top_3_underserved_regions.csv
4. geospatial_analysis_map.html
5. geospatial_analysis_report.txt

Charts generated:

1. 01_revenue_by_state.png
2. 02_users_by_state.png
3. 03_users_vs_revenue.png

Open geospatial_analysis_map.html in a browser
to view the interactive map.
""")

print("=" * 60)
print("                  THANK YOU!")
print("=" * 60)