import os
import pandas as pd
import requests

print("=" * 70)
print("SKYFINDER MASTER DATASET BUILDER")
print("=" * 70)

# ============================================================
# PATHS
# ============================================================

METADATA_PATH = "data/external/skyfinder/metadata/complete_table_with_mcr.csv"

IMAGE_ROOT = "data/external/skyfinder/images"

OUTPUT_FOLDER = "data/external/skyfinder/final"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading Metadata...")

metadata = pd.read_csv(METADATA_PATH)

print("Metadata Loaded Successfully!")

print("\nRows :", len(metadata))

print("Columns :", len(metadata.columns))

# ============================================================
# DETECT CAMERA FOLDERS
# ============================================================

print("\nScanning Camera Folders...")

camera_folders = []

total_images = 0

for folder in sorted(os.listdir(IMAGE_ROOT)):

    folder_path = os.path.join(IMAGE_ROOT, folder)

    if not os.path.isdir(folder_path):
        continue

    images = [

        file

        for file in os.listdir(folder_path)

        if file.lower().endswith(".jpg")

    ]

    print(f"Camera {folder:>6} : {len(images)} images")

    camera_folders.append(folder)

    total_images += len(images)

print("\nDetected Cameras :", len(camera_folders))

print("Total Images :", total_images)

print("\nCamera IDs")

for camera in camera_folders:

    print(camera)

# ============================================================
# BUILD MASTER METADATA
# ============================================================

print("\n" + "=" * 70)
print("MATCHING METADATA WITH DOWNLOADED IMAGES")
print("=" * 70)

master_df = []

for camera in camera_folders:

    print(f"\nProcessing Camera {camera}...")

    image_folder = os.path.join(IMAGE_ROOT, camera)

    image_files = [

        file

        for file in os.listdir(image_folder)

        if file.lower().endswith(".jpg")

    ]

    camera_metadata = metadata[
        metadata["CamId"] == int(camera)
    ].copy()

    camera_metadata = camera_metadata[
        camera_metadata["Filename"].isin(image_files)
    ]

    print("Metadata Rows :", len(camera_metadata))

    master_df.append(camera_metadata)

# ============================================================
# MERGE ALL CAMERAS
# ============================================================

master_df = pd.concat(

    master_df,

    ignore_index=True

)

print("\n" + "=" * 70)
print("MASTER METADATA CREATED")
print("=" * 70)

print("\nRows :", len(master_df))

print("Columns :", len(master_df.columns))

print("\nFirst Five Rows\n")

print(master_df.head())
# ============================================================
# SAVE MASTER METADATA
# ============================================================

OUTPUT_PATH = "data/external/skyfinder/final/master_metadata.csv"

master_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nMaster Metadata Saved!")

print(OUTPUT_PATH)

# ============================================================
# STAGE 3
# CREATE CLEAN COPY
# ============================================================

print("\n" + "=" * 70)
print("STAGE 3 : DATA CLEANING")
print("=" * 70)

print("\nCreating Backup Copy...")

clean_df = master_df.copy()

print("Done!")

print("\nShape of clean_df :")

print(clean_df.shape)
# ============================================================
# STEP 3.2
# REPLACE INVALID VALUES
# ============================================================

print("\n" + "=" * 70)
print("STEP 3.2 : REPLACING INVALID VALUES")
print("=" * 70)

print("\nReplacing -9999 with NaN...")

clean_df.replace(-9999, pd.NA, inplace=True)

print("Done!")

print("\nChecking Missing Values...\n")

important_columns = [

    "TempM",

    "Hum",

    "VisM",

    "PressureM",

    "Latitude",

    "Longitude",

    "WspdM"

]

for column in important_columns:

    missing = clean_df[column].isna().sum()

    print(f"{column:<12} : {missing}")
# ============================================================
# STEP 3.3
# REMOVE ROWS WITH MISSING REQUIRED VALUES
# ============================================================

print("\n" + "=" * 70)
print("STEP 3.3 : REMOVING INCOMPLETE ROWS")
print("=" * 70)

rows_before = len(clean_df)

required_columns = [

    "Filename",
    "TempM",
    "Hum",
    "VisM",
    "Latitude",
    "Longitude",
    "Date",
    "Hour",
    "Min"

]

clean_df = clean_df.dropna(subset=required_columns)

rows_after = len(clean_df)

print("\nRows Before :", rows_before)

print("Rows After  :", rows_after)

print("Rows Removed :", rows_before - rows_after)
# ============================================================
# STEP 3.4
# REMOVE DUPLICATE IMAGES
# ============================================================

print("\n" + "=" * 70)
print("STEP 3.4 : REMOVING DUPLICATE IMAGES")
print("=" * 70)

rows_before = len(clean_df)

clean_df = clean_df.drop_duplicates(
    subset="Filename",
    keep="first"
)

rows_after = len(clean_df)

print("\nRows Before :", rows_before)
print("Rows After  :", rows_after)
print("Duplicates Removed :", rows_before - rows_after)
# ============================================================
# STEP 3.5A
# INSPECT DAYLIGHT COLUMN
# ============================================================

print("\n" + "=" * 70)
print("STEP 3.5A : INSPECTING DAYLIGHT VALUES")
print("=" * 70)

print("\nDaylight Statistics\n")

print(clean_df["daylight"].describe())

print("\nUnique Sample Values\n")

print(
    sorted(clean_df["daylight"].unique())[:20]
)

print("\nMaximum Value :",
      clean_df["daylight"].max())

print("Minimum Value :",
      clean_df["daylight"].min())
# ============================================================
# STEP 3.5B
# INSPECT HOUR DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("STEP 3.5B : HOUR DISTRIBUTION")
print("=" * 70)

hour_counts = clean_df["Hour"].value_counts().sort_index()

print("\nImages per Hour\n")

print(hour_counts)

print("\nTotal Hours :", len(hour_counts))
# ============================================================
# STAGE 4
# STEP 4.1
# KEEP REQUIRED COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("STAGE 4 : CONVERT TO PRL FORMAT")
print("=" * 70)

print("\nSTEP 4.1 : Keeping Required Columns")

required_columns = [

    "Filename",

    "Date",

    "Hour",

    "Min",

    "TempM",

    "Hum",

    "VisM",

    "Latitude",

    "Longitude"

]

clean_df = clean_df[required_columns]

print("\nRemaining Columns")

print(clean_df.columns.tolist())

print("\nShape")

print(clean_df.shape)
# ============================================================
# STEP 4.2
# RENAME COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4.2 : RENAMING COLUMNS")
print("=" * 70)

clean_df.rename(

    columns={

        "Filename": "image_name",

        "Date": "date",

        "Hour": "hour",

        "Min": "minute",

        "TempM": "temperature",

        "Hum": "relative_humidity",

        "VisM": "visibility_km",

        "Latitude": "latitude",

        "Longitude": "longitude"

    },

    inplace=True

)

print("\nColumns After Renaming\n")

print(clean_df.columns.tolist())

print("\nShape")

print(clean_df.shape)
# ============================================================
# STEP 4.3
# ADD SOLAR INTENSITY COLUMN
# ============================================================

print("\n" + "=" * 70)
print("STEP 4.3 : ADDING SOLAR INTENSITY COLUMN")
print("=" * 70)

clean_df["solar_intensity"] = pd.NA

print("\nColumn Added Successfully!")

print("\nColumns")

print(clean_df.columns.tolist())

print("\nShape")

print(clean_df.shape)
# ============================================================
# STEP 4.4
# CREATE TIME COLUMN
# ============================================================

print("\n" + "=" * 70)
print("STEP 4.4 : CREATING TIME COLUMN")
print("=" * 70)

clean_df["time"] = (

    clean_df["hour"].astype(int).astype(str).str.zfill(2)

    + ":"

    + clean_df["minute"].astype(int).astype(str).str.zfill(2)

)

print("\nTime Column Created Successfully!")

print("\nFirst Five Times\n")

print(clean_df["time"].head())

print("\nShape")

print(clean_df.shape)
# ============================================================
# STEP 4.5
# REORDER COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4.5 : REORDERING COLUMNS")
print("=" * 70)

column_order = [

    "image_name",

    "date",

    "time",

    "hour",

    "minute",

    "temperature",

    "relative_humidity",

    "solar_intensity",

    "visibility_km",

    "latitude",

    "longitude"

]

clean_df = clean_df[column_order]

print("\nFinal Column Order\n")

for col in clean_df.columns:

    print(col)

print("\nShape")

print(clean_df.shape)
# ============================================================
# STAGE 5
# STEP 5.1
# FIND UNIQUE LOCATIONS
# ============================================================

print("\n" + "=" * 70)
print("STAGE 5 : SOLAR RADIATION")
print("=" * 70)

print("\nSTEP 5.1 : FINDING UNIQUE CAMERA LOCATIONS")

locations = (

    clean_df[

        ["latitude", "longitude"]

    ]

    .drop_duplicates()

    .reset_index(drop=True)

)

print("\nUnique Locations :", len(locations))

print("\nLocations\n")

print(locations)
# ============================================================
# STEP 5.2
# FIND UNIQUE LOCATION-YEAR COMBINATIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 5.2 : FINDING LOCATION-YEAR COMBINATIONS")
print("=" * 70)

print("\nFirst 10 Date Values\n")

print(clean_df["date"].head(10))

print("\nDate Data Type")

print(clean_df["date"].dtype)
# Convert date to datetime
# Convert MATLAB datenum to pandas datetime
# MATLAB datenum → datetime
clean_df["date"] = pd.to_datetime(
    clean_df["date"] - 719529,
    unit="D"
)

# Keep only the calendar date
clean_df["date"] = clean_df["date"].dt.floor("D")

print("\nConverted Dates\n")
print(clean_df["date"].head(10))
# Create year column
clean_df["year"] = clean_df["date"].dt.year

# Find unique location-year combinations
location_years = (
    clean_df[
        ["latitude", "longitude", "year"]
    ]
    .drop_duplicates()
    .sort_values(
        ["latitude", "longitude", "year"]
    )
    .reset_index(drop=True)
)

print("\nUnique Location-Year Combinations :", len(location_years))

print("\nLocation-Year Table\n")

print(location_years)
# ============================================================
# STEP 5.3
# CREATE NASA REQUEST TABLE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5.3 : CREATING NASA REQUEST TABLE")
print("=" * 70)

requests_df = location_years.copy()

requests_df["start_date"] = (
    requests_df["year"].astype(str) + "0101"
)

requests_df["end_date"] = (
    requests_df["year"].astype(str) + "1231"
)

requests_df.insert(
    0,
    "request_id",
    range(1, len(requests_df) + 1)
)

print("\nNASA Requests\n")

print(requests_df)
# ============================================================
# SAVE REQUEST TABLE
# ============================================================

REQUEST_PATH = (
    "data/external/skyfinder/final/"
    "nasa_power_requests.csv"
)

requests_df.to_csv(
    REQUEST_PATH,
    index=False
)

print("\nNASA Request Table Saved!")

print(REQUEST_PATH)

# ============================================================
# NASA POWER DOWNLOADER
# ============================================================

def download_solar(latitude, longitude, year):

    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=ALLSKY_SFC_SW_DWN"
        f"&community=RE"
        f"&longitude={longitude}"
        f"&latitude={latitude}"
        f"&start={year}0101"
        f"&end={year}1231"
        f"&format=JSON"
    )

    response = requests.get(url)

    if response.status_code != 200:
        print(f"Download Failed : {response.status_code}")
        return None

    return response.json()
print("\nDownloading NASA POWER Data...\n")

solar_rows = []
for row in requests_df.itertuples():

    print(f"Downloading {row.request_id}/{len(requests_df)}")

    data = download_solar(
        row.latitude,
        row.longitude,
        row.year
    )

    if data is None:
        continue

    parameter = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]

    for date, value in parameter.items():

        solar_rows.append({

            "latitude": row.latitude,

            "longitude": row.longitude,

            "date": date,

            "solar_intensity": value

        })

solar_df = pd.DataFrame(solar_rows)

print("\nSolar Data")

print(solar_df.head())

print("\nRows")

print(len(solar_df))
solar_df.to_csv(

    "data/external/skyfinder/final/solar_data.csv",

    index=False

)

print("\nSolar Data Saved!")
# ============================================================
# STEP 5.5
# MERGE SOLAR DATA
# ============================================================
print("\nDataset Date Sample")
print(clean_df["date"].head())

print("\nSolar Date Sample")
print(solar_df["date"].head())
print("\n" + "=" * 70)
print("STEP 5.5 : MERGING SOLAR DATA")
print("=" * 70)
solar_df["date"] = pd.to_datetime(
    solar_df["date"],
    format="%Y%m%d"
).dt.floor("D")
clean_df["latitude"] = clean_df["latitude"].round(6)
clean_df["longitude"] = clean_df["longitude"].round(6)

solar_df["latitude"] = solar_df["latitude"].round(6)
solar_df["longitude"] = solar_df["longitude"].round(6)
print("\n========== MERGE DEBUG ==========")

print("\nClean Dataset Sample")
print(clean_df[["latitude", "longitude", "date"]].head())

print("\nSolar Dataset Sample")
print(solar_df[["latitude", "longitude", "date"]].head())

print("\nClean Data Types")
print(clean_df[["latitude", "longitude", "date"]].dtypes)

print("\nSolar Data Types")
print(solar_df[["latitude", "longitude", "date"]].dtypes)
master_dataset = pd.merge(
    clean_df,
    solar_df,
    how="left",
    on=[
        "latitude",
        "longitude",
        "date"
    ],
    validate="many_to_one"
)
matched = master_dataset["solar_intensity_y"].notna().sum()

print("\nMatched Rows :", matched)

print("Missing Rows :", len(master_dataset) - matched)
master_dataset["solar_intensity"] = (

    master_dataset["solar_intensity_y"]

)
master_dataset.drop(

    columns=[

        "solar_intensity_x",

        "solar_intensity_y",

        "year"

    ],

    inplace=True,

    errors="ignore"

)
print("\nFinal Shape")

print(master_dataset.shape)

print("\nMissing Solar Values")

print(

    master_dataset["solar_intensity"]

    .isna()

    .sum()

)
master_dataset.to_csv(

    "data/external/skyfinder/final/skyfinder_master_dataset.csv",

    index=False

)

print("\nMASTER DATASET SAVED!")

print(

    "data/external/skyfinder/final/skyfinder_master_dataset.csv"

)

