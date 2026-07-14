import os
import requests
import pandas as pd

print("=" * 70)
print("DOWNLOADING SOLAR RADIATION FROM NASA POWER")
print("=" * 70)

# =====================================================
# INPUT
# =====================================================

INPUT_PATH = "data/external/skyfinder/final/solar_requests_{CAMERA_ID}.csv"

OUTPUT_FOLDER = "data/external/skyfinder/final"

OUTPUT_FILE = "solar_data_{CAMERA_ID}.csv"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# =====================================================
# LOAD REQUESTS
# =====================================================

print("\nLoading Requests...")

df = pd.read_csv(INPUT_PATH)

print("Rows :", len(df))

# =====================================================
# GET LATITUDE/LONGITUDE
# =====================================================

latitude = df["latitude"].iloc[0]
longitude = df["longitude"].iloc[0]

print("\nLatitude :", latitude)
print("Longitude :", longitude)

# =====================================================
# YEAR
# =====================================================

START_DATE = "20130101"
END_DATE = "20131231"

# =====================================================
# NASA URL
# =====================================================

url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=ALLSKY_SFC_SW_DWN"
    "&community=RE"
    f"&latitude={latitude}"
    f"&longitude={longitude}"
    f"&start={START_DATE}"
    f"&end={END_DATE}"
    "&format=JSON"
)

print("\nDownloading Solar Radiation...")

response = requests.get(url)

print("Status Code :", response.status_code)

# =====================================================
# CHECK RESPONSE
# =====================================================

if response.status_code != 200:

    print("NASA API Error")

    exit()

data = response.json()

# =====================================================
# EXTRACT VALUES
# =====================================================

solar = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]

solar_df = pd.DataFrame(

    solar.items(),

    columns=["date", "solar_intensity"]

)

# =====================================================
# FORMAT DATE
# =====================================================

solar_df["date"] = pd.to_datetime(

    solar_df["date"],

    format="%Y%m%d"

).dt.strftime("%Y-%m-%d")

# =====================================================
# SAVE
# =====================================================

save_path = os.path.join(

    OUTPUT_FOLDER,

    OUTPUT_FILE

)

solar_df.to_csv(

    save_path,

    index=False

)

print("\nSaved Successfully!")

print(save_path)

print("\nRows :", len(solar_df))

print("\nFirst Five Rows\n")

print(solar_df.head())