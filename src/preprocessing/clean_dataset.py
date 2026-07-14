import pandas as pd

# ==========================================================
# FILES
# ==========================================================

INPUT_FILE = "data/prl/metadata/prl_dataset_final.xlsx"

OUTPUT_FILE = "data/prl/metadata/prl_dataset_clean.xlsx"

# ==========================================================

df = pd.read_excel(INPUT_FILE)

# ----------------------------------------------------------
# DATE
# ----------------------------------------------------------

df["date"] = pd.to_datetime(df["date"])

# ----------------------------------------------------------
# DAY
# ----------------------------------------------------------

df["day"] = df["date"].dt.day

# ----------------------------------------------------------
# HOUR
# ----------------------------------------------------------

df["time"] = df["time"].astype(str)

df["hour"] = df["time"].str.split(":").str[0].astype(int)

# ----------------------------------------------------------
# TEMPERATURE
# ----------------------------------------------------------

df["temperature_C"] = (
    df["temperature_C"]
    .astype(str)
    .str.replace("°C", "", regex=False)
    .str.strip()
)

df["temperature_C"] = pd.to_numeric(
    df["temperature_C"],
    errors="coerce"
)

# ----------------------------------------------------------
# HUMIDITY
# ----------------------------------------------------------

df["relative_humidity_%"] = (
    df["relative_humidity_%"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
)

df["relative_humidity_%"] = pd.to_numeric(
    df["relative_humidity_%"],
    errors="coerce"
)

# ----------------------------------------------------------
# SOLAR
# ----------------------------------------------------------

df["solar_intensity_Wm2"] = (
    df["solar_intensity_Wm2"]
    .astype(str)
    .str.replace("w/m²", "", regex=False)
    .str.replace("W/m²", "", regex=False)
    .str.strip()
)

df["solar_intensity_Wm2"] = pd.to_numeric(
    df["solar_intensity_Wm2"],
    errors="coerce"
)

# ----------------------------------------------------------
# VISIBILITY
# ----------------------------------------------------------

df["visibility_km"] = (
    df["visibility_km"]
    .astype(str)
    .str.replace("Km", "", regex=False)
    .str.replace("km", "", regex=False)
    .str.strip()
)

df["visibility_km"] = pd.to_numeric(
    df["visibility_km"],
    errors="coerce"
)

# ----------------------------------------------------------
# COLUMN ORDER
# ----------------------------------------------------------

df = df[
    [
        "image_name",
        "date",
        "day",
        "time",
        "hour",
        "temperature_C",
        "relative_humidity_%",
        "solar_intensity_Wm2",
        "visibility_km",
        "latitude",
        "longitude",
        "distance_m",
    ]
]

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\nDataset cleaned successfully!")
print(df.head())