import pandas as pd

print("=" * 70)
print("SKYFINDER DATASET EXPLORER")
print("=" * 70)

# =====================================================
# LOAD DATASET
# =====================================================

csv_path = "data/external/skyfinder/metadata/complete_table_with_mcr.csv"

print("\nLoading Metadata...")

df = pd.read_csv(csv_path)

print("Metadata Loaded Successfully!")

# =====================================================
# BASIC INFORMATION
# =====================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Rows :", len(df))
print("Columns :", len(df.columns))

print("\nColumn Names:\n")
print(df.columns.tolist())

# =====================================================
# CAMERAS
# =====================================================

print("\n" + "=" * 70)
print("CAMERA INFORMATION")
print("=" * 70)

num_cameras = df["CamId"].nunique()

print("Total Cameras :", num_cameras)

camera_counts = (
    df["CamId"]
    .value_counts()
    .sort_index()
)

print("\nImages Per Camera:\n")
print(camera_counts)

# =====================================================
# MISSING VALUES
# =====================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)

# =====================================================
# TEMPERATURE
# =====================================================

print("\n" + "=" * 70)
print("TEMPERATURE")
print("=" * 70)

print(df["TempM"].describe())

# =====================================================
# HUMIDITY
# =====================================================

print("\n" + "=" * 70)
print("HUMIDITY")
print("=" * 70)

print(df["Hum"].describe())

# =====================================================
# VISIBILITY
# =====================================================

print("\n" + "=" * 70)
print("VISIBILITY")
print("=" * 70)

print(df["VisM"].describe())

# =====================================================
# PRESSURE
# =====================================================

print("\n" + "=" * 70)
print("PRESSURE")
print("=" * 70)

print(df["PressureM"].describe())

# =====================================================
# WIND SPEED
# =====================================================

print("\n" + "=" * 70)
print("WIND SPEED")
print("=" * 70)

print(df["WspdM"].describe())

# =====================================================
# WEATHER CONDITIONS
# =====================================================

print("\n" + "=" * 70)
print("WEATHER CONDITIONS")
print("=" * 70)

print(df["Conds"].value_counts())

# =====================================================
# DAY / NIGHT
# =====================================================

print("\n" + "=" * 70)
print("DAY / NIGHT")
print("=" * 70)

print("Daylight Images :", df["daylight"].sum())
print("Night Images    :", df["night"].sum())

# =====================================================
# FOG
# =====================================================

print("\n" + "=" * 70)
print("FOG")
print("=" * 70)

print(df["Fog"].value_counts())

# =====================================================
# RAIN
# =====================================================

print("\n" + "=" * 70)
print("RAIN")
print("=" * 70)

print(df["Rain"].value_counts())

# =====================================================
# SNOW
# =====================================================

print("\n" + "=" * 70)
print("SNOW")
print("=" * 70)

print(df["Snow"].value_counts())

# =====================================================
# LATITUDE
# =====================================================

print("\n" + "=" * 70)
print("LATITUDE")
print("=" * 70)

print(df["Latitude"].describe())

# =====================================================
# LONGITUDE
# =====================================================

print("\n" + "=" * 70)
print("LONGITUDE")
print("=" * 70)

print(df["Longitude"].describe())

# =====================================================
# SAVE CAMERA SUMMARY
# =====================================================

camera_summary = (
    df.groupby("CamId")
      .agg(
          Images=("Filename", "count"),
          Latitude=("Latitude", "first"),
          Longitude=("Longitude", "first"),
          MeanTemp=("TempM", "mean"),
          MeanHumidity=("Hum", "mean"),
          MeanVisibility=("VisM", "mean")
      )
      .reset_index()
)

output_path = "results/external/camera_summary.csv"

camera_summary.to_csv(
    output_path,
    index=False
)

print("\nCamera Summary Saved!")

print("\nOutput File :")
print(output_path)

print("\nDone!")