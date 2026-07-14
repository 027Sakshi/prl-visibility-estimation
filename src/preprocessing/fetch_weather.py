import pandas as pd
import requests
import time

# =======================================================
# LOCATION
# =======================================================

LATITUDE = 23.036
LONGITUDE = 72.544

# =======================================================

metadata = pd.read_csv(
    "data/prl/metadata/image_metadata.csv"
)

weather_data = []

for index, row in metadata.iterrows():

    date = row["date"]

    hour = int(row["time"].split(":")[0])

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={date}"
        f"&end_date={date}"
        "&hourly="
        "temperature_2m,"
        "relative_humidity_2m,"
        "shortwave_radiation"
        "&timezone=Asia/Kolkata"
    )

    response = requests.get(url)

    data = response.json()

    weather_data.append({

        "image_name": row["image_name"],

        "date": date,

        "time": row["time"],

        "temperature":

        data["hourly"]["temperature_2m"][hour],

        "humidity":

        data["hourly"]["relative_humidity_2m"][hour],

        "solar_intensity":

        data["hourly"]["shortwave_radiation"][hour],

        "latitude": LATITUDE,

        "longitude": LONGITUDE,

        "distance": 265

    })

    print(f"{index+1}/{len(metadata)} Done")

    time.sleep(0.2)

df = pd.DataFrame(weather_data)

df.to_csv(

    "data/prl/metadata/weather_data.csv",

    index=False

)

print("\nWeather Dataset Created Successfully!")