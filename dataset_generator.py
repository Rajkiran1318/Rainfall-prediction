import pandas as pd
import random
from datetime import datetime, timedelta

zones = {
    "Coastal": [6, 7, 8, 9],
    "Inland": [7, 8],
    "Plateau": [6, 7, 8]
}

rows = []
start = datetime(2000, 1, 1)
end = datetime(2024, 12, 31)

current = start
while current <= end:
    for zone, monsoon_months in zones.items():
        for hour in range(24):
            rain_prob = 0.15

            if current.month in monsoon_months:
                rain_prob += 0.35

            if hour in range(16, 22):
                rain_prob += 0.1

            rain = 1 if random.random() < rain_prob else 0

            rows.append([
                zone,
                current.day,
                current.month,
                current.weekday(),
                hour,
                1 if current.month in monsoon_months else 0,
                rain
            ])
    current += timedelta(days=1)

df = pd.DataFrame(rows, columns=[
    "zone", "day", "month", "weekday", "hour", "is_monsoon", "rain"
])

df.to_csv("rain_history.csv", index=False)
print("Dataset created")
