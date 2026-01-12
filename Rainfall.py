import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

df = pd.read_csv("rain_history.csv")

required_cols = ["city", "day", "month", "weekday", "hour", "rain"]
if not all(col in df.columns for col in required_cols):
    print("Dataset format incorrect")
    exit()

city_zone_map = {
    "chennai": "Coastal",
    "mumbai": "Coastal",
    "kolkata": "Coastal",
    "delhi": "Inland",
    "bangalore": "Plateau"
}

df["zone"] = df["city"].str.lower().map(city_zone_map).fillna("Inland")

zone_encoder = LabelEncoder()
df["zone_encoded"] = zone_encoder.fit_transform(df["zone"])

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

X = df[
    [
        "zone_encoded",
        "day",
        "month",
        "weekday",
        "is_monsoon",
        "hour_sin",
        "hour_cos"
    ]
]

y = df["rain"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=18,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy * 100:.2f}%")

def predict_rain(city, date_input, time_input, period):
    zone = city_zone_map.get(city.lower(), "Inland")
    zone_encoded = zone_encoder.transform([zone])[0]

    hour = int(time_input.split(":")[0])
    if period.lower() == "pm" and hour != 12:
        hour += 12
    if period.lower() == "am" and hour == 12:
        hour = 0

    dt = datetime.strptime(date_input, "%Y-%m-%d")

    features = pd.DataFrame(
        [[
            zone_encoded,
            dt.day,
            dt.month,
            dt.weekday(),
            1 if dt.month in [6, 7, 8, 9] else 0,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24)
        ]],
        columns=X.columns
    )

    prob = model.predict_proba(features)[0][1]

    if prob >= 0.5:
        print(f"🌧 Rain likely in {city} ({prob * 100:.1f}%)")
    else:
        print(f"☀ No rain expected in {city} ({prob * 100:.1f}%)")

city = input("Enter city name: ")
tomorrow = datetime.now() + timedelta(days=1)
predict_rain(city, tomorrow.strftime("%Y-%m-%d"), "02:00", "PM")

ask = input("Predict another date/time? (yes/no): ").lower()
if ask == "yes":
    c = input("City: ")
    d = input("Date (YYYY-MM-DD): ")
    t = input("Time (HH:MM): ")
    p = input("AM or PM: ")
    predict_rain(c, d, t, p)
