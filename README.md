# Rainfall Prediction using Machine Learning

This project predicts the probability of rainfall based on historical temporal data such as day, month, weekday, hour, and city. It uses a Random Forest machine learning model to classify whether rain is likely to occur at a given date and time.

## Project Overview

- City-based rainfall prediction
- Supports AM / PM time format
- Uses historical rainfall patterns
- Implements Random Forest classification
- Simple and stable implementation suitable for academic projects

## Dataset Description

The dataset (`rain_history.csv`) contains the following columns:

| Column   | Description |
|---------|-------------|
| city    | Name of the city |
| day     | Day of the month |
| month   | Month number (1–12) |
| weekday | Day of the week (0 = Monday) |
| hour    | Hour of the day (0–23) |
| rain    | Rain occurrence (1 = Rain, 0 = No Rain) |

## Technologies Used

- Python
- Pandas
- Scikit-learn

## How the Model Works

1. The dataset is loaded and validated
2. City names are encoded numerically
3. Features are split into training and testing sets
4. A Random Forest classifier is trained
5. The model predicts rainfall probability for user input

## How to Run the Project

1. Install dependencies:
pip install pandas scikit-learn
2. Run the program:
python rainfall.py
3. Enter city name, date, time, and AM/PM when prompted

## Sample Output
Rain likely in Chennai (63.4%)
No rain expected in Mumbai (41.2%)

## Model Accuracy

The model achieves an accuracy of approximately 65–75%, which is realistic for this dataset and feature set.

## Disclaimer

This project is intended for educational purposes only and is not a real meteorological forecasting system.

## Author

Academic machine learning project
