# Experiment 16 – House Price Prediction System using Linear Regression

**Sample project · CO3/CO4**

## Aim
To build a complete house-price prediction system: data cleaning, feature engineering, linear regression, evaluation, and a web app with a JSON API.

## Dataset
California Housing: 20,640 districts. Features are median income, house age, rooms, bedrooms, population, occupancy, latitude and longitude. The target is median value in units of $100,000.

## Method
- **Cleaning:** remove rows where the census capped the target at $500k, and remove extreme ratio outliers.
- **Feature engineering** (`features.py`, part of the pipeline): rooms per person, bedroom ratio, log population, log income, distance to Los Angeles and San Francisco.
- **Models:** Linear Regression (final), plus Ridge, Lasso and Polynomial+Ridge to study regularization and model complexity. Evaluated with 5-fold CV.

## Run
```bash
pip install -r requirements.txt
python train.py
python app.py
```
Then open http://127.0.0.1:5000. For the API:
```
POST /api/predict {"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.9, "AveBedrms": 1.0, "Population": 322, "AveOccup": 2.6, "Latitude": 37.88, "Longitude": -122.23}
```

## Result
| Model | Test RMSE | MAE | R² |
|---|---|---|---|
| **Linear Regression** | 0.570 | 0.424 | **0.657** |
| Ridge | 0.570 | 0.424 | 0.658 |
| Lasso | 0.573 | 0.427 | 0.654 |
| Polynomial (deg 2) + Ridge | 0.508 | 0.365 | 0.729 |

Median income is the strongest driver. The polynomial model shows that some relationships are non-linear.
