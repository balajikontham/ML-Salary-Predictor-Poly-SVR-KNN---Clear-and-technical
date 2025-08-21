# PolySalary Forecast

A machine learning project that predicts employee salaries based on position levels using multiple regression techniques. This tool helps visualize and forecast salary trends across different organizational levels using various machine learning models.

## Dataset
- The dataset `emp_sal.csv` contains employee salary information with position levels.
- Features: Position Level
- Target: Salary

## Dependencies
- Python 3.x
- NumPy
- Pandas
- scikit-learn
- Matplotlib

## Files
- `code.py`: Main Python script containing the implementation
- `emp_sal.csv`: Dataset file

## Implementation Details

### 1. Linear Regression
- Simple linear regression model to predict salary based on position level
- Visualizes the linear relationship between position level and salary
- Includes a prediction example for position level 6

### 2. Polynomial Regression
- Polynomial regression model with degree 7 for better curve fitting
- Compares the polynomial fit with the linear model
- Includes a prediction example for position level 7

### 3. Support Vector Regression (SVR)
- Implements SVR with default parameters
- Includes a prediction example for position level 6
- Configurable kernel and degree parameters (currently set to polynomial kernel with degree 5)

### 4. K-Nearest Neighbors (KNN) Regression
- KNN regressor implementation for salary prediction
- Flexible number of neighbors parameter (currently using default k=5)

## How to Run
1. Ensure all dependencies are installed
2. Place `emp_sal.csv` in the same directory as the script
3. Run `python code.py`

## Output
- Multiple plots will be generated:
  1. Linear Regression fit
  2. Polynomial Regression fit (degree 7)
- The console will display predicted salaries for different position levels using all implemented models

## Results
- The polynomial regression model provides a more accurate fit for the non-linear relationship between position level and salary compared to the simple linear regression model.
- SVR with polynomial kernel offers an alternative approach to model non-linear relationships
- KNN provides a non-parametric approach to the prediction task

## Note
- The polynomial degree (currently set to 7) can be adjusted in the code to observe different fitting behaviors.
