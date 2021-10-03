import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# Get the data
df = pd.read_csv('data/immigration.csv')

# Get the dependent and independent variables
x = df['Year'].to_numpy().reshape(-1,1)
y = df['Native'].to_numpy()

# Create the model and find a linear regression curve
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)

# Calculate the Y values for the curve
linear_y = model.predict(x)

# Find the average of the Y values for the linear curve
linear_avg = np.average(linear_y)

# Calculate the variance for the incomplete dataset
var = np.var(linear_y, ddof=1)

# Get the standardized variables for the dataset
standardized_y = (y - linear_avg) / var

print(standardized_y)


