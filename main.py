import time
import numpy as np
import pandas as pd
from datetime import datetime as dt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression


def toYearFraction(date_string):
    # https://stackoverflow.com/questions/6451655/how-to-convert-python-datetime-dates-to-decimal-float-years

    date = dt.fromisoformat(date_string)

    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction


# Get the data
df = pd.read_csv('data/immigration.v2.csv', converters={ 'Year': toYearFraction })

# Get the dependent and independent variables
# https://realpython.com/linear-regression-in-python/
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

#print(standardized_y)

f2 = interp1d(x.flatten(), y, kind='cubic')
print(f2(1946.66))