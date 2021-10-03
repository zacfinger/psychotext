import time
import numpy as np
import pandas as pd
from datetime import datetime as dt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression


WORKING_HOURS_PER_YEAR = 2000 # possibly innaccurate for observations before the labor movement
atof = lambda x: float(x.replace(",",""))


def toYearFraction(date_string):
    # https://stackoverflow.com/questions/6451655/how-to-convert-python-datetime-dates-to-decimal-float-years

    # Detect if date is just a year, return "middle of the year" if so
    # i.e., 1850 -> 1850.5
    if(len(date_string) == 4 and float(date_string) >= 1790):
        return float(date_string) + 0.5

    # Detect ISO date pattern
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


def foreignBornModel():
    # Get the data
    df = pd.read_csv('data/us.census.foreign.born.csv', converters={ 'Year': toYearFraction })

    # Get the dependent and independent variables
    # https://realpython.com/linear-regression-in-python/
    x = df['Year'].to_numpy().reshape(-1,1)
    y = df['Native'].to_numpy()

    # Create the model and find a linear regression curve
    model = LinearRegression()
    model.fit(x, y)
    model = LinearRegression().fit(x, y)

    # Calculate the Y values for the linear regression curve
    linear_y = model.predict(x)

    # Calculate the variance for the incomplete linear dataset 
    var = np.var(linear_y, ddof=1)

    # Find the average of the actual Y values
    avg = np.average(y)

    # Get the standardized variables for the dataset
    standardized_y = (y - avg) / var

    # TODO: need to potentially also return sinusoidal function for year out of range queries
    # Each model as a different function on the object
    return interp1d(x.flatten(), standardized_y, kind='cubic')


def relativeWageModel():
    # Get the data
    gdp_df = pd.read_csv('data/USGDP_1790-2021.csv', header=2, converters={ 1: atof })
    wage_df = pd.read_csv('data/USWAGE_1790-2021.csv', header=2)

    # Get the dependent and independent variables
    x = gdp_df['Year'].to_numpy().reshape(-1,1)
    g = gdp_df['Nominal GDP per capita (current dollars)'].to_numpy()
    W = wage_df['Production Workers Hourly Compensation (nominal dollars)'].to_numpy()

    # Calculate average wage relative to GDP per capita for each year
    y = (W * WORKING_HOURS_PER_YEAR)/g

    # Find the average of the actual Y values
    avg = np.average(y)

    standardized_y = (y - avg)

    print(standardized_y)

    # TODO: A lot of code is repeated here
    return interp1d(x.flatten(), standardized_y, kind='cubic')


def statureModel():
    # Get the data and filter other countries
    df = pd.read_csv('data/average-height-of-men-by-year-of-birth.csv', usecols=[1, 2, 3])
    df = df[df['Code'] == 'USA']
    df.drop(['Code'], axis=1, inplace=True)

    # Get dependent and independent variables
    x = df['Year'].to_numpy().reshape(-1,1)
    y = df['Height (Baten and Blum 2015)'].to_numpy()

    # Create the model and find a linear regression curve
    model = LinearRegression()
    model.fit(x, y)
    model = LinearRegression().fit(x, y)

    # Calculate the Y values for the linear regression curve
    linear_y = model.predict(x)

    # Calculate the variance for the incomplete linear dataset 
    var = np.var(linear_y, ddof=1)
    print(var)

    # Find the average of the actual Y values
    avg = np.average(y)
    print(avg)

    # Get the standardized variables for the dataset
    standardized_y = (y - avg) / var

    print(standardized_y)
    
    # TODO: Not the best because of mixed frequency time series in source data
    # Need to learn how to address
    # https://www.youtube.com/watch?v=E4NMZyfao2c
    # Lomb-Scargle periodogram
    return interp1d(x.flatten(), standardized_y, kind='cubic')

statureModel()
