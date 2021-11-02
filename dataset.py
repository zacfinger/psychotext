import numpy as np
import pandas as pd
from cliodynamics import Cliodynamic


WORKING_HOURS_PER_YEAR = 2000   # possibly innaccurate for observations before the labor movement
                                # could be made more dynamic: https://clockify.me/working-hours
atof = lambda x: float(x.replace(",",""))


def foreignBornModel():
    # Get the data
    df = pd.read_csv('data/us.census.foreign.born.csv', index_col = 'Year')

    # Set index to datetime
    df.index = pd.to_datetime(df.index, format='%Y')

    # Invert to measure native born population 
    df['Foreign-Born'] = 100 - df['Foreign-Born']

    c = Cliodynamic("Foreign born", df)

    return c


def relativeWage():
    # Get the data
    gdp_df = pd.read_csv('data/USGDP_1790-2021.csv', header=2, converters={ 1: atof }, index_col = 'Year')
    wage_df = pd.read_csv('data/USWAGE_1790-2021.csv', header=2, index_col = 'Year')

    # Calculate relative wage w = W / g
    # where W = Average annual salary and g = GDP per capita 
    w = (wage_df['Production Workers Hourly Compensation (nominal dollars)'] * WORKING_HOURS_PER_YEAR) / gdp_df['Nominal GDP per capita (current dollars)']

    # Set as new column on dataframe and remove unneeded columns
    wage_df['w'] = w
    wage_df.drop(['Production Workers Hourly Compensation (nominal dollars)'], axis=1, inplace=True)

    # Set index to datetime
    wage_df.index = pd.to_datetime(wage_df.index, format='%Y')

    c = Cliodynamic("Relative wage", wage_df)

    return c


def stature():
    # Get the data
    df = pd.read_csv('data/average-height-of-men-by-year-of-birth.csv', usecols=[1, 2, 3], index_col = 'Year')
    
    # Filter out other countries and remove unneeded columns
    df = df[df['Code'] == 'USA']
    df.drop(['Code'], axis=1, inplace=True)

    # Set index to datetime
    df.index = pd.to_datetime(df.index, format='%Y')

    c = Cliodynamic("Stature", df)

    return c


def lifeExpectancy():
    # Get the data
    df = pd.read_csv('data/life-expectation-at-birth-by-sex.csv', usecols=[1, 2, 3, 4], index_col = 'Year')

    # Set index to datetime
    df.index = pd.to_datetime(df.index, format='%Y')

    # Filter out other countries and remove unneeded columns
    df = df[df['Code'] == 'USA']
    df.drop(['Code'], axis=1, inplace=True)

    # Average male and female values
    df["Life expectancy"] = df.mean(axis=1)

    # Remove unneeded columns
    df.drop(['Female life expectancy at birth (HMD (2018) and others)'], axis=1, inplace=True)
    df.drop(['Male life expectancy at birth (HMD (2018) and others)'], axis=1, inplace=True)

    c = Cliodynamic("Life expectancy", df)

    return c


def marriageAge():
    # Get the data
    df = pd.read_csv('data/median-age-at-first-marriage-1890-to-present.csv', usecols=[0, 2], index_col = 'Year')

    # Set index to datetime
    df.index = pd.to_datetime(df.index, format='%Y')

    c = Cliodynamic("Marriage age", df, invert=True)

    return c


def tuition():
    # Get the data
    tuition_df = pd.read_csv('data/harvard-tuition.csv', index_col = 'Year')
    wage_df = pd.read_csv('data/USWAGE_1790-2021.csv', header=2, index_col = 'Year')

    # Set indices to datetime
    wage_df.index = pd.to_datetime(wage_df.index, format='%Y')
    tuition_df.index = pd.to_datetime(tuition_df.index, format='%Y')

    # Specify daterange for reindex
    idx = pd.date_range(tuition_df.index.values[0], tuition_df.index.values[-1], freq='YS')

    # Reindex to forward fill missing dates
    # https://stackoverflow.com/questions/19324453/add-missing-dates-to-pandas-dataframe
    tuition_df = tuition_df.reindex(idx, method='ffill')

    # Join dataframes
    df = wage_df.join(tuition_df, how='left')

    # Calculate years needed to earn tuition in terms of working class salary
    t = df['Tuition'] / (df['Production Workers Hourly Compensation (nominal dollars)'] * WORKING_HOURS_PER_YEAR)

    df['t'] = t

    # Drop unneeded columns
    df.drop(['Production Workers Hourly Compensation (nominal dollars)'], axis=1, inplace=True)
    df.drop(['Tuition'], axis=1, inplace=True)

    c = Cliodynamic("Tuition", df, invert=True)

    return c


def inequality():
    # Get the data
    df = pd.read_csv('data/inequality.index.peter.turchin.csv', usecols=[0, 4], index_col = 'Year')

    # Set indices to datetime
    df.index = pd.to_datetime(df.index, format='%Y')

    # Create column for logarithmic value
    i = np.log10(df['Inequality Index (Ratio x 1000)'])
    df['Inequality index (log-scale)'] = i

    # Drop unneeded columns
    df.drop(['Inequality Index (Ratio x 1000)'], axis=1, inplace=True)

    c = Cliodynamic("Inequality", df, invert=True)

    return c


def polarization():
    # Get the data
    df = pd.read_csv('data/voteview_polarization_data.csv', usecols=[0, 2, 3], index_col = 'year')

    # Set indices to datetime
    df.index = pd.to_datetime(df.index, format='%Y')

    # Sort by year and average polarization of both chambers
    df.sort_index(inplace=True)
    df = df.groupby(['year']).mean()

    c = Cliodynamic("Polarization", df, invert=True)

    return c