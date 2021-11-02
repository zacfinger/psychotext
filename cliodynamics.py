from datetime import datetime as dt
import time
from gatspy.periodic import LombScargle
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression


class Cliodynamic:
    AVG_DAYS_PER_YEAR = 365.2425  # Days in an average Gregorian year

    def __init__(self, name, df, normalize=True, invert=False):
        self._name              = name      # Title of dynamic
        
        self._df                = df        # Pandas DataFrame to store the 
                                            # time series and corresponding 
                                            # numeric values. Assumes two-column
                                            # DataFrame with a datetime 
                                            # index in column 0
        self._tt                = None      # Numpy arrays to store independent 
        self._yy                = None      # and dependent variables
        self._xx                = None      # Array to store datetimes as floats
        self._h                 = 5         # Timestep value
                                            # (Turchin, 2016)
        self._regular           = False     # Track whether data is missing.
                                            # Determines which fitting 
                                            # functions to use.
        self._normalize         = normalize # False if data is already 
                                            # normalized (i.e., relative wage)
        self._invert            = invert    # Invert the dataset (i.e., average
                                            # age at first marriage).
        
        # Estimation functions that are members of cliodynamic object
        self._interpolate       = None      # SciPy 1D interpolation function
        self._fitFunc           = None      # Sinusoidal regression (Unsym 2017)
        self._gatspyLombScargle = None      # Gatspy Lomb-Scargle implementation
        
        # Detect parameter type of x series 
        # Convert to datetime64['yyyy-MM'dd']
        #print(self._df)

        # Detect irregular time series and get ideal time intervals
        self._computeResampleRule()

        # If series is irregular, resample based on ideal time interval
        self._resample()
        
        # Extract tt and yy values to Numpy arrays
        self._dataFrameToNumpy()

        # Set model's time boundaries to max and min in dataset
        self._min_xx = self._xx.item(np.argmin(self._xx))
        self._max_xx = self._xx.item(np.argmax(self._xx))
        
        # Create standardized model
        self._standardize()


    def eval(self, t):
        """
        Returns the Y value for a given T value for the dynamic if it exists.
        For X values out of the model range, an extrapolated value is provided.
        """
        if t >= self._min_xx and t <= self._max_xx:
            return self._interpolate(t)

        if t < self._min_xx or t > self._max_xx:
            return self._fitFunc(t)


    def _fit_sin(self, tt, yy):
        # https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
        '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
        tt = np.array(tt)
        yy = np.array(yy)
        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
        guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
        guess_amp = np.std(yy) * 2.**0.5
        guess_offset = np.mean(yy)
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

        def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
        popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        f = w/(2.*np.pi)
        fitfunc = lambda t: A * np.sin(w*t + p) + c
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}


    def _toYearFraction(self, date):
        # https://stackoverflow.com/questions/6451655/how-to-convert-python-datetime-dates-to-decimal-float-years
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


    def _convertToDateTimeAndYearFraction(self, ts):
        date = dt.utcfromtimestamp(ts / (1000 * 1000 * 1000))
        return self._toYearFraction(date)
    

    def _computeResampleRule(self):
        """
        Determines if time series is irregular based on deltas between each row in DataFrame.
        If the series is irregular, an optimal resampling value is calculated and stored in self._h
        Assumes index of self._df are ascending and of dtype datetime.
        """
        # TODO: Possibly use largest difference between any two successive elements instead
        # https://stackoverflow.com/questions/33850086/calculating-the-maximum-difference-between-two-adjacent-numbers-in-an-array

        # Compute time deltas between each observation in dataframe and store in list
        # Reference: Nickil Maveli. https://stackoverflow.com/questions/16777570/calculate-time-difference-between-pandas-dataframe-indices
        tt = self._df.index.to_series().diff().dt.days.div(self.AVG_DAYS_PER_YEAR).values.tolist()
        del tt[0]               # Remove zero index of list with 'NaN' as value
        
        # Round all time deltas to nearest integer 
        #tt = [round(t) for t in tt]

        # Check if all elements in list are identical
        # Reference: kennytm. https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
        self._regular = tt[:-1] == tt[1:]
        
        if not self._regular:
            # Set h value by which series will be resampled, most common time difference
            # Reference: David Dao. https://stackoverflow.com/questions/10797819/finding-the-mode-of-a-list
            self._h = round(max(set(tt), key=tt.count))
    

    def _dataFrameToNumpy(self):
        # Extract x and y values
        tt = self._df.index.values.flatten()
        self._tt = tt
        self._yy = self._df.values.flatten()
        
        # Convert tt array to integer using toYearFraction
        vfunc = np.vectorize(self._convertToDateTimeAndYearFraction, otypes=[float])
        self._xx = vfunc(tt)
    

    def _resample(self):
        # https://stackoverflow.com/questions/25234941/python-regularise-irregular-time-series-with-linear-interpolation
        if not self._regular:
            self._df = self._df.resample('d').interpolate().resample(str(self._h) + 'YS').asfreq().dropna()


    def _standardize(self):
        """
        Standardize dataset according to Turchin's methodology.
        Create interpolation and sinusoidal regression functions on standardized data.
        """
        # Reshape time series to be two-dimensional
        # Ref: https://realpython.com/linear-regression-in-python/
        tt = self._xx.reshape(-1, 1)

        # Do not normalize on already normalized datasets
        # Potentially use https://stackoverflow.com/questions/45834276/numpyzero-mean-data-and-standardization
        if self._normalize:
            # Create the model and find a linear regression curve
            model = LinearRegression()
            model.fit(tt, self._yy)
            model = LinearRegression().fit(tt, self._yy)

            # Calculate the Y values for the linear regression curve
            linear_y = model.predict(tt)

            # Calculate the variance for the incomplete linear dataset 
            # TODO: Determine to use STD or VAR
            var = np.std(linear_y, ddof=1)

            # Find the average of the actual Y values
            avg = np.average(self._yy)

            # Get the standardized variables for the dataset
            # Ref: https://www.statology.org/detrend-data/
            standardized_y = (self._yy - avg) / var
        
            self._yy = standardized_y
        
        if self._invert:
            self._yy = -1 * self._yy

        # Compute the prediction functions
        self._interpolate = interp1d(self._xx, self._yy, kind='cubic')
        self._fitFunc = self._fit_sin(self._xx, self._yy)["fitfunc"]

        # calculate lomb-scargle best fit line
        ls = LombScargle().fit(self._xx, self._yy)
        ls.optimizer.period_range = (0.2, 1.3)
        self._gatspyLombScargle = ls.predict
    

    def toDataFrameWithDateTimeIndex(self):
        df = pd.DataFrame(data={'Year':self._tt, '{0} (standardized)'.format(self._name):self._yy})
        return df.set_index('Year')
    

    def toDataFrameWithIntegerIndex(self):
        df = pd.DataFrame(data={'Year':self._xx, '{0} (standardized)'.format(self._name):self._yy})
        return df.set_index('Year')