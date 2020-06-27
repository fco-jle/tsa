import pandas as pd
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from utils import DataSamples, PlotStyle
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


PlotStyle.cyber()


class TimeSeriesAnalyzer:
    def __init__(self, df):
        self._df = df

    @property
    def df(self):
        """DataFrame"""
        return self._df

    def get_columns(self, *keywords):
        """Find columns in the inner DataFrame based on teh provided keywords"""
        cols = self._df.columns
        for kw in keywords:
            cols = [c for c in cols if kw in c]
        return cols

    def sma(self, column, window, min_periods=None, center=False, win_type=None, on=None,
            axis=0, closed=None, fill_with_expanding=True):
        """Simple Moving Average"""
        c_name = f'{column}_sma_{window}'
        self.df[c_name] = self.df[column].rolling(window=window, min_periods=min_periods,
                                                  center=center, win_type=win_type, on=on,
                                                  axis=axis, closed=closed).mean()
        if fill_with_expanding:
            self.df['expand'] = self.df[column].expanding().mean()
            self.df.loc[self.df[c_name].isna(), c_name] = self.df.loc[self.df[c_name].isna(), 'expand']
            self.df.drop('expand', axis=1, inplace=True)

    def ewma(self, column, com=None, span=None, halflife=None, alpha=None, min_periods=0,
             adjust=True, ignore_na=False, axis=0):
        """Exponential Weighted Moving Average"""
        self.df[f'{column}_ewma_{span}'] = self.df[column].ewm(com=com, span=span, halflife=halflife,
                                                               alpha=alpha, min_periods=min_periods, adjust=adjust,
                                                               ignore_na=ignore_na, axis=axis).mean()

    def seasonal_decompose(self, column, append_to_df=False, **kwargs):
        """Basic seasonal decomposition into trend/seasonal/residual"""
        sd = seasonal_decompose(self.df[column], **kwargs)
        df = pd.DataFrame(data={f'sd_trend_{column}': sd.trend,
                                f'sd_seasonal_{column}': sd.seasonal,
                                f'sd_residual_{column}': sd.resid},
                          index=self.df.index)
        if append_to_df:
            self._df = pd.concat([self._df, df], axis=1)
        else:
            return df

    def hp_filter(self, column, lamb=1600):
        """Hodrick Prescott filter, splits series into cyclic and trend components"""
        cycle, trend = hpfilter(self.df[column], lamb=lamb)
        self.df[f'hp_cycle_{column}'] = cycle
        self.df[f'hp_trend_{column}'] = trend

    def holt_winters(self, column, trend=None, damped=False, seasonal=None, seasonal_periods=None,
                     dates=None, freq=None, smoothing_level=None):
        """Holt Winters Model"""

        x = self.df[column].astype(float)
        if trend is None:
            model = SimpleExpSmoothing(x)
            c_name = f'hw_simple_{round(smoothing_level,2)}_{column}'
            fitted_model = model.fit(smoothing_level=smoothing_level, optimized=True)

        else:
            model = ExponentialSmoothing(x, trend=trend, damped=damped, seasonal=seasonal,
                                         seasonal_periods=seasonal_periods, dates=dates, freq=freq)
            c_name = f'hw_exp_{trend}'
            if seasonal is not None:
                c_name += f'_{seasonal}'
            if seasonal_periods is not None:
                c_name += f'_{seasonal_periods}'
            c_name += f'_{column}'
            fitted_model = model.fit(smoothing_level=smoothing_level)

        self.df[c_name] = fitted_model.fittedvalues

    def lag_plot(self, column, lag):
        lag_plot(self.df[column], lag=lag)

    def autocorrelation_plot(self, column):
        autocorrelation_plot(self.df[column])

    def acf_plot(self, column, lags, **kwargs):
        plot_acf(self.df[column], lags=lags, **kwargs)

    def pacf_plot(self, column, lags, **kwargs):
        plot_pacf(self.df[column], lags=lags, **kwargs)


if __name__ == '__main__':
    win = 12
    alp = 2 / (win + 1)
    col = 'Thousands of Passengers'
    data = DataSamples.load('airline_passengers', index_col='Month', parse_dates=True)
    tsa = TimeSeriesAnalyzer(data)
    tsa.seasonal_decompose(col)
    tsa.sma(col, window=12)
    tsa.ewma(col, span=12)
    tsa.hp_filter(col, lamb=1600)
    tsa.holt_winters(col, smoothing_level=alp)
    tsa.holt_winters(col, trend='add')
    tsa.holt_winters(col, trend='mul')
    tsa.holt_winters(col, trend='mul', seasonal='mul', seasonal_periods=12)
    tsa.lag_plot(col, lag=24)
    tsa.autocorrelation_plot(col)
    tsa.acf_plot(col, lags=40)
    tsa.pacf_plot(col, lags=40)
