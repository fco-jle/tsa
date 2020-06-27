from statsmodels.tsa.holtwinters import ExponentialSmoothing
from utils import DataSamples, PlotStyle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
PlotStyle.cyber()


class HoltWintersModel:
    def __init__(self, time_series: pd.Series, test_size: float = None):
        self.ts = time_series
        self.name = time_series.name
        self.s_train = None
        self.s_test = None
        self.models = []
        if test_size is not None:
            self.train_test_split(test_size)

    def train_test_split(self, test_size=0.2):
        idx = int(len(self.ts) * (1 - test_size))
        self.s_train = self.ts.iloc[:idx].copy()
        self.s_test = self.ts.iloc[idx:].copy()

    def check_train_test_exist(self):
        if self.s_train is None or self.s_test is None:
            raise ValueError('Train and Test sets do not exist, please run train_test_split method.')

    def create_model(self, trend='mul', seasonal='mul', seasonal_periods=12, forecast_periods=36):
        self.check_train_test_exist()
        model_code = f'hw_{trend}_{seasonal}_{seasonal_periods}'
        model = ExponentialSmoothing(self.s_train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
        model = model.fit()
        forecast = model.forecast(forecast_periods)
        mae, mse, rmse = self.evaluate_forecast(self.s_test, forecast)
        model_data = {'name': 'holt_winters',
                      'code': model_code,
                      'trend': trend,
                      'seasonal': seasonal,
                      'seasonal_periods': seasonal_periods,
                      'model': model,
                      'forecast': forecast,
                      'mae': mae,
                      'mse': mse,
                      'rmse': rmse}
        self.models.append(model_data)

    @staticmethod
    def evaluate_forecast(y_true, y_pred):
        eval_df = pd.DataFrame(data=y_true).copy()
        eval_df.columns = ['true']
        eval_df['forecast'] = y_pred
        eval_df = eval_df.dropna()
        mae = mean_absolute_error(eval_df['true'], eval_df['forecast'])
        mse = mean_squared_error(eval_df['true'], eval_df['forecast'])
        rmse = np.sqrt(mse)
        return mae, mse, rmse

    def plot_models(self, plot_train_set=True, figsize=(16, 6), model_index=None):
        fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)

        if plot_train_set:
            ax.plot(self.s_train, label='Train')

        ax.plot(self.s_test, label='Test')

        if model_index is None:
            for m in self.models:
                ax.plot(m['forecast'], label=m["code"])
        else:
            ax.plot(self.models[model_index]['forecast'], label=self.models[model_index]['code'])

        ax.set_xlabel(self.ts.index.name)
        ax.set_ylabel(self.name)
        ax.set_title(f'Holt Winters - {self.name}')
        ax.legend()

    @property
    def models_summary(self):
        res = []
        for m in self.models:
            res.append((m['code'], m['rmse'], m['mae'], m['mse'], m['trend'], m['seasonal'], m['seasonal_periods']))
        res = pd.DataFrame(res, columns=['code', 'rmse', 'mae', 'mse', 'trend', 'seasonal', 'seasonal_periods'])
        return res

    def get_models(self):
        return self.models

    def get_best_model(self, metric='rmse'):
        assert metric in ['rmse', 'mse', 'mae']
        best_model = sorted(self.models, key=lambda x: x[metric])[0]
        print(f'Best model "{best_model["code"]}" with {metric} {best_model[metric]}')
        return best_model


if __name__ == '__main__':
    data = DataSamples.load('airline_passengers', index_col='Month', parse_dates=True).dropna()
    s = data['Thousands of Passengers']

    hwm = HoltWintersModel(s, test_size=0.2)
    hwm.create_model(trend='mul', seasonal='mul', seasonal_periods=12, forecast_periods=36)
    hwm.create_model(trend='add', seasonal='add', seasonal_periods=12, forecast_periods=36)
    hwm.create_model(trend='add', seasonal='mul', seasonal_periods=12, forecast_periods=36)
    hwm.create_model(trend='mul', seasonal='add', seasonal_periods=12, forecast_periods=36)
    hwm.plot_models(plot_train_set=True, model_index=None)
    bm = hwm.get_best_model()
