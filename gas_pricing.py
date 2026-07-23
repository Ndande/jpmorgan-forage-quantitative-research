import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

class NaturalGasPricer:
    """
    Estimates and forecasts natural gas prices using Meta's Prophet library.
    Supports price estimation for any date including interpolation and 
    24-month forward forecasting with seasonal pattern recognition.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.forecast_df = None
        self._prepare_data()
        self._train_model()
    
    def _prepare_data(self):
        df = pd.read_csv(self.data_path)
        df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
        df = df.sort_values(by='Dates', ascending=True).reset_index(drop=True)
        self.prophet_df = df.rename(columns={'Dates': 'ds', 'Prices': 'y'})
    
    def _train_model(self):
        self.model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
        self.model.fit(self.prophet_df)
    
    def get_price(self, date_str):
        date = pd.to_datetime(date_str)
        
        # Check if it's a weekend
        is_weekend = date.weekday() >= 5
        
        # Get forecast for that date
        future_date = pd.DataFrame({'ds': [date]})
        result = self.model.predict(future_date)
        price = result['yhat'].values[0]
        
        # Adjust for weekends slightly lower demand
        if is_weekend:
            price = price * 0.995
        
        return round(price, 2)
    
    def forecast(self, months=24):
        # --- FORECAST 24 MONTHS AHEAD ---
        future = self.model.make_future_dataframe(periods=24, freq='ME')
        forecast = self.model.predict(future)
        self.forecast_df = forecast

    def plot(self):
        # --- PLOT ---
        self.model.plot(self.forecast_df)
        plt.title('Natural Gas Price Forecast - 24 Months')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    pricer = NaturalGasPricer('Nat_Gas.csv')
    pricer.forecast()
    pricer.plot()
    
    test_dates = [
        '2021-06-15',
        '2022-01-10', 
        '2025-01-15',
        '2025-07-20',
        '2026-01-01',
    ]
    
    print("\n--- Price Estimates ---")
    for date in test_dates:
        price = pricer.get_price(date)
        day = pd.to_datetime(date).strftime('%A')
        print(f"{date} ({day}): ${price}")