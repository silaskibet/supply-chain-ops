"""
supply_chain_utils.py
---------------------
Utility functions for KPC Supply Chain Optimization and Predictive Maintenance.
Includes modules for data generation, Prophet forecasting, inventory calculations,
and PuLP linear programming optimization.

Author: Silas Kibet
Team: NULL_TERMINATORS
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
import pulp

def generate_kpc_demand_data(start_date='2025-01-01', end_date='2025-12-31'):
    """
    Generates a synthetic daily demand dataset for KPC throughput including 
    trend, weekly/yearly seasonality, and holiday external regressors.
    """
    np.random.seed(42)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(date_range)

    trend = np.linspace(50000, 65000, n)
    weekly_seasonality = 8000 * np.sin(2 * np.pi * date_range.dayofweek / 7)
    yearly_seasonality = 12000 * np.cos(2 * np.pi * date_range.dayofyear / 365)
    noise = np.random.normal(0, 2500, n)

    # Peak season effects (April and December)
    holidays_effect = np.where((date_range.month == 12) | (date_range.month == 4), 15000, 0)

    demand = trend + weekly_seasonality + yearly_seasonality + holidays_effect + noise
    demand = np.maximum(demand, 10000)

    df = pd.DataFrame({
        'ds': date_range,
        'y': demand,
        'is_peak_season': np.where(holidays_effect > 0, 1, 0)
    })
    return df

def run_prophet_forecast(df_history, periods=60):
    """
    Trains a Prophet model with an external regressor and returns the forecast
    along with MAPE and RMSE evaluation metrics.
    """
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.add_regressor('is_peak_season')
    model.fit(df_history)

    future = model.make_future_dataframe(periods=periods)
    future['is_peak_season'] = np.where((future['ds'].dt.month == 12) | (future['ds'].dt.month == 4), 1, 0)
    
    forecast = model.predict(future)
    
    # Evaluate model
    train_eval = forecast.iloc[:len(df_history)]
    mape = mean_absolute_percentage_error(df_history['y'], train_eval['yhat'])
    rmse = root_mean_squared_error(df_history['y'], train_eval['yhat'])
    
    metrics = {'MAPE': mape, 'RMSE': rmse}
    
    return model, forecast, metrics

def calculate_inventory_parameters(df_history, forecast, lead_time_days=5, z_score=1.645):
    """
    Calculates Safety Stock and Reorder Point (ROP) based on forecast variance
    and a defined service level Z-score (default 1.645 for 95%).
    """
    # Calculate forecast standard deviation of residuals
    train_eval = forecast.iloc[:len(df_history)]
    forecast_residuals = df_history['y'] - train_eval['yhat']
    std_dev_demand = np.std(forecast_residuals)

    # Future demand average
    average_daily_demand = forecast['yhat'].tail(60).mean()
    
    # Calculations
    safety_stock = z_score * std_dev_demand * np.sqrt(lead_time_days)
    reorder_point = (average_daily_demand * lead_time_days) + safety_stock
    
    return {
        'average_daily_demand': average_daily_demand,
        'demand_std_dev': std_dev_demand,
        'safety_stock': safety_stock,
        'reorder_point': reorder_point
    }

def optimize_distribution_network():
    """
    Formulates and solves a Linear Programming distribution problem using PuLP.
    Minimizes transportation costs between KPC Terminals and Depots.
    """
    prob = pulp.LpProblem("KPC_Product_Distribution", pulp.LpMinimize)

    terminals = ['Mombasa_Terminal', 'Nairobi_Terminal']
    depots = ['Nakuru', 'Eldoret', 'Kisumu']

    supply = {'Mombasa_Terminal': 1200000, 'Nairobi_Terminal': 900000}
    demand_depots = {'Nakuru': 500000, 'Eldoret': 700000, 'Kisumu': 600000}

    costs = {
        ('Mombasa_Terminal', 'Nakuru'): 15,
        ('Mombasa_Terminal', 'Eldoret'): 22,
        ('Mombasa_Terminal', 'Kisumu'): 28,
        ('Nairobi_Terminal', 'Nakuru'): 8,
        ('Nairobi_Terminal', 'Eldoret'): 14,
        ('Nairobi_Terminal', 'Kisumu'): 18
    }

    # Decision variables
    route_vars = pulp.LpVariable.dicts("Route", ((t, d) for t in terminals for d in depots), lowBound=0, cat='Continuous')

    # Objective Function
    prob += pulp.lpSum(route_vars[t, d] * costs[t, d] for t in terminals for d in depots)

    # Constraints
    for t in terminals:
        prob += pulp.lpSum(route_vars[t, d] for d in depots) <= supply[t], f"Supply_{t}"
    for d in depots:
        prob += pulp.lpSum(route_vars[t, d] for t in terminals) >= demand_depots[d], f"Demand_{d}"

    prob.solve()
    
    results = {
        'status': pulp.LpStatus[prob.status],
        'total_cost': pulp.value(prob.objective),
        'routes': {v.name: v.varValue for v in prob.variables() if v.varValue > 0}
    }
    
    return results

if __name__ == "__main__":
    # Quick test run of the module
    print("Generating Data...")
    df = generate_kpc_demand_data()
    
    print("Running Forecast...")
    model, forecast, metrics = run_prophet_forecast(df)
    print(f"MAPE: {metrics['MAPE']:.2%}")
    
    print("Calculating Inventory Parameters...")
    inv_params = calculate_inventory_parameters(df, forecast)
    print(f"Safety Stock: {inv_params['safety_stock']:,.2f}")
    
    print("Running Linear Optimization...")
    lp_results = optimize_distribution_network()
    print(f"Optimization Status: {lp_results['status']} | Cost: KES {lp_results['total_cost']:,.2f}")