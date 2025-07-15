# import numpy as np
# from datetime import datetime
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import plotly.graph_objs as go
# import math
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# import json
# import plotly

# @staticmethod
# def ARIMA_ALGO(df, quote):
#     uniqueVals = df["Code"].unique()  
#     len(uniqueVals)
#     df=df.set_index("Code")
#     # for daily basis
#     def parser(x):
#         if isinstance(x, str):
#             return datetime.strptime(x, '%Y-%m-%d')
#         else:
#             # Handle non-string values (e.g., NaN) appropriately
#             return None
#     def arima_model(train, test):
#         history = [x for x in train]
#         predictions = list()
#         for t in range(len(test)):
#             model = ARIMA(history, order=(6,1 ,0))
#             model_fit = model.fit()
#             output = model_fit.forecast()
#             yhat = output[0]
#             predictions.append(yhat)
#             obs = test[t]
#             history.append(obs)
#         return predictions
#     for company in uniqueVals[:10]:
#         data=(df.loc[company,:]).reset_index()
#         data['Price'] = data['Close']
#         Quantity_date = data[['Price','Date']]
#         Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
#         Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
#         Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
#         Quantity_date = Quantity_date.drop(['Date'],axis =1)
        
#         traces = []
#         for col in Quantity_date.columns:
#             trace = go.Scatter(x=Quantity_date.index, y=Quantity_date[col], mode='lines', name=col)
#             traces.append(trace)

#         # Create the Plotly figure
#         fig_trend = go.Figure(traces)

#         # Update layout to make the graph interactive
#         fig_trend.update_layout(title='Trends',
#                         xaxis_title='Date',
#                         yaxis_title='Price',
#                         template='plotly_dark',  # Use dark theme for better appeal
#                         legend=dict(x=0, y=1))

#         trend_graph_json = json.dumps(fig_trend, cls=plotly.utils.PlotlyJSONEncoder)

#         quantity = Quantity_date.values
#         size = int(len(quantity) * 0.80)
#         train, test = quantity[0:size], quantity[size:len(quantity)]
#         # fit in model
#         predictions = arima_model(train, test)
        
#         # Plot graph using Plotly
#         fig_arima = go.Figure()

#         # Add actual price trace
#         fig_arima.add_trace(go.Scatter(x=np.arange(len(test)), y=test.flatten(), mode='lines', name='Actual Price'))

#         # Add predicted price trace
#         fig_arima.add_trace(go.Scatter(x=np.arange(len(test)), y=np.array(predictions).flatten(), mode='lines', name='Predicted Price'))

#         # Update layout
#         fig_arima.update_layout(title='ARIMA Prediction',
#                           xaxis_title='Date',
#                           yaxis_title='Price',
#                           template='plotly_dark',  # Use dark theme for better appeal
#                           legend=dict(x=0, y=1))
        
#         arima_graph_json = json.dumps(fig_arima, cls=plotly.utils.PlotlyJSONEncoder)
        
        
#         # Calculate MAE
#         mae_arima = mean_absolute_error(test, predictions)
#         print("ARIMA MAE:", mae_arima)

#         print()
#         print("##############################################################################")
#         arima_pred=predictions[-2]
#         print("Tomorrow's",quote," Closing Price Prediction by ARIMA:",arima_pred)
#         # rmse calculation
#         error_arima = math.sqrt(mean_squared_error(test, predictions))
#         print("ARIMA RMSE:",error_arima)
#         print("##############################################################################")
        
#         # Forecasting Prediction for next 7 days
    
#         forecast_arima = arima_model(train, test)
            
#         return arima_pred, mae_arima, error_arima, forecast_arima[-7:], arima_graph_json, trend_graph_json
    
    
    
import numpy as np
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objs as go
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import json
import plotly
from plotly.offline import plot

def ARIMA_ALGO(df, quote):
    uniqueVals = df["Code"].unique()
    df = df.set_index("Code")
    
    def parser(x):
        if isinstance(x, str):
            return datetime.strptime(x, '%Y-%m-%d')
        else:
            return None
    
    def arima_model(train, test):
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(6, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
        return predictions
    
    for company in uniqueVals[:10]:
        data = (df.loc[company, :]).reset_index()
        data['Price'] = data['Close']
        Quantity_date = data[['Price', 'Date']]
        Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
        Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
        Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
        Quantity_date = Quantity_date.drop(['Date'], axis=1)

        traces = []
        for col in Quantity_date.columns:
            trace = go.Scatter(x=Quantity_date.index, y=Quantity_date[col], mode='lines', name=col)
            traces.append(trace)

        fig_trend = go.Figure(traces)
        fig_trend.update_layout(title='Trends',
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark',
                                legend=dict(x=0, y=1))
        trend_graph_json = json.dumps(fig_trend, cls=plotly.utils.PlotlyJSONEncoder)
        trend_graph_html = plot(fig_trend, output_type='div', include_plotlyjs=False)

        quantity = Quantity_date.values
        size = int(len(quantity) * 0.80)
        train, test = quantity[0:size], quantity[size:len(quantity)]
        
        predictions = arima_model(train, test)

        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(x=data['Date'][size:], y=test.flatten(), mode='lines', name='Actual Price'))
        fig_arima.add_trace(go.Scatter(x=data['Date'][size:], y=np.array(predictions).flatten(), mode='lines', name='Predicted Price'))

        fig_arima.update_layout(title='ARIMA Prediction',
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark',
                                xaxis=dict(tickformat='%b %d %Y', tickmode='linear', dtick=86400000.0*7, tickangle=-45), # Weekly ticks
                                legend=dict(x=0, y=1))
        
        arima_graph_json = json.dumps(fig_arima, cls=plotly.utils.PlotlyJSONEncoder)
        arima_graph_html = plot(fig_arima, output_type='div', include_plotlyjs=False)
        
        mae_arima = mean_absolute_error(test, predictions)
        print("ARIMA MAE:", mae_arima)

        print("##############################################################################")
        arima_pred = predictions[-2]
        print("Tomorrow's", quote, " Closing Price Prediction by ARIMA:", arima_pred)
        
        error_arima = math.sqrt(mean_squared_error(test, predictions))
        print("ARIMA RMSE:", error_arima)
        print("##############################################################################")
        
        forecast_arima = arima_model(train, test)
        
        return arima_pred, mae_arima, error_arima, forecast_arima[-7:], arima_graph_json, trend_graph_json, trend_graph_html, arima_graph_html

