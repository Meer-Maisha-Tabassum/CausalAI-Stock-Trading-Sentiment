# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import plotly.graph_objs as go
# import math
# import json
# import plotly 

# @staticmethod
# def LIN_REG_ALGO(df, quote):
#     # No of days to be forecasted in future
#     forecast_out = int(7)
#     # Price after n days
#     df['Close after n days'] = df['Close'].shift(-forecast_out)
#     # New df with only relevant data
#     df_new = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'OBV', 'ADL', 'ADX', 'Aroon Oscillator', 'MACD', 'RSI', '%K', '%D', 'Close after n days']]
#     df_new = df_new.dropna()

#     # Structure data for train, test & forecast
#     # labels of known data, discard last 35 rows
#     y = np.array(df_new.iloc[:-forecast_out,-1])
#     y=np.reshape(y, (-1,1))
#     # all cols of known data except labels, discard last 35 rows
#     X=np.array(df_new.iloc[:-forecast_out,0:-1])
#     # Unknown, X to be forecasted
#     X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])

#     # Training, testing to plot graphs, check accuracy
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#     # Feature Scaling===Normalization
#     sc = MinMaxScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)

#     X_to_be_forecasted = sc.transform(X_to_be_forecasted)

#     # Training
#     clf = LinearRegression(n_jobs=-1)
#     clf.fit(X_train, y_train)

#     print("Shape of X_train:", X_train.shape)

#     # Testing
#     y_test_pred = clf.predict(X_test)
#     y_test_pred=y_test_pred*(1.04)

#     fig = go.Figure()

#     # Add actual price trace
#     fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test.flatten(), mode='lines', name='Actual Price'))

#     # Add predicted price trace
#     fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test_pred.flatten(), mode='lines', name='Predicted Price'))

#     # Update layout
#     fig.update_layout(title='Linear Regression Prediction',
#                       xaxis_title='Index',
#                       yaxis_title='Price',
#                       template='plotly_dark',  # Use dark theme for better appeal
#                       legend=dict(x=0, y=1))

#     lr_graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

#     error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
    
#     # Calculate MAE
#     mae_lr = mean_absolute_error(y_test, y_test_pred)
#     print("Linear Regression MAE:", mae_lr)
    
#     # Forecasting
#     forecast_lr = clf.predict(X_to_be_forecasted)
#     forecast_lr=forecast_lr*(1.04)
#     mean=forecast_lr.mean()
#     lr_pred=forecast_lr[0,0]
#     print()
#     print("##############################################################################")
#     print("Tomorrow's ",quote," Closing Price Prediction by Linear Regression: ",lr_pred)
#     print("Linear Regression RMSE:",error_lr)
#     print("##############################################################################")

#     return df, lr_pred, mae_lr, forecast_lr, mean, error_lr, lr_graph_json


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objs as go
import math
import json
import plotly
from plotly.offline import plot

def LIN_REG_ALGO(df, quote):
    forecast_out = int(7)
    df['Close after n days'] = df['Close'].shift(-forecast_out)
    df_new = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'OBV', 'ADL', 'ADX', 'Aroon Oscillator', 'MACD', 'RSI', '%K', '%D', 'Close after n days']]
    df_new = df_new.dropna()

    y = np.array(df_new.iloc[:-forecast_out, -1])
    y = np.reshape(y, (-1, 1))
    X = np.array(df_new.iloc[:-forecast_out, 1:-1])
    X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 1:-1])

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, df_new.iloc[:-forecast_out, 0], test_size=0.2, shuffle=False)

    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_to_be_forecasted = sc.transform(X_to_be_forecasted)

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)

    print("Shape of X_train:", X_train.shape)

    y_test_pred = clf.predict(X_test)
    y_test_pred = y_test_pred * 1.04

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dates_test, y=y_test.flatten(), mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=dates_test, y=y_test_pred.flatten(), mode='lines', name='Predicted Price'))

    fig.update_layout(title='Linear Regression Prediction',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      template='plotly_dark',
                      xaxis=dict(tickformat='%b %d %Y', tickmode='linear', dtick=86400000.0*7, tickangle=-45),
                      legend=dict(x=0, y=1))

    lr_graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    lr_graph_html = plot(fig, output_type='div', include_plotlyjs=False)

    error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_lr = mean_absolute_error(y_test, y_test_pred)
    print("Linear Regression MAE:", mae_lr)

    forecast_lr = clf.predict(X_to_be_forecasted)
    forecast_lr = forecast_lr * 1.04
    mean = forecast_lr.mean()
    lr_pred = forecast_lr[0, 0]
    print()
    print("##############################################################################")
    print("Tomorrow's ", quote, " Closing Price Prediction by Linear Regression: ", lr_pred)
    print("Linear Regression RMSE:", error_lr)
    print("##############################################################################")

    return df, lr_pred, mae_lr, forecast_lr, mean, error_lr, lr_graph_json, lr_graph_html

