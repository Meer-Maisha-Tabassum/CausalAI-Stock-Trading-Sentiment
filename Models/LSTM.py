import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense, Dropout
import plotly.graph_objs as go
import math
import json
import plotly 
from plotly.offline import plot



class lstm_model:
    def __init__(self):
        import tensorflow as tf
        tf.config.experimental_run_functions_eagerly(True)
    
    @staticmethod
    def lstm_algo(df, quote):
        print("IN lstm")
        # Split data into training set and test set
        dataset_train=df.iloc[0:int(0.8*len(df)),:]
        dataset_test=df.iloc[int(0.8*len(df)):,:]
        
        
        ############# NOTE #################
        # TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
        # HERE N=7
        training_set=df.iloc[:,4:5].values# 1:2, to store as numpy array else Series obj will be stored
        
        # Feature Scaling
        sc=MinMaxScaler(feature_range=(0,1))# Scaled values btween 0,1
        training_set_scaled=sc.fit_transform(training_set)
        # In scaling, fit_transform for training, transform for test
        
        # Creating data stucture with 7 timesteps and 1 output. 
        # 7 timesteps meaning storing trends from 7 days before current day to predict 1 next output
        X_train=[]# memory with 7 days from day i
        y_train=[]# day i
        for i in range(7,len(training_set_scaled)):
            X_train.append(training_set_scaled[i-7:i,0])
            y_train.append(training_set_scaled[i,0])
        # Convert list to numpy arrays
        X_train=np.array(X_train)
        y_train=np.array(y_train)
        X_forecast=np.array(X_train[-1,1:])
        X_forecast=np.append(X_forecast,y_train[-1])
        # Reshaping: Adding 3rd dimension
        X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))#.shape 0=row,1=col
        X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))

        # Initialise RNN
        regressor=Sequential()
        
        # Add first LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
        regressor.add(Dropout(0.1))
        
        # Add 2nd LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        
        # Add 3rd LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        
        # Add 4th LSTM layer
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))
        
        # Add o/p layer
        regressor.add(Dense(units=1))
        
        # Compile
        regressor.compile(optimizer='adam',loss='mean_squared_error')
        
        # Training
        regressor.fit(X_train,y_train,epochs=25,batch_size=32 )
        
        # Testing
        real_stock_price=dataset_test.iloc[:,4:5].values
        print("running1")
        
        # To predict, we need stock prices of 7 days before the test set
        # So combine train and test set to get the entire data set
        dataset_total=pd.concat((dataset_train['Close'],dataset_test['Close']),axis=0) 
        testing_set=dataset_total[len(dataset_total)-len(dataset_test)-7:].values
        testing_set=testing_set.reshape(-1,1)
        
        # Feature scaling
        testing_set=sc.transform(testing_set)
        
        # Create data structure
        X_test=[]
        for i in range(7,len(testing_set)):
            X_test.append(testing_set[i-7:i,0])
        # Convert list to numpy arrays
        X_test=np.array(X_test)
        
        # Reshaping: Adding 3rd dimension
        X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        print("running2")
        # Testing Prediction
        predicted_stock_price=regressor.predict(X_test)
        
        # Getting original prices back from scaled values
        predicted_stock_price=sc.inverse_transform(predicted_stock_price)
        print("running3")
        
        fig = go.Figure()

        # Add actual price trace
        fig.add_trace(go.Scatter(x=dataset_test['Date'], y=real_stock_price.flatten(), mode='lines', name='Actual Price'))

        # Add predicted price trace
        fig.add_trace(go.Scatter(x=dataset_test['Date'], y=predicted_stock_price.flatten(), mode='lines', name='Predicted Price'))

        # Update layout
        fig.update_layout(title='LSTM Prediction',
                          xaxis_title='Date',
                          yaxis_title='Price',
                          template='plotly_dark',  # Use dark theme for better appeal
                          xaxis=dict(tickformat='%b %d %Y', tickmode='linear', dtick=86400000.0*7, tickangle=-45),  # Rotate labels
                          legend=dict(x=0, y=1))
        
        lstm_graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        lstm_graph_html = plot(fig, output_type='div', include_plotlyjs=False)

        
        error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        
        # Forecasting Prediction
        forecasted_stock_price=regressor.predict(X_forecast)
        
        # Getting original prices back from scaled values
        forecasted_stock_price=sc.inverse_transform(forecasted_stock_price)
        lstm_pred=forecasted_stock_price[0,0]
        print()
        
        # Calculate MAE
        mae_lstm = mean_absolute_error(real_stock_price, predicted_stock_price)
        print("LSTM MAE:", mae_lstm)
        
        print("##############################################################################")
        print("Tomorrow's ",quote," Closing Price Prediction by LSTM: ",lstm_pred)
        print("LSTM RMSE:",error_lstm)
        print("##############################################################################")

        # Forecasting Prediction for next 7 days
        forecast_lstm = [lstm_pred]  # Start with tomorrow's prediction
        # Forecasting for 7 days
        for i in range(7):
            X_forecast = np.array(X_train[-1, 1:])
            X_forecast = np.append(X_forecast, y_train[-1])
            X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))
            forecasted_price = regressor.predict(X_forecast)
            # Invert scaling transformation
            forecasted_price_original = sc.inverse_transform(forecasted_price)
            forecast_lstm.append(forecasted_price_original[0, 0])
            # Updating X_train and y_train for the next prediction
            X_train = np.append(X_train, X_forecast, axis=0)
            y_train = np.append(y_train, forecasted_price)

        # Return the LSTM forecast
        return lstm_pred, mae_lstm, error_lstm, forecast_lstm[-7:], lstm_graph_json,lstm_graph_html
    
    
    



