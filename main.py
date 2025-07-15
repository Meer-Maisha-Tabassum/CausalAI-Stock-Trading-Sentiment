#**************** IMPORT PACKAGES ********************
from flask import Flask, render_template, request, jsonify, send_from_directory
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf
import tweepy
import preprocessor as p
import re
from sklearn.linear_model import LinearRegression
from Data import Get_Data as data
from Stock_Indicators import OBV as obv
from Stock_Indicators import ADLine as ad_line
from Stock_Indicators import ADX as adx
from Stock_Indicators import AroonOscillator as aroon_oscillator
from Stock_Indicators import MACD as macd
from Stock_Indicators import RSI as rsi
from Stock_Indicators import StochasticOscillator as stochastic_oscillator
from Models import Arima as arima
from Models.LSTM import lstm_model
from Models import Linear_Regression as lin_reg
from Sentiment_Analysis import News_Sentiment as news_sentiment
from Sentiment_Analysis import Recommendation as recommendation
from Causal_AI import ADF_Test as adf
from Causal_AI import Granger_Causation as granger
from Causal_AI.Causal_Model import causal_model
import threading
import warnings
import os
from waitress import serve
import json

# Ignore Warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

global result

@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    symbol = request.args.get('symbol')
    print(symbol)
    if symbol:
        data = insertintotable(symbol)
        # print(data)
        return render_template('prediction.html', quote=data['quote'], open=data['open'], high=data['high'],
                                low=data['low'], close=data['close'], adj_close=data['adj_close'], volume=data['volume'],
                                forecast_set_arima=data['forecast_set_arima'], forecast_set_lstm=data['forecast_set_lstm'],
                                forecast_set_lr=data['forecast_set_lr'],trend_graph_json=data['trend_graph_json'], arima_graph_json=data['arima_graph_json'],
                                lstm_graph_json=data['lstm_graph_json'], lr_graph_json=data['lr_graph_json'], arima_pred=data['arima_pred'],
                                lstm_pred=data['lstm_pred'], lr_pred=data['lr_pred'], error_arima=data['error_arima'], error_lstm=data['error_lstm'],
                                error_lr=data['error_lr'], mae_arima=data['mae_arima'], mae_lstm=data['mae_lstm'], mae_lr=data['mae_lr'], trend_graph_html=data['trend_graph_html'],
                                arima_graph_html=data['arima_graph_html'], lstm_graph_html=data['lstm_graph_html'], lr_graph_html=data['lr_graph_html'])
    return render_template('prediction.html', quote=result['quote'], open=result['open'], high=result['high'],
                            low=result['low'], close=result['close'], adj_close=result['adj_close'], volume=result['volume'],
                            forecast_set_arima=result['forecast_set_arima'], forecast_set_lstm=result['forecast_set_lstm'],
                            forecast_set_lr=result['forecast_set_lr'],trend_graph_json=result['trend_graph_json'], arima_graph_json=result['arima_graph_json'],
                            lstm_graph_json=result['lstm_graph_json'], lr_graph_json=result['lr_graph_json'], arima_pred=result['arima_pred'],
                            lstm_pred=result['lstm_pred'], lr_pred=result['lr_pred'], error_arima=result['error_arima'], error_lstm=result['error_lstm'],
                            error_lr=result['error_lr'], mae_arima=result['mae_arima'], mae_lstm=result['mae_lstm'], mae_lr=result['mae_lr'], trend_graph_html=result['trend_graph_html'],
                            arima_graph_html=result['arima_graph_html'], lstm_graph_html=result['lstm_graph_html'], lr_graph_html=result['lr_graph_html'])



@app.route('/sentiment_analysis')
def sentiment_analysis():
    symbol = request.args.get('symbol')
    
    if symbol:
        data = insertintotable(symbol)
        print("HERE")
        return render_template('sentiment_analysis.html', quote=data['quote'], news_headlines=data['news_headlines'], polarity=data['polarity'],
                               idea=data['idea'], decision=data['decision'], sentiment_graph_html=data['sentiment_graph_html'], sentiment_graph_json=data['sentiment_graph_json'])
    return render_template('sentiment_analysis.html', quote=result['quote'], news_headlines=result['news_headlines'], polarity=result['polarity'],
                            idea=result['idea'], decision=result['decision'], sentiment_graph_html=result['sentiment_graph_html'], sentiment_graph_json=result['sentiment_graph_json'])

@app.route('/causality')
def causality():
    symbol = request.args.get('symbol')
    if symbol:
        data = insertintotable(symbol)
        return render_template('causality.html', quote=data['quote'], granger_results_html=data['granger_results_html'],
                               causal_estimate_regression=data['causal_estimate_regression'], causal_estimate_regression_pos_neg=data['causal_estimate_regression_pos_neg'],
                               causal_estimate_regression_increase_decrease=data['causal_estimate_regression_increase_decrease'], causal_estimate_iv=data['causal_estimate_iv'],
                               causal_estimate_iv_pos_neg=data['causal_estimate_iv_pos_neg'], causal_estimate_iv_increase_decrease=data['causal_estimate_iv_increase_decrease'],
                               granger_graph_json=data['granger_graph_json'], granger_graph_html=data['granger_graph_html'])                            
    return render_template('causality.html', quote=result['quote'], granger_results_html=result['granger_results_html'],
                               causal_estimate_regression=result['causal_estimate_regression'], causal_estimate_regression_pos_neg=result['causal_estimate_regression_pos_neg'],
                               causal_estimate_regression_increase_decrease=result['causal_estimate_regression_increase_decrease'], causal_estimate_iv=result['causal_estimate_iv'],
                               causal_estimate_iv_pos_neg=result['causal_estimate_iv_pos_neg'], causal_estimate_iv_increase_decrease=result['causal_estimate_iv_increase_decrease'],
                               granger_graph_json=result['granger_graph_json'], granger_graph_html=result['granger_graph_html'])

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

@app.route('/insertintotable', methods=['POST'])
def call_insertintotable():
    data = request.get_json()
    symbol = data['symbol']
    call_result = insertintotable(symbol)
    json_result = jsonify(call_result)
    return json_result

def convert_to_serializable(data):
    if isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(v) for v in data]
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    elif isinstance(data, (np.ndarray, pd.Series)):
        return data.tolist()
    elif isinstance(data, pd.DataFrame):
        return data.to_dict(orient='records')
    elif pd.isna(data):
        return None
    else:
        return data

def serialize_causal_estimate(estimate):
    if hasattr(estimate, 'get_confidence_intervals'):
        ci = estimate.get_confidence_intervals()
        return {
            'value': convert_to_serializable(estimate.value),
            'stderr': convert_to_serializable(estimate.get_standard_error()),
            'p_value': convert_to_serializable(estimate.test_stat_significance()),
            'ci_lower': convert_to_serializable(ci[0][0] if isinstance(ci, list) and len(ci) > 0 else None),
            'ci_upper': convert_to_serializable(ci[0][1] if isinstance(ci, list) and len(ci) > 0 else None),
            'refute_results': serialize_refute_results(estimate.refute_results) if hasattr(estimate, 'refute_results') else None
        }
    else:
        return {
            'refutation_type': estimate.refutation_type,
            'estimated_effect': convert_to_serializable(estimate.estimated_effect),
            'new_effect': convert_to_serializable(estimate.new_effect),
            'p_value': convert_to_serializable(getattr(estimate, 'p_value', None))
        }

def serialize_refute_results(refute_results):
    if refute_results is None:
        return None
    return {
        'estimated_effect': convert_to_serializable(refute_results.estimated_effect),
        'new_effect': convert_to_serializable(refute_results.new_effect),
        'p_value': convert_to_serializable(getattr(refute_results, 'p_value', None)),
        'refutation_type': refute_results.refutation_type
    }

def clean_data(data):
    # Replace infinite values with NaNs
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with NaNs
    data.dropna(inplace=True)
    
    return data

def insertintotable(nm):
    quote = nm
    try:
        data.get_historical(quote)
    except Exception as e:
        return {"error": "An error occurred while fetching historical data: {}".format(e)}

    df = pd.read_csv(f'{quote}.csv')
    today_stock = df.iloc[-1:].to_dict(orient='records')[0]
    df = df.dropna()
    df = obv.calculate_obv(df)
    df = ad_line.calculate_ad_line(df)
    df = adx.calculate_adx(df)
    df = aroon_oscillator.calculate_aroon_oscillator(df)
    df = macd.calculate_macd(df)
    df = rsi.calculate_rsi(df)
    df = stochastic_oscillator.calculate_stochastic_oscillator(df)

    code_list = [quote] * len(df)
    df2 = pd.DataFrame(code_list, columns=['Code'])
    df2 = pd.concat([df2, df], axis=1)
    df = df2

    try:
        arima_pred, mae_arima, error_arima, forecast_arima, arima_graph_json, trend_graph_json, trend_graph_html, arima_graph_html = arima.ARIMA_ALGO(df, quote)
        lstm = lstm_model()
        lstm_pred, mae_lstm, error_lstm, forecast_lstm, lstm_graph_json, lstm_graph_html = lstm.lstm_algo(df, quote)
        df, lr_pred, mae_lr, forecast_lr, mean, error_lr, lr_graph_json, lr_graph_html = lin_reg.LIN_REG_ALGO(df, quote)
    except Exception as e:
        return {"error": "An error occurred during the prediction process: {}".format(e)}

    compound_score, sentiment_graph_json, news_headlines, sentiment_graph_html = news_sentiment.get_news_sentiment(quote)
    global_polarity = news_sentiment.get_news_sentiment(quote)
    idea, decision, polarity = recommendation.recommending(df, global_polarity, today_stock, mean, quote)

    df = df.dropna()
    combined = pd.DataFrame()

    if adf.varify_adf_test(df['Close']) > 0.05:
        df_transformed_close = df['Close'].diff().dropna()
        adf.adf_test(df_transformed_close)
        if adf.varify_adf_test(df_transformed_close) < 0.05:
            combined = pd.concat([combined, df_transformed_close], axis=1)
    else:
        adf.adf_test(df['Close'])
        combined = pd.concat([combined, df['Close']], axis=1)

    if adf.varify_adf_test(df['RSI']) > 0.05:
        df_transformed_rsi = df['RSI'].diff().dropna()
        adf.adf_test(df_transformed_rsi)
        if adf.varify_adf_test(df_transformed_rsi) < 0.05:
            combined = pd.concat([combined, df_transformed_rsi], axis=1)
    else:
        adf.adf_test(df['RSI'])
        combined = pd.concat([combined, df['RSI']], axis=1)

    if adf.varify_adf_test(df['MACD']) > 0.05:
        df_transformed_macd = df['MACD'].diff().dropna()
        adf.adf_test(df_transformed_macd)
        if adf.varify_adf_test(df_transformed_macd) < 0.05:
            combined = pd.concat([combined, df_transformed_macd], axis=1)
    else:
        adf.adf_test(df['MACD'])
        combined = pd.concat([combined, df['MACD']], axis=1)

    if adf.varify_adf_test(df['Volume']) > 0.05:
        df_transformed_volume = df['Volume'].diff().dropna()
        adf.adf_test(df_transformed_volume)
        if adf.varify_adf_test(df_transformed_volume) < 0.05:
            combined = pd.concat([combined, df_transformed_volume], axis=1)
    else:
        adf.adf_test(df['Volume'])
        combined = pd.concat([combined, df['Volume']], axis=1)
        
    combined = combined.dropna()
    granger_results, granger_graph_json, granger_graph_html = granger.grangers_causation_matrix(combined, variables=combined.columns)
    granger_results_html = granger_results.to_html(classes='table table-bordered table-hover')
    
        
    if adf.varify_adf_test(df['Open']) > 0.05:
        df_transformed_volume = df['Open'].diff().dropna()
        adf.adf_test(df_transformed_volume)
        if adf.varify_adf_test(df_transformed_volume) < 0.05:
            combined = pd.concat([combined, df_transformed_volume], axis=1)
    else:
        adf.adf_test(df['Open'])
        combined = pd.concat([combined, df['Open']], axis=1)
        
    if adf.varify_adf_test(df['High']) > 0.05:
        df_transformed_volume = df['High'].diff().dropna()
        adf.adf_test(df_transformed_volume)
        if adf.varify_adf_test(df_transformed_volume) < 0.05:
            combined = pd.concat([combined, df_transformed_volume], axis=1)
    else:
        adf.adf_test(df['High'])
        combined = pd.concat([combined, df['High']], axis=1)
        
    if adf.varify_adf_test(df['Low']) > 0.05:
        df_transformed_volume = df['Low'].diff().dropna()
        adf.adf_test(df_transformed_volume)
        if adf.varify_adf_test(df_transformed_volume) < 0.05:
            combined = pd.concat([combined, df_transformed_volume], axis=1)
    else:
        adf.adf_test(df['Low'])
        combined = pd.concat([combined, df['Low']], axis=1)
    
    if adf.varify_adf_test(df['OBV']) > 0.05:
        df_transformed_volume = df['OBV'].diff().dropna()
        adf.adf_test(df_transformed_volume)
        if adf.varify_adf_test(df_transformed_volume) < 0.05:
            combined = pd.concat([combined, df_transformed_volume], axis=1)
    else:
        adf.adf_test(df['OBV'])
        combined = pd.concat([combined, df['OBV']], axis=1)
    
    if adf.varify_adf_test(df['ADX']) > 0.05:
        df_transformed_volume = df['ADX'].diff().dropna()
        adf.adf_test(df_transformed_volume)
        if adf.varify_adf_test(df_transformed_volume) < 0.05:
            combined = pd.concat([combined, df_transformed_volume], axis=1)
    else:
        adf.adf_test(df['ADX'])
        combined = pd.concat([combined, df['ADX']], axis=1)
        
    if adf.varify_adf_test(df['ADL']) > 0.05:
        df_transformed_volume = df['ADL'].diff().dropna()
        adf.adf_test(df_transformed_volume)
        if adf.varify_adf_test(df_transformed_volume) < 0.05:
            combined = pd.concat([combined, df_transformed_volume], axis=1)
    else:
        adf.adf_test(df['ADL'])
        combined = pd.concat([combined, df['ADL']], axis=1)

    
    combined = clean_data(combined)
    
    causal_analysis = causal_model(
        data=combined,
        treatment='RSI',
        outcome='Close',
        common_causes=['MACD','OBV','ADL', 'ADX'],
        instruments=['Volume']
    )
    
    causal_analysis.create_model()
    causal_model_image_path = causal_analysis.view_model()
    causal_analysis.identify_effect()
    causal_analysis.estimate_effects()
    causal_analysis.refute_estimates()
    causal_analysis.print_results()
    causal_estimate_regression = causal_analysis.get_estimate_regression()
    causal_estimate_iv = causal_analysis.get_estimate_iv()
    causal_estimate_regression_pos_neg = causal_analysis.get_pos_neg_lr()
    causal_estimate_regression_increase_decrease = causal_analysis.get_increase_decrease_lr()
    causal_estimate_iv_pos_neg = causal_analysis.get_pos_neg_iv()
    causal_estimate_iv_increase_decrease = causal_analysis.get_increase_decrease_iv()

    response = {
        "quote": convert_to_serializable(quote),
        "open": convert_to_serializable(round(float(today_stock['Open']), 2)),
        "close": convert_to_serializable(round(float(today_stock['Close']), 2)),
        "adj_close": convert_to_serializable(round(float(today_stock['Adj Close']), 2)),
        "high": convert_to_serializable(round(float(today_stock['High']), 2)),
        "low": convert_to_serializable(round(float(today_stock['Low']), 2)),
        "volume": convert_to_serializable(round(float(today_stock['Volume']), 2)),
        "arima_pred": convert_to_serializable(round(float(arima_pred), 2)),
        "lstm_pred": convert_to_serializable(round(float(lstm_pred), 2)),
        "lr_pred": convert_to_serializable(round(float(lr_pred), 2)),
        "error_arima": convert_to_serializable(round(float(error_arima), 2)),
        "error_lstm": convert_to_serializable(round(float(error_lstm), 2)),
        "error_lr": convert_to_serializable(round(float(error_lr), 2)),
        "mae_arima": convert_to_serializable(round(float(mae_arima), 2)),
        "mae_lstm": convert_to_serializable(round(float(mae_lstm), 2)),
        "mae_lr": convert_to_serializable(round(float(mae_lr), 2)),
        "forecast_set_arima": convert_to_serializable(forecast_arima),
        "forecast_set_lstm": convert_to_serializable(forecast_lstm),
        "forecast_set_lr": convert_to_serializable(forecast_lr),
        "granger_results_html": convert_to_serializable(granger_results_html),
        "news_headlines": convert_to_serializable(news_headlines),
        "idea": convert_to_serializable(idea),
        "decision": convert_to_serializable(decision),
        "arima_graph_json": arima_graph_json,
        "trend_graph_json": trend_graph_json,
        "lstm_graph_json": lstm_graph_json,
        "lr_graph_json": lr_graph_json,
        "granger_graph_json": granger_graph_json,
        "sentiment_graph_json": sentiment_graph_json,
        "trend_graph_html": trend_graph_html,
        "arima_graph_html": arima_graph_html,
        "lstm_graph_html": lstm_graph_html,
        "lr_graph_html": lr_graph_html,
        "granger_graph_html": granger_graph_html,
        "sentiment_graph_html": sentiment_graph_html,
        "causal_model_image": causal_model_image_path,
        "polarity" : convert_to_serializable(polarity),
        "causal_estimate_regression": convert_to_serializable(round(causal_estimate_regression, 3)),
        "causal_estimate_iv": convert_to_serializable(round(causal_estimate_iv, 3)),
        "causal_estimate_regression_pos_neg" : convert_to_serializable(causal_estimate_regression_pos_neg),
        "causal_estimate_regression_increase_decrease" : convert_to_serializable(causal_estimate_regression_increase_decrease),
        "causal_estimate_iv_pos_neg" : convert_to_serializable(causal_estimate_iv_pos_neg),
        "causal_estimate_iv_increase_decrease" : convert_to_serializable(causal_estimate_iv_increase_decrease)
    }
    global result 
    result = response
    # print(result)
    
    return response

if __name__ == '__main__':
    app.run()
    serve(app, host='0.0.0.0', port=5000)
