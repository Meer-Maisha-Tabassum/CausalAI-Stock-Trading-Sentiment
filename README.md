-----

# Causal AI Driven Algorithmic Trading Strategy Optimization: A Deep Learning Approach

This project presents a sophisticated algorithmic trading framework that moves beyond traditional correlation-based analysis by integrating causal inference, deep learning, and Natural Language Processing (NLP). The system analyzes historical stock data, computes a wide range of technical indicators, assesses market sentiment from financial news, and uses a structural causal model to understand the underlying drivers of price movements. By combining these insights with predictive models like LSTMs, the project aims to generate more robust and explainable trading recommendations.

-----

## 📋 Key Features

  * **Causal Inference Engine**: Utilizes the `DoWhy` library to build a formal causal graph, identifying the true drivers of stock price changes among various factors.
  * **NLP-Powered Sentiment Analysis**: Scrapes real-time financial news and employs NLTK's VADER to quantify market sentiment, using it as a key variable in the decision-making process.
  * **Multi-Model Forecasting**: Implements three distinct predictive models for robust price forecasting:
      * **LSTM (Deep Learning)**: A Recurrent Neural Network to capture complex temporal patterns in stock data.
      * **ARIMA**: A classical statistical model for time-series forecasting.
      * **Linear Regression**: A baseline model using multiple technical indicators as features.
  * **Comprehensive Technical Analysis**: Automatically calculates a suite of over 10 technical indicators, including MACD, RSI, ADX, On-Balance Volume (OBV), Stochastic Oscillator, and more.
  * **Data-Driven Recommendations**: A final recommendation module synthesizes the outputs from predictive models and sentiment analysis to provide a clear BUY, SELL, or HOLD signal.
  * **Interactive Visualizations**: Generates dynamic and interactive charts using Plotly for model predictions, sentiment distribution, and causality analysis, which are embedded in a web interface.

-----

## 🛠️ Tech Stack & Libraries

  * **Backend Framework**: Flask (inferred from `main.py`)
  * **Data Manipulation & Analysis**: Pandas, NumPy
  * **Machine Learning / Deep Learning**: scikit-learn, TensorFlow
  * **Causal Inference**: `DoWhy`, `statsmodels`
  * **Natural Language Processing (NLP)**: `NLTK (VADER)`, `BeautifulSoup`
  * **Data Visualization**: Plotly, Matplotlib
  * **Data Acquisition**: `yfinance`, Alpha Vantage API

-----

## ⚙️ Methodology & Workflow

The project follows a multi-stage pipeline from data acquisition to final trading recommendation:

1.  **Data Acquisition**: Fetches historical daily stock data (Open, High, Low, Close, Volume) for a given ticker symbol using the `yfinance` library.
2.  **Feature Engineering**: Enriches the raw data by calculating a comprehensive set of technical indicators. Each indicator is designed to capture a different aspect of market momentum, volatility, or trend strength.
3.  **NLP Sentiment Analysis**:
      * Financial news headlines for the selected stock are scraped from Finviz using `BeautifulSoup`.
      * The headlines are processed using NLTK's VADER sentiment analyzer to generate a compound sentiment score, quantifying the overall market mood as a numerical value.
4.  **Causal Analysis**: This is the core of the project's innovation.
      * First, a **Granger Causality** test is performed to identify statistically significant relationships between the time series of different indicators and the closing price.
      * A **Structural Causal Model** is then formally defined using `DoWhy`. This model represents our hypothesis about how different variables influence each other. For instance, we can define RSI as a *treatment* variable, the 'Close' price as the *outcome*, and MACD as a *common cause* (confounder).
      * The model estimates the causal effect of the treatment on the outcome and runs refutation tests to validate the robustness of the causal claim.
5.  **Predictive Modeling**: The engineered features are fed into three parallel models to forecast future stock prices. Each model provides a unique perspective on the potential price movement.
6.  **Recommendation Synthesis**: The final module (`Recommendation.py`) integrates all the information. It considers the predicted price direction from the models along with the NLP sentiment score to generate a final, actionable trading decision.

-----

## 🗣️ Natural Language Processing (NLP) Integration

The NLP component is critical for incorporating the influence of public perception and news events into the trading strategy.

  * **Objective**: To quantify market sentiment from unstructured text data (news headlines) and use this sentiment as a causal and predictive variable.
  * **Tools & Process**:
    1.  The `News_Sentiment.py` script uses `BeautifulSoup` to scrape the latest news headlines from the financial analysis website Finviz.
    2.  It then leverages the **NLTK (Natural Language Toolkit)** library, specifically the **VADER (Valence Aware Dictionary and sEntiment Reasoner)** tool. VADER is highly effective for short, sentiment-rich texts like headlines.
    3.  VADER analyzes each headline and calculates a `compound` score ranging from -1 (most negative) to +1 (most positive). These scores are aggregated to produce a single, powerful metric representing the current news sentiment for the stock.
    4.  This sentiment score is then passed to the recommendation engine, which uses it to adjust the final trading signal. For example, a strong 'BUY' signal from the technical models might be downgraded to 'HOLD' if the news sentiment is overwhelmingly negative.

-----

## 🚀 Getting Started

### Prerequisites

  * Python 3.8+
  * Pip for package management

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-username/Causal-AI-Trading.git
    cd Causal-AI-Trading
    ```

2.  Create and activate a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  Execute the main application script:
    ```bash
    python main.py
    ```
2.  Open your web browser and navigate to `http://127.0.0.1:5000`.
3.  Enter a valid stock ticker symbol (e.g., 'MSFT', 'GOOGL') and submit to see the full analysis.

-----

## 📁 File Descriptions

| File                       | Description                                                                                                       |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `main.py`                  | The main Flask application file; serves the web interface and orchestrates the analysis pipeline.              |
| `Get_Data.py`              | Fetches historical stock data from online sources like `yfinance`.                                 |
| `Causal_Model.py`          | Implements the structural causal model using the `DoWhy` library to estimate and refute causal effects. |
| `Granger_Causation.py`     | Performs Granger Causality tests to identify potential predictive relationships between time series. |
| `News_Sentiment.py`        | Scrapes news headlines and performs sentiment analysis using NLTK VADER.                    |
| `LSTM.py`                  | Contains the implementation of the LSTM deep learning model for price prediction.                        |
| `Arima.py`                 | Implements the ARIMA time-series forecasting model.                                              |
| `Linear_Regression.py`     | Implements the Linear Regression model using technical indicators as features.           |
| `Recommendation.py`        | Synthesizes model predictions and sentiment scores to generate a final trading recommendation.  |
| `ADF_Test.py`              | Contains functions to perform the Augmented Dickey-Fuller test for stationarity.              |
| `ADLine.py`                | Calculates the Accumulation/Distribution Line (ADL) indicator.                                   |
| `ADX.py`                   | Calculates the Average Directional Index (ADX) indicator.                                           |
| `AroonOscillator.py`       | Calculates the Aroon Oscillator indicator.                                           |
| `MACD.py`                  | Calculates the Moving Average Convergence Divergence (MACD) indicator.                         |
| `OBV.py`                   | Calculates the On-Balance Volume (OBV) indicator.                                               |
| `RSI.py`                   | Calculates the Relative Strength Index (RSI) indicator.                                         |
| `StochasticOscillator.py`  | Calculates the Stochastic Oscillator (%K and %D) indicators.                |
| `MSFT.csv`                 | Sample CSV data file for Microsoft stock.                                                          |
| `defaultSymbol.csv`        | A CSV file containing a list of default stock symbols.                                         |

-----

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
