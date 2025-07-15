@staticmethod
# Function to calculate Moving Average Convergence Divergence (MACD)
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    exp1 = df['Close'].ewm(span=short_window, adjust=False).mean()
    exp2 = df['Close'].ewm(span=long_window, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    df['MACD'] = macd
    df['MACD Signal'] = signal
    df['MACD Histogram'] = macd - signal
    return df