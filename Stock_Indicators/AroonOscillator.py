@staticmethod
# Function to calculate Aroon Oscillator
def calculate_aroon_oscillator(df, window=25):
    df['Aroon Up'] = df['High'].rolling(window=window).apply(lambda x: x.argmax()) / window * 100
    df['Aroon Down'] = df['Low'].rolling(window=window).apply(lambda x: x.argmin()) / window * 100
    df['Aroon Oscillator'] = df['Aroon Up'] - df['Aroon Down']
    df.drop(['Aroon Up', 'Aroon Down'], axis=1, inplace=True)
    return df