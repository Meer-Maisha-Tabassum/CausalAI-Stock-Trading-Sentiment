@staticmethod
# Function to calculate Accumulation/distribution (A/D) line
def calculate_ad_line(df):
    df['ADL'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['ADL'] = df['ADL'].cumsum()
    return df