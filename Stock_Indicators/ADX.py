import numpy as np

@staticmethod
# Function to calculate Average Directional Index (ADX)
def calculate_adx(df, window=14):
    df['H-L'] = abs(df['High'] - df['Low'])
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['DMplus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), df['High'] - df['High'].shift(1), 0)
    df['DMminus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), df['Low'].shift(1) - df['Low'], 0)
    df['TRn'] = df['TR'].rolling(window=window).sum()
    df['DMplusn'] = df['DMplus'].rolling(window=window).sum()
    df['DMminusn'] = df['DMminus'].rolling(window=window).sum()
    df['DIplus'] = (df['DMplusn'] / df['TRn']) * 100
    df['DIminus'] = (df['DMminusn'] / df['TRn']) * 100
    df['DX'] = (abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus'])) * 100
    df['ADX'] = df['DX'].rolling(window=window).mean()
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR', 'DMplus', 'DMminus', 'TRn', 'DMplusn', 'DMminusn'], axis=1, inplace=True)
    return df