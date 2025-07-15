# import pandas as pd

# @staticmethod
# def recommending(df, global_polarity_tuple, today_stock, mean, quote):
#     # Extract the compound sentiment score from the tuple
#     global_polarity = global_polarity_tuple[0]
    
#     if isinstance(today_stock, dict):
#         today_stock = pd.DataFrame([today_stock])

#     if today_stock.iloc[-1]['Close'] < mean:
#         if global_polarity <= 0:
#             idea="RISE"
#             decision="BUY"
#             polarity="POSITIVE" 
#             print()
#             print("##############################################################################")
#             print("According to the ML Predictions and Sentiment Analysis of News, a",idea,"in",quote,"stock is expected => ",decision)
#         elif global_polarity > 0:
#             idea="FALL"
#             decision="SELL"
#             polarity="NEGETIVE"
#             print()
#             print("##############################################################################")
#             print("According to the ML Predictions and Sentiment Analysis of News, a",idea,"in",quote,"stock is expected => ",decision)
#     else:
#         idea="FALL"
#         decision="SELL"
#         polarity="NEGETIVE"
#         print()
#         print("##############################################################################")
#         print("According to the ML Predictions and Sentiment Analysis of, a",idea,"in",quote,"stock is expected => ",decision)
#     return idea, decision, polarity



import pandas as pd

def recommending(df, global_polarity_tuple, today_stock, mean, quote):
    # Extract the compound sentiment score from the tuple
    global_polarity = global_polarity_tuple[0]
    
    # Convert today_stock to DataFrame if it is a dictionary
    if isinstance(today_stock, dict):
        today_stock = pd.DataFrame([today_stock])
    
    # Initialize variables
    idea = ""
    decision = ""
    polarity = ""

    # Determine sentiment polarity based on the compound score
    if global_polarity > 0:
        polarity = "POSITIVE"
    elif global_polarity < 0:
        polarity = "NEGATIVE"
    else:
        polarity = "NEUTRAL"

    # Check the stock's closing price against the mean
    if today_stock.iloc[-1]['Close'] < mean:
        if polarity == "POSITIVE":
            idea = "RISE"
            decision = "BUY"
        else:
            idea = "FALL"
            decision = "SELL"
    else:
        if polarity == "POSITIVE":
            idea = "RISE"
            decision = "HOLD"
        else:
            idea = "FALL"
            decision = "SELL"
    
    # Log or print the recommendation (optional)
    print("##############################################################################")
    print(f"According to the ML Predictions and Sentiment Analysis of News, a {idea} in {quote} stock is expected => {decision}")
    
    # Return the recommendation
    return idea, decision, polarity
