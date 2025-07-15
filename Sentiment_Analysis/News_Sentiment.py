# import plotly.graph_objs as go
# from urllib.request import urlopen, Request
# from bs4 import BeautifulSoup
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import plotly.graph_objects as go
# import json
# import plotly 

# @staticmethod
# def get_news_sentiment(symbol):
#     try:
#         finviz_url = 'https://finviz.com/quote.ashx?t='
#         # Define the URL for the stock symbol
#         url = finviz_url + symbol

#         # Set up the request headers
#         req = Request(url=url, headers={'user-agent': 'my-app'})

#         # Open the URL and read the page content
#         response = urlopen(req).read()

#         # Parse the HTML content using BeautifulSoup
#         soup = BeautifulSoup(response, 'html.parser')

#         # Extract news headlines from the HTML
#         news_table = soup.find(id='news-table')
#         news_list = news_table.find_all('tr')

#         # Combine all headlines into a single string
#         headlines = ' '.join([row.a.text for row in news_list])

#         # Initialize VADER sentiment analyzer
#         sid = SentimentIntensityAnalyzer()

#         # Analyze sentiment of the headlines
#         sentiment_scores = sid.polarity_scores(headlines)

#         # Extract the compound score (overall sentiment polarity)
#         compound_score = sentiment_scores['compound']

#         # Extract individual scores for visualization
#         positive_score = sentiment_scores['pos']
#         negative_score = sentiment_scores['neg']
#         neutral_score = sentiment_scores['neu']

#         # Create an interactive pie chart
#         fig = go.Figure(data=[go.Pie(labels=['Positive', 'Negative', 'Neutral'],
#                                     values=[positive_score, negative_score, neutral_score],
#                                     hole=0.3)])
#         fig.update_layout(title='Sentiment Distribution in News Headlines',
#                         template='plotly_dark')
        
#         sentiment_graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

#         # Return the compound score as the overall sentiment polarity and the Plotly figure
#         return compound_score, sentiment_graph_json
    
#     except Exception as e:
#         print("An error occurred while fetching news sentiment for {}: {}".format(symbol, e))
#         return None, None



# import plotly.graph_objs as go
# from urllib.request import urlopen, Request
# from bs4 import BeautifulSoup
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import json
# import plotly

# def get_news_sentiment(symbol):
#     try:
#         finviz_url = 'https://finviz.com/quote.ashx?t='
#         url = finviz_url + symbol
#         req = Request(url=url, headers={'user-agent': 'my-app'})
#         response = urlopen(req).read()
#         soup = BeautifulSoup(response, 'html.parser')
#         news_table = soup.find(id='news-table')
#         if not news_table:
#             raise ValueError(f"News table not found for symbol: {symbol}")

#         news_list = news_table.find_all('tr')
#         headlines = ' '.join([row.a.text for row in news_list if row.a])
#         if not headlines:
#             raise ValueError(f"No headlines found for symbol: {symbol}")

#         sid = SentimentIntensityAnalyzer()
#         sentiment_scores = sid.polarity_scores(headlines)
#         compound_score = sentiment_scores['compound']
#         positive_score = sentiment_scores['pos']
#         negative_score = sentiment_scores['neg']
#         neutral_score = sentiment_scores['neu']

#         fig = go.Figure(data=[go.Pie(labels=['Positive', 'Negative', 'Neutral'],
#                                      values=[positive_score, negative_score, neutral_score],
#                                      hole=0.3)])
#         fig.update_layout(title='Sentiment Distribution in News Headlines',
#                           template='plotly_dark')

#         sentiment_graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
#         return compound_score, sentiment_graph_json

#     except Exception as e:
#         print(f"An error occurred while fetching news sentiment for {symbol}: {e}")
#         # Return default values in case of error
#         return 0, None



import plotly.graph_objs as go
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import plotly
from plotly.offline import plot
def get_news_sentiment(symbol):
    try:
        finviz_url = 'https://finviz.com/quote.ashx?t='
        url = finviz_url + symbol
        req = Request(url=url, headers={'user-agent': 'my-app'})
        response = urlopen(req).read()
        soup = BeautifulSoup(response, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table:
            raise ValueError(f"News table not found for symbol: {symbol}")

        news_list = news_table.find_all('tr')
        news_headlines = [row.a.text for row in news_list if row.a][:10]  # Limit to top 10 headlines
        headlines = ' '.join(news_headlines)
        if not headlines:
            raise ValueError(f"No headlines found for symbol: {symbol}")

        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(headlines)
        compound_score = sentiment_scores['compound']
        positive_score = sentiment_scores['pos']
        negative_score = sentiment_scores['neg']
        neutral_score = sentiment_scores['neu']

        fig = go.Figure(data=[go.Pie(labels=['Positive', 'Negative', 'Neutral'],
                                     values=[positive_score, negative_score, neutral_score],
                                     hole=0.3)])
        fig.update_traces(marker=dict(colors=['#4caf50', '#ff5722', '#4680ff']))  # Green for Positive, Red for Negative, Blue for Neutral
        fig.update_layout(title='Sentiment Distribution in News Headlines',
                          template='plotly_dark')

        sentiment_graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        sentiment_graph_html = plot(fig, output_type='div', include_plotlyjs=False)

        return compound_score, sentiment_graph_json, news_headlines, sentiment_graph_html

    except Exception as e:
        print(f"An error occurred while fetching news sentiment for {symbol}: {e}")
        # Return default values in case of error
        return 0, None, []


