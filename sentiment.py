import pandas as pd
import requests
import xmltodict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import data
from datetime import datetime, timedelta

analyzer = SentimentIntensityAnalyzer()

RSS_FEEDS = [
    "https://rss.dw.com/rdf/rss-de-all",
    "https://www.spiegel.de/international/germany/index.rss",
    "https://www.tagesschau.de/xml/rss2/"
]

def load_fictional_sentiment(filename="fictional_sentiment.csv"):
    """Load fictional sentiment data for future dates."""
    try:
        df = pd.read_csv(filename, parse_dates=['date'])
        df.set_index('date', inplace=True)
        return df['sentiment_category']
    except FileNotFoundError:
        print(f"{filename} not found. No fictional sentiment data available.")
        return pd.Series()

def fetch_rss_sentiment(prices):
    """
    Fetch RSS feeds for recent dates and use fictional sentiment for future dates.
    """
    start_date = prices.index.min()
    end_date = prices.index.max()
    today = pd.Timestamp.today().normalize()
    thirty_days_ago = today - pd.Timedelta(days=30)
    
    sentiment_categories = pd.Series(0, index=pd.date_range(start=start_date, end=end_date, freq='D'))
    
    # Load fictional sentiment for future dates
    fictional_sentiment = load_fictional_sentiment()
    for date in sentiment_categories.index:
        if date > today and date in fictional_sentiment.index:
            sentiment_categories[date] = fictional_sentiment[date]
    
    # Fetch real RSS feeds for recent past dates (within 30 days)
    if end_date >= thirty_days_ago:
        for feed in RSS_FEEDS:
            try:
                response = requests.get(feed, timeout=10)
                rss_data = xmltodict.parse(response.content)
                items = rss_data.get('rss', {}).get('channel', {}).get('item', [])
                if not isinstance(items, list):
                    items = [items] if items else []
                for item in items:
                    pub_date_str = item.get('pubDate', today.strftime('%Y-%m-%d'))
                    pub_date = pd.to_datetime(pub_date_str, utc=True).tz_convert(None).normalize()
                    if pub_date < start_date or pub_date > end_date or pub_date < thirty_days_ago:
                        continue
                    title = item.get('title', '')
                    desc = item.get('description', '') or ''
                    text = f"{title} {desc}"
                    score = analyzer.polarity_scores(text)['compound']
                    category = 1 if score > 0.05 else (-1 if score < -0.05 else 0)
                    if pub_date in sentiment_categories.index:
                        sentiment_categories[pub_date] += category
            except Exception as e:
                print(f"Error fetching {feed}: {e}")
    
    # Reindex to match prices index
    return sentiment_categories.reindex(prices.index, method='ffill').fillna(0)

def adjust_returns(returns, sentiment_categories, scale_factor=0.005):
    """
    Adjust returns based on categorized sentiment scores.
    """
    sentiment_impact = sentiment_categories * scale_factor
    adjusted_returns = returns + sentiment_impact.reindex(returns.index, fill_value=0)
    return adjusted_returns