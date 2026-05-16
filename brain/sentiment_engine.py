import requests
from transformers import pipeline

# --- 1. INITIALIZE THE NEURAL SENTIMENT ANALYZER ---
# Using a FinBERT model (specifically trained for financial language)
print(">>> Loading Financial NLP Model...")
sentiment_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def get_market_sentiment(pair="EURUSD"):
    # In a production environment, you would use a paid API like Bloomberg or Reuters.
    # For now, we will use a free News API or RSS feed.
    url = f"https://newsapi.org/v2/everything?q={pair}&apiKey=YOUR_API_KEY"
    
    # Mocking headlines for demonstration (Replace with actual API call)
    headlines = [
        "ECB signals potential rate cut as inflation cools",
        "Euro holds steady despite manufacturing slowdown",
        "Dollar strengthens on hawkish Fed comments"
    ]
    
    scores = []
    for text in headlines:
        result = sentiment_pipe(text)[0]
        # FinBERT labels: 'positive', 'negative', 'neutral'
        val = 1.0 if result['label'] == 'positive' else (-1.0 if result['label'] == 'negative' else 0)
        scores.append(val * result['score'])
        
    avg_score = sum(scores) / len(scores) if scores else 0
    return avg_score # Returns -1.0 (Very Bearish) to +1.0 (Very Bullish)