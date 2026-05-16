import time
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import requests
from transformers import pipeline

# --- 1. THE V3 APEX ARCHITECTURE (Attention Mechanism) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NexusV3_Attention(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, num_layers=3):
        super(NexusV3_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.attention_weights = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x) 
        attn_scores = self.attention_weights(lstm_out) 
        attn_probs = torch.softmax(attn_scores, dim=1) 
        context_vector = torch.sum(attn_probs * lstm_out, dim=1) 
        return self.fc(context_vector)

# --- 2. LIVE TECHNICAL INDICATOR ENGINE ---
def add_technical_indicators(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['BB_Upper'] = sma20 + (std20 * 2)
    df['BB_Lower'] = sma20 - (std20 * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    return df

# --- 3. SENTIMENT GUARD (NLP VETO) ---
print(">>> Initializing Financial NLP Model on GPU...")
# FinBERT is the gold standard for financial sentiment
sentiment_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=0 if torch.cuda.is_available() else -1)

def get_market_sentiment(pair="EURUSD"):
    # Note: In a live environment, you would use a News API key here.
    # This mock mimics the structure of real institutional sentiment feeds.
    headlines = [
        f"Federal Reserve maintains hawkish tone, pressure on {pair} remains",
        f"Positive economic data from Eurozone supports {pair} recovery",
        f"Geopolitical tensions drive safe-haven demand for Dollar over Euro"
    ]
    
    try:
        results = sentiment_pipe(headlines)
        scores = []
        for res in results:
            # Convert FinBERT labels to numerical weights
            val = 1.0 if res['label'] == 'positive' else (-1.0 if res['label'] == 'negative' else 0)
            scores.append(val * res['score'])
        return sum(scores) / len(scores)
    except Exception as e:
        print(f"⚠️ Sentiment Engine Error: {e}")
        return 0 # Neutral default

# --- 4. MAIN LIVE INFERENCE LOOP ---
def run_live_inference():
    print(">>> Waking up V3 Apex AI Brain...")
    
    model = NexusV3_Attention().to(device)
    try:
        model.load_state_dict(torch.load("brain/weights/nexus_apex_120.pth", weights_only=True))
        model.eval()
        print("✅ Apex Weights loaded successfully.")
    except Exception as e:
        print(f"❌ Could not load weights: {e}")
        return

    if not mt5.initialize():
        print("❌ MT5 init failed.")
        return

    pair = "EURUSD"
    # These match your successful 60% ROI backtest thresholds
    buy_threshold = 0.50610
    sell_threshold = 0.49033
    
    # Track sentiment to avoid hitting APIs too frequently
    last_sentiment_check = 0
    current_mood = 0

    print(f"\n>>> NEXUS V3 APEX ONLINE. Monitoring {pair} Live...")

    try:
        while True:
            # 1. Grab 200 candles to ensure indicators have enough 'warm up' data
            rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_M15, 0, 200)
            if rates is None:
                time.sleep(1)
                continue
                
            df = pd.DataFrame(rates)
            df = add_technical_indicators(df)
            df.dropna(inplace=True)
            
            # 2. Slice the 120-candle window for the Attention Brain
            df_window = df.tail(120)
            feature_cols = ['open', 'high', 'low', 'close', 'tick_volume', 
                            'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'BB_Width']
            
            live_features = df_window[feature_cols].values
            
            # 3. Normalize for the GPU
            mean = np.mean(live_features, axis=0)
            std = np.std(live_features, axis=0) + 1e-8
            norm_live = (live_features - mean) / std
            
            X_tensor = torch.tensor(norm_live, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 4. PREDICT TECHNICAL PROBABILITY
            with torch.no_grad():
                raw_out = model(X_tensor)
                probability = torch.sigmoid(raw_out).item()
            
            # 5. SENTIMENT REFRESH (Every 15 minutes)
            if time.time() - last_sentiment_check > 900:
                current_mood = get_market_sentiment(pair)
                last_sentiment_check = time.time()

            current_price = df['close'].iloc[-1]
            status = "⏳ WAITING"
            
            # 6. APEX VETO LOGIC
            # We filter for technical probability + fundamental alignment
            if probability >= buy_threshold:
                if current_mood > -0.2: # Veto if headlines are aggressively bearish
                    status = "🟢 BUY SIGNAL"
                else:
                    status = "⚠️ VETOED (Bearish Sentiment)"
            elif probability <= sell_threshold:
                if current_mood < 0.2: # Veto if headlines are aggressively bullish
                    status = "🔴 SELL SIGNAL"
                else:
                    status = "⚠️ VETOED (Bullish Sentiment)"
            
            # Display output
            conf = probability if probability > 0.5 else (1 - probability)
            print(f"[{time.strftime('%H:%M:%S')}] Price: {current_price:.5f} | Mood: {current_mood:+.2f} | Prob: {probability:.5f} | {status} ({conf*100:.2f}%)")
            
            # Sleep until next check (approx 1 minute)
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n>>> Shutting down Nexus AI...")
        mt5.shutdown()

if __name__ == "__main__":
    run_live_inference()