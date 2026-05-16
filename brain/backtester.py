import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
import os

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

def run_apex_autocalibrated_backtest():
    print(">>> Initializing Apex Auto-Calibrator...")
    model = NexusV3_Attention().to(device)
    weights_path = "brain/weights/nexus_apex_120.pth"
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    data_path = "data_lake/mega_captures/EURUSD/*.parquet"
    df = pd.read_parquet(glob.glob(data_path)[0])
    
    # --- Indicator Engineering ---
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12, ema26 = df['close'].ewm(span=12).mean(), df['close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    sma20, std20 = df['close'].rolling(20).mean(), df['close'].rolling(20).std()
    df['BB_Upper'], df['BB_Lower'] = sma20 + (std20 * 2), sma20 - (std20 * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df.dropna(inplace=True)

    feature_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'BB_Width']
    raw_features = df[feature_cols].values
    norm_features = (raw_features - np.mean(raw_features, axis=0)) / (np.std(raw_features, axis=0) + 1e-8)

    # --- STEP 1: THE PRE-PASS (Finding the Confidence Spectrum) ---
    print(">>> Scanning market for confidence peaks...")
    all_probs = []
    for i in range(120, len(df), 10): # Sampling every 10th candle for speed
        window = torch.tensor(norm_features[i-120:i], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(window)).item()
            all_probs.append(prob)
    
    # Find the top 5% and bottom 5% thresholds dynamically
    buy_threshold = np.percentile(all_probs, 95)
    sell_threshold = np.percentile(all_probs, 5)
    
    print(f"✅ Calibration Complete.")
    print(f">>> Buy Threshold (95th Percentile): {buy_threshold:.5f}")
    print(f">>> Sell Threshold (5th Percentile): {sell_threshold:.5f}")

    # --- STEP 2: THE REAL BACKTEST ---
    balance = 10000.0
    initial_balance = balance
    risk_per_trade = 0.02 
    tp_pips, sl_pips = 60, 30 
    pip_size = 0.0001
    
    wins, losses = 0, 0
    in_trade = False
    
    print(f"\n>>> Running Backtest with Dynamic Thresholds...")

    for i in range(120, len(df)):
        current_price = df['close'].iloc[i]

        if in_trade:
            price_move = (current_price - entry_price) if trade_type == "BUY" else (entry_price - current_price)
            if price_move >= (tp_pips * pip_size):
                balance += (initial_balance * risk_per_trade * 2); wins += 1; in_trade = False
            elif price_move <= -(sl_pips * pip_size):
                balance -= (initial_balance * risk_per_trade); losses += 1; in_trade = False
            continue

        window = torch.tensor(norm_features[i-120:i], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(window)).item()

        if prob >= buy_threshold:
            in_trade, trade_type, entry_price = True, "BUY", current_price
        elif prob <= sell_threshold:
            in_trade, trade_type, entry_price = True, "SELL", current_price

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    roi = ((balance / initial_balance) - 1) * 100
    
    print("\n" + "="*45)
    print(f"      APEX AUTO-CALIBRATED RESULTS")
    print("="*45)
    print(f"Total Trades   : {total_trades}")
    print(f"Win Rate       : {win_rate:.2f}%")
    print(f"Final Balance  : ${balance:,.2f}")
    print(f"Total ROI      : {roi:.2f}%")
    print("="*45)

if __name__ == "__main__":
    run_apex_autocalibrated_backtest()