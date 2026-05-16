import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> APEX ENLIGHTENMENT TRAINING: {device}")

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

def create_sequences(features, targets, seq_length=120): # DOUBLED CONTEXT
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        xs.append(features[i:(i + seq_length), :])
        ys.append(targets[i + seq_length])
    return np.array(xs), np.array(ys)

def train_model():
    model = NexusV3_Attention().to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    data_path = "data_lake/mega_captures/EURUSD/*.parquet"
    files = glob.glob(data_path)
    
    # 30-Hour context requires more epochs for the Attention to stabilize
    for epoch in range(25): 
        for file in files:
            df = pd.read_parquet(file)
            df = add_technical_indicators(df)
            df['Target'] = (df['close'].shift(-1) > df['close']).astype(float)
            df.dropna(inplace=True)
            
            feature_cols = ['open', 'high', 'low', 'close', 'tick_volume', 
                            'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'BB_Width']
            
            raw_features = df[feature_cols].values
            raw_targets = df['Target'].values
            
            mean, std = np.mean(raw_features, axis=0), np.std(raw_features, axis=0) + 1e-8 
            norm_features = (raw_features - mean) / std
            
            X, y = create_sequences(norm_features, raw_targets, seq_length=120)
            
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
            
            loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=512, shuffle=True)
            
            correct, total = 0, 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                acc_preds = (torch.sigmoid(predictions) > 0.5).float()
                correct += (acc_preds == batch_y).sum().item()
                total += batch_y.size(0)
            
            print(f"Epoch {epoch+1} | {file.split('/')[-1]} | Win Rate: {(correct/total)*100:.2f}%")

    os.makedirs("brain/weights", exist_ok=True)
    torch.save(model.state_dict(), "brain/weights/nexus_apex_120.pth")
    print("\n✅ APEX WEIGHTS SAVED: brain/weights/nexus_apex_120.pth")

if __name__ == "__main__":
    train_model()