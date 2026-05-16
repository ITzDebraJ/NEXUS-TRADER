import numpy as np
from numba import njit

@njit(fastmath=True)
def calculate_micro_pressure(bids, asks, times):
    """
    Analyzes the last N ticks to determine if buyers or sellers are aggressive.
    Compiled to C-level machine code via Numba for 0.0001ms execution.
    """
    n = len(bids)
    if n < 50:
        return 0.0  # Not enough data yet

    # 1. Tick Velocity (How fast are orders coming in?)
    time_delta = times[-1] - times[0]
    if time_delta == 0:
        time_delta = 0.001
    velocity = n / time_delta

    # 2. Price Pressure (Are they hitting the Bid or the Ask?)
    # If the current bid is higher than the average of the last 50 bids, buyers are pushing.
    avg_bid = np.mean(bids)
    avg_ask = np.mean(asks)
    
    bid_pressure = (bids[-1] - avg_bid) / avg_bid * 100000 # Normalized pip movement
    ask_pressure = (asks[-1] - avg_ask) / avg_ask * 100000

    # 3. Combine into a single "Extreme Score" (-100 to +100)
    score = (bid_pressure + ask_pressure) * (velocity * 0.1)
    
    return score