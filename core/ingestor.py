import MetaTrader5 as mt5
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.memory_map import NexusMemory, MAX_TICKS, SYMBOL_MAP

ACTIVE_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]

def run_ingestor():
    print(">>> Initializing Multi-Asset MT5 Connection...")
    if not mt5.initialize():
        print("❌ MT5 init failed. Is MetaTrader 5 open?")
        return

    for sym in ACTIVE_SYMBOLS:
        if not mt5.symbol_select(sym, True):
            print(f"❌ Failed to activate {sym} in MT5 Market Watch.")

    memory = NexusMemory(create=True)
    print(f"🚀 V8 INGESTOR ONLINE: Blasting {len(ACTIVE_SYMBOLS)} pairs into RAM...")
    
    tick_index = 0
    last_times = {sym: 0 for sym in ACTIVE_SYMBOLS}

    try:
        while True:
            for sym in ACTIVE_SYMBOLS:
                tick = mt5.symbol_info_tick(sym)
                
                if tick and tick.time_msc != last_times[sym]:
                    last_times[sym] = tick.time_msc
                    
                    # Write to RAM with the correct symbol ID
                    memory.buffer[tick_index]['symbol_id'] = SYMBOL_MAP[sym]
                    memory.buffer[tick_index]['time'] = tick.time
                    memory.buffer[tick_index]['bid'] = tick.bid
                    memory.buffer[tick_index]['ask'] = tick.ask
                    memory.buffer[tick_index]['volume'] = tick.volume
                    
                    tick_index = (tick_index + 1) % MAX_TICKS
            
            # Sleep 1ms to prevent maxing out the CPU core
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n>>> Manual shutdown detected...")
    finally:
        memory.close()
        mt5.shutdown()
        print(">>> Ingestor Offline.")

if __name__ == "__main__":
    run_ingestor()