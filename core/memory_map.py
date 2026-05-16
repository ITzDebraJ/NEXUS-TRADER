import numpy as np
from multiprocessing import shared_memory

# Upgraded to handle multiple pairs simultaneously
MAX_TICKS = 50000 

# Map strings to integers for lightning-fast C-level processing
SYMBOL_MAP = {
    "EURUSD": 0,
    "GBPUSD": 1,
    "USDJPY": 2
}
REVERSE_MAP = {v: k for k, v in SYMBOL_MAP.items()}

# Added 'symbol_id' so the Brain knows which currency it's looking at
TICK_DTYPE = np.dtype([
    ('symbol_id', 'i4'), 
    ('time', 'f8'), 
    ('bid', 'f8'), 
    ('ask', 'f8'), 
    ('volume', 'f8'),
    ('last', 'f8')
])

class NexusMemory:
    def __init__(self, create=False):
        self.name = "nexus_shared_ticks_v2"
        self.size = MAX_TICKS * TICK_DTYPE.itemsize
        
        if create:
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            except FileExistsError:
                self.shm = shared_memory.SharedMemory(name=self.name)
        else:
            self.shm = shared_memory.SharedMemory(name=self.name)
            
        self.buffer = np.ndarray((MAX_TICKS,), dtype=TICK_DTYPE, buffer=self.shm.buf)

    def close(self):
        self.shm.close()