import customtkinter as ctk
import sys
import os
import numpy as np
import socket

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.memory_map import NexusMemory, SYMBOL_MAP

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class NexusDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Nexus Extreme Radar")
        self.geometry("380x520") # Slightly taller to fit all the new data
        self.attributes("-topmost", True) 

        try:
            self.memory = NexusMemory(create=False)
            status_text = "● RAM LINKED [MULTI-PAIR]"
            status_color = "#2ECC71" 
        except FileNotFoundError:
            status_text = "● NO DATA"
            status_color = "#E74C3C" 
            self.memory = None

        # UDP Listener
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 9999))
        self.sock.setblocking(False)

        self.active_symbol = "SCANNING..."
        self.last_displayed_price = 0.0

        # --- UI Layout ---
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(self, text=status_text, text_color=status_color, font=("Consolas", 12, "bold"))
        self.status_label.grid(row=0, column=0, pady=(15,0), sticky="n")

        self.pair_label = ctk.CTkLabel(self, text="AWAITING SIGNAL", font=("Consolas", 24, "bold"), text_color="#F39C12")
        self.pair_label.grid(row=1, column=0, sticky="n")
        
        self.price_label = ctk.CTkLabel(self, text="---.-----", font=("Consolas", 54, "bold"))
        self.price_label.grid(row=1, column=0, pady=(40,0), sticky="n")

        self.signal_frame = ctk.CTkFrame(self, fg_color="#1E1E1E", corner_radius=10)
        self.signal_frame.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")
        
        self.signal_label = ctk.CTkLabel(self.signal_frame, text="ANALYZING MARKETS...", font=("Consolas", 24, "bold"), text_color="#555555")
        self.signal_label.pack(expand=True, pady=(20, 0))

        self.score_label = ctk.CTkLabel(self.signal_frame, text="Pressure: 0.00", font=("Consolas", 14), text_color="gray")
        self.score_label.pack(expand=True, pady=(0, 5))

        # The Actionable Instruction Text
        self.instruction_label = ctk.CTkLabel(self.signal_frame, text="Awaiting market anomaly...", font=("Consolas", 13, "bold"), text_color="#F39C12")
        self.instruction_label.pack(expand=True, pady=(0, 20))

        self.update_ui()

    def update_ui(self):
        # 1. Listen for Brain Signals
        try:
            data, _ = self.sock.recvfrom(1024)
            parts = data.decode('utf-8').split('|')
            
            if len(parts) == 6:
                symbol, signal, score, in_time, out_time, timeframe = parts
                
                self.active_symbol = symbol
                self.pair_label.configure(text=f"► {symbol} ◄")
                self.score_label.configure(text=f"Pressure: {score}")
                
                instruction_text = f"PLACE TRADE: {symbol}\nIN: {in_time} | OUT: {out_time}\nFRAME: {timeframe}"
                self.instruction_label.configure(text=instruction_text)

                if signal == "UP":
                    self.signal_frame.configure(fg_color="#1B5E20") 
                    self.signal_label.configure(text="⬆ GREEN (CALL)", text_color="#2ECC71")
                    self.instruction_label.configure(text_color="#FFFFFF") 
                elif signal == "DOWN":
                    self.signal_frame.configure(fg_color="#B71C1C") 
                    self.signal_label.configure(text="⬇ RED (PUT)", text_color="#E74C3C")
                    self.instruction_label.configure(text_color="#FFFFFF") 
                
        except BlockingIOError:
            pass 

        # 2. Update Live Price for active symbol
        if self.memory and self.active_symbol in SYMBOL_MAP:
            sym_id = SYMBOL_MAP[self.active_symbol]
            valid_ticks = self.memory.buffer[(self.memory.buffer['time'] > 0) & (self.memory.buffer['symbol_id'] == sym_id)]
            
            if len(valid_ticks) > 0:
                latest_tick = valid_ticks[-1]
                current_bid = latest_tick['bid']
                
                if current_bid != self.last_displayed_price:
                    self.price_label.configure(text=f"{current_bid:.5f}")
                    self.last_displayed_price = current_bid

        self.after(16, self.update_ui)

if __name__ == "__main__":
    app = NexusDashboard()
    app.mainloop()