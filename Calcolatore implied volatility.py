# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:50:56 2024

@author: semer
"""

import numpy as np
import scipy.stats as stats
import yfinance as yf
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import mplcursors
from mplfinance.original_flavor import candlestick_ohlc
from scipy.optimize import minimize


def implied_volatility(option_type, market_price, S, K, T, r, q):
    def objective_function(sigma):
        if option_type == "call":
            return (black_scholes_call(S, K, T, r, q, sigma) - market_price)**2
        elif option_type == "put":
            return (black_scholes_put(S, K, T, r, q, sigma) - market_price)**2
    result = minimize(objective_function, x0=0.2, bounds=[(0.001, 3)])
    return result.x[0]

# Funzione per il calcolo dei prezzi delle opzioni
def input_float(entry):
    user_input = entry.get().replace(',', '.')
    try:
        return float(user_input)
    except ValueError:
        messagebox.showerror("Errore di input", "Inserisci un numero valido.")
        return None

def black_scholes_call(S, X, T, r, q, sigma):
    d1 = (np.log(S / X) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * stats.norm.cdf(d1) - X * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

def black_scholes_put(S, X, T, r, q, sigma):
    d1 = (np.log(S / X) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = X * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
    return put_price

def calculate_historical_volatility(ticker, start_date, end_date, T):
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')['Adj Close']
    log_returns = np.log(data / data.shift(1)).dropna()
    volatility = np.std(log_returns) * np.sqrt(252 * T)  # Annualizza la volatilità
    return volatility, data

def plot_price_history(ticker, period, title, candlestick=False):
    end_date = datetime.date.today()
    if period == '6m':
        start_date = end_date - datetime.timedelta(days=180)
    elif period == '1y':
        start_date = end_date - datetime.timedelta(days=365)
    elif period == '3y':
        start_date = end_date - datetime.timedelta(days=1095)

    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')

    figure = Figure(figsize=(9, 4), dpi=100)  # Riduzione delle dimensioni del grafico
    ax = figure.add_subplot(111)

    if candlestick:
        data['Date'] = mdates.date2num(data.index)
        ohlc = data[['Date', 'Open', 'High', 'Low', 'Close']]
        candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red')
        ax.xaxis_date()
        figure.autofmt_xdate()
    else:
        line, = ax.plot(data.index, data['Adj Close'], label="Price")
        mplcursors.cursor(line, hover=True).connect(
            "add", lambda sel: sel.annotation.set_text(
                f"{mdates.num2date(sel.target[0]).strftime('%Y-%m-%d')}\n{sel.target[1]:.2f} €/$"))

    ax.set_title(f"{title} ({ticker})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (€/$)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    figure.tight_layout()

    for widget in graph_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(figure, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def plot_daily_chart():
    ticker = ticker_entry.get().upper()
    data = yf.download(ticker, interval='1m', period='1d')

    if data.empty:
        messagebox.showerror("Error", "No data available for the specified ticker.")
        return

    figure = Figure(figsize=(9, 4), dpi=100)
    ax = figure.add_subplot(111)

    # Controlla se si vuole visualizzare il grafico a candele
    if candlestick_var.get():
        # Aggiungi colonna 'Date' per il formato candlestick
        data['Date'] = mdates.date2num(data.index)
        ohlc = data[['Date', 'Open', 'High', 'Low', 'Close']]

        # Usa il candlestick_ohlc per il grafico a candele
        candlestick_ohlc(ax, ohlc.values, width=0.005, colorup='green', colordown='red')
        ax.xaxis_date()
        figure.autofmt_xdate()

    else:
        # Grafico a linee
        line, = ax.plot(data.index, data['Adj Close'], label="Price")
        current_price = data['Adj Close'].iloc[-1]

        ax.annotate(f"Actual price: {current_price:.2f} €/$", xy=(data.index[-1], current_price),
                    xytext=(data.index[-1], current_price + 1),
                    arrowprops=dict(facecolor='black', shrink=0.05))

        mplcursors.cursor(line, hover=True).connect(
            "add", lambda sel: sel.annotation.set_text(
                f"{mdates.num2date(sel.target[0]).strftime('%Y-%m-%d %H:%M')}\n{sel.target[1]:.2f} €/$"))

    ax.set_title(f"Daily price ({ticker})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (€/$)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    figure.tight_layout()
    figure.autofmt_xdate()

    # Cancella il grafico precedente
    for widget in graph_frame.winfo_children():
        widget.destroy()

    # Visualizza il grafico aggiornato
    canvas = FigureCanvasTkAgg(figure, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def initialize_empty_graph():
    # Cancella il grafico precedente, se esiste
    for widget in graph_frame.winfo_children():
        widget.destroy()

    # Crea un nuovo grafico vuoto
    figure = Figure(figsize=(9, 4), dpi=100)
    ax = figure.add_subplot(111)
    ax.set_title("Stock price")
    ax.set_xlabel("Data")
    ax.set_ylabel("Price (€/$)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    figure.tight_layout()

    canvas = FigureCanvasTkAgg(figure, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def calculate_implied_volatility():
    ticker = ticker_entry.get().upper()
    option_type = option_type_entry.get().lower()
    S = yf.download(ticker, interval='1d')['Adj Close'].iloc[-1]
    K = input_float(strike_entry)
    T = input_float(maturity_entry)
    r = (yf.download("^IRX", interval='1d')['Adj Close'].iloc[-1]) / 100
    q = input_float(dividend_yield_entry)
    market_price = input_float(market_price_entry)

    if K is None or T is None or q is None or market_price is None:
        return

    implied_vol = implied_volatility(option_type, market_price, S, K, T, r, q)
    
    sigma, data = calculate_historical_volatility(ticker, start_date, end_date, T)

    result_label.config(text=f"Volatilità Implicita: {implied_vol:.4f}")
    current_price_label.config(text=f"{S:.4f} €/$")
    
    min_price = data.min()
    max_price = data.max()

    current_price_label.config(text=f"{S:.4f} €/$")
    min_price_label.config(text=f"{min_price:.4f} €/$")
    max_price_label.config(text=f"{max_price:.4f} €/$")
    risk_free_label.config(text=f"{r:.4f}")
    volatility_label.config(text=f"{sigma:.4f}")

    plot_price_history(ticker, '6m', "Stock price 6m")


def clean():
    ticker_entry.delete(0, tk.END)
    strike_entry.delete(0, tk.END)
    maturity_entry.delete(0, tk.END)
    dividend_yield_entry.delete(0, tk.END)
    market_price_entry.delete(0, tk.END)
    option_type_entry.delete(0, tk.END)

    result_label.config(text="")
    current_price_label.config(text="")
    min_price_label.config(text="")
    max_price_label.config(text="")
    risk_free_label.config(text="")
    volatility_label.config(text="")

    initialize_empty_graph()

def update_graph(period, label):
    ticker = ticker_entry.get().upper()
    plot_price_history(ticker, period, label, candlestick_var.get())

def update_financial_data():
    try:
        data = {
            "INDEX": {
                "S&P 500": "^GSPC",
                "DOW JONES": "^DJI",
                "NASDAQ": "^IXIC",
                "FTSE 100": "^FTSE",
                "CAC 40": "^FCHI",
                "DAX": "^GDAXI",
                "EURONEXT 100": "^N100",
                "HANG SENG": "^HSI",
                "NIKKEI 225": "^N225",
                "VIX": "^VIX"
            },
            "COMMODITIES": {
                "CRUDE OIL": "CL=F",
                "BRENT": "BZ=F",
                "NATURAL GAS": "NG=F",
                "GOLD": "GC=F",
                "WHEAT": "KE=F"
            },
            "CURRENCY": {
                "EUR/USD": "EURUSD=X",
                "EUR/CHF": "EURCHF=X",
                "EUR/GBP": "EURGBP=X",
                "GBP/USD": "GBPUSD=X",
                "USD/JPY": "JPY=X"
            }
        }

        for category, symbols in data.items():
            for label, symbol in symbols.items():
                df = yf.download(symbol, interval='1d', period="2d")
                if not df.empty:
                    current_price = df['Adj Close'].iloc[-1]
                    previous_close = df['Adj Close'].iloc[0]
                    daily_change = (current_price - previous_close) / previous_close * 100

                    if category == "INDEX":
                        index_labels[label].config(text=f"{current_price:.2f} ({daily_change:.2f}%)",
                                                   foreground="green" if daily_change >= 0 else "red")
                    elif category == "COMMODITIES":
                        commodity_labels[label].config(text=f"{current_price:.2f} $ ({daily_change:.2f}%)",
                                                       foreground="green" if daily_change >= 0 else "red")
                    elif category == "CURRENCY":
                        currency_labels[label].config(text=f"{current_price:.4f} € ({daily_change:.2f}%)",
                                                      foreground="green" if daily_change >= 0 else "red")
                else:
                    print(f"Errore durante il download dei dati per {label}. Nessun dato disponibile.")

        last_update_label.config(text=f"Last update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        messagebox.showerror("Errore", f"Errore durante l'aggiornamento dei dati finanziari: {e}")

    root.after(60000, update_financial_data)

# Imposta data di inizio e fine per la volatilità storica
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=180)

# Crea l'interfaccia principale
root = tk.Tk()
root.title("Option Calculator")

root.minsize(1200, 800)

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar_y = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

scrollbar_x = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=canvas.xview)
scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

scrollable_frame = tk.Frame(canvas)
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# Sezione input, centrata
input_frame = ttk.Frame(scrollable_frame, padding="10 20 10 10")
input_frame.grid(row=0, column=1, sticky="n")

# Dimensione dei campi input
ttk.Label(input_frame, text="Ticker:", font=('Helvetica', 13, 'bold')).grid(row=0, column=0, sticky="w")
ticker_entry = ttk.Entry(input_frame, font=('Helvetica', 13), width=15)
ticker_entry.grid(row=0, column=1, sticky="w")

ttk.Label(input_frame, text="Strike Price:", font=('Helvetica', 13, 'bold')).grid(row=1, column=0, sticky="w")
strike_entry = ttk.Entry(input_frame, font=('Helvetica', 13), width=15)
strike_entry.grid(row=1, column=1, sticky="w")

ttk.Label(input_frame, text="Maturity (in years):", font=('Helvetica', 13, 'bold')).grid(row=2, column=0, sticky="w")
maturity_entry = ttk.Entry(input_frame, font=('Helvetica', 13), width=15)
maturity_entry.grid(row=2, column=1, sticky="w")

ttk.Label(input_frame, text="Dividend Yield:", font=('Helvetica', 13, 'bold')).grid(row=3, column=0, sticky="w")
dividend_yield_entry = ttk.Entry(input_frame, font=('Helvetica', 13), width=15)
dividend_yield_entry.grid(row=3, column=1, sticky="w")

ttk.Label(input_frame, text="Market Price:", font=('Helvetica', 13, 'bold')).grid(row=4, column=0, sticky="w")
market_price_entry = ttk.Entry(input_frame, font=('Helvetica', 13), width=15)
market_price_entry.grid(row=4, column=1, sticky="w")

ttk.Label(input_frame, text="Option Type:", font=('Helvetica', 13, 'bold')).grid(row=5, column=0, sticky="w")
option_type_entry = ttk.Entry(input_frame, font=('Helvetica', 13), width=15)
option_type_entry.grid(row=5, column=1, sticky="w")

button_frame = ttk.Frame(input_frame, padding="10 10 10 10")
button_frame.grid(row=6, column=1, pady=10)

# Centratura dei pulsanti "Calcola Prezzo" e "Refresh"
ttk.Button(button_frame, text="Calculate", command=calculate_implied_volatility).pack(side=tk.LEFT, padx=20)
ttk.Button(button_frame, text="Clean", command=clean).pack(side=tk.LEFT, padx=20)

# Sezione risultati
result_frame = ttk.Frame(scrollable_frame, padding="8 8 8 8", relief="ridge")
result_frame.grid(row=5, column=1, sticky="nsew", padx=10, columnspan=2)

# Frame per evidenziare i risultati
highlight_frame = ttk.Frame(result_frame, padding="8 8 8 8", relief="ridge", style="Highlight.TFrame")
highlight_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=10)

# Stile per il frame di evidenziazione con sfondo nero
style = ttk.Style()
style.configure("Highlight.TFrame", background="black", relief="flat")
style.configure("Highlight.TLabel", background="black", foreground="orange", font=('Helvetica', 14, 'bold'))

# Etichette per Call Price e Put Price con lo stile Highlight.TLabel
ttk.Label(highlight_frame, text="", style="Highlight.TLabel").grid(row=0, column=0, sticky="w")
result_label = ttk.Label(highlight_frame, text="", style="Highlight.TLabel")
result_label.grid(row=0, column=1, sticky="e")


# Altre etichette di risultato
label_bold = {'font': ('Helvetica', 12, 'bold')}
ttk.Label(result_frame, text="Underlying:", **label_bold).grid(row=1, column=0, sticky="w")
current_price_label = ttk.Label(result_frame, text="")
current_price_label.grid(row=1, column=1, sticky="w")

ttk.Label(result_frame, text="Low 6 months:", **label_bold).grid(row=2, column=0, sticky="w")
min_price_label = ttk.Label(result_frame, text="")
min_price_label.grid(row=2, column=1, sticky="w")

ttk.Label(result_frame, text="High 6 months:", **label_bold).grid(row=3, column=0, sticky="w")
max_price_label = ttk.Label(result_frame, text="")
max_price_label.grid(row=3, column=1, sticky="w")

ttk.Label(result_frame, text="Risk free:", **label_bold).grid(row=4, column=0, sticky="w")
risk_free_label = ttk.Label(result_frame, text="")
risk_free_label.grid(row=4, column=1, sticky="w")

ttk.Label(result_frame, text="Historical Volatility 6 months:", **label_bold).grid(row=5, column=0, sticky="w")
volatility_label = ttk.Label(result_frame, text="")
volatility_label.grid(row=5, column=1, sticky="w")

graph_frame = ttk.Frame(result_frame, padding="10 10 10 10", relief="sunken")
graph_frame.grid(row=6, column=0, columnspan=2, sticky="nsew")

initialize_empty_graph()

graph_button_frame = ttk.Frame(scrollable_frame, padding="10 10 10 10")
graph_button_frame.grid(row=7, column=1, pady=10)

ttk.Button(graph_button_frame, text="Daily", command=plot_daily_chart).pack(side=tk.LEFT, padx=5)
ttk.Button(graph_button_frame, text="6 months", command=lambda: update_graph('6m', "Stock price - 6m")).pack(side=tk.LEFT, padx=5)
ttk.Button(graph_button_frame, text="1 year", command=lambda: update_graph('1y', "Stock price - 1y")).pack(side=tk.LEFT, padx=5)
ttk.Button(graph_button_frame, text="3 year", command=lambda: update_graph('3y', "Stock price - 3y")).pack(side=tk.LEFT, padx=5)

candlestick_var = tk.BooleanVar()
candlestick_checkbutton = ttk.Checkbutton(graph_button_frame, text="Candlestick chart", variable=candlestick_var, command=lambda: update_graph('6m', "Prezzo dell'azione negli ultimi 6 mesi"))
candlestick_checkbutton.pack(side=tk.LEFT, padx=5)

# Sezione dati finanziari
label_bold2 = {'font': ('Helvetica', 14, 'bold')}
label_bold3 = {'font': ('Helvetica', 12, 'bold')}
financial_data_frame = ttk.Frame(scrollable_frame, padding="10 80 10 10", relief="flat")
financial_data_frame.grid(row=0, column=3, rowspan=8, sticky="nsew", padx=80)

# Sezione indici
index_frame = ttk.Frame(financial_data_frame, padding="20 20 20 35", relief="flat")
index_frame.grid(row=0, column=0, sticky="nsew")
tk.Label(index_frame, text="INDEX", **label_bold2, anchor='center', bg='black', fg='white').grid(row=0, column=0, columnspan=2, sticky="ew")

index_labels = {}
for idx, (label, _) in enumerate({
    "S&P 500": "^GSPC",
    "DOW JONES": "^DJI",
    "NASDAQ": "^IXIC",
    "FTSE 100": "^FTSE",
    "CAC 40": "^FCHI",
    "DAX": "^GDAXI",
    "EURONEXT 100": "^N100",
    "HANG SENG": "^HSI",
    "NIKKEI 225": "^N225",
    "VIX": "^VIX"
}.items()):
    ttk.Label(index_frame, text=f"{label}:", **label_bold3).grid(row=idx + 1, column=0, sticky="ew", padx=5)
    index_labels[label] = ttk.Label(index_frame, text="", **label_bold3, anchor='center')
    index_labels[label].grid(row=idx + 1, column=1, sticky="ew", padx=5)

# Sezione materie prime
commodity_frame = ttk.Frame(financial_data_frame, padding="20 20 20 35", relief="flat")
commodity_frame.grid(row=1, column=0, sticky="nsew")
tk.Label(commodity_frame, text="COMMODITIES", **label_bold2, anchor='center', bg='black', fg='white').grid(row=0, column=0, columnspan=2, sticky="ew")

commodity_labels = {}
for idx, (label, _) in enumerate({
    "CRUDE OIL": "CL=F",
    "BRENT": "BZ=F",
    "NATURAL GAS": "NG=F",
    "GOLD": "GC=F",
    "WHEAT": "KE=F"
}.items()):
    ttk.Label(commodity_frame, text=f"{label}:", **label_bold3).grid(row=idx + 1, column=0, sticky="ew", padx=5)
    commodity_labels[label] = ttk.Label(commodity_frame, text="", **label_bold3, anchor='center')
    commodity_labels[label].grid(row=idx + 1, column=1, sticky="ew", padx=5)

# Sezione valute
currency_frame = ttk.Frame(financial_data_frame, padding="20 20 20 60", relief="flat")
currency_frame.grid(row=2, column=0, sticky="nsew")
tk.Label(currency_frame, text="CURRENCY", **label_bold2, anchor='center', bg='black', fg='white').grid(row=0, column=0, columnspan=2, sticky="ew")

currency_labels = {}
for idx, (label, _) in enumerate({
    "EUR/USD": "EURUSD=X",
    "EUR/CHF": "EURCHF=X",
    "EUR/GBP": "EURGBP=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X"
}.items()):
    ttk.Label(currency_frame, text=f"{label}:", **label_bold3).grid(row=idx + 1, column=0, sticky="ew", padx=5)
    currency_labels[label] = ttk.Label(currency_frame, text="", **label_bold3, anchor='center')
    currency_labels[label].grid(row=idx + 1, column=1, sticky="ew", padx=5)

# Pulsante per aggiornare e ultima riga (aggiornamento)
ttk.Button(financial_data_frame, text="Update", command=update_financial_data).grid(row=3, column=0, pady=10, sticky="ew")
last_update_label = ttk.Label(financial_data_frame, text="", anchor='center')
last_update_label.grid(row=4, column=0, columnspan=2, pady=5)

root.after(60000, update_financial_data)
update_financial_data()

root.mainloop()