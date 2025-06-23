
import yfinance as yf
import numpy as np
import scipy.stats as stats
import scipy.stats as si
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import mplcursors
from mplfinance.original_flavor import candlestick_ohlc
from scipy.optimize import minimize
from curl_cffi import requests as curl_requests


# =========================== #
# BLACK-SCHOLES PRICING MODEL #
# =========================== #

#Define the objective function to minimize the difference between the markte price and the Black-Scholes Price
def implied_volatility(option_type, market_price, S, K, T, r, q):
    def objective_function(sigma):
        if option_type == "call":
            return (black_scholes_call(S, K, T, r, q, sigma) - market_price)**2
        elif option_type == "put":
            return (black_scholes_put(S, K, T, r, q, sigma) - market_price)**2
    result = minimize(objective_function, x0=0.2, bounds=[(0.001, 3)])
    return result.x[0]

#Function to get the mandatory data to calculate implied volatility (calling previous function) 
def calculate_implied_volatility():
    ticker = ticker_entry.get().upper()
    option_type = option_type_entry.get().lower()
    S = yf.download(ticker, interval='1d')['Close'].iloc[-1].squeeze()
    K = input_float(strike_entry)
    T = input_float(maturity_entry)
    r = (yf.download("^IRX", interval='1d')['Close'].iloc[-1]).squeeze() / 100
    q = input_float(dividend_yield_entry)
    market_price = input_float(market_price_entry)

    if K is None or T is None or q is None or market_price is None:
        return

    implied_vol = implied_volatility(option_type, market_price, S, K, T, r, q)
    
    sigma, data = calculate_historical_volatility(ticker, start_date, end_date, T)

    result_label.config(text=f"VolatilitÃ  Implicita: {implied_vol:.4f}")
    current_price_label.config(text=f"{S:.4f} €/$")
    data = data.squeeze()
    sigma = sigma.squeeze()
    
    min_price = data.min()
    max_price = data.max()

    current_price_label.config(text=f"{S:.4f} €/$")
    min_price_label.config(text=f"{min_price:.4f} €/$")
    max_price_label.config(text=f"{max_price:.4f} €/$")
    risk_free_label.config(text=f"{r:.4f}")
    volatility_label.config(text=f"{sigma:.4f}")

    plot_price_history(ticker, '6m', "Stock price - 6m")
    
    greeks = black_scholes_greeks(S, K, T, r, sigma, option_type)
        # Aggiornamento delle etichette per le greche
    for greek, value in greeks.items():
        greeks_labels[greek].config(text=f"{value:.4f}")
        result_label.config(text=f"Implied Volatility: {implied_vol:.4f}")
        # Aggiornamento delle altre etichette
    

#Replace commas with dots and convert the input to a float
def input_float(entry):
    user_input = entry.get().replace(',', '.')
    try:
        return float(user_input)
    except ValueError:
        messagebox.showerror("Errore di input", "Inserisci un numero valido.")
        return None

#Calculate the price for a Call option using the Black-Scholes Formula
def black_scholes_call(S, X, T, r, q, sigma):
    d1 = (np.log(S / X) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * stats.norm.cdf(d1) - X * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

#Calculate the price for a Put option using the Black-Scholes Formula
def black_scholes_put(S, X, T, r, q, sigma):
    d1 = (np.log(S / X) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = X * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
    return put_price


#Calculate Option Greeks
def black_scholes_greeks(S, K, T, r, sigma, option_type):
    """Calculate option Greeks using the Black-Scholes model."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        delta = si.norm.cdf(d1)
        theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * si.norm.cdf(d2))
    elif option_type == "put":
        delta = -si.norm.cdf(-d1)
        theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * si.norm.cdf(-d2))
    else:
        raise ValueError("Invalid option type. Choose 'CALL' or 'PUT'")
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * (si.norm.cdf(d2) if option_type == "CALL" else -si.norm.cdf(-d2))
    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}

# ============================= #
# DATA RETRIVIAL AND PROCESSING #
# ============================= #

#Calculate the historical annual volatility for a given ticker
def calculate_historical_volatility(ticker, start_date, end_date, T):
    data = (yf.download(ticker, start=start_date, end=end_date, interval='1d').squeeze())['Close']
    log_returns = np.log(data / data.shift(1)).dropna()
    volatility = np.std(log_returns) * np.sqrt(252 * T)  # Annualizing the daily volatility
    return volatility, data

#Create a dictionary, show the current major financial assets prices and returns and update it every 60 seconds
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
                "EUR/CNH": "EURCNH=X",
                "EUR/JPY": "EURJPY=X",
                "USD/CHF": "CHF=X",
                "USD/JPY": "JPY=X",
                "USD/CNY": "USDCNY=X",
                "GBP/USD": "GBPUSD=X",
                "GBP/JPY": "GBPJPY=X"
            }
        }

        for category, symbols in data.items():
            for label, symbol in symbols.items():
                df = yf.download(symbol, interval='1d', period="5d")
                if not df.empty:
                    current_price = df['Close'].iloc[-1].squeeze()
                    previous_close = df['Close'].iloc[0].squeeze()
                    daily_change = (current_price - previous_close) / previous_close * 100

                    if category == "INDEX":
                        index_labels[label].config(text=f"{current_price:.2f} ({daily_change:.2f}%)",
                                                   foreground="green" if daily_change >= 0 else "red")
                    elif category == "COMMODITIES":
                        commodity_labels[label].config(text=f"{current_price:.2f} $ ({daily_change:.2f}%)",
                                                       foreground="green" if daily_change >= 0 else "red")
                    elif category == "CURRENCY":
                        currency_labels[label].config(text=f"{current_price:.4f} ({daily_change:.2f}%)",
                                                      foreground="green" if daily_change >= 0 else "red")
                else:
                    print(f"Errore durante il download dei dati per {label}. Nessun dato disponibile.")

        last_update_label.config(text=f"Last update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        messagebox.showerror("Errore", f"Errore durante l'aggiornamento dei dati finanziari: {e}")

    root.after(60000, update_financial_data)


# ====================== #
# GRAPH AND UI FUNCTIONS #
# ====================== #

#Function to create the historical price plot
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
        line, = ax.plot(data.index, data['Close'], label="Price")
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

#Function to plot di intraday price chart
def plot_daily_chart():
    ticker = ticker_entry.get().upper()
    data = yf.download(ticker, interval='1m', period='1d').squeeze()

    if data.empty:
        messagebox.showerror("Error", "No data available for the specified ticker.")
        return

    figure = Figure(figsize=(9, 4), dpi=100)
    ax = figure.add_subplot(111)

    # Check For Candlestick Bottom
    if candlestick_var.get():
        data['Date'] = mdates.date2num(data.index)
        ohlc = data[['Date', 'Open', 'High', 'Low', 'Close']]
        candlestick_ohlc(ax, ohlc.values, width=0.005, colorup='green', colordown='red')
        ax.xaxis_date()
        figure.autofmt_xdate()

    else:
        # Line chart
        line, = ax.plot(data.index, data['Close'], label="Price")
        current_price = data['Close'].iloc[-1]
        current_price = current_price.squeeze()

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

    # Delate the previous graph
    for widget in graph_frame.winfo_children():
        widget.destroy()

    # Show the updated chart
    canvas = FigureCanvasTkAgg(figure, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

#Function to create the initial empty graph at stat up
def initialize_empty_graph():
    # Delete the previous graph if it exist
    for widget in graph_frame.winfo_children():
        widget.destroy()

    # Create a new empty graph
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

#Clear the entries field for initial input
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
    # Clear the Greek labels
    for label in greeks_labels.values():
        label.config(text="")

    initialize_empty_graph()

#Update the graph calling the plot_price_history function
def update_graph(period, label):
    ticker = ticker_entry.get().upper()
    plot_price_history(ticker, period, label, candlestick_var.get())


# Set initial and final date to calculate historical volatility
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=180)

# ==================== #
#   CREATING THE  UI   #
# ==================== #

#Initialize main application window
root = tk.Tk()
root.title("Option Calculator")
root.minsize(1200, 800)

# Create a Notebook widget
notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# Create frames for each tab
tab1 = ttk.Frame(notebook)
#tab2 = ttk.Frame(notebook)

# Add tabs to the notebook
notebook.add(tab1, text="Implied Volatility")
#notebook.add(tab2, text="Pricing")

#Create a scrollable main frame		
main_frame = tk.Frame(tab1)
main_frame.pack(fill=tk.BOTH, expand=True)
canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

#Add vertical and horizontal scrollbars
scrollbar_y = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
scrollbar_x = tk.Scrollbar(tab1, orient=tk.HORIZONTAL, command=canvas.xview)
scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

#Create a scrollable frame inside the canvas
scrollable_frame = tk.Frame(canvas)
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")


# Centered Input Section
input_frame = ttk.Frame(scrollable_frame, padding="200 50 50 10")
input_frame.grid(row=0, column=1, sticky="n")

# Input field Size
ttk.Label(input_frame, text="Ticker:", font=('Helvetica', 12, 'bold')).grid(row=0, column=0, sticky="w")
ticker_entry = ttk.Entry(input_frame, font=('Helvetica', 12), width=15)
ticker_entry.grid(row=0, column=1, sticky="w")

ttk.Label(input_frame, text="Strike Price:", font=('Helvetica', 12, 'bold')).grid(row=1, column=0, sticky="w")
strike_entry = ttk.Entry(input_frame, font=('Helvetica', 12), width=15)
strike_entry.grid(row=1, column=1, sticky="w")

ttk.Label(input_frame, text="Maturity (in years):", font=('Helvetica', 12, 'bold')).grid(row=2, column=0, sticky="w")
maturity_entry = ttk.Entry(input_frame, font=('Helvetica', 12), width=15)
maturity_entry.grid(row=2, column=1, sticky="w")

ttk.Label(input_frame, text="Dividend Yield:", font=('Helvetica', 12, 'bold')).grid(row=3, column=0, sticky="w")
dividend_yield_entry = ttk.Entry(input_frame, font=('Helvetica', 12), width=15)
dividend_yield_entry.grid(row=3, column=1, sticky="w")

ttk.Label(input_frame, text="Market Price:", font=('Helvetica', 12, 'bold')).grid(row=4, column=0, sticky="w")
market_price_entry = ttk.Entry(input_frame, font=('Helvetica', 12), width=15)
market_price_entry.grid(row=4, column=1, sticky="w")

ttk.Label(input_frame, text="Option Type:", font=('Helvetica', 12, 'bold')).grid(row=5, column=0, sticky="w")
option_type_entry = ttk.Entry(input_frame, font=('Helvetica', 12), width=15)
option_type_entry.grid(row=5, column=1, sticky="w")

#Create action buttons
button_frame = ttk.Frame(input_frame, padding="4 4 4 4")
button_frame.grid(row=6, column=0, pady=5)

#Center buttons "Calculate" e "Clean"
ttk.Button(button_frame, text="Calculate", command=calculate_implied_volatility).pack(side=tk.LEFT, padx=20)
ttk.Button(button_frame, text="Clean", command=clean).pack(side=tk.LEFT, padx=5)

# Results section
result_frame = ttk.Frame(scrollable_frame, padding="5 5 5 5", relief="ridge")
result_frame.grid(row=4, column=1, sticky="nsew", padx=10, columnspan=6)

# Highlighted result frame
highlight_frame = ttk.Frame(result_frame, padding="8 8 8 8", relief="ridge", style="Highlight.TFrame")
highlight_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=10)
style = ttk.Style()
style.configure("Highlight.TFrame", background="black", relief="flat")
style.configure("Highlight.TLabel", background="black", foreground="orange", font=('Helvetica', 14, 'bold'))

# Define results labels
# Define styles
style = ttk.Style()
style.configure("TLabel", font=('Helvetica', 11))
style.configure("Value.TLabel", font=('Helvetica', 11, 'bold'), foreground="Black")
ttk.Label(highlight_frame, text="", style="Highlight.TLabel").grid(row=0, column=0, sticky="w")
result_label = ttk.Label(highlight_frame, text="", style="Highlight.TLabel")
result_label.grid(row=0, column=1, sticky="w")

label_bold = {'font': ('Helvetica', 12, 'bold')}
ttk.Label(result_frame, text="Underlying:", **label_bold).grid(row=1, column=0, sticky="w")
current_price_label = ttk.Label(result_frame, text="", style="Value.TLabel")
current_price_label.grid(row=1, column=0, sticky="n")

ttk.Label(result_frame, text="Low 6 months:", **label_bold).grid(row=2, column=0, sticky="w")
min_price_label = ttk.Label(result_frame, text="", style="Value.TLabel")
min_price_label.grid(row=2, column=0, sticky="n")

ttk.Label(result_frame, text="High 6 months:", **label_bold).grid(row=3, column=0, sticky="w")
max_price_label = ttk.Label(result_frame, text="", style="Value.TLabel")
max_price_label.grid(row=3, column=0, sticky="n")

ttk.Label(result_frame, text="Risk free:", **label_bold).grid(row=4, column=0, sticky="w")
risk_free_label = ttk.Label(result_frame, text="", style="Value.TLabel")
risk_free_label.grid(row=4, column=0, sticky="n")

ttk.Label(result_frame, text="AHV 6 months:", **label_bold).grid(row=5, column=0, sticky="w")
volatility_label = ttk.Label(result_frame, text="", style="Value.TLabel")
volatility_label.grid(row=5, column=0, sticky="n")

# Adding new labels for Greeks
greeks_labels = {}
greeks_info = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
label_bold = {'font': ('Helvetica', 12, 'bold')}
for idx, greek in enumerate(greeks_info):
    ttk.Label(result_frame, text=f"{greek}:", **label_bold).grid(row=1+idx, column=1, sticky="w")
    greeks_labels[greek] = ttk.Label(result_frame, text="", style="Value.TLabel")
    greeks_labels[greek].grid(row=1+idx, column=1, sticky="n")

# Graph section
graph_frame = ttk.Frame(result_frame, padding="10 10 10 10", relief="sunken")
graph_frame.grid(row=10, column=0, columnspan=2, sticky="nsew")

initialize_empty_graph()

graph_button_frame = ttk.Frame(scrollable_frame, padding="10 10 10 10")
graph_button_frame.grid(row=7, column=1, pady=10)

ttk.Button(graph_button_frame, text="Daily", command=plot_daily_chart).pack(side=tk.LEFT, padx=5)
ttk.Button(graph_button_frame, text="6 months", command=lambda: update_graph('6m', "Stock price - 6m")).pack(side=tk.LEFT, padx=5)
ttk.Button(graph_button_frame, text="1 year", command=lambda: update_graph('1y', "Stock price - 1y")).pack(side=tk.LEFT, padx=5)
ttk.Button(graph_button_frame, text="3 year", command=lambda: update_graph('3y', "Stock price - 3y")).pack(side=tk.LEFT, padx=5)

candlestick_var = tk.BooleanVar()
candlestick_checkbutton = ttk.Checkbutton(graph_button_frame, text="Candlestick chart", variable=candlestick_var, command=lambda: update_graph('6m', "Stock price - 6m"))
candlestick_checkbutton.pack(side=tk.LEFT, padx=5)


# Financial Section
label_bold2 = {'font': ('Helvetica', 14, 'bold')}
label_bold3 = {'font': ('Helvetica', 12, 'bold')}
financial_data_frame = ttk.Frame(scrollable_frame, padding="5 60 10 10", relief="flat")
financial_data_frame.grid(row=0, column=10, rowspan=15, sticky="nsew", padx=80)

# Index Section
index_frame = ttk.Frame(financial_data_frame, padding="5 20 20 35", relief="flat")
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

# Commodities Section
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

# Forex Section
currency_frame = ttk.Frame(financial_data_frame, padding="20 20 20 5", relief="flat")
currency_frame.grid(row=2, column=0, sticky="nsew")
tk.Label(currency_frame, text="CURRENCY", **label_bold2, anchor='center', bg='black', fg='white').grid(row=0, column=0, columnspan=2, sticky="ew")

currency_labels = {}
for idx, (label, _) in enumerate({
    "EUR/USD": "EURUSD=X",
    "EUR/CHF": "EURCHF=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/CNH": "EURCNH=X",
    "EUR/JPY": "EURJPY=X",
    "USD/CHF": "CHF=X",
    "USD/JPY": "JPY=X",
    "USD/CNY": "USDCNY=X",
    "GBP/USD": "GBPUSD=X",
    "GBP/JPY": "GBPJPY=X"
}.items()):
    ttk.Label(currency_frame, text=f"{label}:", **label_bold3).grid(row=idx + 1, column=0, sticky="ew", padx=5)
    currency_labels[label] = ttk.Label(currency_frame, text="", **label_bold3, anchor='center')
    currency_labels[label].grid(row=idx + 1, column=1, sticky="ew", padx=5)

# Upadate Financial data every 60 seconds
ttk.Button(financial_data_frame, text="Update", command=update_financial_data).grid(row=3, column=0, pady=10, sticky="ew")
last_update_label = ttk.Label(financial_data_frame, text="", anchor='center')
last_update_label.grid(row=4, column=0, columnspan=2, pady=5)

root.after(60000, update_financial_data)
update_financial_data()

root.mainloop()
