import ccxt
import pandas as pd
import pandas_ta as ta
import requests
import time
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────
SYMBOLS = [
    'ADA/USDT', 'APT/USDT', 'AVAX/USDT', 'BNB/USDT', 'BTC/USDT',
    'DOT/USDT', 'ETH/USDT', 'INJ/USDT', 'NEAR/USDT', 'SOL/USDT', 'XRP/USDT'
]
NTFY_TOPIC  = 'talosbot-adam-7x4k'
TIMEFRAME   = '1h'
HTF         = '4h'

# Strategy settings — match PineScript exactly
SQ_ATR_LEN  = 14
SQ_SMA_LEN  = 20
SQ_THRESH   = 0.85
SQ_LOOKBACK = 20
RSI_MIN_L   = 45
RSI_MAX_L   = 72
RSI_MIN_S   = 28
RSI_MAX_S   = 55
VOL_MULT    = 1.4
ST_MULT     = 3.0
ST_ATR_P    = 9
MAX_SL_PCT  = 5.0

# ── EXCHANGE ─────────────────────────────────────────
exchange = ccxt.binance({'enableRateLimit': True})

# Prevents duplicate notifications for the same signal
sent_signals = set()

# ── NTFY ─────────────────────────────────────────────
def send_ntfy(title, message, priority='high', tags='chart_with_upwards_trend'):
    try:
        requests.post(
            f'https://ntfy.sh/{NTFY_TOPIC}',
            data=message.encode('utf-8'),
            headers={
                'Title': title,
                'Priority': priority,
                'Tags': tags
            },
            timeout=10
        )
        print(f"  Sent: {title}")
    except Exception as e:
        print(f"  ntfy failed: {e}")

# ── DATA ──────────────────────────────────────────────
def fetch_ohlcv(symbol, timeframe, limit=300):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ── INDICATORS ────────────────────────────────────────
def add_indicators(df):
    df['ema21']   = ta.ema(df['close'], length=21)
    df['ema50']   = ta.ema(df['close'], length=50)
    df['ema200']  = ta.ema(df['close'], length=200)
    df['rsi']     = ta.rsi(df['close'], length=14)
    df['vol_sma'] = df['volume'].rolling(20).mean()
    df['atr']     = ta.atr(df['high'], df['low'], df['close'], length=SQ_ATR_LEN)
    df['atr_sma'] = df['atr'].rolling(SQ_SMA_LEN).mean()
    df['squeeze'] = df['atr'] < df['atr_sma'] * SQ_THRESH
    df['had_sq']  = df['squeeze'].rolling(SQ_LOOKBACK).sum() > 0

    st = ta.supertrend(df['high'], df['low'], df['close'],
                       length=ST_ATR_P, multiplier=ST_MULT)
    dir_col = [c for c in st.columns if c.startswith('SUPERTd')][0]
    df['st_bull']       = st[dir_col] == 1
    df['st_flip_long']  = df['st_bull'] & ~df['st_bull'].shift(1).fillna(False)
    df['st_flip_short'] = ~df['st_bull'] & df['st_bull'].shift(1).fillna(True)
    return df

# ── SIGNAL CHECK ──────────────────────────────────────
def check_signals(symbol):
    try:
        df     = fetch_ohlcv(symbol, TIMEFRAME, limit=300)
        df_htf = fetch_ohlcv(symbol, HTF, limit=100)

        df              = add_indicators(df)
        df_htf['ema50'] = ta.ema(df_htf['close'], length=50)
        htf_bull        = df_htf['close'].iloc[-1] > df_htf['ema50'].iloc[-1]

        # Current candle — aligns with what TradingView shows live
        c         = df.iloc[-1]
        candle_ts = str(c['timestamp'])

        body     = abs(c['close'] - c['open'])
        top_wick = c['high'] - max(c['open'], c['close'])
        bot_wick = min(c['open'], c['close']) - c['low']
        clean_l  = c['close'] > c['open'] and (body == 0 or top_wick < body * 0.5)
        clean_s  = c['close'] < c['open'] and (body == 0 or bot_wick  < body * 0.5)

        vol_ok   = c['volume'] > c['vol_sma'] * VOL_MULT
        rsi_ok_l = RSI_MIN_L <= c['rsi'] <= RSI_MAX_L
        rsi_ok_s = RSI_MIN_S <= c['rsi'] <= RSI_MAX_S

        window    = df.iloc[-SQ_LOOKBACK - 2:-1]
        sq_low    = window['low'].min()
        sq_high   = window['high'].max()
        sl_dist_l = (c['close'] - sq_low)  / c['close'] * 100
        sl_dist_s = (sq_high - c['close']) / c['close'] * 100
        sl_ok_l   = 0.3 < sl_dist_l <= MAX_SL_PCT
        sl_ok_s   = 0.3 < sl_dist_s <= MAX_SL_PCT

        trend_ok_l = c['close'] > c['ema50'] and htf_bull
        trend_ok_s = c['close'] < c['ema50'] and not htf_bull

        raw_l = trend_ok_l and c['st_flip_long']  and c['had_sq'] and rsi_ok_l and vol_ok and clean_l
        raw_s = trend_ok_s and c['st_flip_short'] and c['had_sq'] and rsi_ok_s and vol_ok and clean_s

        enter_long  = raw_l and sl_ok_l
        enter_short = raw_s and sl_ok_s
        sl_wide_l   = raw_l and not sl_ok_l
        sl_wide_s   = raw_s and not sl_ok_s

        ticker = symbol.replace('/', '')

        if enter_long:
            key = (ticker, 'LONG', candle_ts)
            if key not in sent_signals:
                sent_signals.add(key)
                tp  = c['close'] + (c['close'] - sq_low) * 2
                msg = (f"Timeframe: 1H MANUAL\n"
                       f"Entry:  {c['close']:.4f}\n"
                       f"SL:     {sq_low:.4f}  ({sl_dist_l:.1f}%)\n"
                       f"TP:     {tp:.4f}  (2R)\n"
                       f"Vol:    {c['volume'] / c['vol_sma']:.1f}x avg\n"
                       f"RSI:    {c['rsi']:.1f}")
                send_ntfy(f"MANUAL LONG - {ticker} 1H", msg,
                          priority='urgent', tags='green_circle')

        elif enter_short:
            key = (ticker, 'SHORT', candle_ts)
            if key not in sent_signals:
                sent_signals.add(key)
                tp  = c['close'] - (sq_high - c['close']) * 2
                msg = (f"Timeframe: 1H MANUAL\n"
                       f"Entry:  {c['close']:.4f}\n"
                       f"SL:     {sq_high:.4f}  ({sl_dist_s:.1f}%)\n"
                       f"TP:     {tp:.4f}  (2R)\n"
                       f"Vol:    {c['volume'] / c['vol_sma']:.1f}x avg\n"
                       f"RSI:    {c['rsi']:.1f}")
                send_ntfy(f"MANUAL SHORT - {ticker} 1H", msg,
                          priority='urgent', tags='red_circle')

        elif sl_wide_l:
            key = (ticker, 'SL_WIDE_L', candle_ts)
            if key not in sent_signals:
                sent_signals.add(key)
                send_ntfy(f"MANUAL SL WIDE - {ticker} 1H",
                          f"Long signal valid but SL is {sl_dist_l:.1f}% - skip",
                          priority='default', tags='warning')

        elif sl_wide_s:
            key = (ticker, 'SL_WIDE_S', candle_ts)
            if key not in sent_signals:
                sent_signals.add(key)
                send_ntfy(f"MANUAL SL WIDE - {ticker} 1H",
                          f"Short signal valid but SL is {sl_dist_s:.1f}% - skip",
                          priority='default', tags='warning')

    except Exception as e:
        print(f"  Error on {symbol}: {e}")

# ── MAIN LOOP ─────────────────────────────────────────
def run_scan():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Scanning {len(SYMBOLS)} pairs...")
    for symbol in SYMBOLS:
        print(f"  Checking {symbol}...")
        check_signals(symbol)
        time.sleep(1)
    print("Scan complete.")

print("Talos Manual Signal Scanner starting...")
send_ntfy("Talos Scanner Online",
          "Manual strategy signal scanner is running.",
          priority='low', tags='white_check_mark')

while True:
    run_scan()
    print(f"Next scan in 5 minutes...")
    time.sleep(300)
