#!/usr/bin/env python3
"""
Cryptocurrency Trading AI Signal Generator
Generates 2-5 daily trading signals using technical analysis
Supports multiple exchanges via API keys
"""

import os
import sys
import asyncio
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Try to import required packages with fallbacks
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("Warning: ccxt not installed. Install with: pip install ccxt")

try:
    import talib
    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False
    print("Warning: TA-Lib not installed. Using simplified indicators.")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("Warning: aiohttp not installed. Install with: pip install aiohttp")

try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

# ============================================================================
# ENUMS AND CONFIGURATION
# ============================================================================

class Exchange(Enum):
    BINANCE = "binance"
    KUCOIN = "kucoin"
    BYBIT = "bybit"
    OKX = "okx"
    COINBASE = "coinbase"
    GATEIO = "gateio"
    HUOBI = "huobi"
    BITGET = "bitget"

class TimeFrame(Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class RiskLevel(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"

@dataclass
class Config:
    """Configuration for the trading AI"""
    # Exchange settings
    exchange: Exchange = Exchange.BINANCE
    timeframe: TimeFrame = TimeFrame.ONE_DAY
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""  # For exchanges like OKX/KuCoin
    
    # Trading settings
    max_signals_per_day: int = 5
    min_volume_usdt: float = 1000000
    min_price: float = 0.01
    risk_level: RiskLevel = RiskLevel.MODERATE
    
    # Analysis settings
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    bb_std_dev: float = 2.0
    lookback_period: int = 100
    
    # Risk management
    default_stop_loss_pct: float = 3.0
    default_take_profit_pct: float = 6.0
    max_position_size_pct: float = 2.0
    
    # Alert settings
    enable_telegram: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    # File output
    save_to_json: bool = True
    save_to_csv: bool = True
    output_directory: str = "trading_signals"
    
    def __post_init__(self):
        """Load from environment variables if not set"""
        if not self.api_key:
            self.api_key = os.getenv(f"{self.exchange.value.upper()}_API_KEY", "")
        if not self.api_secret:
            self.api_secret = os.getenv(f"{self.exchange.value.upper()}_API_SECRET", "")
        if not self.passphrase and self.exchange.value in ['kucoin', 'okx']:
            self.passphrase = os.getenv(f"{self.exchange.value.upper()}_PASSPHRASE", "")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class CryptoSymbol:
    """Represents a cryptocurrency trading pair"""
    symbol: str
    base: str
    quote: str = "USDT"
    rank: int = 0
    volume_24h: float = 0.0
    price_change_24h: float = 0.0
    market_cap: float = 0.0
    price: float = 0.0
    
    def __str__(self):
        return f"{self.base}/{self.quote}"

@dataclass
class TradingSignal:
    """Represents a trading signal"""
    id: str
    symbol: str
    signal_type: SignalType
    strategy: str
    confidence: float  # 0.0 to 1.0
    price: float
    timestamp: datetime
    indicators: Dict[str, float] = field(default_factory=dict)
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward_ratio: float = 0.0
    volume: float = 0.0
    reason: str = ""
    
    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "signal": self.signal_type.value,
            "strategy": self.strategy,
            "confidence": round(self.confidence, 3),
            "price": round(self.price, 4),
            "timestamp": self.timestamp.isoformat(),
            "entry": round(self.entry_price, 4),
            "stop_loss": round(self.stop_loss, 4),
            "take_profit": round(self.take_profit, 4),
            "risk_reward": f"1:{self.risk_reward_ratio:.1f}",
            "volume": round(self.volume, 2),
            "reason": self.reason,
            "indicators": {k: round(v, 4) for k, v in self.indicators.items()}
        }

# ============================================================================
# DATA FETCHER
# ============================================================================

class CryptoDataFetcher:
    """Fetches cryptocurrency data from various sources"""
    
    def __init__(self, config: Config):
        self.config = config
        self.exchange = None
        self.session = None
        self._init_exchange()
        
    def _init_exchange(self):
        """Initialize exchange connection"""
        if not CCXT_AVAILABLE:
            print("CCXT not available. Using fallback data sources.")
            return
            
        exchange_config = {
            'apiKey': self.config.api_key,
            'secret': self.config.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }
        
        if self.config.exchange in [Exchange.OKX, Exchange.KUCOIN]:
            exchange_config['password'] = self.config.passphrase
            
        try:
            self.exchange = getattr(ccxt, self.config.exchange.value)(exchange_config)
            # Test connection if API keys are provided
            if self.config.api_key and self.config.api_secret:
                try:
                    self.exchange.fetch_balance()
                    print(f"✓ Connected to {self.config.exchange.value.upper()}")
                except:
                    print(f"⚠️  API keys may be invalid for {self.config.exchange.value.upper()}")
            else:
                print(f"ℹ️  Using public API for {self.config.exchange.value.upper()}")
        except Exception as e:
            print(f"❌ Failed to initialize exchange: {e}")
            self.exchange = None
    
    async def fetch_top_cryptocurrencies(self, limit: int = 50) -> List[CryptoSymbol]:
        """Fetch top cryptocurrencies by market cap"""
        cryptos = []
        
        # Try CoinGecko API first (no API key needed)
        if AIOHTTP_AVAILABLE:
            try:
                url = "https://api.coingecko.com/api/v3/coins/markets"
                params = {
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': limit,
                    'page': 1,
                    'sparkline': 'false'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            for idx, coin in enumerate(data):
                                symbol = coin['symbol'].upper()
                                crypto = CryptoSymbol(
                                    symbol=symbol,
                                    base=symbol,
                                    quote='USDT',
                                    rank=idx + 1,
                                    volume_24h=coin.get('total_volume', 0),
                                    price_change_24h=coin.get('price_change_percentage_24h', 0),
                                    market_cap=coin.get('market_cap', 0),
                                    price=coin.get('current_price', 0)
                                )
                                cryptos.append(crypto)
                            print(f"✓ Fetched {len(cryptos)} cryptos from CoinGecko")
                            return cryptos
            except Exception as e:
                print(f"CoinGecko API error: {e}")
        
        # Fallback: Use exchange markets
        if self.exchange:
            try:
                markets = self.exchange.load_markets()
                # Filter for USDT pairs and sort by volume
                usdt_pairs = [m for m in markets.values() 
                             if m['quote'] == 'USDT' and m['active']]
                usdt_pairs.sort(key=lambda x: x.get('quoteVolume', 0), reverse=True)
                
                for idx, market in enumerate(usdt_pairs[:limit]):
                    base = market['base']
                    crypto = CryptoSymbol(
                        symbol=f"{base}/USDT",
                        base=base,
                        quote='USDT',
                        rank=idx + 1,
                        volume_24h=market.get('quoteVolume', 0),
                        price=market.get('last', 0)
                    )
                    cryptos.append(crypto)
                print(f"✓ Fetched {len(cryptos)} cryptos from exchange")
            except Exception as e:
                print(f"Exchange markets error: {e}")
        
        # Ultimate fallback: hardcoded list
        if not cryptos:
            cryptos = self._get_default_cryptos(limit)
            print("⚠️  Using default crypto list")
        
        return cryptos
    
    def _get_default_cryptos(self, limit: int) -> List[CryptoSymbol]:
        """Get default cryptocurrency list"""
        default_pairs = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
            "SOL/USDT", "DOGE/USDT", "DOT/USDT", "MATIC/USDT", "SHIB/USDT",
            "TRX/USDT", "AVAX/USDT", "UNI/USDT", "ATOM/USDT", "LINK/USDT",
            "ETC/USDT", "XLM/USDT", "BCH/USDT", "ALGO/USDT", "VET/USDT",
            "FIL/USDT", "THETA/USDT", "XTZ/USDT", "AAVE/USDT", "EOS/USDT",
            "MKR/USDT", "SNX/USDT", "COMP/USDT", "YFI/USDT", "SAND/USDT"
        ]
        
        cryptos = []
        for idx, pair in enumerate(default_pairs[:limit]):
            base = pair.split('/')[0]
            crypto = CryptoSymbol(
                symbol=pair,
                base=base,
                quote='USDT',
                rank=idx + 1,
                volume_24h=1000000 * (limit - idx)  # Simulated volume
            )
            cryptos.append(crypto)
        
        return cryptos
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = None, limit: int = None) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol"""
        if timeframe is None:
            timeframe = self.config.timeframe.value
        if limit is None:
            limit = self.config.lookback_period
        
        # Try exchange first
        if self.exchange and self.exchange.has['fetchOHLCV']:
            try:
                ohlcv = await self.exchange.fetch_ohlcv_async(
                    symbol, 
                    timeframe=timeframe, 
                    limit=limit
                )
                
                df = pd.DataFrame(
                    ohlcv, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            except Exception as e:
                print(f"Exchange OHLCV error for {symbol}: {e}")
        
        # Fallback: Generate synthetic data for testing
        return self._generate_synthetic_data(symbol, limit)
    
    def _generate_synthetic_data(self, symbol: str, periods: int) -> pd.DataFrame:
        """Generate synthetic price data for testing"""
        np.random.seed(hash(symbol) % 10000)
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
        base_price = np.random.uniform(10, 1000)
        
        returns = np.random.normal(0.001, 0.02, periods)
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, periods)),
            'high': prices * (1 + np.random.uniform(0, 0.03, periods)),
            'low': prices * (1 - np.random.uniform(0, 0.03, periods)),
            'close': prices,
            'volume': np.random.lognormal(10, 1, periods)
        }, index=dates)
        
        return df
    
    async def fetch_current_price(self, symbol: str) -> float:
        """Fetch current price for a symbol"""
        if self.exchange:
            try:
                ticker = await self.exchange.fetch_ticker_async(symbol)
                return ticker['last']
            except:
                pass
        
        # Fallback: Use CoinGecko
        if AIOHTTP_AVAILABLE:
            try:
                coin_id = symbol.split('/')[0].lower()
                url = f"https://api.coingecko.com/api/v3/simple/price"
                params = {'ids': coin_id, 'vs_currencies': 'usd'}
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data.get(coin_id, {}).get('usd', 0)
            except:
                pass
        
        return 0.0

# ============================================================================
# TECHNICAL ANALYZER
# ============================================================================

class TechnicalAnalyzer:
    """Performs technical analysis on price data"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        if df.empty or len(df) < 50:
            return df
            
        df = df.copy()
        
        # Price basics
        df['returns'] = df['close'].pct_change()
        df['high_52'] = df['high'].rolling(52).max()
        df['low_52'] = df['low'].rolling(52).min()
        
        # Moving Averages
        df = self._calculate_moving_averages(df)
        
        # Oscillators
        df = self._calculate_oscillators(df)
        
        # Volume indicators
        df = self._calculate_volume_indicators(df)
        
        # Volatility indicators
        df = self._calculate_volatility_indicators(df)
        
        # Trend indicators
        df = self._calculate_trend_indicators(df)
        
        return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages"""
        # Use TA-Lib if available
        if TA_LIB_AVAILABLE:
            df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
            df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
            df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)
            df['EMA_12'] = talib.EMA(df['close'], timeperiod=12)
            df['EMA_26'] = talib.EMA(df['close'], timeperiod=26)
            df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
        else:
            # Manual calculations
            df['SMA_20'] = df['close'].rolling(20).mean()
            df['SMA_50'] = df['close'].rolling(50).mean()
            df['SMA_200'] = df['close'].rolling(200).mean()
            df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Moving average cross signals
        df['MA_CROSS_20_50'] = df['SMA_20'] > df['SMA_50']
        df['MA_CROSS_12_26'] = df['EMA_12'] > df['EMA_26']
        
        return df
    
    def _calculate_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate oscillator indicators"""
        # RSI
        if TA_LIB_AVAILABLE:
            df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        else:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        if TA_LIB_AVAILABLE:
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
        else:
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Stochastic
        if TA_LIB_AVAILABLE:
            df['STOCH_K'], df['STOCH_D'] = talib.STOCH(
                df['high'], df['low'], df['close'],
                fastk_period=14, slowk_period=3, slowd_period=3
            )
        else:
            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            df['STOCH_K'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
            df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
        
        # Williams %R
        df['WILLIAMS_R'] = (df['high'].rolling(14).max() - df['close']) / \
                          (df['high'].rolling(14).max() - df['low'].rolling(14).min()) * -100
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        df['VOLUME_SMA'] = df['volume'].rolling(20).mean()
        df['VOLUME_RATIO'] = df['volume'] / df['VOLUME_SMA']
        
        # OBV
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        
        # Volume Price Trend
        df['VPT'] = (df['volume'] * (df['close'].diff() / df['close'].shift(1))).cumsum()
        
        return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        # Bollinger Bands
        if TA_LIB_AVAILABLE:
            df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(
                df['close'], timeperiod=20,
                nbdevup=self.config.bb_std_dev,
                nbdevdn=self.config.bb_std_dev
            )
        else:
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            df['BB_UPPER'] = sma20 + (std20 * self.config.bb_std_dev)
            df['BB_MIDDLE'] = sma20
            df['BB_LOWER'] = sma20 - (std20 * self.config.bb_std_dev)
        
        df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE']
        df['BB_POSITION'] = (df['close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])
        
        # ATR
        if TA_LIB_AVAILABLE:
            df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        else:
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(14).mean()
        
        df['ATR_PCT'] = df['ATR'] / df['close'] * 100
        
        return df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators"""
        # ADX
        if TA_LIB_AVAILABLE:
            df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        else:
            # Simplified ADX calculation
            up_move = df['high'].diff()
            down_move = df['low'].diff().abs() * -1
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            tr = self._true_range(df)
            atr = tr.rolling(14).mean()
            
            plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['ADX'] = dx.rolling(14).mean()
        
        # Parabolic SAR
        if TA_LIB_AVAILABLE:
            df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
        else:
            # Simplified SAR
            df['SAR'] = df['close'].rolling(4).mean()
        
        return df
    
    def _true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    """Generates trading signals based on technical analysis"""
    
    def __init__(self, config: Config):
        self.config = config
        self.analyzer = TechnicalAnalyzer(config)
        
    async def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> List[TradingSignal]:
        """Analyze a symbol and generate signals"""
        if df.empty or len(df) < 50:
            return []
        
        # Calculate indicators
        df = self.analyzer.calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        
        # Strategy 1: Trend Following with MA Cross
        if latest['MA_CROSS_20_50'] and not prev['MA_CROSS_20_50']:
            confidence = 0.75 if latest['close'] > latest['SMA_200'] else 0.6
            signals.append(self._create_signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strategy="MA Trend Cross",
                confidence=confidence,
                price=latest['close'],
                reason=f"MA Cross: SMA20 crossed above SMA50. Trend: {'Bullish' if latest['close'] > latest['SMA_200'] else 'Neutral'}",
                indicators={
                    'SMA_20': latest['SMA_20'],
                    'SMA_50': latest['SMA_50'],
                    'SMA_200': latest['SMA_200']
                }
            ))
        
        # Strategy 2: RSI Oversold/Overbought
        if latest['RSI'] < self.config.rsi_oversold:
            # Check for bullish divergence
            if self._check_bullish_divergence(df, 'RSI'):
                signals.append(self._create_signal(
                    symbol=symbol,
                    signal_type=SignalType.STRONG_BUY,
                    strategy="RSI Divergence",
                    confidence=0.85,
                    price=latest['close'],
                    reason=f"RSI oversold ({latest['RSI']:.1f}) with bullish divergence",
                    indicators={'RSI': latest['RSI']}
                ))
            else:
                signals.append(self._create_signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strategy="RSI Oversold",
                    confidence=0.7,
                    price=latest['close'],
                    reason=f"RSI oversold ({latest['RSI']:.1f})",
                    indicators={'RSI': latest['RSI']}
                ))
        
        # Strategy 3: Bollinger Band Squeeze Breakout
        if latest['BB_WIDTH'] < 0.1 and latest['VOLUME_RATIO'] > 1.5:
            if latest['close'] > latest['BB_UPPER']:
                signals.append(self._create_signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strategy="BB Squeeze Breakout",
                    confidence=0.8,
                    price=latest['close'],
                    reason="Bullish breakout from Bollinger Band squeeze",
                    indicators={
                        'BB_WIDTH': latest['BB_WIDTH'],
                        'VOLUME_RATIO': latest['VOLUME_RATIO'],
                        'BB_POSITION': latest['BB_POSITION']
                    }
                ))
        
        # Strategy 4: MACD Crossover
        if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            signals.append(self._create_signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strategy="MACD Crossover",
                confidence=0.65,
                price=latest['close'],
                reason="MACD crossed above signal line",
                indicators={
                    'MACD': latest['MACD'],
                    'MACD_SIGNAL': latest['MACD_signal']
                }
            ))
        
        # Strategy 5: Volume Breakout
        if latest['VOLUME_RATIO'] > 2.0 and latest['close'] > latest['high'].rolling(20).max().shift(1):
            signals.append(self._create_signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strategy="Volume Breakout",
                confidence=0.75,
                price=latest['close'],
                reason="High volume breakout above 20-day high",
                indicators={
                    'VOLUME_RATIO': latest['VOLUME_RATIO'],
                    'PRICE_HIGH_20': latest['high'].rolling(20).max()
                }
            ))
        
        # Strategy 6: Support Bounce
        support_level = self._find_support_level(df)
        if support_level > 0 and abs(latest['close'] - support_level) / support_level < 0.02:
            signals.append(self._create_signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strategy="Support Bounce",
                confidence=0.6,
                price=latest['close'],
                reason=f"Price bouncing off support at ${support_level:.2f}",
                indicators={'SUPPORT_LEVEL': support_level}
            ))
        
        # Add risk management parameters
        for signal in signals:
            signal.entry_price = signal.price
            signal.stop_loss = self._calculate_stop_loss(signal)
            signal.take_profit = self._calculate_take_profit(signal)
            signal.risk_reward_ratio = self._calculate_risk_reward(signal)
        
        return signals
    
    def _create_signal(self, symbol: str, signal_type: SignalType, strategy: str,
                      confidence: float, price: float, reason: str, indicators: Dict) -> TradingSignal:
        """Create a trading signal object"""
        signal_id = f"{symbol}_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return TradingSignal(
            id=signal_id,
            symbol=symbol,
            signal_type=signal_type,
            strategy=strategy,
            confidence=confidence,
            price=price,
            timestamp=datetime.now(),
            indicators=indicators,
            reason=reason
        )
    
    def _check_bullish_divergence(self, df: pd.DataFrame, indicator: str = 'RSI') -> bool:
        """Check for bullish divergence between price and indicator"""
        if len(df) < 30:
            return False
        
        prices = df['close'].iloc[-20:].values
        indicators = df[indicator].iloc[-20:].values if indicator in df.columns else df['RSI'].iloc[-20:].values
        
        # Find two most recent lows
        low_idx1 = len(prices) - 1 - np.argmin(prices[::-1])
        low_idx2 = len(prices) - 1 - np.argmin(prices[:low_idx1][::-1]) if low_idx1 > 0 else -1
        
        if low_idx2 < 0:
            return False
        
        # Check for divergence: lower price low but higher indicator low
        price_low1 = prices[low_idx1]
        price_low2 = prices[low_idx2]
        ind_low1 = indicators[low_idx1]
        ind_low2 = indicators[low_idx2]
        
        return price_low1 < price_low2 and ind_low1 > ind_low2
    
    def _find_support_level(self, df: pd.DataFrame, lookback: int = 50) -> float:
        """Find recent support level using pivot lows"""
        if len(df) < lookback:
            return 0.0
        
        prices = df['low'].iloc[-lookback:].values
        
        # Find local minima
        support_levels = []
        for i in range(2, len(prices) - 2):
            if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
               prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                support_levels.append(prices[i])
        
        if not support_levels:
            return 0.0
        
        # Return the most recent significant support
        return sorted(support_levels)[-1] if len(support_levels) > 1 else support_levels[-1]
    
    def _calculate_stop_loss(self, signal: TradingSignal) -> float:
        """Calculate stop loss based on signal type and confidence"""
        if signal.signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
            sl_pct = self.config.default_stop_loss_pct * 0.8  # Tighter SL for strong signals
        else:
            sl_pct = self.config.default_stop_loss_pct
        
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return signal.price * (1 - sl_pct / 100)
        else:
            return signal.price * (1 + sl_pct / 100)
    
    def _calculate_take_profit(self, signal: TradingSignal) -> float:
        """Calculate take profit based on signal type and confidence"""
        if signal.signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
            tp_pct = self.config.default_take_profit_pct * 1.2  # Higher TP for strong signals
        else:
            tp_pct = self.config.default_take_profit_pct
        
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return signal.price * (1 + tp_pct / 100)
        else:
            return signal.price * (1 - tp_pct / 100)
    
    def _calculate_risk_reward(self, signal: TradingSignal) -> float:
        """Calculate risk-reward ratio"""
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            risk = signal.price - signal.stop_loss
            reward = signal.take_profit - signal.price
        else:
            risk = signal.stop_loss - signal.price
            reward = signal.price - signal.take_profit
        
        if risk > 0:
            return reward / risk
        return 1.0

# ============================================================================
# TRADING AI CORE
# ============================================================================

class CryptoTradingAI:
    """Main trading AI class that coordinates everything"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_fetcher = CryptoDataFetcher(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.signals: List[TradingSignal] = []
        
        # Create output directory
        os.makedirs(self.config.output_directory, exist_ok=True)
    
    async def generate_signals(self, max_symbols: int = 20) -> List[TradingSignal]:
        """Generate trading signals for multiple cryptocurrencies"""
        print(f"\n{'='*80}")
        print("🚀 CRYPTO TRADING AI SIGNAL GENERATOR")
        print(f"{'='*80}")
        print(f"Exchange: {self.config.exchange.value.upper()}")
        print(f"Timeframe: {self.config.timeframe.value}")
        print(f"Risk Level: {self.config.risk_level.value}")
        print(f"{'='*80}\n")
        
        # Fetch top cryptocurrencies
        print("📊 Fetching cryptocurrency data...")
        cryptos = await self.data_fetcher.fetch_top_cryptocurrencies(limit=max_symbols)
        
        # Filter by minimum volume
        cryptos = [c for c in cryptos if c.volume_24h >= self.config.min_volume_usdt]
        print(f"Analyzing {len(cryptos)} cryptocurrencies with sufficient volume...\n")
        
        all_signals = []
        
        # Analyze each symbol
        for i, crypto in enumerate(cryptos):
            try:
                print(f"  [{i+1}/{len(cryptos)}] Analyzing {crypto.symbol}...", end='\r')
                
                # Fetch OHLCV data
                df = await self.data_fetcher.fetch_ohlcv(
                    crypto.symbol,
                    timeframe=self.config.timeframe.value,
                    limit=self.config.lookback_period
                )
                
                if not df.empty:
                    # Generate signals
                    signals = await self.signal_generator.analyze_symbol(crypto.symbol, df)
                    
                    # Add volume information
                    for signal in signals:
                        signal.volume = crypto.volume_24h
                    
                    all_signals.extend(signals)
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"\nError analyzing {crypto.symbol}: {str(e)[:50]}")
                continue
        
        print("\n" + "="*80)
        
        # Filter and sort signals
        self.signals = self._filter_and_sort_signals(all_signals)
        
        # Limit to max signals per day
        self.signals = self.signals[:self.config.max_signals_per_day]
        
        return self.signals
    
    def _filter_and_sort_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter and sort signals by confidence and risk-reward"""
        if not signals:
            return []
        
        # Remove duplicates (same symbol and strategy)
        unique_signals = {}
        for signal in signals:
            key = (signal.symbol, signal.strategy)
            if key not in unique_signals or signal.confidence > unique_signals[key].confidence:
                unique_signals[key] = signal
        
        # Filter by minimum confidence based on risk level
        min_confidence = {
            RiskLevel.LOW: 0.7,
            RiskLevel.MODERATE: 0.6,
            RiskLevel.HIGH: 0.5
        }.get(self.config.risk_level, 0.6)
        
        filtered = [s for s in unique_signals.values() 
                   if s.confidence >= min_confidence and s.risk_reward_ratio >= 1.0]
        
        # Sort by confidence * risk-reward ratio
        filtered.sort(key=lambda x: x.confidence * x.risk_reward_ratio, reverse=True)
        
        return filtered
    
    def display_signals(self, signals: List[TradingSignal] = None):
        """Display signals in a formatted way"""
        signals = signals or self.signals
        
        if not signals:
            print("\n❌ No trading signals generated today.")
            print("Market conditions may be unfavorable or indicators are neutral.")
            return
        
        print(f"\n{'='*80}")
        print(f"✅ TRADING SIGNALS GENERATED: {len(signals)}")
        print(f"{'='*80}\n")
        
        for i, signal in enumerate(signals):
            signal_dict = signal.to_dict()
            
            # Get emojis based on signal type
            if signal.signal_type == SignalType.STRONG_BUY:
                emoji, color = "🔥", "GREEN"
            elif signal.signal_type == SignalType.BUY:
                emoji, color = "🟢", "GREEN"
            elif signal.signal_type == SignalType.STRONG_SELL:
                emoji, color = "💀", "RED"
            elif signal.signal_type == SignalType.SELL:
                emoji, color = "🔴", "RED"
            else:
                emoji, color = "⚪", "WHITE"
            
            print(f"{emoji} SIGNAL #{i+1}: {signal.symbol}")
            print(f"   Type: {signal.signal_type.value} | Confidence: {signal.confidence:.1%}")
            print(f"   Strategy: {signal.strategy}")
            print(f"   Current Price: ${signal.price:.4f}")
            print(f"   Entry: ${signal.entry_price:.4f}")
            print(f"   Stop Loss: ${signal.stop_loss:.4f} ({abs((signal.stop_loss/signal.price-1)*100):.1f}%)")
            print(f"   Take Profit: ${signal.take_profit:.4f} ({abs((signal.take_profit/signal.price-1)*100):.1f}%)")
            print(f"   Risk/Reward: 1:{signal.risk_reward_ratio:.1f}")
            print(f"   Volume (24h): ${signal.volume:,.0f}")
            print(f"   Reason: {signal.reason}")
            print(f"   Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 60)
    
    def save_signals(self, signals: List[TradingSignal] = None):
        """Save signals to files"""
        signals = signals or self.signals
        
        if not signals:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Convert to dictionaries
        signal_dicts = [s.to_dict() for s in signals]
        
        # Save to JSON
        if self.config.save_to_json:
            json_file = os.path.join(self.config.output_directory, f"signals_{timestamp}.json")
            with open(json_file, 'w') as f:
                json.dump({
                    'metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'exchange': self.config.exchange.value,
                        'timeframe': self.config.timeframe.value,
                        'risk_level': self.config.risk_level.value,
                        'total_signals': len(signals)
                    },
                    'signals': signal_dicts
                }, f, indent=2)
            print(f"\n💾 Signals saved to JSON: {json_file}")
        
        # Save to CSV
        if self.config.save_to_csv:
            csv_file = os.path.join(self.config.output_directory, f"signals_{timestamp}.csv")
            df = pd.DataFrame(signal_dicts)
            df.to_csv(csv_file, index=False)
            print(f"💾 Signals saved to CSV: {csv_file}")
        
        return signal_dicts
    
    async def send_alerts(self, signals: List[TradingSignal] = None):
        """Send alerts via configured channels"""
        signals = signals or self.signals
        
        if not signals or not self.config.enable_telegram:
            return
        
        # Telegram alerts
        if self.config.telegram_bot_token and self.config.telegram_chat_id:
            await self._send_telegram_alert(signals)
    
    async def _send_telegram_alert(self, signals: List[TradingSignal]):
        """Send alert to Telegram"""
        try:
            if not AIOHTTP_AVAILABLE:
                print("⚠️  aiohttp not installed for Telegram alerts")
                return
            
            message = f"🚀 *Crypto Trading Signals* ({datetime.now().strftime('%Y-%m-%d')})\n\n"
            
            for i, signal in enumerate(signals[:3]):  # Send top 3 signals
                action = "BUY" if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else "SELL"
                message += f"*{i+1}. {signal.symbol} - {action}*\n"
                message += f"Price: ${signal.price:.2f}\n"
                message += f"Strategy: {signal.strategy}\n"
                message += f"Confidence: {signal.confidence:.0%}\n"
                message += f"RR: 1:{signal.risk_reward_ratio:.1f}\n\n"
            
            message += f"_Total signals: {len(signals)}_\n"
            message += "#TradingSignals #Crypto"
            
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            params = {
                'chat_id': self.config.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params) as response:
                    if response.status == 200:
                        print("✅ Telegram alert sent successfully")
                    else:
                        print(f"⚠️  Failed to send Telegram alert: {response.status}")
                        
        except Exception as e:
            print(f"❌ Telegram alert error: {e}")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading AI Signal Generator')
    
    # Exchange arguments
    parser.add_argument('--exchange', type=str, default='binance',
                       choices=[e.value for e in Exchange],
                       help='Exchange to use (default: binance)')
    parser.add_argument('--api-key', type=str, default='',
                       help='Exchange API key')
    parser.add_argument('--api-secret', type=str, default='',
                       help='Exchange API secret')
    parser.add_argument('--passphrase', type=str, default='',
                       help='Exchange passphrase (for OKX/KuCoin)')
    
    # Trading arguments
    parser.add_argument('--timeframe', type=str, default='1d',
                       choices=[tf.value for tf in TimeFrame],
                       help='Chart timeframe (default: 1d)')
    parser.add_argument('--max-signals', type=int, default=5,
                       help='Maximum signals per day (default: 5)')
    parser.add_argument('--risk-level', type=str, default='MODERATE',
                       choices=[rl.value for rl in RiskLevel],
                       help='Risk level (default: MODERATE)')
    
    # Analysis arguments
    parser.add_argument('--rsi-overbought', type=int, default=70,
                       help='RSI overbought threshold (default: 70)')
    parser.add_argument('--rsi-oversold', type=int, default=30,
                       help='RSI oversold threshold (default: 30)')
    
    # Alert arguments
    parser.add_argument('--telegram', action='store_true',
                       help='Enable Telegram alerts')
    parser.add_argument('--telegram-token', type=str, default='',
                       help='Telegram bot token')
    parser.add_argument('--telegram-chat-id', type=str, default='',
                       help='Telegram chat ID')
    
    # Output arguments
    parser.add_argument('--no-save', action='store_true',
                       help='Disable saving signals to files')
    parser.add_argument('--output-dir', type=str, default='trading_signals',
                       help='Output directory for saved signals')
    
    # Test mode
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with synthetic data')
    parser.add_argument('--demo', action='store_true',
                       help='Show demo signals without fetching data')
    
    return parser.parse_args()

async def main():
    """Main function"""
    args = parse_args()
    
    # Create configuration
    config = Config(
        exchange=Exchange(args.exchange),
        timeframe=TimeFrame(args.timeframe),
        api_key=args.api_key,
        api_secret=args.api_secret,
        passphrase=args.passphrase,
        max_signals_per_day=args.max_signals,
        risk_level=RiskLevel(args.risk_level),
        rsi_overbought=args.rsi_overbought,
        rsi_oversold=args.rsi_oversold,
        enable_telegram=args.telegram,
        telegram_bot_token=args.telegram_token or os.getenv('TELEGRAM_BOT_TOKEN', ''),
        telegram_chat_id=args.telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID', ''),
        save_to_json=not args.no_save,
        save_to_csv=not args.no_save,
        output_directory=args.output_dir
    )
    
    # Initialize AI
    ai = CryptoTradingAI(config)
    
    # Run in demo mode
    if args.demo:
        print("\n🎮 DEMO MODE - Showing sample signals\n")
        demo_signals = [
            TradingSignal(
                id="DEMO_001",
                symbol="BTC/USDT",
                signal_type=SignalType.STRONG_BUY,
                strategy="MA Trend Cross",
                confidence=0.85,
                price=45000.50,
                timestamp=datetime.now(),
                entry_price=45000.50,
                stop_loss=43650.48,
                take_profit=47700.53,
                risk_reward_ratio=2.0,
                volume=25000000,
                reason="Strong uptrend with MA crossover and high volume"
            ),
            TradingSignal(
                id="DEMO_002",
                symbol="ETH/USDT",
                signal_type=SignalType.BUY,
                strategy="RSI Divergence",
                confidence=0.72,
                price=2500.75,
                timestamp=datetime.now(),
                entry_price=2500.75,
                stop_loss=2375.71,
                take_profit=2650.80,
                risk_reward_ratio=1.8,
                volume=15000000,
                reason="Bullish RSI divergence in oversold territory"
            )
        ]
        ai.display_signals(demo_signals)
        return
    
    # Generate signals
    try:
        signals = await ai.generate_signals(max_symbols=25)
        
        # Display results
        ai.display_signals(signals)
        
        # Save to files
        if signals:
            ai.save_signals(signals)
            
            # Send alerts
            if config.enable_telegram and config.telegram_bot_token:
                await ai.send_alerts(signals)
        
        print(f"\n{'='*80}")
        print("✅ Signal generation complete!")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# QUICK START FUNCTIONS
# ============================================================================

def quick_start():
    """Quick start function for daily use"""
    print("🚀 Crypto Trading AI - Quick Start")
    print("="*50)
    
    # Check requirements
    if not CCXT_AVAILABLE:
        print("Installing required packages...")
        os.system("pip install ccxt pandas numpy aiohttp python-dotenv")
        print("Please restart the script.")
        return
    
    # Create default config
    config = Config()
    
    # Ask for API keys
    use_api = input("\nUse exchange API keys? (y/n): ").lower() == 'y'
    if use_api:
        config.exchange = Exchange(input("Exchange (binance/kucoin/bybit/okx): ").lower() or 'binance')
        config.api_key = input(f"{config.exchange.value.upper()} API Key: ")
        config.api_secret = input(f"{config.exchange.value.upper()} API Secret: ")
        if config.exchange in [Exchange.OKX, Exchange.KUCOIN]:
            config.passphrase = input("Passphrase: ")
    
    # Ask for Telegram
    use_telegram = input("\nEnable Telegram alerts? (y/n): ").lower() == 'y'
    if use_telegram:
        config.enable_telegram = True
        config.telegram_bot_token = input("Telegram Bot Token: ")
        config.telegram_chat_id = input("Telegram Chat ID: ")
    
    # Run AI
    ai = CryptoTradingAI(config)
    
    print("\n" + "="*50)
    print("Starting analysis...")
    
    # Run async main
    asyncio.run(ai.generate_signals())
    ai.display_signals()
    
    if ai.signals:
        ai.save_signals()

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Setup environment and create .env template"""
    env_template = """# Exchange API Keys
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

KUCOIN_API_KEY=your_kucoin_api_key_here
KUCOIN_API_SECRET=your_kucoin_api_secret_here
KUCOIN_PASSPHRASE=your_kucoin_passphrase_here

BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_API_SECRET=your_bybit_api_secret_here

OKX_API_KEY=your_okx_api_key_here
OKX_API_SECRET=your_okx_api_secret_here
OKX_PASSPHRASE=your_okx_passphrase_here

# Telegram Alerts
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Trading Settings
RISK_LEVEL=MODERATE
MAX_SIGNALS=5
TIMEFRAME=1d
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_template)
    
    print("✅ Created .env.template file")
    print("📝 Rename to .env and fill in your API keys")
    print("🔒 Never commit .env file to version control!")

# ============================================================================
# ENTRY POINTS
# ============================================================================

if __name__ == "__main__":
    # Check if running with arguments
    if len(sys.argv) > 1:
        asyncio.run(main())
    else:
        # Interactive mode
        print("\n" + "="*80)
        print("🤖 CRYPTO TRADING AI SIGNAL GENERATOR")
        print("="*80)
        print("\nOptions:")
        print("  1. Quick Start (Interactive)")
        print("  2. Run with defaults")
        print("  3. Setup environment")
        print("  4. Show demo")
        print("  5. Exit")
        
        choice = input("\nSelect option (1-5): ")
        
        if choice == '1':
            quick_start()
        elif choice == '2':
            asyncio.run(main())
        elif choice == '3':
            setup_environment()
        elif choice == '4':
            args = type('Args', (), {'demo': True})()
            asyncio.run(main())
        else:
            print("Goodbye!")