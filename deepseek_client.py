```py
"""
DeepSeek V4 API Client for Technical Analysis and Trading Signals.

This module provides a robust client for interacting with the DeepSeek V4 API
to generate technical analysis and trading signals for cryptocurrency trading.
It includes comprehensive error handling, input validation, rate limiting,
and logging capabilities.

Author: Trading Bot Team
Version: 1.0.0
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import aiohttp
import backoff
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# Configure logging
logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Enumeration of possible trading signal types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    ERROR = "ERROR"


class TimeFrame(str, Enum):
    """Supported timeframes for technical analysis."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


class AnalysisType(str, Enum):
    """Types of analysis that can be requested."""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    COMPREHENSIVE = "comprehensive"


@dataclass
class TechnicalIndicator:
    """Represents a single technical indicator value."""
    name: str
    value: float
    signal: SignalType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingSignal:
    """Complete trading signal with analysis details."""
    symbol: str
    signal: SignalType
    confidence: float  # 0.0 to 1.0
    price: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    timeframe: TimeFrame = TimeFrame.HOUR_1
    indicators: List[TechnicalIndicator] = field(default_factory=list)
    analysis_text: str = ""
    risk_score: float = 0.5  # 0.0 (low risk) to 1.0 (high risk)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DeepSeekConfig(BaseModel):
    """Configuration for DeepSeek API client."""
    
    api_key: str = Field(..., min_length=32, description="DeepSeek API key")
    api_secret: str = Field(..., min_length=32, description="DeepSeek API secret")
    base_url: str = Field(
        default="https://api.deepseek.com/v4",
        description="Base URL for DeepSeek API"
    )
    timeout: int = Field(default=30, ge=5, le=120, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    rate_limit_per_minute: int = Field(default=60, ge=1, le=300, description="API rate limit")
    
    @validator("api_key")
    def validate_api_key(cls, v: str) -> str:
        """Validate API key format."""
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        return v.strip()
    
    @validator("api_secret")
    def validate_api_secret(cls, v: str) -> str:
        """Validate API secret format."""
        if not v or not v.strip():
            raise ValueError("API secret cannot be empty")
        return v.strip()
    
    class Config:
        """Pydantic configuration."""
        frozen = True  # Make config immutable


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, max_calls: int, period: float = 60.0):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds (default: 60 seconds)
        """
        self.max_calls = max_calls
        self.period = period
        self.tokens = max_calls
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """
        Acquire a token for API call.
        
        Returns:
            bool: True if token acquired, False if rate limited
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            
            # Refill tokens based on elapsed time
            if elapsed >= self.period:
                self.tokens = self.max_calls
                self.last_refill = now
            else:
                refill_amount = (elapsed / self.period) * self.max_calls
                self.tokens = min(self.max_calls, self.tokens + refill_amount)
                self.last_refill = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            return False


class DeepSeekClient:
    """
    Client for interacting with DeepSeek V4 API.
    
    This client handles authentication, request/response processing,
    rate limiting, retries with exponential backoff, and comprehensive
    error handling for generating technical analysis and trading signals.
    
    Attributes:
        config (DeepSeekConfig): Client configuration
        session (aiohttp.ClientSession): HTTP session for API calls
        rate_limiter (RateLimiter): Rate limiter instance
        _last_request_time (float): Timestamp of last API request
    
    Example:
        ```python
        config = DeepSeekConfig(
            api_key="your-api-key",
            api_secret="your-api-secret"
        )
        
        async with DeepSeekClient(config) as client:
            signal = await client.generate_trading_signal(
                symbol="BTCUSDT",
                timeframe=TimeFrame.HOUR_4,
                price_data=price_df
            )
            print(f"Signal: {signal.signal}, Confidence: {signal.confidence}")
        ```
    """
    
    def __init__(self, config: DeepSeekConfig):
        """
        Initialize DeepSeek client.
        
        Args:
            config (DeepSeekConfig): Client configuration
            
        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(
            max_calls=config.rate_limit_per_minute,
            period=60.0
        )
        self._last_request_time: float = 0.0
        
        logger.info(
            f"Initialized DeepSeek client with base URL: {config.base_url}"
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self) -> None:
        """Ensure HTTP session exists and is open."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.debug("Created new HTTP session")
    
    async def close(self) -> None:
        """Close the HTTP session gracefully."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("Closed HTTP session")
    
    def _generate_signature(self, payload: Dict[str, Any]) -> str:
        """
        Generate HMAC-SHA256 signature for request authentication.
        
        Args:
            payload (Dict[str, Any]): Request payload to sign
            
        Returns:
            str: Hex-encoded signature string
            
        Raises:
            ValueError: If payload is empty or invalid
        """
        if not payload:
            raise ValueError("Payload cannot be empty for signature generation")
        
        # Sort payload keys for consistent signing
        sorted_payload = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        
        # Create HMAC-SHA256 signature
        signature = hmac.new(
            key=self.config.api_secret.encode('utf-8'),
            msg=sorted_payload.encode('utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()
        
        logger.debug(f"Generated signature for payload of length {len(sorted_payload)}")
        return signature
    
    def _validate_price_data(self, price_data: pd.DataFrame) -> None:
        """
        Validate price data format and content.
        
        Args:
            price_data (pd.DataFrame): OHLCV price data
            
        Raises:
            ValueError: If price data is invalid or missing required columns
            TypeError: If price_data is not a DataFrame
        """
        if not isinstance(price_data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(price_data).__name__}")
        
        if price_data.empty:
            raise ValueError("Price data DataFrame is empty")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check for required columns (case-insensitive)
        df_columns_lower = [col.lower() for col in price_data.columns]
        
        for col in required_columns:
            if col not in df_columns_lower:
                raise ValueError(f"Missing required column '{col}' in price data")
        
        # Check for NaN values in critical columns
        critical_cols_idx = [df_columns_lower.index(col) for col in required_columns]
        
        for idx in critical_cols_idx:
            col_name = price_data.columns[idx]
            if price_data[col_name].isna().any():
                raise ValueError(f"Column '{col_name}' contains NaN values")
        
        # Validate numeric types
        for idx in critical_cols_idx:
            col_name = price_data.columns[idx]
            if not np.issubdtype(price_data[col_name].dtype, np.number):
                raise TypeError(
                    f"Column '{col_name}' must be numeric, got {price_data[col_name].dtype}"
                )
        
        # Validate minimum data points for analysis (at least 50 candles)
        if len(price_data) < 50:
            raise ValueError(
                f"Insufficient data points: {len(price_data)}. Minimum required: 50"
            )
        
        logger.debug(f"Price data validated successfully: {len(price_data)} rows")
    
    def _prepare_analysis_payload(
        self,
        symbol: str,
        timeframe: TimeFrame,
        price_data: pd.DataFrame,
        analysis_type: AnalysisType = AnalysisType.TECHNICAL,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare the API request payload for analysis.
        
        Args:
            symbol (str): Trading pair symbol (e.g., "BTCUSDT")
            timeframe (TimeFrame): Analysis timeframe
            price_data (pd.DataFrame): OHLCV price data for analysis
            analysis_type (AnalysisType): Type of analysis to perform
            additional_context (Optional[Dict[str, Any]]): Additional context
            
        Returns:
            Dict[str, Any]: Prepared payload dictionary
            
        Raises:
            ValueError: If symbol is invalid or parameters are malformed
        """
        # Validate symbol format (alphanumeric + slash or just alphanumeric)
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        # Normalize symbol to uppercase and remove spaces
        symbol = symbol.strip().upper()
        
        if not all(c.isalnum() or c in ['/', '-', '_'] for c in symbol):
            raise ValueError(f"Symbol '{symbol}' contains invalid characters")
        
        # Convert price data to list format for API consumption
        df_lower_cols = {col.lower(): col for col in price_data.columns}
        
        ohlcv_data = []
        
        for idx in range(len(price_data)):
            row = {
                "timestamp": int(price_data.index[idx].timestamp()) 
                    if isinstance(price_data.index[idx], datetime) 
                    else int(idx),
                "open": float(price_data.iloc[idx][df_lower_cols['open']]),
                "high": float(price_data.iloc[idx][df_lower_cols['high']]),
                "low": float(price_data.iloc[idx][df_lower_cols['low']]),
                "close": float(price_data.iloc[idx][df_lower_cols['close']]),
                "volume": float(price_data.iloc[idx][df_lower_cols['volume']]),
            }
            
            ohlcv_data.append(row)
        
        # Build the payload structure according to DeepSeek V4 API spec
        payload = {
            "symbol": symbol,
            "timeframe": timeframe.value,
            "analysis_type": analysis_type.value,
            "market_data": {
                "ohlcv": ohlcv_data,
                "current_price": float(price_data.iloc[-1][df_lower_cols['close']]),
                "volume_24h": float(price_data.iloc[-24:][df_lower_cols['volume']].sum()) 
                    if len(price_data) >= 24 else float(price_data[df_lower_cols['volume']].sum()),
                "high_24h": float(price_data.iloc[-24:][df_lower_cols['high']].max())
                    if len(price_data) >= 24 else float(price_data[df_lower_cols['high']].max()),
                "low_24h": float(price_data.iloc[-24:][df_lower_cols['low']].min())
                    if len(price_data) >= 24 else float(price_data[df_lower_cols['low']].min()),
                "data_points": len(ohlcv_data),
                "start_timestamp": ohlcv_data[0]["timestamp"],
                "end_timestamp": ohlcv_data[-1]["timestamp"],
            },
            "analysis_parameters": {
                "include_indicators": [
                    "rsi", "macd", "bollinger_bands", "moving_averages",
                    "volume_profile", "support_resistance", "trend_lines",
                    "ichimoku", "fibonacci", "elliott_wave"
                ],
                "signal_strength_threshold": 0.7,
                "confidence_calculation": "weighted_average",
                "risk_assessment": True,
                "pattern_recognition": True,
                "market_regime_detection": True,
                "volatility_analysis": True,
            },
            "output_format": {
                "include_raw_indicators": True,
                "include_chart_patterns": True,
                "include_market_context": True,
                "include_trade_recommendations": True,
                "include_risk_warnings": True,
                "natural_language_summary": True,
                "max_tokens": 2000,
                "temperature": 0.3,
            },
            "metadata": {
                "client_version": "1.0.0",
                "request_timestamp": datetime.utcnow().isoformat(),
                "source": "trading_bot",
                **additional_context if additional_context else {}
            }
        }
        
        logger.debug(f"Prepared analysis payload for {symbol} ({timeframe.value})")
        
        return payload
    
    def _parse_signal_response(self, response_data: Dict[str, Any], symbol: str) -> TradingSignal:
        """
        Parse and validate the API response into a TradingSignal object.
        
        Args:
            response_data (Dict[str, Any]): Raw API response data
            symbol (str): Trading pair symbol
            
        Returns:
            TradingSignal: Parsed and validated trading signal
            
        Raises:
            ValueError: If response data is malformed or missing required fields
            KeyError: If required fields are missing from response
        """
        
        def safe_get(dictionary: Dict[str, Any], *keys: str) -> Any:
            """Safely get nested dictionary value."""
            
            current = dictionary
            
            for key in keys:
                if not isinstance(current, dict):
                    return None
                
                current = current.get(key)
                
                if current is None:
                    return None
            
            return current
        
        
         # Extract signal type from response
        
         signal_str=safe_get(response_data,"signal","type")or safe_get(response_data,"analysis","signal")or"NEUTRAL"
        
        
         try :
             signal_type=SignalType(signal_str.upper())
         except (ValueError ,AttributeError ):
             logger.warning(f"Unknown signal type '{signal_str}', defaulting to NEUTRAL")
             signal_type=SignalType.NEUTRAL
        
        
         # Extract confidence score
        
         confidence=safe_get(response_data,"signal","confidence")or safe_get(response_data,"analysis","confidence")or0.5
        
         try :
             confidence=float(confidence)
             confidence=max(0.0 ,min(1.0 ,confidence))
         except (TypeError ,ValueError ):
             confidence=0.5
        
        
         # Extract current price
        
         price=safe_get(response_data,"market","current_price")or safe_get(response_data,"price")or0.0
        
         try :
             price=float(price)
         except (TypeError ,ValueError ):
             price=0.0
        
        
         # Extract risk score
        
         risk_score=safe_get(response_data,"risk","score")or safe_get(response_data,"analysis","risk_score")or0.5
        
         try :
             risk_score=float(risk_score)
             risk_score=max(0.0 ,min(1.0 ,risk_score))
         except (TypeError ,ValueError ):
             risk_score=0.5
        
        
         # Parse technical indicators
        
         indicators=[]
         raw_indicators=safe_get(response_data,"indicators")or safe_get(response_data,"analysis","indicators")or[]
        
        
         if isinstance(raw_indicators ,list ):
             for ind in raw_indicators :
                 try :
                     indicator_name=str(ind.get("name","unknown"))
                     indicator_value=float(ind.get("value",0))
                     indicator_signal_str=str(ind.get("signal","NEUTRAL")).upper()
                    
                     try :
                         indicator_signal=SignalType(indicator_signal_str)
                     except ValueError :
                         indicator_signal=SignalType.NEUTRAL
                    
                     indicator=TechnicalIndicator(
                         name=indicator_name ,
                         value=indicator_value ,
                         signal=indicator_signal ,
                         timestamp=datetime .utcnow(),
                         metadata={k:v for k,v in ind.items()if k not in["name","value","signal"]}
                     )
                    
                     indicators.append(indicator)
                 except (ValueError ,TypeError ,KeyError )as e :
                     logger.warning(f"Failed to parse indicator:{e}")
        
        
         # Extract natural language analysis text
        
         analysis_text=safe_get(response_data,"analysis","summary")or safe_get(response_data,"summary")or""
        
        
         # Extract timeframe
        
         timeframe_str=safe_get(response_data,"timeframe")or"1h"
        
         try :
             timeframe=TimeFrame(timeframe_str)
         except ValueError :
             timeframe=TimeFrame.HOUR_1
        
        
         # Build final trading signal
        
         signal=TradingSignal(
             symbol=symbol ,
             signal=signal_type ,
             confidence=confidence ,
             price=price ,
             timestamp=datetime .utcnow(),
             timeframe=timeframe ,
             indicators=indicators ,
             analysis_text=str(analysis_text),
             risk_score=risk_score ,
             metadata={
                 k:v for k,v in response_data.items()
                 if k not in["signal","indicators","analysis","market","risk","timeframe","summary"]
             }
         )
        
        
         logger.info(
             f"Parsed signal for {symbol}: {signal.signal.value} "
             f"(confidence:{signal.confidence:.2f}, risk:{signal.risk_score:.2f})"
         )
        
        
         return signal
    
    
     @retry(
         stop=stop_after_attempt(3),
         wait=wait_exponential(multiplier=1 ,min=2 ,max=30),
         retry=retry_if_exception_type((aiohttp.ClientError ,asyncio.TimeoutError)),
         before_sleep=before_sleep_log(logger ,logging.WARNING),
     )
     async def _make_api_request(
         self ,
         endpoint :str ,
         payload :Dict[str ,Any]
     )->Dict[str ,Any]:
         """
          Make authenticated API request to DeepSeek with retry logic .
         
          Args :
              endpoint :API endpoint path (e .g .,"/analyze")
              payload :Request payload dictionary
         
          Returns :
              Dict[str ,Any]:API response data
         
          Raises :
              aiohttp .ClientError :On HTTP client errors after retries exhausted
              asyncio .TimeoutError :On request timeout after retries exhausted  
              ValueError :On invalid response format  
              RuntimeError :On authentication failure  
          """
          await self._ensure_session()
        
        
          # Apply rate limiting  
          acquired=await self.rate_limiter.acquire()
        
          if not acquired :
              wait_time=self.config.timeout/self.config.max_retries  
              logger.warning(f"Rate limited ,waiting {wait_time:.1f}s before retry")
              await asyncio.sleep(wait_time)
        
        
          # Prepare authenticated request  
          url=urljoin(self.config.base_url ,endpoint)
        
        
          # Add authentication headers  
          timestamp=str(int(time.time()*1000))
          nonce=hashlib.sha256(f"{timestamp}{self.config.api_key}".encode()).hexdigest()[:16]
        
        
          auth_payload={
              **payload ,
              "_auth":{"timestamp":timestamp ,"nonce":nonce}
          }
        
        
          signature=self._generate_signature(auth_payload)
        
        
          headers={
              "Content-Type":"application/json",
              "X-API-Key":self.config.api_key ,
              "X-Signature":signature ,
              "X-Timestamp":timestamp ,
              "X-Nonce":nonce ,
              "User-Agent":"DeepSeekTradingBot/1.0.0",
          }
        
        
          logger.debug(f"Making API request to {url} ({endpoint})")
        
        
          try :
              async with self.session.post(
                  url ,
                  json=auth_payload ,
                  headers=headers ,
                  timeout=aiohttp.ClientTimeout(total=self.config.timeout)
              )as response :
                  response_text=await response.text()
                
                
                  # Log response status  
                  logger.debug(f"API response status:{response.status}")
                
                
                  # Handle HTTP errors  
                  if response.status==401 :
                      raise RuntimeError("Authentication failed :Invalid API credentials")
                  elif response.status==429 :
                      retry_after=int(response.headers.get("Retry-After",30))
                      logger.warning(f"Rate limited by server .Retrying after {retry_after}s")
                      await asyncio.sleep(retry_after)
                      raise aiohttp.ClientError("Rate limited by server")
                  elif response.status==503 :
                      raise aiohttp.ClientError("Service temporarily unavailable")
                  elif response.status>=500 :
                      raise aiohttp.ClientError(f"Server error:{response.status}")
                  elif response.status>=400 :
                      error_detail=f"Client error {response.status}:{response_text[:500]}"
                      logger.error(error_detail)
                      raise ValueError(error_detail)
                
                
                  # Parse JSON response  
                  try :
                      response_data=json.loads(response_text)
                  except json.JSONDecodeError as e :
                      raise ValueError(f"Invalid JSON response:{e}")
                
                
                  # Validate response structure  
                  if not isinstance(response_data ,dict):
                      raise ValueError("Response must be a JSON object")
                
                
                  # Check for API-level errors  
                  api_error=response_data.get("error")or response_data.get("errors")
                
                  if api_error :
                      error_msg=str(api_error.get("message",api_error))
                      error_code=str(api_error.get("code","UNKNOWN"))
                      logger.error(f"API error [{error_code}]:{error_msg}")
                      raise ValueError(f"API error:{error_msg}")
                
                
                  logger.debug(f"Successfully received API response ({len(response_text)} bytes)")
                
                
                  return response_data
        
          except asyncio.TimeoutError :
              logger.error(f"Request timeout after {self.config.timeout}s")
              raise
        
          except aiohttp.ClientError as e :
              logger.error(f"HTTP client error:{e}")
              raise
        
          except Exception as e :
              logger.error(f"Unexpected error during API request:{e}")
              raise
    
    
     async def generate_trading_signal(
         self ,
         symbol :str ,
         timeframe :TimeFrame ,
         price_data :pd.DataFrame ,
         analysis_type :AnalysisType =AnalysisType.TECHNICAL ,
         additional_context :Optional[Dict[str ,Any]]=None ,
     )->TradingSignal :
         """
          Generate comprehensive trading signal using DeepSeek V4 .
         
          This method validates inputs ,prepares the analysis payload ,
          makes the API request with retry logic ,and parses the response .
         
          Args :
              symbol :Trading pair symbol (e .g .,"BTCUSDT")
              timeframe :Analysis timeframe  
              price_data :OHLCV price data as pandas DataFrame  
              analysis_type :Type of analysis to perform  
              additional_context :Optional additional context dictionary  
         
          Returns :
              TradingSignal :Complete trading signal with analysis details  
         
          Raises :
              ValueError :If inputs are invalid or API returns malformed response  
              RuntimeError :If authentication fails  
              ConnectionError :If unable to reach API after retries  
         
          Example :
              >>> df=get_price_history("BTCUSDT","4h",100)
              >>> signal=await client.generate_trading_signal("BTCUSDT",TimeFrame.HOUR_4 ,df)
              >>> print(f"{signal.signal.value}:{signal.confidence:.2%}")
          """
          start_time=time.monotonic()
        
        
          try :
              # Validate inputs  
              logger.info(f"Generating trading signal for {symbol} ({timeframe.value})")
            
            
              # Validate price data  
              self._validate_price_data(price_data)
            
            
              # Prepare analysis payload  
              payload=self._prepare_analysis_payload(
                  symbol=symbol ,
                  timeframe=timeframe ,
                  price_data=price_data ,
                  analysis_type=analysis_type ,
                  additional_context=additional_context ,
              )
            
            
              # Make API request  
              endpoint="/analyze/trading-signal"
            
            
              response_data=await self._make_api_request(endpoint ,payload)
            
            
              # Parse response into TradingSignal  
              signal=self._parse_signal_response(response_data ,symbol)
            
            
              elapsed_time=time.monotonic()-start_time
            
            
              logger.info(
                  f"Generated signal for {symbol}:{signal.signal.value} "
                  f"(confidence:{signal.confidence:.2f})in {elapsed_time:.2f}s"
              )
            
            
              return signal
        
          except ValidationError as e :
              logger.error(f"Input validation error:{e}")
              raise ValueError(f"Invalid input parameters:{e}")
        
          except ValueError as e :
              logger.error(f"Value error generating signal:{e}")
            
              # Return error signal instead of raising exception  
              return TradingSignal(
                  symbol=symbol ,
                  signal=SignalType.ERROR ,
                  confidence=0.0 ,
                  price=float(price_data.iloc[-1]["close"])if len(price_data)>0 else 0.0 ,
                  timestamp=datetime .utcnow(),
                  timeframe=timeframe ,
                  analysis_text=f"Error generating signal:{str(e)}",
                  risk_score=1.0 ,
                  metadata={"error":str(e)}
              )
        
          except Exception as e :
              logger.critical(f"Critical error generating trading signal:{e}",exc_info=True)
            
              return TradingSignal(
                  symbol=symbol ,
                  signal=SignalType.ERROR ,
                  confidence=0.0 ,
                  price=float(price_data.iloc[-1]["close"])if len(price_data)>0 else 0.0 ,
                  timestamp=datetime .utcnow(),
                  timeframe=timeframe ,
                  analysis_text=f"Critical system error:{str(e)}",
                  risk_score=1.0 ,
                  metadata={"error":"critical_system_error"}
              )
    
    
     async def batch_generate_signals(
         self ,
         symbols :List[str],
         timeframe :TimeFrame ,
         price_data_dict :Dict[str ,pd.DataFrame],
         analysis_type :AnalysisType =AnalysisType.TECHNICAL ,
     )->Dict[str ,TradingSignal]:
         """
          Generate trading signals for multiple symbols concurrently .
         
          Args :
              symbols :List of trading pair symbols  
              timeframe :Analysis timeframe  
              price_data_dict :Dictionary mapping symbols to their OHLCV DataFrames  
              analysis_type :Type of analysis to perform  
         
          Returns :
              Dict[str ,TradingSignal]:Dictionary mapping symbols to their signals  
         
          Raises :
              ValueError :If symbols list is empty or mismatched with data dict  
         
          Example :
              >>> data_dict={"BTCUSDT":df1,"ETHUSDT":df2}
              >>> signals=await client.batch_generate_signals(["BTCUSDT","ETHUSDT"],TimeFrame.HOUR_4 ,data_dict)
          """
          if not symbols :
              raise ValueError("Symbols list cannot be empty")
        
        
          if set(symbols)!=set(price_data_dict.keys()):
              missing=[s for s in symbols if s not in price_data_dict]
              extra=[s for s in price_data_dict if s not in symbols]
            
              error_msg=[]
            
              if missing :
                  error_msg.append(f"Missing data for symbols:{missing}")
            
              if extra :
                  error_msg.append(f"Extra symbols without request:{extra}")
            
            
              raise ValueError(";".join(error_msg))
        
        
          tasks=[]
        
        
          for symbol in symbols :
              task=self.generate_trading_signal(
                  symbol=symbol ,
                  timeframe=timeframe ,
                  price_data=price_data_dict[symbol],
                  analysis_type=analysis_type ,
                  additional_context={"batch_request":True}
              )
            
            
              tasks.append((symbol ,task))
        
        
          results={}
        
        
          # Execute all tasks concurrently with timeout  
          timeout=aiohttp.ClientTimeout(total=self.config.timeout*2)
        
        
          async def execute_with_timeout(symbol ,coro ):
               try :
                   result=await asyncio.wait_for(coro ,timeout=self.config.timeout*2)
                   return symbol ,result ,None  
               except Exception as e :
                   return symbol ,None ,str(e)
        
        
          pending=[execute_with_timeout(sym ,task)for sym ,task in tasks]
        
        
          completed_results=[]
        
        
          try :
               completed_results.extend(
                   await asyncio.gather(*pending ,return_exceptions=True)
               )
          except Exception as e :
               logger.error(f"Batch execution error:{e}")
        
        
          for item in completed_results :
               if isinstance(item ,tuple )and len(item)==3 :
                   symbol ,result ,error_str=None ,None ,None 
                
                   try :
                       symbol ,result ,error_str=None ,None ,None 
                       symbol ,result ,error_str=None ,None ,None 
                    
                       # Unpack tuple safely  
                       symbol=str(item[0])
                       result=TradingSignal(**item[1])if isinstance(item[1],dict )else item[1]
                       error_str=str(item[2])if item[2]else None 
                   except Exception as unpack_err :
                       logger.error(f"Failed to unpack batch result:{unpack_err}")
                       continue 
                
                
                   if error_str or result is None or result.signal==SignalType.ERROR :
                       logger.warning(f"Failed to generate signal for {symbol}:{error_str}")
                    
                       results[symbol]=TradingSignal(
                           symbol=symbol ,
                           signal=SignalType.ERROR ,
                           confidence=0.0 ,
                           price=0.0 ,
                           timestamp=datetime .utcnow(),
                           timeframe=timeframe ,
                           analysis_text=f"Batch generation failed:{error_str}"if error_str else"Unknown error",
                           risk_score=1.0 ,
                           metadata={"batch_error":error_str}
                       )
                   else :
                       results[symbol]=result
        
        
          logger.info(
               f"Batch generation complete:{len(results)}/{len(symbols)} signals generated"
          )
        
        
          return results
    
    
     async def health_check(self)->Dict[str ,Any]:
         """
          Check API health and connectivity .
         
          Returns :
               Dict[str ,Any]:Health check results including latency and status  
         
          Raises :
               ConnectionError :If unable to connect to API  
               RuntimeError :If authentication fails during health check  
         
          Example :
               >>> health_status=await client.health_check()
               >>> print(f"API Status:{health_status['status']}")
          """
          start_time=time.monotonic()
        
        
          try :
               await self._ensure_session()
            
            
               url=f"{self.config.base_url}/health"
               headers={
                   "X-API-Key":self.config.api_key ,
                   "User-Agent":"DeepSeekTradingBot/1.0.0",
               }
            
            
               async with self.session.get(url ,headers=headers)as response :
                   latency=(time.monotonic()-start_time)*1000  # Convert to ms 
                
                
                   if response.status==200 :
                       try :
                           data=await response.json()
                       except Exception :
                           data={"status":"healthy"}
                    
                    
                       result={
                           "status":"healthy",
                           "latency_ms":round(latency ,2),
                           "api_version":"v4",
                           "timestamp":datetime .utcnow().isoformat(),
                           **data 
                       }
                    
                    
                       logger.info(f"Health check passed (latency:{latency:.2f}ms)")
                    
                    
                       return result 
                
                   elif response.status==401 :
                       raise RuntimeError("Authentication failed during health check")
                
                   else :
                       raise ConnectionError(
                           f"Health check failed with status {response.status}"
                       )
        
          except asyncio.TimeoutError :
               elapsed=(time.monotonic()-start_time)*1000
            
            
               logger.error(f"Health check timed out after {elapsed:.2f}ms")
            
            
               return{
                   "status":"unreachable",
                   "latency_ms":round(elapsed ,2),
                   "error":"Connection timeout",
                   "timestamp":datetime .utcnow().isoformat(),
               }
        
          except Exception as e :
               elapsed=(time.monotonic()-start_time)*1000
            
            
               logger.error(f"Health check failed:{e}")
            
            
               return{
                   "status":"error",
                   "latency_ms":round(elapsed ,2),
                   "error":str(e),
                   "timestamp":datetime .utcnow().isoformat(),
               }


# Convenience function for quick signal generation

async def get_trading_signal(
     api_key :str ,
     api_secret :str ,
     symbol :str ="BTCUSDT",
     timeframe :TimeFrame =TimeFrame.HOUR_4 ,
     price_df :Optional[pd.DataFrame]=None ,
)->TradingSignal:
     """
      Quick convenience function to generate a trading signal .
     
      This function creates a temporary client instance and generates a signal .
      Useful for simple scripts and testing .
     
      Args :
           api_key :DeepSeek API key  
           api_secret :DeepSeek API secret  
           symbol :Trading pair symbol (default:"BTCUSDT")
           timeframe :Analysis timeframe (default:HOUR_4)
           price_df :OHLCV DataFrame .If None,a simple test DataFrame is created .
     
      Returns :
           TradingSignal :Generated trading signal  
     
      Example :
           >>> import pandas as pd 
           >>> df=get_test_price_data()
           >>> signal=await get_trading_signal("key","secret",price_df=df)
           >>> print(signal.signal.value,signal.confidence)
      """
      config=None 
      try :
           config_obj=None 
           config_obj=None 
           config_obj=None 
           config_obj=None