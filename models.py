"""
Pydantic models for request/response validation and data schemas.

This module defines all data models used throughout the trading bot application,
including API request/response schemas, database models, and WebSocket message formats.
All models use Pydantic for robust validation and serialization.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, ConfigDict, field_validator
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated


# ─── Enums ───────────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    """Trading order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Trading order types supported by Binance."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"


class TimeInForce(str, Enum):
    """Order time-in-force instructions."""
    GTC = "GTC"  # Good till cancelled
    IOC = "IOC"  # Immediate or cancel
    FOK = "FOK"  # Fill or kill


class SignalType(str, Enum):
    """Trading signal types from AI analysis."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class Interval(str, Enum):
    """Kline/candlestick intervals."""
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class MessageRole(str, Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class WebSocketEvent(str, Enum):
    """WebSocket event types."""
    PRICE_UPDATE = "price_update"
    SIGNAL_GENERATED = "signal_generated"
    ORDER_EXECUTED = "order_executed"
    ERROR = "error"
    CONNECTION_STATUS = "connection_status"


# ─── Custom Types ────────────────────────────────────────────────────────────

# Decimal type with proper validation for financial amounts
PriceDecimal = Annotated[
    Decimal,
    BeforeValidator(lambda v: Decimal(str(v)) if not isinstance(v, Decimal) else v),
]


# ─── Base Models ─────────────────────────────────────────────────────────────

class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            Decimal: str,
            datetime: lambda dt: dt.isoformat(),
        },
        str_strip_whitespace=True,
        str_min_length=1,
        extra="forbid",  # Reject unknown fields
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


# ─── API Request Models ──────────────────────────────────────────────────────

class ChatRequest(BaseSchema):
    """Request model for chat/completion endpoint."""
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User message to the AI assistant",
        examples=["Analyze BTCUSDT for potential buy signals"],
    )
    
    conversation_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Conversation ID for maintaining chat history",
        examples=["conv_abc123"],
    )
    
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context for the AI model (e.g., market data)",
        examples=[{"symbol": "BTCUSDT", "interval": "1h"}],
    )


class MarketDataRequest(BaseSchema):
    """Request model for fetching market data."""
    
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        pattern=r"^[A-Z0-9]{5,20}$",
        description="Trading pair symbol (e.g., BTCUSDT)",
        examples=["BTCUSDT"],
    )
    
    interval: Interval = Field(
        ...,
        description="Candlestick interval",
        examples=[Interval.HOUR_1],
    )
    
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of candles to fetch",
        examples=[100],
    )
    
    start_time: Optional[int] = Field(
        None,
        ge=0,
        description="Start time in milliseconds since epoch",
        examples=[1700000000000],
    )
    
    end_time: Optional[int] = Field(
        None,
        ge=0,
        description="End time in milliseconds since epoch",
        examples=[1700086400000],
    )


class OrderRequest(BaseSchema):
    """Request model for placing a trading order."""
    
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        pattern=r"^[A-Z0-9]{5,20}$",
        description="Trading pair symbol",
        examples=["BTCUSDT"],
    )
    
    side: OrderSide = Field(..., description="Order side")
    
    type: OrderType = Field(..., description="Order type")
    
    quantity: PriceDecimal = Field(
        ...,
        gt=Decimal("0"),
        description="Order quantity",
        examples=["0.001"],
    )
    
    price: Optional[PriceDecimal] = Field(
        None,
        gt=Decimal("0"),
        description="Order price (required for LIMIT orders)",
        examples=["50000.00"],
    )
    
    stop_price: Optional[PriceDecimal] = Field(
        None,
        gt=Decimal("0"),
        description="Stop price (required for stop orders)",
        examples=["48000.00"],
    )
    
    time_in_force: Optional[TimeInForce] = Field(
        None,
        description="Time-in-force for limit orders",
        examples=[TimeInForce.GTC],
    )
    
    @field_validator("price")
    @classmethod
    def validate_price_for_limit(cls, v: Optional[Decimal], info) -> Optional[Decimal]:
        """Ensure price is provided for LIMIT orders."""
        if info.data.get("type") == OrderType.LIMIT and v is None:
            raise ValueError("Price is required for LIMIT orders")
        return v
    
    @field_validator("stop_price")
    @classmethod
    def validate_stop_price(cls, v: Optional[Decimal], info) -> Optional[Decimal]:
        """Ensure stop_price is provided for stop orders."""
        order_type = info.data.get("type")
        if order_type in (OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT,
                          OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT):
            if v is None:
                raise ValueError(f"Stop price is required for {order_type.value} orders")
        return v


class AnalysisRequest(BaseSchema):
    """Request model for AI technical analysis."""
    
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        pattern=r"^[A-Z0-9]{5,20}$",
        description="Trading pair symbol",
        examples=["BTCUSDT"],
    )
    
    interval: Interval = Field(
        default=Interval.HOUR_1,
        description="Analysis timeframe",
        examples=[Interval.HOUR_1],
    )
    
    include_news: bool = Field(
        default=False,
        description="Include recent news in analysis",
    )
    
    custom_prompt: Optional[str] = Field(
        None,
        max_length=2000,
        description="Custom instructions for the AI model",
        examples=["Focus on RSI divergence patterns"],
    )


# ─── Response Models ─────────────────────────────────────────────────────────

class ChatResponse(BaseSchema):
    """Response model for chat/completion endpoint."""
    
    message: str = Field(..., description="AI assistant response")
    
    conversation_id: str = Field(..., description="Conversation ID")
    
    tokens_used: int = Field(..., ge=0, description="Number of tokens used")
    
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    
    model_version: str = Field(
        ...,
        description="AI model version used",
        examples=["deepseek-v4-20240101"],
    )


class MarketDataResponse(BaseSchema):
    """Response model for market data."""
    
    symbol: str = Field(..., description="Trading pair symbol")
    
    interval: Interval = Field(..., description="Candlestick interval")
    
    candles: List[Dict[str, Any]] = Field(
        ...,
        description="List of candlestick data",
        examples=[[{
            "open_time": 1700000000000,
            "open": "50000.00",
            "high": "51000.00",
            "low": "49000.00",
            "close": "50500.00",
            "volume": "100.50",
            "close_time": 1700086400000,
            "quote_volume": "5075250.00",
            "trades": 15000,
            "taker_buy_base": 50.25,
            "taker_buy_quote": 2537625.00,
            "ignore": False
        }]],
    )
    
    total_candles: int = Field(..., ge=0, description="Total number of candles returned")


class OrderResponse(BaseSchema):
    """Response model for order operations."""
    
    order_id: int = Field(..., description="Binance order ID")
    
    symbol: str = Field(..., description="Trading pair symbol")
    
    client_order_id: str = Field(..., description="Client-assigned order ID")
    
    transact_time: int = Field(..., ge=0, description="Transaction time in milliseconds")
    
    price: PriceDecimal = Field(..., description="Order price")
    
    orig_qty: PriceDecimal = Field(..., description="Original order quantity")
    
    executed_qty: PriceDecimal = Field(..., description="Executed quantity")
    
    cummulative_quote_qty: PriceDecimal = Field(
        ...,
        description="Cumulative quote asset quantity",
    )
    
    status: str = Field(..., description="Order status")
    
    type: OrderType = Field(..., description="Order type")
    
    side: OrderSide = Field(..., description="Order side")
    
    fills: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Order fill details",
    )


class AnalysisResponse(BaseSchema):
    """Response model for AI analysis results."""
    
    symbol: str = Field(..., description="Trading pair symbol")
    
    interval: Interval = Field(..., description="Analysis timeframe")
    
    signal: SignalType = Field(..., description="Trading signal")
    
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)",
        examples=[0.85],
    )
    
    analysis_text: str = Field(..., description="Detailed analysis explanation")
    
    indicators_summary: Dict[str, Any] = Field(
        ...,
        description="Summary of technical indicators used",
        examples=[{
            "rsi": 45.2,
            "macd": {"value": 150.5, "signal": 148.2},
            "moving_averages": {"ma50": 49000, "ma200": 47000},
            "support_resistance": {"support": 48000, "resistance": 52000},
            "volume_trend": "increasing",
            "volatility": {"current": 2.5, "average": 2.0},
            "pattern_detected": ["bullish_flag", "double_bottom"],
            "market_sentiment": {"bullish": 55, "bearish": 45},
            "funding_rate": 0.01,
            "open_interest_change": 2.5
        }],
        
        

)
    
risk_assessment: Dict[str, Any] = Field(
...,
description="Risk assessment metrics",
examples=[{
"risk_level": "medium",
"stop_loss_suggested": 48000.00,
"take_profit_suggested": 52000.00,
"risk_reward_ratio": 2.5,
"max_drawdown_potential": 5.0
}],
)

timestamp: datetime = Field(
default_factory=datetime.utcnow,
description="Analysis timestamp",
)


class ErrorResponse(BaseSchema):
"""Standard error response model."""

error_code: str = Field(..., description="Error code identifier")

message: str = Field(..., description="Human-readable error message")

details: Optional[Dict[str, Any]] = Field(
None,
description="Additional error details",
)

timestamp: datetime = Field(
default_factory=datetime.utcnow,
description="Error timestamp",
)

request_id: Optional[str] = Field(
None,
description="Request ID for tracing",
)


# ─── WebSocket Models ────────────────────────────────────────────────────────

class WebSocketMessage(BaseSchema):
"""Base WebSocket message model."""

event_type: WebSocketEvent = Field(..., description="Type of WebSocket event")

data: Dict[str, Any] = Field(..., description="Event payload data")

timestamp: datetime = Field(
default_factory=datetime.utcnow,
description="Message timestamp",
)


class PriceUpdateMessage(WebSocketMessage):
"""WebSocket message for real-time price updates."""

event_type: WebSocketEvent = WebSocketEvent.PRICE_UPDATE

data: Dict[str, Any] = Field(
...,
description="Price update data",
examples=[{
"symbol": "BTCUSDT",
"price": 50500.50,
"change_24h": 2.5,
"volume_24h": 150000.75
}],
)


class SignalMessage(WebSocketMessage):
"""WebSocket message for trading signals."""

event_type: WebSocketEvent = WebSocketEvent.SIGNAL_GENERATED

data: Dict[str, Any] = Field(
...,
description="Signal data",
examples=[{
"symbol": "BTCUSDT",
"signal": "BUY",
"confidence": 0.82,
"reasoning": "Bullish flag pattern detected with increasing volume"
}],
)


# ─── Database Models ─────────────────────────────────────────────────────────

class ConversationHistory(BaseSchema, TimestampMixin):
"""Model for chat conversation history."""

conversation_id: str = Field(
...,
max_length=100,
description="Unique conversation identifier",
)

messages: List[Dict[str, Any]] = Field(
default_factory=list,
description="List of messages in the conversation",
examples=[[{
"role": MessageRole.USER.value,
"content": "Analyze BTCUSDT",
"timestamp": datetime.now().isoformat()
}]],
)

user_id: Optional[str] = Field(
None,
max_length=100,
description="User identifier (if authenticated)",
)

metadata: Dict[str, Any] = Field(
default_factory=dict,
description="Additional conversation metadata",
)


class TradingSignal(BaseSchema, TimestampMixin):
"""Model for storing trading signals."""

signal_id: str = Field(
...,
max_length=100,
description="Unique signal identifier",
)

symbol: str = Field(..., max_length=20)

interval: Interval

signal_type: SignalType

confidence_score: float

analysis_text: str

indicators_summary: Dict[str, Any]

risk_assessment: Dict[str, Any]

executed_order_id: Optional[int] = Field(
None,
description="Order ID if signal was executed",
)

is_active: bool = Field(default=True)

expires_at: Optional[datetime] = None


# ─── Configuration Models ────────────────────────────────────────────────────

class TradingConfig(BaseSchema):
"""Trading bot configuration model."""

max_position_size_usdt: PriceDecimal

min_position_size_usdt: PriceDecimal

max_leverage: int

stop_loss_percentage: float

take_profit_percentage: float

allowed_symbols: List[str]

trading_enabled: bool

max_daily_trades: int

risk_per_trade_percentage: float


class AIConfig(BaseSchema):
"""AI model configuration."""

model_name: str

temperature: float

max_tokens: int

top_p: float

frequency_penalty: float

presence_penalty: float


# ─── Utility Functions ──────────────────────────────────────────────────────

def create_error_response(
error_code: str,
message: str,
details: Optional[Dict[str, Any]] = None
) -> ErrorResponse:
"""Create a standardized error response.

Args:
error_code: Unique error code identifier.
message: Human-readable error message.
details: Optional additional error details.

Returns:
ErrorResponse instance.
"""
return ErrorResponse(
error_code=error_code,
message=message,
details=details or {},
)


def create_success_response(data: Any) -> Dict[str, Any]:
"""Create a standardized success response wrapper.

Args:
data: Response data to wrap.

Returns:
Dictionary with success status and data.
"""
return {
"success": True,
"data": data if isinstance(data, dict) else data.model_dump(),
"timestamp": datetime.utcnow().isoformat(),
}