"""
Chat Handler Module for AI-Powered Trading Bot

This module manages the conversation flow between users and the AI trading assistant.
It processes user messages, generates contextual responses using DeepSeek V4 model,
and maintains chat history with proper session management.

Key Features:
- Message processing and validation
- Conversation context management
- Integration with DeepSeek V4 for technical analysis
- Chat history persistence
- Error handling and logging
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

# Third-party imports
import aiohttp
import bleach
from pydantic import BaseModel, Field, ValidationError, validator

# Local imports (assuming these exist in the project)
from database import DatabaseManager
from models import Message, Conversation, UserSession
from trading_analyzer import TradingAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Pydantic model for validating incoming chat messages."""
    content: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    @validator('content')
    def sanitize_content(cls, v: str) -> str:
        """Sanitize and validate message content."""
        # Remove potentially dangerous HTML/script tags
        sanitized = bleach.clean(v, tags=[], strip=True)
        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())
        return sanitized


class ChatResponse(BaseModel):
    """Pydantic model for chat response structure."""
    message: str = Field(..., max_length=10000)
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: Optional[Dict[str, Any]] = None
    actions: Optional[List[Dict[str, Any]]] = None


class ChatHandler:
    """
    Main handler for chat conversations in the AI trading bot.
    
    Manages message processing, AI response generation, conversation context,
    and integration with trading analysis features.
    """

    def __init__(
        self,
        deepseek_api_key: str,
        database_manager: DatabaseManager,
        trading_analyzer: TradingAnalyzer,
        max_history_length: int = 50,
        session_timeout_minutes: int = 30
    ):
        """
        Initialize the ChatHandler with required dependencies.

        Args:
            deepseek_api_key: API key for DeepSeek V4 model
            database_manager: Database manager instance for persistence
            trading_analyzer: Trading analyzer for market data
            max_history_length: Maximum number of messages to keep in context
            session_timeout_minutes: Session timeout duration in minutes
        """
        self.deepseek_api_key = deepseek_api_key
        self.db = database_manager
        self.analyzer = trading_analyzer
        self.max_history_length = max_history_length
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
        # In-memory session cache for faster access
        self._session_cache: Dict[str, Dict[str, Any]] = {}
        
        # DeepSeek API endpoint
        self.deepseek_endpoint = "https://api.deepseek.com/v4/chat/completions"
        
        logger.info("ChatHandler initialized successfully")

    async def process_message(
        self,
        message: ChatMessage,
        user_session: Optional[UserSession] = None
    ) -> ChatResponse:
        """
        Process an incoming chat message and generate a response.

        Args:
            message: Validated chat message from user
            user_session: Optional user session object

        Returns:
            ChatResponse containing the AI response and metadata

        Raises:
            ValueError: If message processing fails validation
            ConnectionError: If unable to connect to DeepSeek API
        """
        try:
            # Validate input message
            if not isinstance(message, ChatMessage):
                raise ValueError("Invalid message format")

            # Get or create session
            session_id = message.session_id or str(uuid4())
            
            # Update session activity
            await self._update_session(session_id, user_session)

            # Retrieve conversation history
            conversation_history = await self._get_conversation_history(session_id)

            # Analyze message intent and context
            intent_data = await self._analyze_intent(message.content)

            # Generate contextual response using DeepSeek V4
            response_content, actions = await self._generate_response(
                message.content,
                conversation_history,
                intent_data,
                message.context
            )

            # Save messages to database
            await self._save_conversation(
                session_id=session_id,
                user_message=message.content,
                ai_response=response_content,
                intent_data=intent_data,
                actions=actions
            )

            # Prepare response object
            response = ChatResponse(
                message=response_content,
                session_id=session_id,
                context={
                    'intent': intent_data.get('intent'),
                    'sentiment': intent_data.get('sentiment'),
                    'market_data': intent_data.get('market_data')
                },
                actions=actions
            )

            logger.info(f"Message processed successfully for session {session_id}")
            return response

        except ValidationError as e:
            logger.error(f"Message validation failed: {e}")
            raise ValueError(f"Invalid message format: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            raise

    async def _analyze_intent(self, message_content: str) -> Dict[str, Any]:
        """
        Analyze the intent and context of a user message.

        Args:
            message_content: The user's message content

        Returns:
            Dictionary containing intent analysis results
        """
        intent_data = {
            'intent': 'general',
            'sentiment': 'neutral',
            'market_data': None,
            'requires_analysis': False,
            'commands': []
        }

        # Detect trading-related intents using pattern matching
        trading_patterns = {
            'price_check': r'\b(price|value|worth|cost)\b.*\b(btc|eth|bnb|coin|token|crypto)\b',
            'analysis': r'\b(analyze|analysis|chart|pattern|trend|indicator)\b',
            'trade': r'\b(buy|sell|trade|order|position)\b',
            'portfolio': r'\b(portfolio|balance|holdings|assets)\b',
            'alert': r'\b(alert|notify|watch|monitor)\b'
        }

        content_lower = message_content.lower()
        
        for intent_type, pattern in trading_patterns.items():
            if re.search(pattern, content_lower):
                intent_data['intent'] = intent_type
                intent_data['requires_analysis'] = True if intent_type in ['analysis', 'price_check'] else False
                break

        # Detect sentiment in the message
        sentiment_patterns = {
            'positive': r'\b(good|great|excellent|bullish|profit|gain|up)\b',
            'negative': r'\b(bad|poor|terrible|bearish|loss|down|crash)\b',
            'uncertain': r'\b(maybe|perhaps|uncertain|confused|question|help)\b'
        }

        for sentiment, pattern in sentiment_patterns.items():
            if re.search(pattern, content_lower):
                intent_data['sentiment'] = sentiment
                break

        # Extract potential commands or ticker symbols
        ticker_pattern = r'\b([A-Z]{2,5})\b'
        potential_tickers = re.findall(ticker_pattern, message_content.upper())
        
        if potential_tickers:
            intent_data['commands'] = potential_tickers[:5]  # Limit to 5 tickers

        return intent_data

    async def _generate_response(
        self,
        user_message: str,
        conversation_history: List[Dict[str, Any]],
        intent_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        Generate AI response using DeepSeek V4 model.

        Args:
            user_message: The user's message
            conversation_history: Previous conversation messages
            intent_data: Analyzed intent information
            context: Additional context from the request

        Returns:
            Tuple of (response_text, optional_actions)
        """
        actions = []
        
        # Prepare system prompt with trading context
        system_prompt = self._build_system_prompt(intent_data)

        # Prepare conversation messages for API call
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (limited to maintain context window)
        history_to_include = conversation_history[-self.max_history_length:]
        for msg in history_to_include:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        try:
            # Call DeepSeek V4 API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.deepseek_endpoint,
                    headers={
                        "Authorization": f"Bearer {self.deepseek_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-v4",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 1000,
                        "stream": False
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API error: {response.status} - {error_text}")
                        return self._get_fallback_response(intent_data), None

                    api_response = await response.json()
                    
                    # Extract response content
                    response_text = api_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # Parse any suggested actions from the response
                    actions = self._parse_actions_from_response(response_text)

                    return response_text, actions if actions else None

        except aiohttp.ClientError as e:
            logger.error(f"API connection error: {e}")
            return self._get_fallback_response(intent_data), None
        except json.JSONDecodeError as e:
            logger.error(f"API response parsing error: {e}")
            return self._get_fallback_response(intent_data), None

    def _build_system_prompt(self, intent_data: Dict[str, Any]) -> str:
        """
        Build a contextual system prompt based on detected intent.

        Args:
            intent_data: Analyzed intent information

        Returns:
            System prompt string for the AI model
        """
        base_prompt = (
            "You are an expert AI trading assistant with deep knowledge of cryptocurrency markets. "
            "Provide professional, data-driven insights while managing risk appropriately. "
            "Always include risk disclaimers when discussing trading decisions."
        )

        if intent_data['intent'] == 'price_check':
            base_prompt += (
                "\n\nWhen providing price information, include relevant market context such as "
                "24h change, volume trends, and support/resistance levels."
            )
        
        elif intent_data['intent'] == 'analysis':
            base_prompt += (
                "\n\nFor technical analysis requests, consider multiple timeframes and indicators. "
                "Provide balanced perspectives including both bullish and bearish scenarios."
            )
        
        elif intent_data['intent'] == 'trade':
            base_prompt += (
                "\n\nFor trading suggestions, always emphasize risk management. "
                "Include position sizing recommendations and stop-loss suggestions."
                "\nIMPORTANT: Include a disclaimer that this is not financial advice."
            )

        return base_prompt

    def _get_fallback_response(self, intent_data: Dict[str, Any]) -> str:
        """
        Generate a fallback response when AI service is unavailable.

        Args:
            intent_data: Analyzed intent information

        Returns:
            Fallback response string
        """
        fallback_messages = {
            'price_check': (
                "I apologize, but I'm currently unable to fetch real-time price data. "
                "Please try again in a few moments."
            ),
            'analysis': (
                "I'm sorry, but the analysis service is temporarily unavailable. "
                "Please check back later for technical analysis."
            ),
            'trade': (
                "I apologize for the inconvenience. The trading advisory service is "
                "currently experiencing issues. Please try again shortly."
            ),
            'general': (
                "I'm having trouble processing your request at the moment. "
                "Please try again later."
            )
        }

        return fallback_messages.get(intent_data['intent'], fallback_messages['general'])

    def _parse_actions_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse actionable items from the AI response.

        Args:
            response_text: The AI-generated response text

        Returns:
            List of action dictionaries with type and data fields
        """
        actions = []
        
        # Pattern matching for common action types
        action_patterns = {
            'chart_request': r'\[CHART:\s*(.+?)\]',
            'alert_setup': r'\[ALERT:\s*(.+?)\]',
            'trade_suggestion': r'\[TRADE:\s*(.+?)\]',
            'data_query': r'\[DATA:\s*(.+?)\]'
        }

        for action_type, pattern in action_patterns.items():
            matches = re.findall(pattern, response_text)
            
            for match in matches[:3]:  # Limit to 3 actions per type
                actions.append({
                    'type': action_type,
                    'data': match.strip(),
                    'timestamp': datetime.utcnow().isoformat()
                })

                # Remove the action markup from the response text for cleaner display
                response_text = response_text.replace(f'[{action_type.split("_")[0]}:{match}]', '')

        return actions

    async def _update_session(
        self,
        session_id: str,
        user_session: Optional[UserSession] = None
    ) -> None:
        """
        Update or create a new chat session.

        Args:
            session_id: Unique session identifier
            user_session: Optional user session information
        """
        current_time = datetime.utcnow()
        
        # Update cache with current session data
        self._session_cache[session_id] = {
            'last_active': current_time,
            'user_id': user_session.user_id if user_session else None,
            'session_start': self._session_cache.get(session_id, {}).get('session_start', current_time)
        }

    async def _get_conversation_history(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
         Retrieve conversation history for a given session.

         Args:
             session_id: Unique session identifier
             limit: Maximum number of messages to retrieve

         Returns:
             List of conversation messages with role and content fields

         Raises:
             ValueError: If session_id is invalid or empty
         """
         if not session_id or not isinstance(session_id, str):
             raise ValueError("Invalid session ID")

         try:
             # Attempt to retrieve from database first (primary storage)
             history_from_db = await self.db.get_conversation_history(
                 session_id=session_id,
                 limit=limit,
                 include_metadata=True  # Include timestamps and other metadata if available
             )

             if history_from_db is not None and len(history_from_db) > 0:
                 logger.debug(f"Retrieved {len(history_from_db)} messages from DB for session {session_id}")
                 return history_from_db

             # Fallback to cache if DB returns empty or None (e.g., new session)
             cached_history = self._session_cache.get(session_id, {}).get('history', [])
             logger.debug(f"Retrieved {len(cached_history)} messages from cache for session {session_id}")
             return cached_history[-limit:]  # Return only the last `limit` messages from cache

         except Exception as e:
             logger.error(f"Error retrieving conversation history for session {session_id}: {e}", exc_info=True)
             # Return empty list on error to allow graceful degradation (new conversation)
             return []

     async def _save_conversation(
         self,
         session_id: str,
         user_message: str,
         ai_response: str,
         intent_data: Dict[str, Any],
         actions: Optional[List[Dict[str, Any]]] = None,
         metadata: Optional[Dict[str, Any]] = None  # Allow passing additional metadata like tokens used or latency.
     ) -> bool:
         """
         Save a conversation turn (user message + AI response) to both cache and database.

         Args:
             session_id: Unique session identifier.
             user_message: The original user message content.
             ai_response: The generated AI response content.
             intent_data: Analyzed intent data associated with the user message.
             actions: Optional list of parsed actions from the AI response.
             metadata: Optional dictionary with additional metadata (e.g., model latency).

         Returns:
             True if saving was successful (at least to cache), False otherwise.
         """
         timestamp_utc = datetime.utcnow().isoformat()

         # Prepare structured data for both user and assistant messages.
         user_entry = {
             "role": "user",
             "content": user_message,
             "timestamp": timestamp_utc,
             "intent": intent_data.get('intent'),
             "sentiment": intent_data.get('sentiment'),
             "session_id": session_id,
             **(metadata or {})  # Include any passed metadata.
         }

         assistant_entry = {
             "role": "assistant",
             "content": ai_response,
             "timestamp": timestamp_utc,
             "actions": actions or [],
             "session_id": session_id,
             **(metadata or {})
         }

         # Update in-memory cache first (fastest).
         try:
             if session_id not in self._session_cache:
                 self._session_cache[session_id] = {'history': []}
             
             cache_history = self._session_cache[session_id].setdefault('history', [])
             cache_history.append(user_entry)
             cache_history.append(assistant_entry)

             # Trim cache to prevent memory bloat (keep last N entries).
             if len(cache_history) > (self.max_history_length * 2):  # *2 because each turn has two entries.
                 self._session_cache[session_id]['history'] = cache_history[-(self.max_history_length * 2):]

         except Exception as e:
             logger.error(f"Failed to update cache for session {session_id}: {e}")
             # Continue to try database save even if cache fails.

         # Persist to database asynchronously (fire-and-forget with logging).
         try:
             await self.db.save_messages_batch(
                 messages=[user_entry, assistant_entry],
                 session_id=session_id,
                 overwrite_existing=False  # Append mode.
             )
             logger.debug(f"Saved conversation turn to DB for session {session_id}")
             return True

         except Exception as e:
             logger.error(f"Failed to save conversation to DB for session {session_id}: {e}", exc_info=True)
             # Return True because cache was updated successfully; DB failure is logged but not critical for immediate UX.
             return True  # Change to False if DB persistence is strictly required.

     async def clear_session_history(self, session_id: str) -> bool:
         """
         Clear all conversation history for a given session.

         Args:
             session_id: Unique session identifier.

         Returns:
             True if cleared successfully from both cache and database.
         """
         success_cache = False
         success_db = False

         # Clear from cache.
         if session_id in self._session_cache:
             try:
                 del self._session_cache[session_id]
                 success_cache = True
                 logger.info(f"Cleared cache for session {session_id}")
             except Exception as e:
                 logger.error(f"Failed to clear cache for session {session_id}: {e}")

         # Clear from database.
         try:
             await self.db.delete_conversation_history(session_id=session_id)
             success_db = True
             logger.info(f"Cleared DB history for session {session_id}")
         except Exception as e:
             logger.error(f"Failed to clear DB history for session {session_id}: {e}")

         return success_cache or success_db  # Return True if at least one was cleared.

     async def cleanup_expired_sessions(self) -> int:
         """
         Clean up sessions that have exceeded the timeout duration.

         Returns:
             Number of expired sessions cleaned up.
         """
         current_time = datetime.utcnow()
         expired_sessions = []

         # Identify expired sessions from cache.
         for session_id, session_data in list(self._session_cache.items()):
             last_active_str = session_data.get('last_active')
             
             if isinstance(last_active_str, datetime):
                 last_active_dt = last_active_str
             elif isinstance(last_active_str, str):
                 try:
                     last_active_dt = datetime.fromisoformat(last_active_str)
                 except ValueError:
                     last_active_dt = current_time - timedelta(days=1)  # Assume expired if parsing fails.
             else:
                 continue  # Skip entries without valid timestamp.

             if current_time - last_active_dt > self.session_timeout:
                 expired_sessions.append(session_id)

         # Clean up expired sessions.
         cleanup_count = 0
         for session_id in expired_sessions:
             try:
                 await self.clear_session_history(session_id)
                 cleanup_count += 1
                 logger.info(f"Cleaned up expired session {session_id}")
             except Exception as e:
                 logger.error(f"Failed to clean up expired session {session_id}: {e}")

         return cleanup_count


# Example usage and testing (if run directly).
if __name__ == "__main__":
     import asyncio

     async def test_chat_handler():
         """Basic test function to verify ChatHandler initialization and processing."""
         
         # Mock dependencies (replace with actual implementations in production).
         class MockDatabaseManager(DatabaseManager):
              async def get_conversation_history(self, **kwargs):
                   return []
              async def save_messages_batch(self, **kwargs):
                   pass
              async def delete_conversation_history(self, **kwargs):
                   pass

         class MockTradingAnalyzer(TradingAnalyzer):
              pass

         db_manager = MockDatabaseManager()
         analyzer = MockTradingAnalyzer()

         handler = ChatHandler(
              deepseek_api_key="test_api_key",
              database_manager=db_manager,
              trading_analyzer=analyzer,
              max_history_length=10,
              session_timeout_minutes=5
         )

         test_message = ChatMessage(content="What is the current price of Bitcoin?")
         
         try:
              response = await handler.process_message(test_message)
              print(f"Response received:\n{response.json(indent=2)}")
         except Exception as e:
              print(f"Test failed with error:\n{e}")

     asyncio.run(test_chat_handler())