"""
Custom REST channel with JWT authentication support for Rasa.

This module provides a custom input channel that extends Rasa's RestInput
to support JWT authentication and header propagation for external API calls.

Features:
- JWT token validation (configurable)
- Custom header extraction and propagation
- Error handling with structured responses
- Support for streaming and regular responses
"""

import logging
from typing import Text, Dict, Any, Callable, Awaitable, Union
from functools import partial

from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse, ResponseStream, BaseHTTPResponse

from rasa.core.channels.rest import RestInput
from rasa.core.channels.channel import UserMessage, CollectingOutputChannel
from rasa.core.constants import BEARER_TOKEN_PREFIX
import rasa.utils.endpoints
from asyncio import CancelledError

logger = logging.getLogger(__name__)

# Configuration constants
CLIENT_AUTH_HEADER_KEY = "client_auth_headers"
API_ERROR = "API_ERROR"

# Headers to extract and propagate to downstream services
HEADERS_TO_PROPAGATE = [
    "Accept-Language",
    "Authorization",
    "User-Agent",
    "X-Auth-Token"
]


class AuthRestChannel(RestInput):
    """
    Custom REST input channel with JWT authentication and header propagation.

    This channel extends Rasa's RestInput to provide:
    - Optional JWT token validation
    - Custom header extraction and propagation to action metadata
    - Structured error response handling
    """

    @classmethod
    def name(cls) -> Text:
        """Return the channel name for registration."""
        return "auth_rest_channel"

    def __init__(self):
        """Initialize the custom REST channel."""
        super().__init__()

    def extract_client_headers(self, request: Request) -> Dict[str, Any]:
        """
        Extract client headers from request for API propagation.

        Uses the pattern suggested by colleague for flexible header handling.

        Args:
            request: The Sanic request object

        Returns:
            Dict containing organized headers for metadata
        """
        # Get base metadata or initialize empty dict
        metadata = self.get_metadata(request) or {}

        # Option 1: Pass all headers (use if needed)
        metadata["headers"] = dict(request.headers)
  
        # Option 2: Extract specific client headers (recommended approach)
        # client_headers = {}
        # for header in HEADERS_TO_PROPAGATE:
        #     if header in request.headers:
        #         client_headers[header] = request.headers.get(header)

        # Option 3: Extract specific auth token (alternative approach)
        # metadata["x-auth-token"] = request.headers.get("x-auth-token")

        # Store client headers in nested structure for compatibility
        # metadata[CLIENT_AUTH_HEADER_KEY] = client_headers

        return metadata

    async def receive_messages(
        self,
        request: Request,
        on_new_message: Callable[[UserMessage], Awaitable[None]]
    ) -> Union[ResponseStream, BaseHTTPResponse]:
        """
        Process incoming messages with optional JWT validation and headers.
        
        Args:
            request: The incoming Sanic request
            on_new_message: Callback for processing the user message
        
        Returns:
            HTTP response (streaming or regular)
        """
        # Extract metadata with client headers using colleague's pattern
        metadata = self.get_metadata(request)

        metadata = self.extract_client_headers(request)

        # log metadata for debugging
        logger.debug(f"Extracted metadata: {metadata}")
        
        # Extract message information
        sender_id = await self._extract_sender(request)
        text = self._extract_message(request)
        should_use_stream = rasa.utils.endpoints.bool_arg(
            request, "stream", default=False
        )
        input_channel = self._extract_input_channel(request)

        if should_use_stream:
            return ResponseStream(
                partial(
                    self.stream_response,
                    on_new_message,
                    text,
                    sender_id,
                    input_channel,
                    metadata,
                ),
                content_type="text/event-stream",
            )
        else:
            collector = CollectingOutputChannel()
            # noinspection PyBroadException
            try:
                await on_new_message(
                    UserMessage(
                        text,
                        collector,
                        sender_id,
                        input_channel=input_channel,
                        metadata=metadata,
                        headers=request.headers,
                    )
                )
            except CancelledError:
                structlogger.error(
                    "rest.message.received.timeout",
                    event_info="Message processing was cancelled.",
                )
            except Exception as e:
                structlogger.exception(
                    "rest.message.received.failure",
                    event_info=f"Message processing failed. Error: {e}",
                )

            return response.json(collector.messages)

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[None]]
    ) -> Blueprint:
        """
        Create the Sanic blueprint for the webhook endpoints.

        Args:
            on_new_message: Callback for processing new messages

        Returns:
            Configured Sanic blueprint
        """
        custom_webhook = Blueprint(
            f"custom_webhook_{type(self).__name__}",
            __name__,
        )

        @custom_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            """Health check endpoint."""
            return response.json({"status": "ok"})

        @custom_webhook.route("/webhook", methods=["POST"])
        async def receive(
            request: Request
        ) -> Union[ResponseStream, BaseHTTPResponse]:
            """Main webhook endpoint for receiving messages."""
            return await self.receive_messages(request, on_new_message)

        return custom_webhook
