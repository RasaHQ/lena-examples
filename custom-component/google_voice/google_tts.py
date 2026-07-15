"""Google Cloud Text-to-Speech (Chirp 3 HD) bidirectional streaming TTS engine.

Docs:
    https://docs.cloud.google.com/text-to-speech/docs/create-audio-text-streaming
    https://docs.cloud.google.com/text-to-speech/docs/reference/rpc

Bidirectional streaming synthesis (``TextToSpeech.StreamingSynthesize``) is
only available over gRPC — there is no REST equivalent — and only for
Chirp 3: HD voices. ``TextToSpeechAsyncClient.streaming_synthesize`` takes an
async generator of ``StreamingSynthesizeRequest`` messages: the first message
must carry a ``StreamingSynthesizeConfig`` (voice + output format) and no
text; every following message must carry only a ``StreamingSynthesisInput``
text chunk. Google streams back ``StreamingSynthesizeResponse`` messages with
raw (headerless) audio as soon as each chunk is ready, so text keeps
streaming in while audio streams out on the same call.
"""

import asyncio
import contextlib
from typing import Any, AsyncIterator, Dict, List, Optional

import structlog

from rasa.core.channels.voice_stream.audio_bytes import AudioFormat, RasaAudioBytes
from rasa.core.channels.voice_stream.tts.config import StreamingConfig
from rasa.core.channels.voice_stream.tts.tts_engine import (
    StreamState,
    TTSEngine,
    TTSEngineConfig,
    TTSError,
    TTSLanguageMapEntry,
)

from .helpers import (
    google_tts_encoding,
    grpc_channel_state,
    regional_endpoint,
    resolve_project_id,
)

structlogger = structlog.get_logger()

DEFAULT_VOICE = "en-US-Chirp3-HD-Charon"


class GoogleTTSConfig(TTSEngineConfig):
    """Configuration for the Google Cloud Text-to-Speech streaming engine.

    Per-language voice name (e.g. ``"en-US-Chirp3-HD-Charon"``) and language
    code come from ``language_map`` entries (``language`` / ``voice``), same
    pattern as the built-in Azure, Cartesia, Deepgram, and Rime TTS engines.

    Streaming synthesis only works with Chirp 3: HD voices (Preview);
    non-HD voices will fail at synthesis time.

    Attributes:
        project_id: Google Cloud project ID used for quota/billing. Falls
            back to the ``GOOGLE_CLOUD_PROJECT`` environment variable, and
            then Application Default Credentials, when unset.
        location: Google Cloud region serving the voices, e.g. ``"us"`` ->
            ``us-texttospeech.googleapis.com``. Leave unset (or ``"global"``)
            for the default global endpoint. Mirrors the ASR engine's
            ``location`` for data-residency parity.

    Note:
        No ``speaking_rate`` field: Google's ``StreamingAudioConfig`` has no
        pace-control field (only the unary ``AudioConfig`` does), so it
        can't be honored for streaming synthesis.
    """

    project_id: Optional[str] = None
    location: Optional[str] = None


class GoogleTTS(TTSEngine[GoogleTTSConfig]):
    """Google Cloud Text-to-Speech (Chirp 3 HD) bidirectional streaming engine.

    Each bot response drives one ``streaming_synthesize`` gRPC call: text
    chunks pushed via ``send_text_chunk`` land on an internal queue that
    feeds the request generator, and audio chunks received from Google are
    forwarded to another queue that ``stream_audio`` drains. Interrupting
    playback (barge-in) cancels the underlying call so Google stops
    generating audio immediately, rather than waiting for it to finish.
    """

    required_packages = ("google.cloud.texttospeech_v1",)
    streaming_input: bool = True

    @classmethod
    def name(cls) -> str:
        """Return the name identifier for this TTS engine."""
        return "google"

    def __init__(
        self,
        rasa_language: str,
        format: AudioFormat,
        config: Optional[GoogleTTSConfig] = None,
        additional_languages: Optional[List[str]] = None,
    ):
        super().__init__(rasa_language, format, config, additional_languages)
        self._client: Optional[Any] = None
        self._text_queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        self._audio_queue: "asyncio.Queue[Optional[RasaAudioBytes]]" = asyncio.Queue()
        self._synthesis_task: Optional[asyncio.Task] = None
        # Kept so barge-in can cancel the in-flight call directly instead of
        # waiting for the response stream to drain.
        self._active_call: Optional[Any] = None

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self, config: Optional[GoogleTTSConfig] = None) -> None:
        """Create the Text-to-Speech async client.

        Honours ``location`` and ``project_id`` so TTS resolves the same way
        as ASR; both default to the global endpoint / Application Default
        Credentials when unset.
        """
        from google.api_core.client_options import ClientOptions
        from google.cloud.texttospeech_v1 import TextToSpeechAsyncClient

        self._client = TextToSpeechAsyncClient(
            client_options=ClientOptions(
                api_endpoint=regional_endpoint("texttospeech", self.config.location),
                quota_project_id=resolve_project_id(self.config.project_id),
            )
        )

    async def close_connection(self) -> None:
        """Cancel any in-flight synthesis and release the gRPC channel."""
        await self._cancel_active_synthesis()
        if self._client is not None:
            await self._client.transport.close()
            self._client = None

    async def set_language(self, rasa_language: str) -> bool:
        """Update the TTS language for the next synthesis call."""
        if not await super().set_language(rasa_language):
            return False
        # Recycle the client under the engine lock so a concurrent
        # send_text_chunk can't observe it being torn down.
        await self._reconnect_with_lock()
        return True

    # -- Text in --------------------------------------------------------

    async def prepare_response(
        self, streaming_config: Optional[StreamingConfig] = None
    ) -> None:
        """Start a new ``streaming_synthesize`` call for the next response."""
        await self._cancel_active_synthesis()
        self._text_queue = asyncio.Queue()
        self._audio_queue = asyncio.Queue()
        self.stop_streaming_output_audio_chunks = False
        self._synthesis_task = asyncio.create_task(self._run_synthesis())

    async def send_text_chunk(self, text: str) -> None:
        """Queue a text chunk for the request generator to forward to Google."""
        async with self._get_engine_lock():
            await self._text_queue.put(text)

    async def signal_text_done(self) -> None:
        """Signal that no more text chunks will be sent for this response."""
        async with self._get_engine_lock():
            await self._text_queue.put(None)

    async def _request_generator(
        self,
    ) -> AsyncIterator[Any]:
        """Yield the config request, then text chunks as they are queued.

        Google requires exactly one config-only request before any text
        request on the stream; every request after that carries only text.
        """
        from google.cloud import texttospeech_v1

        yield texttospeech_v1.StreamingSynthesizeRequest(
            streaming_config=self._build_streaming_config()
        )
        while True:
            text = await self._text_queue.get()
            if text is None:
                return
            yield texttospeech_v1.StreamingSynthesizeRequest(
                input=texttospeech_v1.StreamingSynthesisInput(text=text)
            )

    # -- Audio out --------------------------------------------------------

    async def signal_interrupt(self) -> None:
        """Cancel the in-flight synthesis call and unblock ``stream_audio``."""
        structlogger.debug("google_tts.signal_interrupt")
        await self._cancel_active_synthesis()
        await self._audio_queue.put(None)

    async def stop_streaming(self) -> None:
        """Stop or defer-interrupt the current response, per ``StreamState``."""
        await super().stop_streaming()
        if self.stream_state == StreamState.SENDING_RESPONSE_CHUNKS:
            # Audio is still being generated; defer the actual cancel to
            # send_response_chunk_end(), matching Cartesia/Rime/Deepgram.
            self.stream_state = StreamState.INTERRUPTED
        elif self.stream_state == StreamState.RESPONSE_CHUNKS_SENT:
            self.stream_state = StreamState.NO_STREAMING
            await self.signal_interrupt()

    async def stream_audio(self) -> AsyncIterator[RasaAudioBytes]:
        """Yield audio chunks until a ``None`` sentinel signals completion."""
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                return
            yield chunk

    async def synthesize(
        self, text: str, config: Optional[GoogleTTSConfig] = None
    ) -> AsyncIterator[RasaAudioBytes]:
        """Generate speech from a single block of text via streaming synthesis."""
        await self.prepare_response()
        await self.send_text_chunk(text)
        await self.signal_text_done()
        async for chunk in self.stream_audio():
            yield chunk

    def engine_bytes_to_rasa_audio_bytes(self, chunk: bytes) -> RasaAudioBytes:
        """Convert the generated TTS audio bytes into Rasa audio bytes."""
        return RasaAudioBytes(chunk, format=self.audio_format)

    async def _run_synthesis(self) -> None:
        """Drive one ``streaming_synthesize`` call and relay audio chunks.

        A ``None`` sentinel always ends ``stream_audio``. Timeouts bound both
        opening the call and waiting for each audio chunk.
        """
        if self._client is None:
            structlogger.error("google_tts.synthesis.not_connected")
            await self._audio_queue.put(None)
            return
        timeout = self.config.timeout
        try:
            call = self._client.streaming_synthesize(
                requests=self._request_generator()
            )
            stream = await asyncio.wait_for(call, timeout=timeout)
            self._active_call = stream
            structlogger.debug(
                "google_tts.synthesis.call_opened",
                channel_state=grpc_channel_state(self._client),
            )
            responses = stream.__aiter__()
            while True:
                try:
                    response = await asyncio.wait_for(
                        responses.__anext__(), timeout=timeout
                    )
                except StopAsyncIteration:
                    break
                audio_bytes = self.engine_bytes_to_rasa_audio_bytes(
                    response.audio_content
                )
                await self._audio_queue.put(audio_bytes)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            structlogger.error(
                "google_tts.synthesis.timeout",
                timeout_seconds=timeout,
                channel_state=grpc_channel_state(self._client),
            )
        except Exception as e:
            structlogger.error(
                "google_tts.synthesis.error",
                error=str(e),
                channel_state=grpc_channel_state(self._client),
                exc_info=True,
            )
        finally:
            self._active_call = None
            await self._audio_queue.put(None)

    async def _cancel_active_synthesis(self) -> None:
        """Cancel the in-flight gRPC call and its driving task, if any."""
        if self._active_call is not None:
            self._active_call.cancel()
            self._active_call = None
        if self._synthesis_task is not None and not self._synthesis_task.done():
            self._synthesis_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._synthesis_task
        self._synthesis_task = None

    # -- Request building -------------------------------------------------

    def _build_streaming_config(
        self,
    ) -> Any:
        """Build the ``StreamingSynthesizeConfig`` for the current language/voice.

        Raises:
            TTSError: When no voice is configured for the current language.
        """
        from google.cloud import texttospeech_v1

        language_config = self.current_language_config
        if not language_config.voice:
            raise TTSError(
                f"No voice configured for language "
                f"'{language_config.rasa_language_key}'. Add 'voice' to the "
                f"language_map entry in your TTS config, e.g. "
                f"'{DEFAULT_VOICE}'."
            )

        return texttospeech_v1.StreamingSynthesizeConfig(
            voice=texttospeech_v1.VoiceSelectionParams(
                name=language_config.voice,
                language_code=language_config.engine_language_key or "en-US",
            ),
            streaming_audio_config=texttospeech_v1.StreamingAudioConfig(
                audio_encoding=google_tts_encoding(self.audio_format),
                sample_rate_hertz=self.audio_format.sample_rate,
            ),
        )

    # -- Rasa engine factory hooks ------------------------------------------

    @staticmethod
    def get_default_config(rasa_language: str) -> GoogleTTSConfig:
        """Return the default config for *rasa_language*."""
        return GoogleTTSConfig(
            language_map={
                rasa_language: TTSLanguageMapEntry(
                    language="en-US",
                    voice=DEFAULT_VOICE,
                ),
            },
        )

    @classmethod
    def from_config_dict(
        cls,
        config: Dict,
        format: AudioFormat,
        rasa_language: str,
        additional_languages: Optional[List[str]] = None,
    ) -> "GoogleTTS":
        """Create a ``GoogleTTS`` instance from a raw config dictionary."""
        return cls(
            rasa_language=rasa_language,
            format=format,
            config=GoogleTTSConfig.model_validate(config),
            additional_languages=additional_languages,
        )
