"""Google Cloud Speech-to-Text v2 (Chirp 3) streaming ASR engine.

Docs:
    https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3
    https://docs.cloud.google.com/speech-to-text/docs/streaming-recognize

Streaming recognition (``Speech.StreamingRecognize``) is only available over
gRPC — there is no REST equivalent. ``SpeechAsyncClient.streaming_recognize``
takes an async generator of ``StreamingRecognizeRequest`` messages: the first
message carries the ``StreamingRecognitionConfig``, every following message
carries a raw audio chunk. It returns an async iterable of
``StreamingRecognizeResponse`` messages as Google transcribes the audio, so
requests and responses flow over the wire concurrently on the same call.
"""

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Literal, Optional

import structlog

from rasa.core.channels.voice_stream.asr.asr_engine import (
    ASRConfigError,
    ASREngine,
    ASREngineConfig,
    ASRLanguageMapEntry,
)
from rasa.core.channels.voice_stream.asr.asr_event import (
    ASREvent,
    NewTranscript,
    UserIsSpeaking,
)
from rasa.core.channels.voice_stream.audio_bytes import AudioFormat, RasaAudioBytes
from rasa.shared.exceptions import ConnectionException

from .helpers import (
    google_asr_encoding,
    grpc_channel_state,
    regional_endpoint,
    resolve_project_id,
)

if TYPE_CHECKING:
    from google.cloud import speech_v2

structlogger = structlog.get_logger()

DEFAULT_MODEL = "chirp_3"

EndpointingSensitivity = Literal["standard", "short", "supershort"]


class GoogleASRConfig(ASREngineConfig):
    """Configuration for the Google Cloud Speech-to-Text v2 (Chirp 3) engine.

    The per-language recognition language code and model identifier (e.g.
    ``"chirp_3"``) are configured per Rasa language via ``language_map``
    entries (``language`` / ``model``), following the same pattern as the
    built-in Azure and Deepgram ASR engines.

    Attributes:
        project_id: Google Cloud project ID. Falls back to the
            ``GOOGLE_CLOUD_PROJECT`` environment variable when unset.
        location: Google Cloud region serving the recognizer. ``chirp_3`` is
            GA in the ``us`` and ``eu`` multi-regions.
        recognizer: Full recognizer resource name. Defaults to the implicit
            recognizer for ``project_id``/``location``
            (``projects/{project_id}/locations/{location}/recognizers/_``),
            which requires no prior setup.
        interim_results: Whether Google should stream interim (non-final)
            transcripts, exposed as ``UserIsSpeaking`` events for barge-in
            detection.
        enable_voice_activity_events: Whether Google should emit a
            speech-activity-started event as soon as the user starts
            talking. Used as a barge-in signal when ``interim_results`` is
            disabled or a transcript hasn't arrived yet.
        enable_automatic_punctuation: Whether the model should add
            punctuation and capitalization automatically.
        endpointing_sensitivity: Trade-off between latency and accuracy for
            deciding when an utterance has ended: ``"standard"`` (default,
            long-form/natural conversation), ``"short"`` (single
            sentences/commands), or ``"supershort"`` (single words like
            "yes"/"no").
    """

    project_id: Optional[str] = None
    location: str = "us"
    recognizer: Optional[str] = None
    interim_results: bool = True
    enable_voice_activity_events: bool = True
    enable_automatic_punctuation: bool = True
    endpointing_sensitivity: Optional[EndpointingSensitivity] = None


class GoogleASR(ASREngine[GoogleASRConfig]):
    """Google Cloud Speech-to-Text v2 (Chirp 3) streaming ASR engine.

    Wraps ``SpeechAsyncClient.streaming_recognize``'s bidirectional gRPC
    stream and exposes it through the common
    :class:`~rasa.core.channels.voice_stream.asr.asr_engine.ASREngine`
    interface. Audio chunks pushed via ``send_audio_chunks`` land on an
    internal queue that feeds the request generator driving the call;
    responses are translated into ``ASREvent``s as they arrive.
    """

    required_packages = ("google.cloud.speech_v2",)

    @classmethod
    def name(cls) -> str:
        """Return the name identifier for this ASR engine."""
        return "google"

    def __init__(
        self,
        rasa_language: str,
        format: AudioFormat,
        config: Optional[GoogleASRConfig] = None,
        additional_languages: Optional[List[str]] = None,
    ):
        super().__init__(rasa_language, format, config, additional_languages)
        self._client: Optional[Any] = None
        self._audio_queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
        self._response_stream: Optional[AsyncIterator[Any]] = None

    # -- Connection lifecycle -------------------------------------------

    async def connect(self) -> None:
        """Open the bidirectional gRPC stream to Speech-to-Text v2."""
        from google.api_core.client_options import ClientOptions
        from google.cloud.speech_v2 import SpeechAsyncClient

        self._client = SpeechAsyncClient(
            client_options=ClientOptions(
                api_endpoint=regional_endpoint("speech", self.config.location),
                quota_project_id=self._project_id(),
            )
        )
        self._audio_queue = asyncio.Queue()
        self._response_stream = await self._client.streaming_recognize(
            requests=self._request_generator()
        )

    async def close_connection(self) -> None:
        """End the request stream and release the gRPC channel."""
        await self._audio_queue.put(None)
        self._response_stream = None
        if self._client is not None:
            await self._client.transport.close()
            self._client = None

    async def signal_audio_done(self) -> None:
        """Signal Google that no more audio will be sent on this stream."""
        await self._audio_queue.put(None)

    # -- Audio in ---------------------------------------------------------

    def rasa_audio_bytes_to_engine_bytes(self, chunk: RasaAudioBytes) -> bytes:
        """Convert ``RasaAudioBytes`` to raw bytes for the Google audio stream."""
        return chunk.data

    async def send_audio_chunks(self, chunk: RasaAudioBytes) -> None:
        """Queue an audio chunk for the request generator to forward to Google.

        Held under the engine lock so a mid-call language switch (the base
        ``set_language`` closes and reopens the stream under the same lock)
        can't race this: a chunk either lands on the old queue before the
        switch or on the freshly created queue after it, never on a
        half-torn-down stream.
        """
        async with self._get_engine_lock():
            if self._client is None:
                structlogger.debug(
                    "google_asr.send_audio_chunks.skipped", reason="not_connected"
                )
                return
            await self._audio_queue.put(self.rasa_audio_bytes_to_engine_bytes(chunk))

    async def _request_generator(
        self,
    ) -> AsyncIterator["speech_v2.StreamingRecognizeRequest"]:
        """Yield the initial config request, then audio chunks as queued.

        Google requires exactly one config-only request before any audio
        request on the stream; every request after that carries only audio.
        """
        from google.cloud import speech_v2

        yield self._build_config_request()
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                return
            yield speech_v2.StreamingRecognizeRequest(audio=chunk)

    # -- Transcripts out ----------------------------------------------------

    async def stream_asr_events(self) -> AsyncIterator[ASREvent]:
        """Stream ``ASREvent``s translated from Google's response stream."""
        if self._response_stream is None:
            raise ConnectionException("Google ASR gRPC stream is not connected.")
        try:
            async for response in self._response_stream:
                for event in self.engine_event_to_asr_events(response):
                    yield event
        except Exception as e:
            structlogger.warning(
                "google_asr.stream_asr_events.error",
                error=str(e),
                channel_state=grpc_channel_state(self._client),
                exc_info=True,
            )

    def engine_event_to_asr_events(
        self, response: "speech_v2.StreamingRecognizeResponse"
    ) -> List[ASREvent]:
        """Translate one Google response message into zero or more ``ASREvent``s."""
        from google.cloud import speech_v2

        speech_event_types = speech_v2.StreamingRecognizeResponse.SpeechEventType
        if response.speech_event_type == speech_event_types.SPEECH_ACTIVITY_BEGIN:
            # Fired as soon as voice activity is detected, ahead of any
            # transcript. Used as an early barge-in signal, mirroring
            # Deepgram v2's ``StartOfTurn`` -> ``UserIsSpeaking("")``.
            return [UserIsSpeaking("")]

        events: List[ASREvent] = []
        for result in response.results:
            if not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            if not transcript:
                continue
            if result.is_final:
                events.append(NewTranscript(transcript))
            elif self.config.interim_results:
                events.append(UserIsSpeaking(transcript))
        return events

    # -- Request building -------------------------------------------------

    def _build_config_request(self) -> "speech_v2.StreamingRecognizeRequest":
        """Build the first ``StreamingRecognizeRequest``, carrying the config."""
        from google.cloud import speech_v2

        language_config = self.current_language_config
        recognition_config = speech_v2.RecognitionConfig(
            explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
                encoding=google_asr_encoding(self.audio_format),
                sample_rate_hertz=self.audio_format.sample_rate,
                audio_channel_count=self.audio_format.channels,
            ),
            language_codes=[language_config.engine_language_key or "auto"],
            model=language_config.model or DEFAULT_MODEL,
            features=speech_v2.RecognitionFeatures(
                enable_automatic_punctuation=self.config.enable_automatic_punctuation,
            ),
        )

        return speech_v2.StreamingRecognizeRequest(
            recognizer=self._recognizer_path(),
            streaming_config=speech_v2.StreamingRecognitionConfig(
                config=recognition_config,
                streaming_features=self._build_streaming_features(),
            ),
        )

    def _build_streaming_features(self) -> Any:
        """Build streaming features, tolerating SDKs without endpointing support."""
        from google.cloud import speech_v2

        features_kwargs: Dict[str, Any] = {
            "interim_results": self.config.interim_results,
            "enable_voice_activity_events": self.config.enable_voice_activity_events,
        }
        sensitivity = self.config.endpointing_sensitivity
        if not sensitivity:
            return speech_v2.StreamingRecognitionFeatures(**features_kwargs)

        try:
            sensitivity_enum = (
                speech_v2.StreamingRecognitionFeatures.EndpointingSensitivity
            )
            features_kwargs["endpointing_sensitivity"] = getattr(
                sensitivity_enum,
                f"ENDPOINTING_SENSITIVITY_{sensitivity.upper()}",
            )
            return speech_v2.StreamingRecognitionFeatures(**features_kwargs)
        except (AttributeError, TypeError, ValueError):
            structlogger.warning(
                "google_asr.endpointing_sensitivity.unsupported_sdk",
                configured=sensitivity,
                reason=(
                    "Installed google-cloud-speech does not support this field. "
                    "Upgrade to >=2.33.0 or remove endpointing_sensitivity."
                ),
            )
            features_kwargs.pop("endpointing_sensitivity", None)
            return speech_v2.StreamingRecognitionFeatures(**features_kwargs)

    def _recognizer_path(self) -> str:
        """Return the recognizer resource name, defaulting to the implicit one."""
        if self.config.recognizer:
            return self.config.recognizer
        return (
            f"projects/{self._project_id()}/locations/{self.config.location}"
            f"/recognizers/_"
        )

    def _project_id(self) -> str:
        """Resolve the Google Cloud project ID from config or the environment.

        Raises:
            ASRConfigError: When no project ID is configured anywhere.
        """
        project_id = resolve_project_id(self.config.project_id)
        if not project_id:
            raise ASRConfigError(
                "Google ASR requires a Google Cloud project ID. Set 'project_id' "
                "in the ASR config or the 'GOOGLE_CLOUD_PROJECT' environment "
                "variable."
            )
        return project_id

    # -- Rasa engine factory hooks ------------------------------------------

    @staticmethod
    def get_default_config(rasa_language: str) -> GoogleASRConfig:
        """Return the default config for *rasa_language*."""
        return GoogleASRConfig(
            location="us",
            language_map={
                rasa_language: ASRLanguageMapEntry(
                    language="en-US",
                    model=DEFAULT_MODEL,
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
    ) -> "GoogleASR":
        """Create a ``GoogleASR`` instance from a raw config dictionary."""
        return cls(
            rasa_language=rasa_language,
            format=format,
            config=GoogleASRConfig.model_validate(config),
            additional_languages=additional_languages,
        )
