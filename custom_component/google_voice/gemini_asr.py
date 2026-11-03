"""Gemini 3.5 Transcribe speech-to-text for Rasa voice (single file).

A self-contained Rasa ASR component for Google's Gemini 3.5 Transcribe model.
Copy this one file into your project and reference it in ``credentials.yml``::

    asr:
      name: custom_component.google_voice.gemini_asr.GeminiASR
      location: us                 # global | us | eu
      language_map:
        en-US:
          language: en-US          # or "auto" for automatic language detection
          model: gemini-3.5-transcribe-live

Requirements: ``pip install google-genai``, Application Default Credentials
(e.g. ``gcloud auth application-default login``), and a Google Cloud project
via ``GOOGLE_CLOUD_PROJECT`` or ``project_id``.

Docs:
    https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-transcribe
    https://github.com/GoogleCloudPlatform/generative-ai/blob/main/audio/speech/getting-started/gemini_3_5_transcribe_live.py

How it works: each call opens a transcription-only Gemini Live API session (a
websocket) with ``client.aio.live.connect``. Rasa's audio is sent as 16-bit PCM
with ``session.send_realtime_input``. Gemini answers with
``interim_input_transcription`` (partial text while the user talks, used for
barge-in) and ``input_transcription`` (final text, which Rasa treats as the
user's message). Google caps a session at about 10 minutes of audio, so when
the server closes a session mid-call this component opens a new one.
"""

import os
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, AsyncIterator, Dict, List, Literal, Optional

import structlog
import websockets.exceptions
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

if TYPE_CHECKING:
    from google.genai import live, types

structlogger = structlog.get_logger()

DEFAULT_MODEL = "gemini-3.5-transcribe-live"

# ``language`` value that asks Gemini to detect the spoken language itself.
AUTO_LANGUAGE = "auto"

GOOGLE_CLOUD_PROJECT_ENV_VAR = "GOOGLE_CLOUD_PROJECT"

# Websocket close code for a normal (non-error) session end.
_NORMAL_CLOSE_CODE = 1000

# Sessions that end without a single message from Gemini count as failed
# reconnects. Stop after this many in a row instead of retrying forever against
# a session Google keeps rejecting (wrong model, quota, permissions, ...).
_MAX_EMPTY_RECONNECTS = 3

SpeechSensitivity = Literal["high", "low"]


class GeminiASRConfig(ASREngineConfig):
    """Settings for ``GeminiASR``, i.e. the ``asr:`` block in ``credentials.yml``.

    Language and model are set per Rasa language in ``language_map``
    (``language`` / ``model``), like Rasa's built-in ASR engines. Set
    ``language`` to ``"auto"`` (or leave it out) to let Gemini detect the
    language, including switches mid-sentence.

    Attributes:
        project_id: Google Cloud project ID. Falls back to the
            ``GOOGLE_CLOUD_PROJECT`` environment variable.
        location: Gemini endpoint location: ``global`` (default), ``us`` or
            ``eu``.
        interim_results: Report partial transcripts while the user talks as
            ``UserIsSpeaking`` events, so the bot can stop speaking when
            interrupted (barge-in).
        custom_vocabulary: Names, products or jargon to recognize more
            reliably. Up to 1000 terms; Google recommends about 100.
        start_of_speech_sensitivity: ``high`` detects speech sooner, ``low``
            ignores more background noise.
        end_of_speech_sensitivity: ``high`` finalizes sooner after the user
            stops, ``low`` waits longer.
        silence_duration_ms: Milliseconds of silence before an utterance
            counts as finished.
    """

    project_id: Optional[str] = None
    location: str = "global"
    interim_results: bool = True
    custom_vocabulary: Optional[List[str]] = None
    start_of_speech_sensitivity: Optional[SpeechSensitivity] = None
    end_of_speech_sensitivity: Optional[SpeechSensitivity] = None
    silence_duration_ms: Optional[int] = None


class GeminiASR(ASREngine[GeminiASRConfig]):
    """Rasa ASR engine for Gemini 3.5 Transcribe over the Gemini Live API.

    Holds one Live session at a time. ``send_audio_chunks`` sends Rasa's audio
    to it; ``stream_asr_events`` turns Gemini's messages into Rasa
    ``ASREvent``s and moves on to a new session when the current one is
    replaced (language switch) or closed by Google (time limit).
    """

    required_packages = ("google.genai",)

    @classmethod
    def name(cls) -> str:
        """Return the name identifier for this ASR engine."""
        return "gemini"

    def __init__(
        self,
        rasa_language: str,
        format: AudioFormat,
        config: Optional[GeminiASRConfig] = None,
        additional_languages: Optional[List[str]] = None,
    ):
        """Validate the config; the session itself is opened in ``connect``."""
        super().__init__(rasa_language, format, config, additional_languages)
        # Owns the websocket context manager so ``close_connection`` can exit it.
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional["live.AsyncSession"] = None
        self._session_received_message = False
        self._audio_done = False

    # -- Connection lifecycle -------------------------------------------

    async def connect(self) -> None:
        """Open a Gemini Live transcription session for the current language.

        ``live.connect`` returns only after Google confirms the session setup,
        so no audio is sent too early (which Google warns can cancel the
        session).
        """
        from google import genai

        client = genai.Client(
            enterprise=True, project=self._project_id(), location=self.config.location
        )
        exit_stack = AsyncExitStack()
        self._session = await exit_stack.enter_async_context(
            client.aio.live.connect(
                model=self.current_language_config.model or DEFAULT_MODEL,
                config=self._build_live_config(),
            )
        )
        self._exit_stack = exit_stack
        self._audio_done = False

    async def close_connection(self) -> None:
        """Close the session's websocket. Safe to call more than once."""
        exit_stack, self._exit_stack = self._exit_stack, None
        self._session = None
        if exit_stack is None:
            return
        try:
            await exit_stack.aclose()
        except Exception as e:
            structlogger.debug("gemini_asr.close_connection.error", error=str(e))

    async def signal_audio_done(self) -> None:
        """Tell Gemini that no more audio will be sent on this session."""
        self._audio_done = True
        if self._session is not None:
            await self._session.send_realtime_input(audio_stream_end=True)

    # -- Audio in ---------------------------------------------------------

    def rasa_audio_bytes_to_engine_bytes(self, chunk: RasaAudioBytes) -> bytes:
        """Convert Rasa audio to 16-bit PCM, the only encoding Gemini accepts."""
        return chunk.to_pcm16()

    async def send_audio_chunks(self, chunk: RasaAudioBytes) -> None:
        """Send one audio chunk to the current session.

        Runs under the engine lock, which language switches and reconnects
        also hold while they replace the session. A chunk therefore goes to
        either the old or the new session, never to a half-closed one.
        """
        from google.genai import types

        async with self._get_engine_lock():
            if self._session is None:
                structlogger.debug(
                    "gemini_asr.send_audio_chunks.skipped", reason="not_connected"
                )
                return
            blob = types.Blob(
                data=self.rasa_audio_bytes_to_engine_bytes(chunk),
                mime_type=f"audio/pcm;rate={self.audio_format.sample_rate}",
            )
            try:
                await self._session.send_realtime_input(audio=blob)
            except websockets.exceptions.ConnectionClosed as e:
                # Google ended the session (e.g. the ~10 minute limit). A new
                # one is opened by stream_asr_events; this chunk is dropped.
                structlogger.debug(
                    "gemini_asr.send_audio_chunks.connection_closed", error=str(e)
                )

    # -- Transcripts out ----------------------------------------------------

    async def stream_asr_events(self) -> AsyncIterator[ASREvent]:
        """Yield Rasa ``ASREvent``s for the whole call, across sessions.

        When a session ends, continue with its replacement (after a language
        switch) or open a new one (Google closed it). Stop when the call ends
        or Google keeps rejecting new sessions.
        """
        session = self._session
        if session is None:
            raise ConnectionException("Gemini Live session is not connected.")
        empty_reconnects = 0
        while session is not None:
            self._session_received_message = False
            async for event in self._session_events(session):
                yield event
            empty_reconnects = (
                0 if self._session_received_message else empty_reconnects + 1
            )
            if empty_reconnects > _MAX_EMPTY_RECONNECTS:
                structlogger.error(
                    "gemini_asr.stream_asr_events.giving_up",
                    attempts=empty_reconnects,
                )
                return
            session = await self._next_session(session)

    def engine_event_to_asr_events(
        self, message: "types.LiveServerMessage"
    ) -> List[ASREvent]:
        """Translate one Gemini message into zero or more ``ASREvent``s.

        - Speech started -> ``UserIsSpeaking("")`` (early barge-in signal).
        - Interim transcript -> ``UserIsSpeaking(text)`` if ``interim_results``.
        - Final transcript -> ``NewTranscript(text)``, the user's message.
        """
        from google.genai import types

        if message.go_away is not None:
            structlogger.info(
                "gemini_asr.session.go_away", time_left=message.go_away.time_left
            )

        voice_activity = message.voice_activity
        if (
            voice_activity is not None
            and voice_activity.voice_activity_type
            == types.VoiceActivityType.ACTIVITY_START
        ):
            return [UserIsSpeaking("")]

        content = message.server_content
        if content is None:
            return []

        events: List[ASREvent] = []
        interim = _transcription_text(content.interim_input_transcription)
        if interim and self.config.interim_results:
            events.append(UserIsSpeaking(interim))
        final = _transcription_text(content.input_transcription)
        if final:
            events.append(NewTranscript(final))
        return events

    async def _session_events(
        self, session: "live.AsyncSession"
    ) -> AsyncIterator[ASREvent]:
        """Yield ``ASREvent``s from one session until it closes.

        ``session.receive()`` stops after each completed turn, so we keep
        calling it until the websocket closes (the SDK raises ``APIError``).
        Errors are logged, not raised, so the caller can decide to reconnect.
        """
        from google.genai import errors

        try:
            while True:
                async for message in session.receive():
                    self._session_received_message = True
                    for event in self.engine_event_to_asr_events(message):
                        yield event
        except errors.APIError as e:
            log = (
                structlogger.debug
                if e.code == _NORMAL_CLOSE_CODE
                else structlogger.warning
            )
            log("gemini_asr.session.closed", code=e.code, error=str(e))
        except Exception as e:
            structlogger.warning("gemini_asr.session.error", error=str(e))

    async def _next_session(
        self, ended: "live.AsyncSession"
    ) -> Optional["live.AsyncSession"]:
        """Return the session to read from after *ended* closed.

        Taking the engine lock waits for an in-progress language switch to
        finish, so the session is never seen half-replaced.

        Returns:
            The replacement session, a newly opened one, or ``None`` when the
            call is over or reconnecting failed.
        """
        async with self._get_engine_lock():
            if self._session is None or self._audio_done:
                return None
            if self._session is not ended:
                return self._session
            structlogger.info("gemini_asr.session.reconnecting")
            await self.close_connection()
            try:
                await self.connect()
            except Exception as e:
                structlogger.error("gemini_asr.session.reconnect_failed", error=str(e))
                return None
            return self._session

    # -- Session config -----------------------------------------------------

    def _build_live_config(self) -> "types.LiveConnectConfig":
        """Build a transcription-only session config (text out, no replies)."""
        from google.genai import types

        language = self.current_language_config.engine_language_key
        language_codes = [language] if language and language != AUTO_LANGUAGE else None
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=language_codes,
                custom_vocabulary=self.config.custom_vocabulary,
            ),
            realtime_input_config=self._build_realtime_input_config(),
        )

    def _build_realtime_input_config(self) -> Optional["types.RealtimeInputConfig"]:
        """Build voice-activity-detection tuning, or ``None`` for Google defaults."""
        from google.genai import types

        start = self.config.start_of_speech_sensitivity
        end = self.config.end_of_speech_sensitivity
        silence_ms = self.config.silence_duration_ms
        if start is None and end is None and silence_ms is None:
            return None
        return types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                start_of_speech_sensitivity=(
                    types.StartSensitivity(f"START_SENSITIVITY_{start.upper()}")
                    if start
                    else None
                ),
                end_of_speech_sensitivity=(
                    types.EndSensitivity(f"END_SENSITIVITY_{end.upper()}")
                    if end
                    else None
                ),
                silence_duration_ms=silence_ms,
            )
        )

    def _project_id(self) -> str:
        """Return the project ID from config or ``GOOGLE_CLOUD_PROJECT``.

        Raises:
            ASRConfigError: When no project ID is configured anywhere.
        """
        project_id = self.config.project_id or os.environ.get(
            GOOGLE_CLOUD_PROJECT_ENV_VAR
        )
        if not project_id:
            raise ASRConfigError(
                "Gemini ASR requires a Google Cloud project ID. Set 'project_id' "
                "in the ASR config or the 'GOOGLE_CLOUD_PROJECT' environment "
                "variable."
            )
        return project_id

    # -- Rasa engine factory hooks ------------------------------------------

    @staticmethod
    def get_default_config(rasa_language: str) -> GeminiASRConfig:
        """Return the default config for *rasa_language*."""
        return GeminiASRConfig(
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
    ) -> "GeminiASR":
        """Create a ``GeminiASR`` instance from a raw config dictionary."""
        return cls(
            rasa_language=rasa_language,
            format=format,
            config=GeminiASRConfig.model_validate(config),
            additional_languages=additional_languages,
        )


def _transcription_text(transcription: Optional["types.Transcription"]) -> str:
    """Return the stripped text of a Gemini transcription, or ``""``."""
    if transcription is None or not transcription.text:
        return ""
    return transcription.text.strip()
