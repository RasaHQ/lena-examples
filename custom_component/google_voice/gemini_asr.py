"""Gemini 3.5 Transcribe speech-to-text for Rasa voice (single file).

A self-contained Rasa ASR component for Google's Gemini 3.5 Transcribe model.
Copy this one file into your project and reference it in ``credentials.yml``::

    asr:
      name: custom_component.google_voice.gemini_asr.GeminiASR
      location: us                 # global | us | eu
      language_map:
        en-US:
          language: en-US          # "auto", one code, or "fr-FR, en-US"
          model: gemini-3.5-transcribe-live

Requirements: ``pip install google-genai``, Application Default Credentials
(e.g. ``gcloud auth application-default login``), and a Google Cloud project
via ``GOOGLE_CLOUD_PROJECT`` or ``project_id``.

Docs:
    https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-transcribe
    https://github.com/GoogleCloudPlatform/generative-ai/blob/main/audio/speech/getting-started/gemini_3_5_transcribe_live.py

How it works
------------
Each call opens a transcription-only Gemini Live API session (a websocket)
with ``client.aio.live.connect``. Rasa's audio is sent as 16-bit PCM with
``session.send_realtime_input``. Gemini answers with:

- ``voice_activity`` (speech started)     -> ``UserIsSpeaking("")``
- ``interim_input_transcription`` (partial text while the user talks)
                                          -> ``UserIsSpeaking(text)``
- ``input_transcription`` (final text)    -> ``NewTranscript(text)``

``UserIsSpeaking`` lets the bot stop talking when interrupted (barge-in);
``NewTranscript`` is the user's message. Google caps a session at about 10
minutes, so when the server closes a session mid-call a new one is opened.

Live model features and where they are configured
--------------------------------------------------
=========================  ==================================================
Gemini feature             ``credentials.yml``
=========================  ==================================================
Language hints             ``language_map.<lang>.language``: one code, several
                           comma-separated codes, or ``auto``
Automatic detection        ``language: auto`` (or leave ``language`` out)
Custom vocabulary          ``custom_vocabulary``
Voice activity detection   ``start_of_speech_sensitivity``,
                           ``end_of_speech_sensitivity``,
                           ``prefix_padding_ms``, ``silence_duration_ms``
Region                     ``location``
=========================  ==================================================

Not exposed, because Rasa's voice pipeline can't use them:

- ``SMART`` transcription mode: Google requires manual endpointing (the client
  marks where each utterance starts and ends), which Rasa doesn't do. The
  default ``VERBATIM`` mode is used.
- Diarization and word timestamps: only available in the batch model
  (``gemini-3.5-transcribe``), not in the live one.
"""

import os
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Literal, Optional

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
Location = Literal["global", "us", "eu"]


class GeminiASRConfig(ASREngineConfig):
    """Settings for ``GeminiASR``, i.e. the ``asr:`` block in ``credentials.yml``.

    Language and model are set per Rasa language in ``language_map``
    (``language`` / ``model``), like Rasa's built-in ASR engines. ``language``
    can be one BCP-47 code (``en-US``), several comma-separated codes
    (``fr-FR, en-US``) for callers who mix languages, or ``auto`` (or left
    out) to let Gemini detect the language, including switches mid-sentence.

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
        prefix_padding_ms: Milliseconds of speech required before the start
            of speech is detected. Lower reacts faster but is more easily
            triggered by noise.
        silence_duration_ms: Milliseconds of silence before an utterance
            counts as finished.
    """

    # Connection
    project_id: Optional[str] = None
    location: Location = "global"

    # Transcription
    interim_results: bool = True
    custom_vocabulary: Optional[List[str]] = None

    # Voice activity detection (Google defaults when unset)
    start_of_speech_sensitivity: Optional[SpeechSensitivity] = None
    end_of_speech_sensitivity: Optional[SpeechSensitivity] = None
    prefix_padding_ms: Optional[int] = None
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
        self._stats = _SessionStats()

    # -- Connection lifecycle -------------------------------------------

    async def connect(self) -> None:
        """Open a Gemini Live transcription session for the current language.

        ``live.connect`` returns only after Google confirms the session setup
        (``setup_complete``), so no audio is sent too early, which Google warns
        can cancel the session.
        """
        from google import genai

        session_info = self._session_log_info()
        started = time.monotonic()
        try:
            client = genai.Client(
                enterprise=True,
                project=self._project_id(),
                location=self.config.location,
            )
            exit_stack = AsyncExitStack()
            self._session = await exit_stack.enter_async_context(
                client.aio.live.connect(
                    model=session_info["model"], config=self._build_live_config()
                )
            )
        except Exception as e:
            # Google's errors (e.g. "HTTP 404") rarely say which setting is
            # wrong, so log the settings this session was opened with.
            structlogger.error(
                "gemini_asr.session.connect_failed",
                error=str(e),
                error_type=type(e).__name__,
                project_id=self.config.project_id
                or os.environ.get(GOOGLE_CLOUD_PROJECT_ENV_VAR),
                **session_info,
            )
            raise
        self._exit_stack = exit_stack
        self._audio_done = False
        self._stats = _SessionStats()
        structlogger.info(
            "gemini_asr.session.connected",
            connect_ms=round((time.monotonic() - started) * 1000),
            **session_info,
        )

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
                # Between sessions (reconnect); counted in session.ended.
                self._stats.dropped_chunks += 1
                return
            pcm = self.rasa_audio_bytes_to_engine_bytes(chunk)
            blob = types.Blob(
                data=pcm,
                mime_type=f"audio/pcm;rate={self.audio_format.sample_rate}",
            )
            try:
                await self._session.send_realtime_input(audio=blob)
                self._stats.audio_bytes += len(pcm)
            except websockets.exceptions.ConnectionClosed:
                # Google ended the session (e.g. the ~10 minute limit). A new
                # one is opened by stream_asr_events; this chunk is dropped.
                self._stats.dropped_chunks += 1

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
        empty_sessions_in_a_row = 0
        while session is not None:
            self._session_received_message = False
            async for event in self._session_events(session):
                yield event

            if self._session_received_message:
                empty_sessions_in_a_row = 0
            else:
                empty_sessions_in_a_row += 1
            if empty_sessions_in_a_row > _MAX_EMPTY_RECONNECTS:
                structlogger.error(
                    "gemini_asr.stream_asr_events.giving_up",
                    attempts=empty_sessions_in_a_row,
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
            self._stats.final_transcripts += 1
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
        finally:
            # Also runs when Rasa cancels the task at hangup.
            self._log_session_ended()

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
            structlogger.info(
                "gemini_asr.session.reconnecting",
                received_messages=self._session_received_message,
            )
            await self.close_connection()
            try:
                await self.connect()
            except Exception as e:
                structlogger.error("gemini_asr.session.reconnect_failed", error=str(e))
                return None
            return self._session

    # -- Logging --------------------------------------------------------------

    def _session_log_info(self) -> Dict[str, Any]:
        """Return the settings a session is opened with, for log events."""
        config = self.config
        vad = {
            "start_of_speech_sensitivity": config.start_of_speech_sensitivity,
            "end_of_speech_sensitivity": config.end_of_speech_sensitivity,
            "prefix_padding_ms": config.prefix_padding_ms,
            "silence_duration_ms": config.silence_duration_ms,
        }
        return {
            "model": self.current_language_config.model or DEFAULT_MODEL,
            "location": config.location,
            "rasa_language": self.current_language_config.rasa_language_key,
            "language_codes": self._language_codes() or AUTO_LANGUAGE,
            "sample_rate": self.audio_format.sample_rate,
            "custom_vocabulary_terms": len(config.custom_vocabulary or []),
            "voice_activity": {k: v for k, v in vad.items() if v is not None}
            or "google_defaults",
        }

    def _log_session_ended(self) -> None:
        """Log a per-session summary: useful to spot drops and the time limit."""
        stats = self._stats
        structlogger.info(
            "gemini_asr.session.ended",
            duration_s=round(time.monotonic() - stats.started, 1),
            audio_sent_s=round(
                stats.audio_bytes / (2 * self.audio_format.sample_rate), 1
            ),
            final_transcripts=stats.final_transcripts,
            dropped_chunks=stats.dropped_chunks,
        )

    # -- Session config -----------------------------------------------------

    def _build_live_config(self) -> "types.LiveConnectConfig":
        """Build a transcription-only session config (text out, no replies)."""
        from google.genai import types

        return types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=self._language_codes(),
                custom_vocabulary=self.config.custom_vocabulary,
            ),
            realtime_input_config=self._build_voice_activity_config(),
        )

    def _language_codes(self) -> Optional[List[str]]:
        """Return the language hints for the current language.

        ``None`` (no hints) makes Gemini detect the language automatically.
        """
        language = self.current_language_config.engine_language_key
        if not language or language.strip().lower() == AUTO_LANGUAGE:
            return None
        return [code.strip() for code in language.split(",") if code.strip()]

    def _build_voice_activity_config(self) -> Optional["types.RealtimeInputConfig"]:
        """Build voice-activity-detection tuning, or ``None`` for Google defaults."""
        from google.genai import types

        config = self.config
        start = config.start_of_speech_sensitivity
        end = config.end_of_speech_sensitivity
        if (
            start is None
            and end is None
            and config.prefix_padding_ms is None
            and config.silence_duration_ms is None
        ):
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
                prefix_padding_ms=config.prefix_padding_ms,
                silence_duration_ms=config.silence_duration_ms,
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


@dataclass
class _SessionStats:
    """Counters for one Gemini session, logged in ``gemini_asr.session.ended``."""

    started: float = field(default_factory=time.monotonic)
    audio_bytes: int = 0  # 16-bit PCM actually sent to Gemini
    final_transcripts: int = 0
    dropped_chunks: int = 0  # audio that arrived while no session was open


def _transcription_text(transcription: Optional["types.Transcription"]) -> str:
    """Return the stripped text of a Gemini transcription, or ``""``."""
    if transcription is None or not transcription.text:
        return ""
    return transcription.text.strip()
