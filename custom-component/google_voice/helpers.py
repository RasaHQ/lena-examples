"""Shared Google Cloud helpers for the Chirp ASR/TTS engines.

Both engines need to: (1) map Rasa's ``AudioFormat`` onto Google's encoding
enums, and (2) resolve a project id / regional endpoint and inspect the gRPC
channel for diagnostics. Defined once here so the two engines stay
consistent and don't duplicate this (subtly version-sensitive) logic.
"""

import os
from typing import TYPE_CHECKING, Any, Optional

from rasa.core.channels.voice_stream.asr.asr_engine import ASRConfigError
from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    L16_48KHZ,
    MULAW_8KHZ,
    AudioFormat,
)
from rasa.core.channels.voice_stream.tts.tts_engine import TTSError

if TYPE_CHECKING:
    from google.cloud import speech_v2, texttospeech_v1

GOOGLE_CLOUD_PROJECT_ENV_VAR = "GOOGLE_CLOUD_PROJECT"

# Locations that map to Google's default (global) endpoint, i.e. no host
# override. Everything else is turned into a regional host below.
_GLOBAL_LOCATIONS = frozenset({"", "global"})


def google_asr_encoding(
    audio_format: AudioFormat,
) -> "speech_v2.ExplicitDecodingConfig.AudioEncoding":
    """Return the Speech-to-Text v2 ``AudioEncoding`` for *audio_format*.

    Raises:
        ASRConfigError: When the audio format has no Google Cloud equivalent.
    """
    from google.cloud import speech_v2

    encoding_map = {
        MULAW_8KHZ: speech_v2.ExplicitDecodingConfig.AudioEncoding.MULAW,
        L16_24KHZ: speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
        L16_48KHZ: speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
    }
    encoding = encoding_map.get(audio_format)
    if encoding is None:
        raise ASRConfigError(
            f"Audio format {audio_format} is not supported by Google "
            f"Speech-to-Text."
        )
    return encoding


def google_tts_encoding(
    audio_format: AudioFormat,
) -> "texttospeech_v1.AudioEncoding":
    """Return the streaming ``AudioEncoding`` for Text-to-Speech.

    Streaming synthesis only supports headerless PCM, ALAW, MULAW, and
    OGG_OPUS output (unlike the unary ``SynthesizeSpeech`` call, whose
    ``LINEAR16``/``MULAW`` output is wrapped in a WAV header).

    Raises:
        TTSError: When the audio format has no Google Cloud equivalent.
    """
    from google.cloud import texttospeech_v1

    # StreamingSynthesize only accepts PCM / ALAW / MULAW / OGG_OPUS —
    # LINEAR16 (WAV-wrapped) is for unary SynthesizeSpeech and is rejected
    # with "400 Unsupported audio encoding." for streaming.
    encoding_map = {
        MULAW_8KHZ: texttospeech_v1.AudioEncoding.MULAW,
        L16_24KHZ: texttospeech_v1.AudioEncoding.PCM,
        L16_48KHZ: texttospeech_v1.AudioEncoding.PCM,
    }
    encoding = encoding_map.get(audio_format)
    if encoding is None:
        raise TTSError(
            f"Audio format {audio_format} is not supported by Google "
            f"Text-to-Speech streaming synthesis."
        )
    return encoding


def resolve_project_id(configured: Optional[str]) -> Optional[str]:
    """Return the configured project id, else the ``GOOGLE_CLOUD_PROJECT`` var.

    Returns ``None`` when neither is set; callers decide whether a missing
    project is fatal (ASR needs one for the implicit recognizer path) or just
    means "fall back to Application Default Credentials" (TTS).
    """
    return configured or os.environ.get(GOOGLE_CLOUD_PROJECT_ENV_VAR)


def regional_endpoint(service: str, location: Optional[str]) -> Optional[str]:
    """Return the regional API host for a Google Cloud speech service.

    ``service`` is the host stem — ``"speech"`` or ``"texttospeech"``. A
    ``location`` such as ``"us"`` or ``"eu"`` yields e.g.
    ``"us-speech.googleapis.com"``. Returns ``None`` for the global/default
    endpoint so callers can leave ``ClientOptions.api_endpoint`` unset.
    """
    if not location or location in _GLOBAL_LOCATIONS:
        return None
    return f"{location}-{service}.googleapis.com"


def grpc_channel_state(client: Any) -> Optional[str]:
    """Best-effort connectivity snapshot of a client's gRPC channel.

    Diagnostics only (logged on timeout/error): a ``READY`` channel points at
    a config/request or backend-stall problem, while
    ``CONNECTING``/``TRANSIENT_FAILURE``/``IDLE`` points at a network/channel
    problem. Never raises — the transport's internal attributes are not a
    stable public API across ``google-cloud-*`` versions.
    """
    try:
        return str(client.transport.grpc_channel.get_state())
    except Exception:
        return None
