"""Small Google Cloud helpers used by ``google_tts``."""

import os
from typing import TYPE_CHECKING, Any, Optional

from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    L16_48KHZ,
    MULAW_8KHZ,
    AudioFormat,
)
from rasa.core.channels.voice_stream.tts.tts_engine import TTSError

if TYPE_CHECKING:
    from google.cloud import texttospeech_v1

GOOGLE_CLOUD_PROJECT_ENV_VAR = "GOOGLE_CLOUD_PROJECT"

# Locations served by Google's default endpoint (no regional host).
_GLOBAL_LOCATIONS = frozenset({"", "global"})


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

    # LINEAR16 (WAV-wrapped) is rejected for streaming with
    # "400 Unsupported audio encoding.", so L16 maps to raw PCM.
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

    Returns ``None`` when neither is set, in which case TTS falls back to
    Application Default Credentials.
    """
    return configured or os.environ.get(GOOGLE_CLOUD_PROJECT_ENV_VAR)


def regional_endpoint(service: str, location: Optional[str]) -> Optional[str]:
    """Return the regional API host for a Google Cloud speech service.

    A ``location`` such as ``"eu"`` yields e.g.
    ``"eu-texttospeech.googleapis.com"``. Returns ``None`` for the global
    endpoint so callers can leave ``ClientOptions.api_endpoint`` unset.
    """
    if not location or location in _GLOBAL_LOCATIONS:
        return None
    return f"{location}-{service}.googleapis.com"


def grpc_channel_state(client: Any) -> Optional[str]:
    """Best-effort connectivity snapshot of a client's gRPC channel.

    Diagnostics only (logged on timeout/error): ``READY`` points at a
    config/request or backend problem, ``CONNECTING``/``TRANSIENT_FAILURE``/
    ``IDLE`` at a network problem. Never raises, because the transport
    internals are not a stable public API across ``google-cloud-*`` versions.
    """
    try:
        return str(client.transport.grpc_channel.get_state())
    except Exception:
        return None
