# Google Cloud speech components for Rasa

Speech-to-text (ASR) and text-to-speech (TTS) engines that connect a Rasa
voice assistant to Google Cloud. They plug into
[Rasa's voice streaming](https://rasa.com/docs/rasa-pro/connectors/messaging-and-voice-channels/)
the same way Rasa's built-in Azure or Deepgram engines do. You enable them in
`credentials.yml`, and no Rasa code changes are needed.

There are two speech-to-text options and one text-to-speech component. Use
one ASR option together with the TTS.

| Component | Code | `name:` in `credentials.yml` | Google service |
|---|---|---|---|
| **Speech-to-text, Option A** | `google_asr/` (package) | `custom_component.google_asr.GoogleASR` | [Chirp 3](https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3) **or** [Gemini 3.5 Transcribe](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-transcribe), chosen per language |
| **Speech-to-text, Option B** | `google_voice/gemini_asr.py` (single file) | `custom_component.google_voice.gemini_asr.GeminiASR` | [Gemini 3.5 Transcribe](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-transcribe) only |
| Text-to-speech | `google_voice/google_tts.py` | `custom_component.google_voice.google_tts.GoogleTTS` | [Chirp 3 HD voices](https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd) (streaming) |

The two ASR options are independent. Either can be copied into a project on
its own. See [Choosing a speech-to-text option](#choosing-a-speech-to-text-option).

**Contents**

- [Setup](#setup)
- [Configure `credentials.yml`](#configure-credentialsyml)
- [Choosing a speech-to-text option](#choosing-a-speech-to-text-option)
- [Multilingual assistants](#multilingual-assistants)
- [Configuration reference](#configuration-reference)
- [Tuning latency](#tuning-latency)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [For developers](#for-developers)

## Setup

### 1. Install the Python packages

Install the packages into the same Python environment that runs `rasa run` or
`rasa inspect`:

```bash
pip install -r custom_component/requirements.txt
```

This installs the packages for every component. To install only what you
use:

| You use | Packages |
|---|---|
| Option A (`google_asr/`) with Chirp 3 only | `google-cloud-speech` |
| Option A with a Gemini model | `google-cloud-speech`, `google-genai` |
| Option B (`google_voice/gemini_asr.py`) | `google-genai` |
| TTS (`google_voice/google_tts.py`) | `google-cloud-texttospeech` |

> **pip dependency warnings are expected.** Rasa Pro pins older versions of
> `grpcio`, `protobuf` and `google-auth` than the Google SDKs require, so pip
> prints a conflict warning. This has not affected these components or Rasa's
> voice modules in testing. Still, test against your own Rasa version, and
> consider a dedicated virtual environment or Docker image for this
> assistant.

### 2. Enable the Google Cloud APIs

Run these on the Google Cloud project you want to bill:

```bash
gcloud services enable texttospeech.googleapis.com     # TTS
gcloud services enable speech.googleapis.com           # Chirp 3 (Option A)
gcloud services enable aiplatform.googleapis.com       # Gemini 3.5 Transcribe (Option A or B)
```

### 3. Authenticate

The components use
[Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials).
They don't take an API key. Use one of these:

- **Local development:** run `gcloud auth application-default login`.
- **Service account key:** set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`.
- **On Google Cloud** (GKE, Cloud Run, GCE): attach a service account to the
  workload.

The account needs permission to call the APIs above, for example the
*Cloud Speech Client* role, plus *Vertex AI User* for Gemini.

### 4. Set the project ID

Set the environment variable:

```bash
export GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
```

Alternatively, set `project_id:` in the `asr:` and `tts:` blocks. Speech-to-text
requires a project ID. Text-to-speech uses it only as the billing project.

## Configure `credentials.yml`

Add an `asr:` and a `tts:` block to your voice channel. The example below uses
`inspector`, which is what `rasa inspect --voice` loads. Production channels
(`twilio_media_streams`, `audiocodes_stream`, `genesys`, `jambonz_stream`, …)
take the same `asr:` / `tts:` blocks.

```yaml
inspector:
  server_url: "http://localhost:5005"
  sample_rate: 24000              # inspector default is 48000; 24 kHz suits Google well

  asr:
    name: custom_component.google_asr.GoogleASR   # Option A
    location: us                  # us | eu (Gemini also: global)
    endpointing_sensitivity: short   # Chirp: finalize sooner after the user stops
    language_map:
      en-US:
        language: en-US
        model: chirp_3            # or gemini-3.5-transcribe-live

  tts:
    name: custom_component.google_voice.google_tts.GoogleTTS
    location: us                  # optional; match ASR for data residency
    language_map:
      en-US:
        language: en-US
        voice: en-US-Chirp3-HD-Charon   # must be a Chirp 3 HD voice

  interruptions:
    enabled: true
    min_words: 3
```

For Option B, replace the `asr:` block with this one:

```yaml
  asr:
    name: custom_component.google_voice.gemini_asr.GeminiASR    # Option B
    location: us                  # global (default) | us | eu
    language_map:
      en-US:
        language: en-US           # or "auto" for automatic language detection
        model: gemini-3.5-transcribe-live
```

Start the assistant with `rasa inspect --voice` or `rasa run`. If you leave out
`language_map`, the defaults are `en-US` with `chirp_3` (Option A) or
`gemini-3.5-transcribe-live` (Option B), and the `en-US-Chirp3-HD-Charon`
voice.

> The project root is already on Python's path when Rasa starts, so you don't
> need to set `PYTHONPATH`.

## Choosing a speech-to-text option

| | Option A: `google_asr/` | Option B: `google_voice/gemini_asr.py` |
|---|---|---|
| Google services | Chirp 3 and Gemini 3.5 Transcribe | Gemini 3.5 Transcribe |
| Switch service | change `model:`, per language | — |
| Code | package of 6 small modules | 1 file, about 430 lines |
| Packages | `google-cloud-speech` (+ `google-genai` for Gemini) | `google-genai` |
| Pick it when | you want to compare Chirp and Gemini, keep Chirp as a fallback, or mix them per language | you've settled on Gemini and want the least code to own |

Both options produce the same Gemini behaviour, configuration fields and
automatic reconnects.

### Option A: Chirp 3 or Gemini, per language

The `model` in each `language_map` entry decides which Google service
transcribes that language. To switch, change `model:`. Nothing else in the
config has to change.

| | Chirp 3 | Gemini 3.5 Transcribe |
|---|---|---|
| `model:` | `chirp_3` (default) | `gemini-3.5-transcribe-live` |
| Extra package | none | `google-genai` |
| `location:` | `us`, `eu` | `global`, `us`, `eu` |
| Language detection | set `language:` (or `auto`) | set `language:`, or `auto` to detect automatically (including switching languages mid-sentence) |
| Bias towards domain terms | — | `custom_vocabulary` |
| End-of-speech tuning | `endpointing_sensitivity` | `end_of_speech_sensitivity`, `silence_duration_ms` |
| Connection | gRPC stream | websocket session, about 10 min of audio per session (the component reconnects automatically) |

You can mix services in a multilingual assistant, for example Chirp 3 for
English and Gemini for Spanish. See the next section.

## Multilingual assistants

1. List the extra languages in `config.yml`:

   ```yaml
   language: en-US
   additional_languages:
     - es-US
   ```

2. Add a `language_map` entry for every language to **both** the `asr:` and
   `tts:` blocks. Key each entry by the same language code you used in
   `config.yml`:

   ```yaml
   asr:
     name: custom_component.google_asr.GoogleASR
     language_map:
       en-US:
         language: en-US
         model: chirp_3
       es-US:
         language: es-US
         model: gemini-3.5-transcribe-live   # Option A: services can differ per language
   tts:
     name: custom_component.google_voice.google_tts.GoogleTTS
     language_map:
       en-US:
         language: en-US
         voice: en-US-Chirp3-HD-Charon
       es-US:
         language: es-US
         voice: es-US-Chirp3-HD-Charon
   ```

When the conversation's `language` slot changes, Rasa switches both engines to
the matching entry mid-call. It reconnects to Google with the new language,
voice and, if needed, service. Pick a TTS voice whose locale matches the entry,
e.g. `es-US-Chirp3-HD-*` for `es-US`. The same voice names (Charon, Kore,
Aoede, …) exist in every locale.

## Configuration reference

### Speech-to-text, Option A (`GoogleASR`)

Some fields apply to only one service. The other service ignores them and
logs a `google_asr.config.fields_ignored` warning once.

| Field | Applies to | Default | Description |
|---|---|---|---|
| `language_map` | both | `en-US` → `chirp_3` | Per Rasa language: `language` (BCP-47 code such as `en-US`, or `auto`) and `model` (which also picks the service). |
| `project_id` | both | `GOOGLE_CLOUD_PROJECT` | Google Cloud project. Required (set it here or with the env var). |
| `location` | both | `us` | Region. Chirp 3: `us` or `eu`. Gemini: `global`, `us` or `eu`. Use the one closest to your callers. |
| `interim_results` | both | `true` | Report partial transcripts while the user is talking, so the bot can stop speaking when interrupted (barge-in). |
| `endpointing_sensitivity` | Chirp 3 | Google default (`standard`) | How quickly an utterance is finalized after the user stops: `standard`, `short` (single sentences / commands) or `supershort` (yes/no; may cut people off). Needs `google-cloud-speech>=2.33`. |
| `enable_voice_activity_events` | Chirp 3 | `true` | Signal "user started speaking" before any words are recognized (faster barge-in). |
| `enable_automatic_punctuation` | Chirp 3 | `true` | Add punctuation and capitalization. |
| `recognizer` | Chirp 3 | implicit | Full recognizer resource name. Defaults to `projects/<project>/locations/<location>/recognizers/_`, which needs no setup. |
| `custom_vocabulary` | Gemini | none | List of names, products or jargon to recognize more reliably. Up to 1000 terms; Google recommends about 100. |
| `end_of_speech_sensitivity` | Gemini | Google default | `high` finalizes sooner after the user stops, `low` waits longer. |
| `start_of_speech_sensitivity` | Gemini | Google default | `high` detects speech sooner, `low` ignores more background noise. |
| `silence_duration_ms` | Gemini | Google default | Milliseconds of silence before an utterance counts as finished. |

### Speech-to-text, Option B (`GeminiASR`)

The fields are the same as Option A's Gemini fields. Only the `location`
default differs.

| Field | Default | Description |
|---|---|---|
| `language_map` | `en-US` → `gemini-3.5-transcribe-live` | Per Rasa language: `language` (BCP-47 code, or `auto` to detect automatically) and `model`. |
| `project_id` | `GOOGLE_CLOUD_PROJECT` | Google Cloud project. Required (set it here or with the env var). |
| `location` | `global` | `global`, `us` or `eu`. |
| `interim_results` | `true` | Report partial transcripts while the user is talking (barge-in). |
| `custom_vocabulary` | none | Names, products or jargon to recognize more reliably (up to 1000; about 100 recommended). |
| `end_of_speech_sensitivity` | Google default | `high` finalizes sooner after the user stops, `low` waits longer. |
| `start_of_speech_sensitivity` | Google default | `high` detects speech sooner, `low` ignores more background noise. |
| `silence_duration_ms` | Google default | Milliseconds of silence before an utterance counts as finished. |

### Text-to-speech (`GoogleTTS`)

| Field | Default | Description |
|---|---|---|
| `language_map` | `en-US` → `en-US-Chirp3-HD-Charon` | Per Rasa language: `language` and `voice`. Only Chirp 3 HD voices work, because streaming synthesis is limited to them. |
| `project_id` | `GOOGLE_CLOUD_PROJECT` | Billing project. If neither is set, the credentials' default project is used. |
| `location` | global | Region, e.g. `us` or `eu`. Set it to match the ASR `location` for data residency. |
| `timeout` | `30` | Seconds to wait for Google before giving up on a bot response, so a stalled request can't freeze the call. |

There is no speaking-rate setting, because Google's streaming synthesis
doesn't support one.

## Tuning latency

Speech is usually not the biggest part of response time. LLM command
generation and Rasa's pacing between bot messages often take longer. Compare
`rasa_processing_latency` with `tts_first_byte_latency` in debug logs. These
settings still help:

1. **`sample_rate: 24000`** on the channel sends less audio than the
   inspector's 48 kHz default.
2. **`location:`** set to the region nearest your callers and your Rasa
   server.
3. **End-of-speech tuning** finalizes transcripts sooner. Chirp:
   `endpointing_sensitivity: short`. Gemini: `end_of_speech_sensitivity: high`
   and/or a lower `silence_duration_ms`.
4. **Keep `interim_results` (and, for Chirp, `enable_voice_activity_events`)
   on** for responsive barge-in.
5. **Fixed bot responses are cached.** A repeated response from the domain
   plays almost instantly (logged as `cached=True`).

Outside these components, a faster LLM or a smaller prompt helps, and so
does a lower Rasa voice `min_delay_between_bot_messages` (default about 1 s).

## Troubleshooting

Problems show up as structured log events. Run Rasa with `--debug` for more
detail. The table uses Option A's `google_asr.*` names. Option B
(`google_voice/gemini_asr.py`) logs the matching `gemini_asr.*` events:

| Option A | Option B |
|---|---|
| `google_asr.stream.reconnecting` | `gemini_asr.session.reconnecting` |
| `google_asr.stream.reconnect_failed` | `gemini_asr.session.reconnect_failed` |
| `google_asr.stream_asr_events.giving_up` | `gemini_asr.stream_asr_events.giving_up` |
| `google_asr.gemini.session.closed` | `gemini_asr.session.closed` |

| Log event / error | Meaning | What to do |
|---|---|---|
| `ASRConfigError: ... requires a Google Cloud project ID` | No project configured | Set `GOOGLE_CLOUD_PROJECT` or `project_id`. |
| `ASRConfigError: ... need the 'google-genai' package` | A Gemini model is configured but the SDK isn't installed | `pip install google-genai` |
| `google_asr.stream.reconnecting` (info) | Google closed the stream and a new one is opening | Normal for Gemini about every 10 minutes and after language switches. Frequent occurrences point to network issues. |
| `google_asr.stream_asr_events.giving_up` / `google_asr.stream.reconnect_failed` | Google keeps rejecting the stream | Check credentials, that the API is enabled, `model`, `location` and quota. The preceding warning has Google's error. |
| `google_asr.gemini.session.closed` (warning) | Gemini closed the session with an error code | The `code` and `error` fields come from Google: 4xx is config/permissions, 5xx is on Google's side. |
| `google_asr.chirp.stream.error` | The Chirp gRPC stream failed | Check `channel_state`: `READY` means the network was fine, so look at config or Google's error. `CONNECTING`/`TRANSIENT_FAILURE` means a network problem. |
| `google_asr.config.fields_ignored` | A field for the other service is set | Harmless; remove the field if you don't need it. |
| `google_asr.endpointing_sensitivity.unsupported_sdk` | `google-cloud-speech` is too old for this field | `pip install "google-cloud-speech>=2.33"`; until then the field is ignored. |
| `google_tts.synthesis.timeout` | Google didn't return audio within `timeout` | Same `channel_state` hints as above. A non-HD voice name is a common cause. |
| Bot transcribes nothing, no errors | Usually wrong credentials or project at startup | Look for the first warning after the call starts. Make sure `rasa inspect --voice` has microphone access. |

## Known limitations

- **Chirp 3 HD voices only for TTS.** Google's streaming synthesis (Preview)
  works only with `*-Chirp3-HD-*` voices.
- **Audio formats.** Chirp 3 ASR and TTS support Rasa's 8 kHz μ-law (phone)
  and 24/48 kHz linear PCM formats. Gemini accepts any of Rasa's formats.
- **No Gemini "SMART" formatting mode.** That mode requires the client to mark
  where each utterance starts and ends. Rasa's voice pipeline doesn't do that,
  so the component uses Gemini's default verbatim mode.
- **Gemini voice-activity settings are not documented by Google for this
  model.** `start_of_speech_sensitivity`, `end_of_speech_sensitivity` and
  `silence_duration_ms` are generic Gemini Live API settings. Verify their
  effect before relying on them.
- **Audio during a reconnect is lost.** While a new stream opens (a fraction
  of a second), incoming audio is dropped, so a word spoken exactly then may
  be missed.

## For developers

### Code layout

```
custom_component/
  google_asr/             # Option A: Chirp 3 or Gemini (self-contained package)
    __init__.py           # exports GoogleASR, GoogleASRConfig
    _engine.py            # GoogleASR + GoogleASRConfig: the Rasa-facing component (start here)
    _asr_backend.py       # ASRBackend: base class for one stream to a Google service
    _chirp_backend.py     # ChirpBackend: Speech-to-Text v2 / Chirp 3 over gRPC
    _gemini_backend.py    # GeminiLiveBackend: Gemini 3.5 Transcribe over the Live API websocket
    _helpers.py           # audio-format mapping, project ID, regional endpoints
  google_voice/
    gemini_asr.py         # Option B: GeminiASR + GeminiASRConfig, Gemini only (self-contained file)
    google_tts.py         # TTS: GoogleTTS + GoogleTTSConfig
    helpers.py            # TTS helpers: audio-format mapping, project ID, regional endpoints
  tests/                  # unit tests (no Google credentials needed)
  requirements.txt
```

Each component is self-contained, with no imports between `google_asr/`,
`google_voice/gemini_asr.py` and the TTS files (`google_voice/google_tts.py`,
`google_voice/helpers.py`), so you can ship any one without the others.
`gemini_asr.py` shares a folder with the TTS but doesn't import from it. The
two helper modules are therefore small, deliberate duplicates.
Modules starting with `_` are internal.

### How speech-to-text works (Option A)

`GoogleASR` subclasses Rasa's `ASREngine` and keeps the Rasa-facing logic:
config, language switching, and an event loop that survives reconnects. The
Google API calls happen in a backend. `connect()` creates a `ChirpBackend` or
a `GeminiLiveBackend`, depending on whether the active language's `model`
starts with `gemini-`.

```
Rasa voice channel
   │ audio chunks                      ▲ UserIsSpeaking / NewTranscript
   ▼                                   │
GoogleASR ── send_audio_chunks ──▶ backend ──▶ Google
          ◀─ stream_asr_events ─── backend ◀── transcripts
```

- Each backend owns exactly one stream. When it ends, `stream_asr_events`
  takes the replacement (after `set_language`) or opens a new one (if Google
  closed it). It stops after the call ends, or after 4 streams in a row
  receive nothing from Google.
- A shared lock serializes sending audio and swapping backends, so audio never
  goes to a half-closed stream.
- **Chirp 3**: `StreamingRecognize` is gRPC-only. The first request carries
  the config, and later requests carry audio from a queue. Responses arrive
  while audio is still being sent.
- **Gemini**: `genai.Client(enterprise=True)` plus `client.aio.live.connect`
  opens a transcription-only websocket session. Audio is sent as 16-bit PCM
  (μ-law is converted). `interim_input_transcription` maps to
  `UserIsSpeaking`, and `input_transcription` maps to `NewTranscript`. This
  follows Google's
  [live transcription sample](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/audio/speech/getting-started/gemini_3_5_transcribe_live.py).

To add another Google model family, subclass `ASRBackend`, then pick it in
`GoogleASR._backend_class`.

### How speech-to-text works (Option B)

`google_voice/gemini_asr.py` is Option A's Gemini path folded into a single `ASREngine`
subclass, with no backend layer. `GeminiASR` holds the Live session directly.
`stream_asr_events` follows the same reconnect rules: continue with the
replacement session after a language switch, open a new one when Google closes
the session, and give up after 4 empty sessions in a row. The only external
dependencies are Rasa and `google-genai`.

### How text-to-speech works

Each bot response opens one gRPC `StreamingSynthesize` call (Preview, also
gRPC-only). Text chunks stream in as Rasa produces them, and audio streams
back at the same time. On barge-in, the call is cancelled immediately, so
Google stops generating audio.

### Running the tests

```bash
pip install -r custom_component/requirements.txt pytest
pytest custom_component/tests
```

The tests use fake Google responses and sessions, so they need neither
credentials nor network access.
