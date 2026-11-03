# Google Cloud speech components for Rasa

Speech-to-text (ASR) and text-to-speech (TTS) engines that connect a Rasa
voice assistant to Google Cloud. They plug into
[Rasa's voice streaming](https://rasa.com/docs/rasa-pro/connectors/messaging-and-voice-channels/)
the same way Rasa's built-in Azure or Deepgram engines do. You enable them in
`credentials.yml`, and no Rasa code changes are needed.

| Component | Code | `name:` in `credentials.yml` | Google service |
|---|---|---|---|
| Speech-to-text | `google_voice/gemini_asr.py` | `custom_component.google_voice.gemini_asr.GeminiASR` | [Gemini 3.5 Transcribe](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-transcribe) |
| Text-to-speech | `google_voice/google_tts.py` | `custom_component.google_voice.google_tts.GoogleTTS` | [Chirp 3 HD voices](https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd) (streaming) |

**Contents**

- [Setup](#setup)
- [Configure `credentials.yml`](#configure-credentialsyml)
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

To install only what you use:

| You use | Packages |
|---|---|
| Speech-to-text (`gemini_asr.py`) | `google-genai` |
| Text-to-speech (`google_tts.py`) | `google-cloud-texttospeech` |

> **pip dependency warnings are expected.** Rasa Pro pins older versions of
> `grpcio`, `protobuf` and `google-auth` than the Google SDKs require, so pip
> prints a conflict warning. This has not affected these components or Rasa's
> voice modules in testing. Still, test against your own Rasa version, and
> consider a dedicated virtual environment or Docker image for this
> assistant.

### 2. Enable the Google Cloud APIs

Run these on the Google Cloud project you want to bill:

```bash
gcloud services enable aiplatform.googleapis.com       # Gemini 3.5 Transcribe (ASR)
gcloud services enable texttospeech.googleapis.com     # TTS
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
*Vertex AI User* role for Gemini.

### 4. Set the project ID

Set the environment variable:

```bash
export GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
```

Alternatively, set `project_id:` in the `asr:` and `tts:` blocks. Speech-to-text
requires a project ID. Text-to-speech uses it only as the billing project.

## Configure `credentials.yml`

Add an `asr:` and a `tts:` block to your voice channel. The example below uses
`inspector`, which is what `rasa inspect` loads. Production channels
(`twilio_media_streams`, `audiocodes_stream`, `genesys`, `jambonz_stream`, …)
take the same `asr:` / `tts:` blocks.

```yaml
inspector:
  server_url: "http://localhost:5005"
  sample_rate: 24000              # inspector default is 48000; 24 kHz suits Google well

  asr:
    name: custom_component.google_voice.gemini_asr.GeminiASR
    location: us                  # global (default) | us | eu
    language_map:
      en-US:
        language: en-US           # or "auto" for automatic language detection
        model: gemini-3.5-transcribe-live

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

Start the assistant with `rasa inspect` or `rasa run`. If you leave out
`language_map`, the defaults are `en-US` with `gemini-3.5-transcribe-live`
and the `en-US-Chirp3-HD-Charon` voice.

> The project root is already on Python's path when Rasa starts, so you don't
> need to set `PYTHONPATH`.

## Multilingual assistants

1. List the extra languages in `config.yml`:

   ```yaml
   language: en-US
   additional_languages:
     - fr-FR
   ```

2. Add a `language_map` entry for every language to **both** the `asr:` and
   `tts:` blocks. Key each entry by the same language code you used in
   `config.yml`:

   ```yaml
   asr:
     name: custom_component.google_voice.gemini_asr.GeminiASR
     language_map:
       en-US:
         language: en-US
         model: gemini-3.5-transcribe-live
       fr-FR:
         language: fr-FR
         model: gemini-3.5-transcribe-live
   tts:
     name: custom_component.google_voice.google_tts.GoogleTTS
     language_map:
       en-US:
         language: en-US
         voice: en-US-Chirp3-HD-Charon
       fr-FR:
         language: fr-FR
         voice: fr-FR-Chirp3-HD-Charon
   ```

When the conversation's `language` slot changes, Rasa switches both engines to
the matching entry mid-call and reconnects to Google with the new language and
voice. Pick a TTS voice whose locale matches the entry, e.g.
`fr-FR-Chirp3-HD-*` for `fr-FR`. The same voice names (Charon, Kore, Aoede, …)
exist in every locale.

Set `language: auto` to let Gemini detect the language itself, including
switches mid-sentence.

## Configuration reference

### Speech-to-text (`GeminiASR`)

| Field | Default | Description |
|---|---|---|
| `language_map` | `en-US` → `gemini-3.5-transcribe-live` | Per Rasa language: `language` (BCP-47 code such as `en-US`, or `auto` to detect automatically) and `model`. |
| `project_id` | `GOOGLE_CLOUD_PROJECT` | Google Cloud project. Required (set it here or with the env var). |
| `location` | `global` | `global`, `us` or `eu`. Use the one closest to your callers. |
| `interim_results` | `true` | Report partial transcripts while the user is talking, so the bot can stop speaking when interrupted (barge-in). |
| `custom_vocabulary` | none | List of names, products or jargon to recognize more reliably. Up to 1000 terms; Google recommends about 100. |
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
3. **End-of-speech tuning** finalizes transcripts sooner:
   `end_of_speech_sensitivity: high` and/or a lower `silence_duration_ms`.
4. **Keep `interim_results` on** for responsive barge-in.
5. **Fixed bot responses are cached.** A repeated response from the domain
   plays almost instantly (logged as `cached=True`).

Outside these components, a faster LLM or a smaller prompt helps. Rasa also
waits about 1 s between consecutive bot messages; that delay isn't
configurable from `credentials.yml`.

## Troubleshooting

Problems show up as structured log events. Run Rasa with `--debug` for more
detail.

| Log event / error | Meaning | What to do |
|---|---|---|
| `ASRConfigError: Gemini ASR requires a Google Cloud project ID` | No project configured | Set `GOOGLE_CLOUD_PROJECT` or `project_id`. |
| Missing `google.genai` package at startup | The SDK isn't installed | `pip install google-genai` |
| `gemini_asr.session.reconnecting` (info) | Google closed the session and a new one is opening | Normal about every 10 minutes and after language switches. Frequent occurrences point to network issues. |
| `gemini_asr.stream_asr_events.giving_up` / `gemini_asr.session.reconnect_failed` | Google keeps rejecting the session | Check credentials, that the API is enabled, `model`, `location` and quota. The preceding warning has Google's error. |
| `gemini_asr.session.closed` (warning) | Gemini closed the session with an error code | The `code` and `error` fields come from Google: 4xx is config/permissions, 5xx is on Google's side. |
| `google_tts.synthesis.timeout` | Google didn't return audio within `timeout` | Check `channel_state`: `READY` means the network was fine, so look at config or Google's error. `CONNECTING`/`TRANSIENT_FAILURE` means a network problem. A non-HD voice name is a common cause. |
| Bot transcribes nothing, no errors | Usually wrong credentials or project at startup | Look for the first warning after the call starts. Make sure the browser has microphone access when using `rasa inspect`. |

## Known limitations

- **Chirp 3 HD voices only for TTS.** Google's streaming synthesis (Preview)
  works only with `*-Chirp3-HD-*` voices.
- **Audio formats.** Gemini accepts any of Rasa's audio formats. TTS supports
  Rasa's 8 kHz μ-law (phone) and 24/48 kHz linear PCM formats.
- **No Gemini "SMART" formatting mode.** That mode requires the client to mark
  where each utterance starts and ends. Rasa's voice pipeline doesn't do that,
  so the component uses Gemini's default verbatim mode.
- **Gemini voice-activity settings are not documented by Google for this
  model.** `start_of_speech_sensitivity`, `end_of_speech_sensitivity` and
  `silence_duration_ms` are generic Gemini Live API settings. Verify their
  effect before relying on them.
- **Audio during a reconnect is lost.** While a new session opens (a fraction
  of a second), incoming audio is dropped, so a word spoken exactly then may
  be missed.

## For developers

### Code layout

```
custom_component/
  google_voice/
    gemini_asr.py         # Speech-to-text: GeminiASR + GeminiASRConfig (self-contained file)
    google_tts.py         # Text-to-speech: GoogleTTS + GoogleTTSConfig
    helpers.py            # TTS helpers: audio-format mapping, project ID, regional endpoints
  requirements.txt
```

`gemini_asr.py` doesn't import from `google_tts.py` or `helpers.py`, so you
can ship the speech-to-text on its own: copy `gemini_asr.py` plus the two
`__init__.py` files on its import path.

### How speech-to-text works

`GeminiASR` subclasses Rasa's `ASREngine` and holds one Gemini Live session at
a time.

```
Rasa voice channel
   │ audio chunks                      ▲ UserIsSpeaking / NewTranscript
   ▼                                   │
GeminiASR ── send_audio_chunks ──▶ Gemini Live session ──▶ Google
          ◀─ stream_asr_events ─── Gemini Live session ◀── transcripts
```

- `genai.Client(enterprise=True)` plus `client.aio.live.connect` opens a
  transcription-only websocket session. Audio is sent as 16-bit PCM (μ-law is
  converted). `interim_input_transcription` maps to `UserIsSpeaking`, and
  `input_transcription` maps to `NewTranscript`. This follows Google's
  [live transcription sample](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/audio/speech/getting-started/gemini_3_5_transcribe_live.py).
- When a session ends, `stream_asr_events` continues with the replacement
  session (after a language switch) or opens a new one (if Google closed it).
  It stops after the call ends, or after 4 sessions in a row receive nothing
  from Google.
- A shared lock serializes sending audio and swapping sessions, so audio never
  goes to a half-closed session.

### How text-to-speech works

Each bot response opens one gRPC `StreamingSynthesize` call (Preview,
gRPC-only). Text chunks stream in as Rasa produces them, and audio streams
back at the same time. On barge-in, the call is cancelled immediately, so
Google stops generating audio.
