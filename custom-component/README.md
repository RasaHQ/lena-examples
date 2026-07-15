# Google Cloud Chirp ASR/TTS custom components

Custom Rasa voice-streaming ASR and TTS engines for Google Cloud's Chirp 3
Speech-to-Text and Chirp 3 HD Text-to-Speech, built to plug into
[Rasa's voice-streaming architecture](https://rasa.com/docs/rasa-pro/connectors/messaging-and-voice-channels/)
the same way the built-in Azure/Deepgram/Cartesia/Rime engines do.

## gRPC fact-check

Both Google Cloud Speech-to-Text streaming recognition and Text-to-Speech
bidirectional streaming synthesis are **gRPC-only** — there is no REST
equivalent for either:

- Speech-to-Text: *"Streaming speech recognition is available through gRPC
  only."* — [Transcribe audio from streaming input](https://docs.cloud.google.com/speech-to-text/docs/streaming-recognize)
- Text-to-Speech: *"This method is only available via the gRPC API (not
  REST)."* — [`StreamingSynthesize` RPC reference](https://docs.cloud.google.com/text-to-speech/docs/reference/rpc)

Both engines below use the async gRPC clients
(`SpeechAsyncClient.streaming_recognize`,
`TextToSpeechAsyncClient.streaming_synthesize`), which take an `async`
generator of request messages and return an async-iterable of response
messages, so requests and responses flow concurrently over the same
bidirectional call.

## What's included

```
custom-component/
  google_voice/
    helpers.py         # shared helpers: audio-format mapping, project id,
                        # regional endpoint, gRPC channel state
    google_asr.py      # GoogleASRConfig (pydantic) + GoogleASR(ASREngine) - Speech-to-Text v2 (chirp_3)
    google_tts.py      # GoogleTTSConfig (pydantic) + GoogleTTS(TTSEngine) - Text-to-Speech streaming (Chirp 3 HD)
  requirements.txt
```

Cross-cutting concerns shared by both engines live in `helpers.py` so the
logic is defined once: Rasa `AudioFormat` ↔ Google `AudioEncoding` mapping,
project-id resolution, regional-endpoint selection from `location`, and a
best-effort gRPC `channel_state` probe used in diagnostics.

- **`GoogleASR`** wraps `Speech.StreamingRecognize` (Speech-to-Text v2,
  `chirp_3` model). Audio chunks pushed via `send_audio_chunks` are queued
  and fed to the request generator driving the call; transcripts and
  voice-activity events from the response stream are translated into
  `NewTranscript`/`UserIsSpeaking` events for Rasa's barge-in logic.
- **`GoogleTTS`** wraps `TextToSpeech.StreamingSynthesize` (Preview,
  Chirp 3 HD voices only). Each bot utterance opens one streaming call: text
  chunks are queued and sent as they arrive from the dialogue engine, and
  audio chunks stream back concurrently. Barge-in cancels the in-flight gRPC
  call directly instead of waiting for it to finish.

Both classes follow the exact same interface as the built-in
`rasa/core/channels/voice_stream/asr/azure` and
`rasa/core/channels/voice_stream/tts/*.py` engines (`ASREngine`/`TTSEngine`
subclasses, pydantic `*Config` with a `language_map`, `from_config_dict`,
`get_default_config`).

## Setup

1. **Install the SDKs** into whichever Python environment runs `rasa run` /
   `rasa inspect` for this project:

   ```bash
   pip install -r custom-component/requirements.txt
   ```

   > **Dependency conflict warning:** `rasa-pro`/`rasa-sdk` pin an older
   > `protobuf`/`grpcio` than the Google Cloud SDKs need transitively (via
   > `grpcio-status`). `pip` will print a conflict warning when installing
   > both together — this was verified to still be non-fatal against
   > rasa-pro 3.17.2 (imports and the voice-streaming module load fine with
   > the newer `protobuf`), but re-check this on your own version and
   > consider a dedicated virtual environment for this project.

2. **Authenticate.** Both clients use
   [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials) —
   no engine-specific API key/env var is required by this component. Set one of:
   - `gcloud auth application-default login` (local development), or
   - `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`, or
   - an attached service account (when running on GCP).

3. **Enable the APIs** on your Google Cloud project:

   ```bash
   gcloud services enable speech.googleapis.com texttospeech.googleapis.com
   ```

4. **Set your project ID**, either via the `GOOGLE_CLOUD_PROJECT` environment
   variable or the `project_id` field. ASR requires it (it names the implicit
   recognizer resource); TTS uses it only as the quota/billing project and
   otherwise falls back to Application Default Credentials.

## Usage in `credentials.yml`

Reference the classes by their (importable, if slightly unusual-looking)
module path — the hyphen in `custom-component` is fine for `importlib`, it's
just not valid in an `import` *statement*, and Rasa loads custom ASR/TTS
engines via `importlib` under the hood. This project's root directory is
already on `sys.path` when `rasa run`/`rasa inspect` starts, the same way
`actions_module: "actions"` resolves in `endpoints.yml`, so no `PYTHONPATH`
changes are needed.

```yaml
# credentials.yml — use the `inspector` key for `rasa inspect`
# (for production channels, use audiocodes / twilio / genesys / jambonz /
# vonage / signalwire / browser_audio / ... with the same asr/tts blocks)
# ASR project: set GOOGLE_CLOUD_PROJECT, or pass project_id under asr:.
inspector:
  server_url: "http://localhost:5005"
  # Optional channel-level knobs (not Google-specific):
  sample_rate: 24000  # inspector default is 48000; 24k is Chirp-friendly
  asr:
    name: custom-component.google_voice.google_asr.GoogleASR
    # project_id: your-gcp-project-id  # or rely on GOOGLE_CLOUD_PROJECT
    location: eu  # prefer "eu" if you're in Europe
    endpointing_sensitivity: short
    interim_results: true
    enable_voice_activity_events: true
    # enable_automatic_punctuation: true
    # recognizer: projects/.../locations/.../recognizers/...
    language_map:
      en-US:
        language: en-US
        model: chirp_3
  tts:
    name: custom-component.google_voice.google_tts.GoogleTTS
    location: eu  # optional; match ASR region for data residency
    language_map:
      en-US:
        language: en-US
        voice: en-US-Chirp3-HD-Charon  # must be a Chirp 3: HD voice
  interruptions:
    enabled: true
    min_words: 3
```

Both engines fall back to sensible defaults (`en-US`, `chirp_3`,
`en-US-Chirp3-HD-Charon`, `location: us`) if you omit `language_map`
entirely, matching the built-in engines' pattern of "works out of the box,
override per-language as needed."

### Multilingual assistants

For an assistant that speaks more than one language, list every extra
language under `additional_languages` in `config.yml`, then add a matching
`language_map` entry (keyed by the **same** Rasa language code) to both the
ASR and TTS blocks. This works exactly like the built-in Deepgram/Azure/etc.
engines — Rasa validates that the `language_map` keys line up with
`config.yml`, and switches the active entry automatically whenever the
conversation's language slot changes (via `set_language`, which transparently
reconnects each Google gRPC stream with the new language/voice).

```yaml
# config.yml
language: en-US
additional_languages:
  - es-US
```

```yaml
# credentials.yml (asr/tts blocks)
  asr:
    name: custom-component.google_voice.google_asr.GoogleASR
    location: eu
    language_map:
      en-US:
        language: en-US
        model: chirp_3
      es-US:
        language: es-US
        model: chirp_3
  tts:
    name: custom-component.google_voice.google_tts.GoogleTTS
    location: eu
    language_map:
      en-US:
        language: en-US
        voice: en-US-Chirp3-HD-Charon
      es-US:
        language: es-US
        voice: es-US-Chirp3-HD-Charon
```

Use a Chirp 3: HD voice whose locale matches the entry's `language` (e.g.
`es-US-Chirp3-HD-*` for `es-US`); the same voice "characters" (Charon, Kore,
Aoede, …) are available across locales.

### ASR config (`GoogleASRConfig`)

| Field | Default | Description |
|---|---|---|
| `project_id` | `GOOGLE_CLOUD_PROJECT` | GCP project that owns the Speech recognizer. Required (env or field). |
| `location` | `us` | Multi-region for the recognizer. `chirp_3` is GA in `us` and `eu`. Pick the closest region. |
| `recognizer` | implicit `_` | Full resource name. Defaults to `projects/{project_id}/locations/{location}/recognizers/_` (no prior setup). |
| `language_map` | `en` → `en-US` / `chirp_3` | Per Rasa language: `language` (BCP-47) and `model`. |
| `interim_results` | `true` | Stream partial transcripts as `UserIsSpeaking` for barge-in. |
| `enable_voice_activity_events` | `true` | Emit speech-start ASAP when interim text hasn't arrived yet (barge-in). |
| `enable_automatic_punctuation` | `true` | Model adds punctuation / capitalization. |
| `endpointing_sensitivity` | unset → Google default (`standard`) | How quickly Google finalizes after silence: `standard` (long-form), `short` (commands / single sentences), `supershort` (yes/no / single words — can cut off mid-sentence). Requires `google-cloud-speech>=2.33`; on older SDKs the engine logs a warning and ignores this field (ASR still works). |

### TTS config (`GoogleTTSConfig`)

| Field | Default | Description |
|---|---|---|
| `language_map` | `en-US` → `en-US` / `en-US-Chirp3-HD-Charon` | Per Rasa language: `language` and `voice` (Chirp 3 HD only for streaming). Keys must match `config.yml`. |
| `project_id` | `GOOGLE_CLOUD_PROJECT` → ADC | Quota/billing project. Falls back to the env var, then to Application Default Credentials, when unset. |
| `location` | unset → global | Regional endpoint (e.g. `eu` → `eu-texttospeech.googleapis.com`). Mirrors the ASR `location`; leave unset (or `global`) for the default endpoint. |
| `timeout` | `30` (inherited from `TTSEngineConfig`) | Max seconds to wait for the `streaming_synthesize` call to open and for each audio chunk after that. If Google stalls past this, synthesis is aborted so the turn can finish instead of hanging indefinitely. |

There is no `speaking_rate` field: `StreamingAudioConfig` has no pace-control
field (only the unary `AudioConfig` does), so it can't be honored for
streaming synthesis. Setting it in `credentials.yml` is caught by Rasa's
own "unknown config field" warning rather than accepted here.

### Latency tuning

Speech I/O is usually not the largest part of turn latency — LLM command
generation and Rasa's inter-utterance pacing often dominate (watch
`rasa_processing_latency` / `command_processor` vs
`tts_first_byte_latency` in debug logs). Still useful:

1. **`sample_rate: 24000`** (channel) — less audio than inspector's 48 kHz default; fits Chirp ASR/TTS well.
2. **`location: eu` or `us`** — use the multi-region nearest your callers / bot host.
3. **`endpointing_sensitivity: short`** — finalize ASR sooner after the user stops speaking; use `supershort` only for single-word answers. Needs `google-cloud-speech>=2.33` (`pip install -r custom-component/requirements.txt`).
4. **Keep `interim_results` + `enable_voice_activity_events`** — improves barge-in responsiveness.
5. **Template TTS cache** — repeated domain replies show `cached=True` and ~ms first-byte; first synthesis of a new string still hits Google streaming.

Outside this component: reduce LLM time (model / prompt size), and note Rasa voice's default ~1s
`min_delay_between_bot_messages` between consecutive bot utterances.

## Known limitations

- **Chirp 3 HD voices only for streaming TTS.** `StreamingSynthesize` is a
  Preview feature restricted to Chirp 3: HD voices
  (`*-Chirp3-HD-*`). Non-HD voices will fail at synthesis time — use a
  different TTS engine for those.
- **Streaming session limits.** Very long, uninterrupted `StreamingRecognize`
  calls may eventually be closed server-side; this component does not
  implement automatic mid-call reconnect (Rasa already tears down and
  reconnects both engines whenever the language slot changes, via
  `set_language`, but not purely on a timer).
- **Audio formats.** Only `MULAW_8KHZ` and `L16_24KHZ`/`L16_48KHZ` (Rasa's
  built-in `AudioFormat`s) are supported. ASR maps L16 → Google `LINEAR16`;
  streaming TTS maps L16 → `PCM` (raw PCM — `LINEAR16` is WAV-wrapped and
  rejected for streaming). See `google_voice/helpers.py`.
- **If the bot goes silent mid-call** (no audio; `tts_first_byte_latency_ms`
  stays `None`; eventually a `google_tts.synthesis.timeout` log), the 30s
  `tts.timeout` bounds the stall so the turn can finish instead of freezing
  `ConversationQueue` forever, and the log's `channel_state` tells you where
  to look next: `READY` means the gRPC channel was fine (a config/request
  bug, e.g. an unsupported field passed to `StreamingAudioConfig`, or a
  backend-side stall); `CONNECTING`/`TRANSIENT_FAILURE`/`IDLE` points at a
  network/channel problem instead. Skip `GRPC_TRACE` unless you already know
  which stream ID to isolate — it floods with ASR audio frames.
