# WARD --- Coding Agent Implementation Plan

## 1. Project Goal

Build the WARD road-surface classification application around two
existing models:

-   Original WARD model
-   Fine-tuned WARD model

The application classifies road surface condition as:

``` text
dry
damp
wet
drying
```

The application must support:

``` text
DEV  -> local GPU inference
PROD -> Hugging Face Inference Endpoints
```

The same prediction, fusion, temporal, and UI logic should work in both
modes.

------------------------------------------------------------------------

# 2. Runtime Modes

## DEV Mode

Use the local GPU for model inference.

Purpose:

-   live camera testing
-   local video testing
-   low-latency development
-   debugging
-   performance testing

Run:

``` bash
WARD_MODE=dev streamlit run app.py
```

Pipeline:

``` text
Camera
  ↓
Frame Capture
  ↓
Local GPU Models
  ↓
Fusion
  ↓
Temporal Engine
  ↓
Streamlit
```

## PROD Mode

Use Hugging Face Inference Endpoints.

Run:

``` bash
WARD_MODE=prod streamlit run app.py
```

Pipeline:

``` text
Camera / Image
  ↓
Frame Sampling
  ↓
HF Inference Endpoint(s)
  ↓
Fusion
  ↓
Temporal Engine
  ↓
Streamlit
```

The frontend must not contain separate prediction logic for DEV and
PROD.

------------------------------------------------------------------------

# 3. Models

Use the two existing WARD models.

``` text
Original:
adityakumarxdev/ward-siglip-lora

Fine-tuned:
adityakumarxdev/ward-siglip-lora-2
```

The model IDs must be configurable.

Do not retrain or modify either model as part of this project.

------------------------------------------------------------------------

# 4. Model Label Handling

Do not assume class indexes.

Read:

``` python
model.config.id2label
model.config.label2id
```

The fine-tuned model should provide:

``` text
dry
damp
wet
drying
```

The original model provides the classes it was trained for.

Only use original-model classes that actually exist.

Do not map an unsupported original class to another condition without an
explicit rule.

------------------------------------------------------------------------

# 5. Image Preprocessing

Each model must use its own processor.

Example:

``` python
original_processor = AutoImageProcessor.from_pretrained(...)
fine_processor = AutoImageProcessor.from_pretrained(...)
```

Input images must be converted to:

``` text
RGB
```

Models must be loaded once and reused.

Inference must use:

``` python
model.eval()
```

and inference/no-gradient mode.

------------------------------------------------------------------------

# 6. Prediction Interface

Create a common prediction structure.

Example:

``` python
PredictionResult(
    model="fine_tuned",
    scores={
        "dry": 0.10,
        "damp": 0.60,
        "wet": 0.20,
        "drying": 0.10,
    },
    top_label="damp",
    top_confidence=0.60,
    latency_ms=80,
)
```

The original model should return the same structure even if it has fewer
supported classes.

The rest of the application must not depend on the exact model
implementation.

------------------------------------------------------------------------

# 7. Model Provider Abstraction

Create:

``` text
inference/
    provider.py
    local_provider.py
    hf_provider.py
```

Interface:

``` python
class InferenceProvider:
    def predict(self, image):
        ...
```

DEV:

``` text
LocalInferenceProvider
```

PROD:

``` text
HFInferenceProvider
```

This allows the application to switch between local and remote inference
without changing the frontend.

------------------------------------------------------------------------

# 8. Production Endpoint Configuration

Use environment variables / Streamlit secrets.

Example:

``` text
WARD_MODE=prod

HF_TOKEN=...
HF_ORIGINAL_ENDPOINT=...
HF_FINE_ENDPOINT=...
```

Never hard-code credentials.

The application must gracefully handle:

-   timeout
-   connection failure
-   invalid endpoint response
-   authentication failure
-   HTTP error

Do not let an endpoint failure crash the Streamlit application.

------------------------------------------------------------------------

# 9. Dual-Model Fusion

The fine-tuned model has initial priority.

Initial weighting:

``` text
Original = 30%
Fine-tuned = 70%
```

Use:

``` python
dry_score =
    0.30 * original_dry \
    + 0.70 * finetuned_dry

wet_score =
    0.30 * original_wet \
    + 0.70 * finetuned_wet

damp_score =
    finetuned_damp

drying_score =
    finetuned_drying
```

Normalize the resulting scores before selecting the final class.

The weights must be configurable.

------------------------------------------------------------------------

# 10. Confidence-Aware Fusion

Do not permanently hard-code the 30/70 split.

Implement a configurable weighting layer so it can later use model
confidence.

Initial behavior should remain:

``` text
fine-tuned model has priority
```

The implementation must make the weight calculation independent from the
UI.

------------------------------------------------------------------------

# 11. Model Agreement

Record:

``` text
original_label
fine_label
agreement
```

Examples:

``` text
Original: dry
Fine: dry
Agreement: true
```

``` text
Original: dry
Fine: drying
Agreement: false
```

Model disagreement should not automatically determine the final class.

It should be available to the temporal engine and UI.

------------------------------------------------------------------------

# 12. Temporal Prediction Engine

The final displayed condition should not change solely because one frame
produced a different prediction.

Create:

``` text
temporal/
    engine.py
    smoothing.py
    history.py
```

The temporal engine receives fused predictions.

It maintains a bounded history of recent predictions.

Example:

``` python
deque(maxlen=20)
```

The size must be configurable.

------------------------------------------------------------------------

# 13. Temporal Smoothing

Use probability history instead of only raw label voting.

For example:

``` python
smoothed =
    alpha * current_score \
    + (1 - alpha) * previous_score
```

Make `alpha` configurable.

The purpose is to reduce frame-to-frame prediction noise.

------------------------------------------------------------------------

# 14. Prediction Consistency

Track:

``` text
current stable state
candidate state
candidate streak
candidate confidence
```

Example:

``` text
Stable:
wet

Incoming:
drying
drying
drying
drying
```

If `drying` remains consistent for enough predictions, it should
eventually become the stable state.

The temporal engine must not suppress the same candidate indefinitely.

------------------------------------------------------------------------

# 15. Hysteresis

Prevent:

``` text
wet
drying
wet
drying
wet
```

from appearing rapidly.

Use configurable:

``` text
switch margin
minimum candidate streak
minimum state dwell time
```

Initial values may be:

``` text
switch margin = 0.10
minimum candidate streak = 3
minimum dwell = 2 seconds
```

These values must remain configurable.

------------------------------------------------------------------------

# 16. One-Frame Spike Protection

Example:

``` text
dry
dry
dry
wet
dry
dry
```

The single `wet` prediction should not immediately change the stable
state.

Require persistence unless the new prediction is exceptionally strong
and the configured fast-switch rule allows it.

------------------------------------------------------------------------

# 17. Strong Prediction Override

The temporal engine should not become so conservative that genuine
changes take too long.

Implement a configurable strong-evidence rule.

Example starting behavior:

``` text
confidence >= 0.90
+
candidate appears at least twice
```

may allow a faster state transition.

Keep this separate from normal hysteresis.

------------------------------------------------------------------------

# 18. Ambiguity

Calculate the margin between the top two fused classes:

``` python
margin = top1_score - top2_score
```

Expose:

``` text
ambiguous
```

when the margin is small.

Initial configurable threshold:

``` text
0.10
```

The UI should be able to show:

``` text
Stable
```

or:

``` text
Uncertain
```

based on temporal state and confidence.

------------------------------------------------------------------------

# 19. Drying Handling

`drying` needs special temporal treatment because it represents a
transition.

Do not force a single borderline frame to become stable `drying`.

Use:

``` text
drying candidate
+
persistence
+
confidence
```

before promoting it to the stable state.

------------------------------------------------------------------------

# 20. Temporal State

The temporal engine should produce something similar to:

``` python
TemporalState(
    label="wet",
    confidence=0.82,
    stability=0.91,
    candidate_label=None,
    candidate_streak=0,
    ambiguous=False,
)
```

Keep raw fused scores and temporal state separate.

------------------------------------------------------------------------

# 21. Frame Capture

Use OpenCV for local camera/video capture.

Do not make camera capture wait for inference.

Bad:

``` text
capture
  ↓
inference
  ↓
capture
  ↓
inference
```

Correct:

``` text
Camera
  ↓
Capture Worker
  ↓
Frame Queue
  ↓
Inference Worker
```

------------------------------------------------------------------------

# 22. Bounded Frame Queue

Use a small bounded queue.

Example:

``` python
Queue(maxsize=2)
```

or:

``` python
Queue(maxsize=3)
```

If inference is slower than camera capture:

``` text
drop stale frame
keep newest frame
```

Do not allow an unlimited queue.

Otherwise latency will continuously increase.

Track:

``` text
dropped_frames
```

------------------------------------------------------------------------

# 23. Frame Sampling

Do not necessarily run inference on every camera frame.

Make inference FPS configurable.

Example:

``` text
camera: 30 FPS
inference: 5 FPS
```

The goal is low latency rather than processing every frame.

Display:

``` text
camera FPS
inference FPS
```

------------------------------------------------------------------------

# 24. Prediction History

Keep prediction history separate from the raw frame queue.

``` text
Camera
  ↓
Frame Queue
  ↓
Inference
  ↓
Prediction History
  ↓
Temporal Engine
```

Dropping a stale frame must not corrupt temporal history.

The history itself should be bounded.

------------------------------------------------------------------------

# 25. Recording

Support recording in DEV mode.

Controls:

``` text
Start Recording
Stop Recording
```

Recording must not block inference.

Use a separate video-writing path if necessary.

The recording system is only for the application's live/video
functionality.

------------------------------------------------------------------------

# 26. Streamlit Frontend

Use Streamlit as the frontend.

Main sections:

``` text
WARD
Mode
Input
Live/Uploaded Image
Prediction
Class Scores
Model Comparison
Temporal Status
Weather
Graphs
```

------------------------------------------------------------------------

# 27. Input Controls

Support:

``` text
Upload Image
Upload Video
Camera
```

For camera:

``` text
Start
Stop
```

For recording:

``` text
Start Recording
Stop Recording
```

For session:

``` text
Reset
```

------------------------------------------------------------------------

# 28. Main Prediction UI

Display:

``` text
WARD Condition

WET

Confidence: 82%
Status: Stable
```

Then:

``` text
dry
damp
wet
drying
```

with current fused scores.

------------------------------------------------------------------------

# 29. Model Comparison UI

Show:

``` text
Original WARD
    wet

Fine-tuned WARD
    wet

Fusion
    wet

Agreement
    YES
```

If they disagree:

``` text
Original: dry
Fine-tuned: drying
Agreement: NO
```

------------------------------------------------------------------------

# 30. Temporal UI

Show:

``` text
Current State
Confidence
Stability
Candidate
Candidate Streak
```

Example:

``` text
State: WET
Confidence: 82%
Stability: High

Candidate: DRYING
Streak: 2
```

This makes the temporal behavior understandable.

------------------------------------------------------------------------

# 31. Graphs

Add live graphs for:

### Class probabilities

``` text
dry
damp
wet
drying
```

over time.

### Confidence

``` text
final confidence
```

over time.

### Stability

``` text
temporal stability
```

over time.

### Model comparison

Allow developer mode to compare:

``` text
Original
Fine-tuned
Fusion
```

------------------------------------------------------------------------

# 32. Developer Metrics

Add a developer/debug section.

Show:

``` text
Camera FPS
Inference FPS

Frame queue size
Dropped frames

Original inference latency
Fine inference latency
Fusion latency
Temporal latency
Total latency
```

Also show:

``` text
Original label
Fine label
Final label
Agreement
Ambiguity
```

------------------------------------------------------------------------

# 33. Weather Location

Allow the user to provide:

``` text
Location name
```

or:

``` text
Latitude
Longitude
```

Location is optional.

If no location is supplied:

``` text
Weather context disabled
```

Vision inference must continue normally.

------------------------------------------------------------------------

# 34. Open-Meteo Integration

Create:

``` text
weather/
    client.py
    cache.py
```

Use Open-Meteo for current weather.

Retrieve useful current values such as:

``` text
temperature
humidity
precipitation
rain
showers
snowfall
weather code
cloud cover
wind speed
```

Do not request weather on every frame.

------------------------------------------------------------------------

# 35. Weather Cache

Use a configurable refresh interval.

Initial:

``` text
300 seconds
```

If the weather request fails:

``` text
use last successful result
```

If no previous result exists:

``` text
weather unavailable
```

Weather failure must never stop model inference.

------------------------------------------------------------------------

# 36. Weather UI

Show:

``` text
Location
Temperature
Humidity
Rain
Precipitation
Weather
Last Updated
```

Example:

``` text
Dehradun

24°C
82% humidity
1.2 mm rain
Light rain

Updated 2 min ago
```

------------------------------------------------------------------------

# 37. Weather and Prediction

Weather must not blindly override the vision result.

Do not implement:

``` text
rain = wet
```

The road may be dry despite rain.

Do not implement:

``` text
no rain = dry
```

The road may remain wet after previous rainfall.

Initially weather should be contextual information.

If a later validation experiment proves that a small weather adjustment
improves classification, keep that adjustment behind a configurable
option.

------------------------------------------------------------------------

# 38. Production Networking

PROD mode should handle:

``` text
endpoint latency
timeout
retry
connection failure
invalid response
```

Do not let slow network inference create an unlimited backlog.

For live production inference, prioritize the newest frame.

------------------------------------------------------------------------

# 39. Environment Configuration

Use:

``` text
.env
```

for local development and the deployment platform's secret mechanism for
production.

Example:

``` text
WARD_MODE=dev

HF_TOKEN=
HF_ORIGINAL_ENDPOINT=
HF_FINE_ENDPOINT=
```

Never commit credentials.

------------------------------------------------------------------------

# 40. Project Structure

``` text
ward/
│
├── app.py
├── config.py
├── requirements.txt
├── .env.example
│
├── models/
│   ├── loader.py
│   ├── labels.py
│   └── fusion.py
│
├── inference/
│   ├── provider.py
│   ├── local_provider.py
│   └── hf_provider.py
│
├── temporal/
│   ├── engine.py
│   ├── smoothing.py
│   └── history.py
│
├── video/
│   ├── capture.py
│   ├── frame_queue.py
│   └── recorder.py
│
├── weather/
│   ├── client.py
│   └── cache.py
│
└── ui/
    ├── dashboard.py
    └── charts.py
```

Keep the architecture modular.

Do not put all inference, temporal logic, camera handling, weather
handling, and UI logic inside `app.py`.

------------------------------------------------------------------------

# 41. Configuration

Centralize settings such as:

``` text
WARD_MODE
INFERENCE_FPS
FRAME_QUEUE_SIZE
PREDICTION_HISTORY_SIZE

ORIGINAL_WEIGHT
FINE_WEIGHT

EMA_ALPHA
SWITCH_MARGIN
MIN_CANDIDATE_STREAK
MIN_STATE_DWELL_SECONDS
STRONG_OVERRIDE_THRESHOLD

AMBIGUITY_THRESHOLD

WEATHER_REFRESH_SECONDS
```

Avoid scattered magic numbers.

------------------------------------------------------------------------

# 42. Streamlit Lifecycle

Streamlit reruns the application frequently.

Therefore:

-   cache model loading,
-   do not reload models on every UI interaction,
-   do not repeatedly open the camera,
-   use session state for UI state,
-   explicitly start/stop camera workers,
-   release resources when stopping.

------------------------------------------------------------------------

# 43. Resource Cleanup

When stopping camera/recording:

``` text
stop capture
stop worker
flush/clear queue
release camera
release video writer
```

Avoid orphan threads.

------------------------------------------------------------------------

# 44. Error States

The UI should gracefully show:

``` text
Model loading
Inference unavailable
Camera unavailable
Endpoint unavailable
Weather unavailable
Low confidence
```

The application should continue operating wherever possible.

For example:

``` text
Weather unavailable
```

must not become:

``` text
Application unavailable
```

------------------------------------------------------------------------

# 45. Testing

Create tests for:

## Fusion

-   30/70 fusion
-   score normalization
-   damp/drying handling
-   missing model class handling

## Temporal

-   smoothing
-   candidate streak
-   hysteresis
-   one-frame spike
-   strong override
-   ambiguity
-   state transitions

## Queue

-   bounded size
-   stale frame replacement
-   dropped-frame counter

## Weather

-   successful response
-   cache
-   timeout
-   unavailable weather

## Provider

-   local inference
-   HF inference
-   endpoint failure

------------------------------------------------------------------------

# 46. Evaluation

Create an evaluation module for comparing the current inference
strategies.

Compare:

``` text
Original WARD
Fine-tuned WARD
Static 30/70 Fusion
Confidence-aware Fusion
Temporal Fusion
```

Metrics:

``` text
accuracy
macro precision
macro recall
macro F1
per-class F1
confusion matrix
```

Use the validation set to tune fusion and temporal parameters.

Keep the final test set separate.

------------------------------------------------------------------------

# 47. Development Order

Implement in this exact order:

## Phase 1

Create project structure and configuration.

## Phase 2

Load original and fine-tuned models locally.

## Phase 3

Implement dynamic label mapping.

## Phase 4

Implement independent model prediction.

## Phase 5

Implement 30/70 fusion.

## Phase 6

Implement `PredictionResult`.

## Phase 7

Implement temporal engine.

## Phase 8

Implement local camera and bounded queue.

## Phase 9

Build Streamlit dashboard.

## Phase 10

Add live graphs and developer metrics.

## Phase 11

Add recording.

## Phase 12

Add Open-Meteo and weather cache.

## Phase 13

Add Hugging Face production provider.

## Phase 14

Add DEV/PROD switching.

## Phase 15

Run evaluation and tune configuration.

## Phase 16

Add tests and fix performance/resource issues.

After every phase, keep the application runnable.

------------------------------------------------------------------------

# 48. Definition of Done

-   [ ] Two WARD models load successfully.
-   [ ] Models use independent processors.
-   [ ] Label mappings are dynamic.
-   [ ] Fine-tuned model has priority.
-   [ ] 30/70 fusion works.
-   [ ] Fusion scores are normalized.
-   [ ] Model disagreement is tracked.
-   [ ] Temporal smoothing works.
-   [ ] Candidate streak works.
-   [ ] Hysteresis works.
-   [ ] One-frame spikes are suppressed.
-   [ ] Strong repeated predictions can change the stable state.
-   [ ] Ambiguity is detected.
-   [ ] Frame queue is bounded.
-   [ ] Stale frames are dropped.
-   [ ] Dropped frames are counted.
-   [ ] Camera capture does not wait for inference.
-   [ ] DEV mode uses local GPU.
-   [ ] PROD mode uses HF endpoint(s).
-   [ ] Same application logic works in both modes.
-   [ ] Streamlit dashboard works.
-   [ ] Live graphs work.
-   [ ] Developer metrics work.
-   [ ] Recording works.
-   [ ] Weather location can be configured.
-   [ ] Open-Meteo data is cached.
-   [ ] Weather failure does not break inference.
-   [ ] Endpoint failure does not crash the application.
-   [ ] Models are not reloaded on every Streamlit rerun.
-   [ ] Resources are cleaned up correctly.
-   [ ] Evaluation compares the model strategies.
-   [ ] Tests cover the main pipeline components.
-   [ ] Secrets are not committed.

------------------------------------------------------------------------

# 49. Out of Scope

The coding agent must **not** add unrelated systems.

Do not implement:

-   future dataset generation
-   automatic retraining
-   human annotation workflows
-   multi-camera support
-   road segmentation
-   automatic model fine-tuning
-   new ML architectures
-   user authentication
-   database systems unless specifically required
-   cloud storage unless specifically required
-   mobile application
-   autonomous driving decisions
-   safety certification
-   unrelated analytics

Only implement what is required for the WARD application described
above.

------------------------------------------------------------------------

# 50. Final Pipeline

``` text
                INPUT
                  |
       +----------+----------+
       |                     |
     Image                Camera/Video
       |                     |
       +----------+----------+
                  |
            Frame Sampling
                  |
            Bounded Queue
                  |
          +-------+-------+
          |               |
    Original WARD     Fine WARD
          |               |
          +-------+-------+
                  |
             30/70 Fusion
                  |
          Confidence Analysis
                  |
           Temporal Engine
                  |
        +---------+---------+
        |                   |
   Weather Context      Final State
        |                   |
        +---------+---------+
                  |
             Streamlit UI
                  |
       Prediction + Graphs
       + Metrics + Recording
```

The final system should prioritize the fine-tuned WARD model initially,
use the original model as supporting evidence for `dry`/`wet`, and use
temporal consistency to prevent unstable frame-by-frame classification.
