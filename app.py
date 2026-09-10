import streamlit as st
import tempfile
import os
import subprocess
import threading
import queue
import time

import numpy as np
import soundfile as sf
from transformers import pipeline

import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="VoiceGuard | AI Voice Detector",
    page_icon="🛡️",
    layout="wide"
)

MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"

TARGET_SR = 16000
MAX_DURATION = 30

CHUNK_SECONDS = 3
OVERLAP_SECONDS = 1.5

LIVE_CHUNK_SECONDS = 3
LIVE_STEP_SECONDS = 1.5


# =========================================================
# PROFESSIONAL UI
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0b1020;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        rgba(40,50,100,0.9),
        rgba(15,20,45,0.95)
    );
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    opacity: 0.8;
}

.result-box {
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    margin-top: 15px;
}

.metric-card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}

.warning {
    padding: 15px;
    border-radius: 12px;
    background: rgba(255,180,0,0.08);
    border: 1px solid rgba(255,180,0,0.25);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<h1>🛡️ VoiceGuard</h1>

<p>
AI-Powered Voice Deepfake & Spoof Detection System
</p>

<p>
Analyze uploaded audio, record your voice, or perform near real-time detection.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_detector():

    detector = pipeline(
        "audio-classification",
        model=MODEL_ID
    )

    return detector


# =========================================================
# SCORE EXTRACTION
# =========================================================

def get_ai_score(predictions):

    ai_score = 0.0

    for item in predictions:

        label = str(item["label"]).lower()
        score = float(item["score"])

        if any(word in label for word in [
            "fake",
            "spoof",
            "deepfake",
            "synthetic",
            "generated",
            "artificial",
            "ai"
        ]):

            ai_score = max(ai_score, score)

    return ai_score


# =========================================================
# AUDIO CONVERSION
# =========================================================

def convert_to_wav(input_path):

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )

    output_path = output_file.name

    output_file.close()

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-vn",
        output_path
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    return output_path


# =========================================================
# AUDIO ANALYSIS
# =========================================================

def analyze_audio(audio_bytes, suffix=".wav"):

    detector = load_detector()

    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    input_file.write(audio_bytes)
    input_file.close()

    wav_path = None

    try:

        wav_path = convert_to_wav(input_file.name)

        audio, sr = sf.read(wav_path)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32)

        duration = len(audio) / sr

        if duration > MAX_DURATION:

            audio = audio[:MAX_DURATION * sr]
            duration = MAX_DURATION

        chunk_size = int(CHUNK_SECONDS * sr)
        step_size = int(
            (CHUNK_SECONDS - OVERLAP_SECONDS) * sr
        )

        scores = []

        start = 0

        while start < len(audio):

            end = start + chunk_size

            chunk = audio[start:end]

            if len(chunk) < int(0.7 * chunk_size):
                break

            predictions = detector(
                {
                    "raw": chunk,
                    "sampling_rate": sr
                }
            )

            score = get_ai_score(predictions)

            scores.append(score)

            start += step_size

        if not scores:

            raise ValueError("Could not extract usable audio chunks.")

        scores = np.array(scores)

        max_score = float(np.max(scores))
        mean_score = float(np.mean(scores))

        top_count = max(1, int(np.ceil(len(scores) * 0.30)))

        top_mean = float(
            np.mean(
                np.sort(scores)[-top_count:]
            )
        )

        final_score = (
            0.55 * max_score
            + 0.30 * top_mean
            + 0.15 * mean_score
        )

        final_score = float(
            np.clip(final_score, 0, 1)
        )

        return {
            "ai_score": final_score,
            "max_score": max_score,
            "top_mean": top_mean,
            "mean_score": mean_score,
            "chunks": scores.tolist(),
            "duration": duration
        }

    finally:

        try:
            os.remove(input_file.name)
        except:
            pass

        if wav_path:

            try:
                os.remove(wav_path)
            except:
                pass


# =========================================================
# VERDICT
# =========================================================

def get_verdict(score):

    if score >= 0.85:

        return (
            "AI GENERATED / SPOOF",
            "HIGH"
        )

    elif score >= 0.50:

        return (
            "SUSPICIOUS / POSSIBLY AI",
            "MEDIUM"
        )

    else:

        return (
            "LIKELY HUMAN",
            "LOW"
        )


# =========================================================
# EXPLANATION
# =========================================================

def explain_result(result):

    score = result["ai_score"]
    max_score = result["max_score"]
    mean_score = result["mean_score"]
    chunks = result["chunks"]

    if score >= 0.85:

        reason = (
            "Multiple audio segments contain strong characteristics "
            "associated with spoofed or synthetic speech."
        )

    elif score >= 0.50:

        reason = (
            "Some audio segments show suspicious characteristics, "
            "but the evidence is not strong enough for a high-confidence "
            "AI-generated classification."
        )

    else:

        reason = (
            "Most analyzed segments were classified closer to the "
            "human/bonafide side by the detection model."
        )

    return reason


# =========================================================
# RESULT DISPLAY
# =========================================================

def show_result(result):

    score = result["ai_score"]

    verdict, risk = get_verdict(score)

    confidence = max(score, 1 - score)

    st.markdown("---")

    st.subheader("🔍 Detection Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

            <h4>Verdict</h4>

            <h2>{verdict}</h2>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

            <h4>Risk Level</h4>

            <h2>{risk}</h2>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">

            <h4>Detection Confidence</h4>

            <h2>{confidence * 100:.1f}%</h2>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🤖 AI Probability")

    st.progress(
        min(max(score, 0.0), 1.0)
    )

    st.write(
        f"**Overall AI/Spoof Score:** {score * 100:.1f}%"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Maximum AI Score",
            f"{result['max_score'] * 100:.1f}%"
        )

    with col2:
        st.metric(
            "Top-30% Mean",
            f"{result['top_mean'] * 100:.1f}%"
        )

    with col3:
        st.metric(
            "Average Score",
            f"{result['mean_score'] * 100:.1f}%"
        )

    st.markdown("### 🧠 Why this result?")

    st.info(
        explain_result(result)
    )

    st.markdown("### 📊 Chunk-Level Analysis")

    for i, chunk_score in enumerate(
        result["chunks"],
        start=1
    ):

        st.write(
            f"Chunk {i}: **{chunk_score * 100:.1f}% AI score**"
        )

        st.progress(
            min(max(chunk_score, 0), 1)
        )

    st.markdown(
        f"""
        <div class="warning">

        ⚠️ <b>Important:</b>
        This system provides a machine-learning prediction.
        It should not be treated as absolute proof that a voice is
        AI-generated or human.

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# LIVE AUDIO PROCESSOR
# =========================================================

class LiveAudioProcessor(AudioProcessorBase):

    def __init__(self):

        self.resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=TARGET_SR
        )

        self.buffer = np.array(
            [],
            dtype=np.float32
        )

        self.last_analysis = 0

        self.processing = False

        self.results = queue.Queue()

        self.lock = threading.Lock()

    def recv(self, frame):

        try:

            frames = self.resampler.resample(frame)

            if not isinstance(frames, list):
                frames = [frames]

            for f in frames:

                arr = f.to_ndarray()

                if arr.ndim > 1:
                    arr = arr[0]

                arr = arr.astype(
                    np.float32
                ) / 32768.0

                with self.lock:

                    self.buffer = np.concatenate(
                        [self.buffer, arr]
                    )

                    max_buffer = int(
                        LIVE_CHUNK_SECONDS * TARGET_SR
                    )

                    if len(self.buffer) > max_buffer:

                        self.buffer = self.buffer[
                            -max_buffer:
                        ]

            current_time = time.time()

            enough_audio = (
                len(self.buffer)
                >= LIVE_CHUNK_SECONDS * TARGET_SR
            )

            enough_time = (
                current_time - self.last_analysis
                >= LIVE_STEP_SECONDS
            )

            if (
                enough_audio
                and enough_time
                and not self.processing
            ):

                with self.lock:

                    chunk = self.buffer.copy()

                    overlap_samples = int(
                        (LIVE_CHUNK_SECONDS - LIVE_STEP_SECONDS)
                        * TARGET_SR
                    )

                    if overlap_samples > 0:

                        self.buffer = self.buffer[
                            -overlap_samples:
                        ]

                    else:

                        self.buffer = np.array(
                            [],
                            dtype=np.float32
                        )

                self.processing = True

                self.last_analysis = current_time

                thread = threading.Thread(
                    target=self.analyze_chunk,
                    args=(chunk,),
                    daemon=True
                )

                thread.start()

        except Exception:
            pass

        return frame

    def analyze_chunk(self, chunk):

        try:

            detector = load_detector()

            predictions = detector(
                {
                    "raw": chunk,
                    "sampling_rate": TARGET_SR
                }
            )

            score = get_ai_score(predictions)

            self.results.put(score)

        except Exception as e:

            self.results.put(
                None
            )

        finally:

            self.processing = False


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛡️ VoiceGuard")

st.sidebar.markdown(
    """
### Detection Modes

Choose how you want to analyze the voice.

**Upload / Record**
- Analyze saved audio
- Upload MP3/WAV
- Record microphone audio
- Detailed chunk analysis

**Real-Time Detection**
- Microphone stream
- Rolling 3-second analysis
- Live AI probability
- Live risk classification
"""
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Model: Gustking Wav2Vec2 XLS-R Deepfake Audio Classification"
)


# =========================================================
# MODE SELECTION
# =========================================================

mode = st.radio(
    "Select Detection Mode",
    [
        "📁 Upload / Record",
        "🎙️ Real-Time Detection"
    ],
    horizontal=True
)


# =========================================================
# UPLOAD / RECORD MODE
# =========================================================

if mode == "📁 Upload / Record":

    st.subheader("🎧 Analyze an Audio File")

    uploaded_file = st.file_uploader(
        "Upload voice audio",
        type=[
            "wav",
            "mp3",
            "mpeg",
            "ogg",
            "m4a"
        ]
    )

    st.markdown("### 🎙️ Or record directly")

    recorded_audio = None

    if hasattr(st, "audio_input"):

        recorded_audio = st.audio_input(
            "Record your voice"
        )

    if uploaded_file is not None:

        st.audio(
            uploaded_file
        )

        if st.button(
            "🔎 Analyze Uploaded Audio",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing voice..."
            ):

                try:

                    result = analyze_audio(
                        uploaded_file.getvalue(),
                        suffix=os.path.splitext(
                            uploaded_file.name
                        )[1]
                    )

                    show_result(result)

                except Exception as e:

                    st.error(
                        f"Analysis failed: {e}"
                    )

    elif recorded_audio is not None:

        st.audio(
            recorded_audio
        )

        if st.button(
            "🔎 Analyze Recorded Voice",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing recorded voice..."
            ):

                try:

                    result = analyze_audio(
                        recorded_audio.getvalue(),
                        suffix=".wav"
                    )

                    show_result(result)

                except Exception as e:

                    st.error(
                        f"Analysis failed: {e}"
                    )


# =========================================================
# REAL-TIME MODE
# =========================================================

else:

    st.subheader("🎙️ Real-Time Voice Detection")

    st.info(
        "Start the microphone. VoiceGuard analyzes rolling "
        "3-second audio windows approximately every 1.5 seconds."
    )

    ctx = webrtc_streamer(
        key="voiceguard-live",

        mode=WebRtcMode.SENDONLY,

        audio_processor_factory=LiveAudioProcessor,

        media_stream_constraints={
            "audio": True,
            "video": False
        },

        async_processing=True
    )

    if ctx.state.playing:

        st.success(
            "🟢 Live microphone active"
        )

        score_placeholder = st.empty()

        verdict_placeholder = st.empty()

        risk_placeholder = st.empty()

        chart_placeholder = st.empty()

        live_scores = []

        while ctx.state.playing:

            if ctx.audio_processor:

                try:

                    while True:

                        score = (
                            ctx.audio_processor.results.get_nowait()
                        )

                        if score is None:
                            continue

                        live_scores.append(
                            float(score)
                        )

                except queue.Empty:

                    pass

            if live_scores:

                current_score = (
                    live_scores[-1]
                )

                if current_score >= 0.85:

                    verdict = (
                        "🔴 AI GENERATED / SPOOF"
                    )

                    risk = "HIGH"

                elif current_score >= 0.50:

                    verdict = (
                        "🟠 SUSPICIOUS / POSSIBLY AI"
                    )

                    risk = "MEDIUM"

                else:

                    verdict = (
                        "🟢 LIKELY HUMAN"
                    )

                    risk = "LOW"

                score_placeholder.metric(
                    "Live AI Probability",
                    f"{current_score * 100:.1f}%"
                )

                verdict_placeholder.markdown(
                    f"## {verdict}"
                )

                risk_placeholder.write(
                    f"**Risk Level:** {risk}"
                )

                recent_scores = (
                    live_scores[-20:]
                )

                chart_placeholder.line_chart(
                    recent_scores
                )

            time.sleep(0.2)

    else:

        st.warning(
            "Click START above to begin live detection."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "VoiceGuard • AI Voice Clone & Spoof Detection • "
    "Hackathon Prototype"
)
