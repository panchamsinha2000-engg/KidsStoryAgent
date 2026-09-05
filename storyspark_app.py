import os
import io
import json
import wave
from dataclasses import dataclass, field, asdict
from typing import List

import streamlit as st
from google import genai
from google.genai import types


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="StorySpark — Kids Story Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }
    .story-card {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.25);
        background: rgba(128,128,128,.06);
        line-height: 1.75;
    }
    .scene-card {
        padding: 1rem 1.2rem;
        border-left: 5px solid #888;
        border-radius: 10px;
        margin: .7rem 0;
        background: rgba(128,128,128,.06);
    }
    .small-note {
        font-size: .88rem;
        opacity: .7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Data structures — retained from the notebook
# ------------------------------------------------------------
@dataclass
class StorySpec:
    theme: str
    characters: List[str]
    age_group: str
    moral: str
    tone: str


@dataclass
class Scene:
    id: str
    title: str
    note: str
    characters_present: List[str]


@dataclass
class Paragraph:
    scene_id: str
    text: str = ""
    flags: List[str] = field(default_factory=list)
    score: int = 0
    approved: bool = False
    reading_notes: List[dict] = field(default_factory=list)


# ------------------------------------------------------------
# Gemini client
# ------------------------------------------------------------
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error(
        "GEMINI_API_KEY is not configured. "
        "Set it as an environment variable before starting the app."
    )
    st.code(
        'Windows PowerShell:  $env:GEMINI_API_KEY="YOUR_KEY_HERE"\n'
        'Windows CMD:         set GEMINI_API_KEY=YOUR_KEY_HERE'
    )
    st.stop()

client = genai.Client(api_key=api_key)

# Same model choices as the supplied notebook.
MODEL_CHEAP = os.getenv("STORY_MODEL_CHEAP", "gemini-3.5-flash-lite")
MODEL_QUALITY = os.getenv("STORY_MODEL_QUALITY", "gemini-3.7-flash")
TTS_MODEL = os.getenv("STORY_TTS_MODEL", "gemini-3.1-flash-tts-preview")

# A handful of prebuilt Gemini voices well suited to reading a story aloud.
VOICE_OPTIONS = {
    "Kore — calm & firm": "Kore",
    "Puck — upbeat": "Puck",
    "Leda — youthful": "Leda",
    "Aoede — breezy": "Aoede",
    "Zephyr — bright": "Zephyr",
}


def call_json(model: str, system: str, user: str, max_tokens: int = 700) -> dict:
    """Call Gemini and parse its JSON response."""
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.7,
            response_mime_type="application/json",
        ),
    )
    return json.loads(resp.text)


def _pcm_to_wav_bytes(pcm_data: bytes, channels=1, rate=24000, sample_width=2) -> bytes:
    """Gemini TTS returns raw 16-bit PCM. Wrap it in a WAV header in memory
    so st.audio (and any browser) can play it without writing a file to disk."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def synthesize_speech(text: str, voice_name: str, tone: str, age_group: str) -> bytes:
    """Read-aloud agent: turns the final story text into narrated audio.
    Kept separate from the 5 text/JSON agents above -- TTS is audio-out,
    not text-out, so it doesn't fit the same call_json() shape."""
    style_prompt = (
        f"Read the following children's story aloud in a {tone.lower()} voice, "
        f"suitable for a {age_group} year-old audience. Pace it warmly, and pause "
        f"gently between paragraphs.\n\n{text}"
    )
    resp = client.models.generate_content(
        model=TTS_MODEL,
        contents=style_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
        ),
    )
    pcm_data = resp.candidates[0].content.parts[0].inline_data.data
    return _pcm_to_wav_bytes(pcm_data)


# ------------------------------------------------------------
# The 5-agent pipeline — same logic as the notebook
# ------------------------------------------------------------
def structure_creator(spec: StorySpec) -> List[Scene]:
    system = (
        "You are the Structure Creator agent for a children's story pipeline. "
        "Return ONLY a JSON array, no prose, no markdown fences. "
        "Create exactly 5 scenes following this arc: introduction, discovery, "
        "challenge, turning point (where the moral surfaces), resolution. "
        "Each scene object: {id, title, note, characters_present}. "
        "characters_present must be a subset of the given characters, never invented names."
    )
    user = json.dumps(asdict(spec))
    data = call_json(MODEL_CHEAP, system, user, max_tokens=500)
    return [Scene(**s) for s in data]


def generator(scenes: List[Scene], spec: StorySpec) -> List[Paragraph]:
    system = (
        "You are the Generator agent. Write one paragraph (100-150 words) per scene "
        "for a children's story. Rules: use ONLY the characters listed in that scene's "
        "characters_present -- never introduce a new named character. Stay consistent "
        "with earlier scenes. Match tone and age group. Return ONLY a JSON array of "
        "{scene_id, text}, no prose, no markdown fences."
    )
    user = json.dumps(
        {"spec": asdict(spec), "scenes": [asdict(s) for s in scenes]}
    )
    data = call_json(MODEL_QUALITY, system, user, max_tokens=900)
    return [
        Paragraph(scene_id=d["scene_id"], text=d["text"])
        for d in data
    ]


def reviewer(paragraphs: List[Paragraph], spec: StorySpec) -> List[Paragraph]:
    system = (
        "You are the Reviewer agent. For each paragraph, check: "
        f"(1) vocabulary fits age group {spec.age_group}, "
        "(2) no character or setting contradicts earlier paragraphs, "
        "(3) tone matches the brief. Return ONLY a JSON array of "
        "{scene_id, flags: string[], score: 0-10, approved: bool}. "
        "approved should be true only if score >= 7. No prose, no markdown fences."
    )
    user = json.dumps(
        {"spec": asdict(spec), "paragraphs": [asdict(p) for p in paragraphs]}
    )
    data = call_json(MODEL_CHEAP, system, user, max_tokens=600)
    by_id = {d["scene_id"]: d for d in data}

    for p in paragraphs:
        d = by_id.get(p.scene_id, {})
        p.flags = d.get("flags", [])
        p.score = d.get("score", 0)
        p.approved = d.get("approved", False)

    return paragraphs


def reading_coach(paragraphs: List[Paragraph]) -> List[Paragraph]:
    system = (
        "You are the Reading Coach agent. For each paragraph, add read-aloud notes: "
        "voice changes, pacing, words to emphasize, where to pause. Return ONLY a JSON "
        "array of {scene_id, reading_notes: [{type, detail}]}. "
        "1-3 notes per paragraph. No prose, no markdown fences."
    )
    user = json.dumps({"paragraphs": [asdict(p) for p in paragraphs]})
    data = call_json(MODEL_CHEAP, system, user, max_tokens=600)
    by_id = {d["scene_id"]: d for d in data}

    for p in paragraphs:
        p.reading_notes = by_id.get(p.scene_id, {}).get("reading_notes", [])

    return paragraphs


def stitcher(paragraphs: List[Paragraph]) -> dict:
    approved = [p for p in paragraphs if p.approved]
    blocked = [p for p in paragraphs if not p.approved]

    return {
        "story": "\n\n".join(p.text for p in approved),
        "paragraphs": [asdict(p) for p in approved],
        "limitations": [
            f"{p.scene_id}: {', '.join(p.flags)}"
            for p in blocked
        ],
    }


def run_pipeline(spec: StorySpec) -> dict:
    scenes = structure_creator(spec)
    paragraphs = generator(scenes, spec)
    paragraphs = reviewer(paragraphs, spec)
    paragraphs = reading_coach(paragraphs)
    return stitcher(paragraphs)


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown('<div class="main-title">📚 StorySpark</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    "A quick, fun children's story creator powered by a 5-agent Gemini pipeline."
    "</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Story settings")

    age_group = st.selectbox(
        "Age group",
        ["3-5", "6-8", "9-12"],
        index=1,
    )

    tone = st.selectbox(
        "Tone",
        ["Playful", "Gentle", "Adventurous", "Silly", "Calming"],
        index=1,
    )

    st.divider()
    st.caption("Pipeline")
    st.caption("1️⃣ Structure Creator")
    st.caption("2️⃣ Generator")
    st.caption("3️⃣ Reviewer")
    st.caption("4️⃣ Reading Coach")
    st.caption("5️⃣ Stitcher")
    st.caption("🔊 Read Aloud (optional, after the story is ready)")

    st.divider()
    st.caption(f"Story model: `{MODEL_QUALITY}`")
    st.caption(f"Support model: `{MODEL_CHEAP}`")
    st.caption(f"Narrator model: `{TTS_MODEL}`")


col1, col2 = st.columns(2)

with col1:
    theme = st.text_area(
        "🌟 Story theme",
        value="Two great archers learn about friendship, respect, and choosing kindness even during rivalry",
        height=110,
        placeholder="Example: Two young explorers discover a hidden garden...",
    )

    characters_text = st.text_input(
        "🧑‍🤝‍🧑 Characters",
        value="Arjuna, Karna",
        help="Enter names separated by commas.",
    )

with col2:
    moral = st.text_area(
        "💡 Moral / lesson",
        value="True greatness comes from humility, respect, and doing what is right",
        height=110,
        placeholder="Example: Helping others is more important than winning.",
    )

    st.markdown(
        '<div class="small-note">Tip: Keep the moral specific. '
        "The pipeline uses it at the turning point.</div>",
        unsafe_allow_html=True,
    )

generate = st.button(
    "✨ Generate My Story",
    type="primary",
    use_container_width=True,
)

if generate:
    characters = [
        c.strip()
        for c in characters_text.split(",")
        if c.strip()
    ]

    if not theme.strip() or not characters or not moral.strip():
        st.warning("Please fill in the story theme, characters, and moral.")
        st.stop()

    spec = StorySpec(
        theme=theme.strip(),
        characters=characters,
        age_group=age_group,
        moral=moral.strip(),
        tone=tone,
    )

    progress = st.progress(0, text="Starting the story pipeline...")
    status = st.empty()

    try:
        status.info("🏗️ Structure Creator is designing 5 scenes...")
        progress.progress(20)
        scenes = structure_creator(spec)

        status.info("✍️ Generator is writing the story...")
        progress.progress(40)
        paragraphs = generator(scenes, spec)

        status.info("🔎 Reviewer is checking consistency and age suitability...")
        progress.progress(60)
        paragraphs = reviewer(paragraphs, spec)

        status.info("🎙️ Reading Coach is adding read-aloud guidance...")
        progress.progress(80)
        paragraphs = reading_coach(paragraphs)

        status.info("🧵 Stitcher is assembling the final story...")
        progress.progress(100)
        result = stitcher(paragraphs)

        status.success("Story generated successfully! 🎉")

        # Store for download / rerender
        st.session_state["story_result"] = result
        st.session_state["story_spec"] = spec
        st.session_state["story_scenes"] = scenes

    except Exception as exc:
        progress.empty()
        status.empty()
        st.error("Something went wrong while generating the story.")
        with st.expander("Technical error"):
            st.exception(exc)
        st.stop()


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
if "story_result" in st.session_state:
    result = st.session_state["story_result"]
    scenes = st.session_state.get("story_scenes", [])

    st.divider()
    st.subheader("📖 Your Story")

    if result["story"]:
        st.markdown(
            f'<div class="story-card">{result["story"].replace(chr(10), "<br><br>")}</div>',
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇️ Save story as TXT",
            data=result["story"],
            file_name="my_kids_story.txt",
            mime="text/plain",
        )

        st.divider()
        st.subheader("🔊 Read Aloud")

        spec = st.session_state.get("story_spec")
        rc1, rc2 = st.columns([2, 1])
        with rc1:
            voice_choice = st.selectbox(
                "Narrator voice",
                list(VOICE_OPTIONS.keys()),
                key="voice_choice",
            )
        with rc2:
            st.write("")
            st.write("")
            narrate = st.button("🎙️ Narrate this story", use_container_width=True)

        if narrate:
            with st.spinner("Recording the narration..."):
                try:
                    audio_bytes = synthesize_speech(
                        text=result["story"],
                        voice_name=VOICE_OPTIONS[voice_choice],
                        tone=spec.tone,
                        age_group=spec.age_group,
                    )
                    st.session_state["story_audio"] = audio_bytes
                except Exception as exc:
                    st.error("Couldn't generate narration.")
                    with st.expander("Technical error"):
                        st.exception(exc)

        if "story_audio" in st.session_state:
            st.audio(st.session_state["story_audio"], format="audio/wav")
            st.download_button(
                "⬇️ Save narration as WAV",
                data=st.session_state["story_audio"],
                file_name="my_kids_story.wav",
                mime="audio/wav",
            )
    else:
        st.warning(
            "No paragraphs were approved by the Reviewer, so the Stitcher "
            "did not assemble a final story."
        )

    st.divider()
    st.subheader("🎙️ Reading-aloud coach")

    for p in result["paragraphs"]:
        with st.expander(f"{p['scene_id']}"):
            for note in p.get("reading_notes", []):
                st.markdown(
                    f"**{note.get('type', 'Note')}** — {note.get('detail', '')}"
                )

    if result["limitations"]:
        st.divider()
        st.subheader("⚠️ Review limitations")
        for limitation in result["limitations"]:
            st.warning(limitation)

    with st.expander("🗺️ Story structure"):
        for scene in scenes:
            st.markdown(
                f"""
                <div class="scene-card">
                <strong>{scene.title}</strong><br>
                {scene.note}<br>
                <small>Characters: {", ".join(scene.characters_present)}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("🔧 Debug / pipeline output"):
        st.json(result)