"""
✨ AI Timetable Generator
A Streamlit app that turns a plain-English prompt into a beautifully
formatted, downloadable timetable using Groq's free LLM API.

Run locally:
    pip install -r requirements.txt
    export GROQ_API_KEY="your_key_here"      # (Linux/Mac)
    setx GROQ_API_KEY "your_key_here"        # (Windows)
    streamlit run app.py
"""

import os
import io
import json
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from groq import Groq


# ─────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Timetable Generator",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

MODEL_OPTIONS = {
    "Llama 3.3 70B (best quality)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (fastest)": "llama-3.1-8b-instant",
}

# A palette used to color-code distinct activities in the grid view
PALETTE = [
    "#6C5CE7", "#00B894", "#0984E3", "#E17055", "#FDCB6E",
    "#E84393", "#00CEC9", "#D63031", "#636E72", "#A29BFE",
    "#55EFC4", "#FAB1A0", "#74B9FF", "#FF7675", "#81ECEC",
]


# ─────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — "Beautiful UI"
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #1f1147 0%, #0f0c29 45%, #0b0b1a 100%);
    }

    /* Hero header */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a29bfe, #74b9ff, #55efc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .hero-subtitle {
        color: #c8c8e0;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        margin-bottom: 1.2rem;
    }

    /* Text areas / inputs */
    .stTextArea textarea, .stTextInput input {
        background: rgba(255,255,255,0.08) !important;
        color: #f5f5fa !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6c5ce7, #0984e3);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 14px rgba(108, 92, 231, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(108, 92, 231, 0.55);
        color: white;
    }

    .stDownloadButton>button {
        background: linear-gradient(90deg, #00b894, #00cec9);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1440, #100c2b);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Timetable grid table */
    .timetable-wrapper {
        overflow-x: auto;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.12);
    }
    table.timetable {
        border-collapse: collapse;
        width: 100%;
        min-width: 700px;
    }
    table.timetable th {
        background: rgba(255,255,255,0.08);
        color: #eaeaf5;
        padding: 10px 8px;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 2px solid rgba(255,255,255,0.15);
    }
    table.timetable td {
        padding: 6px;
        vertical-align: top;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .time-label {
        color: #9d9dc0;
        font-size: 0.78rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .slot-card {
        border-radius: 10px;
        padding: 8px 10px;
        color: white;
        font-size: 0.82rem;
        font-weight: 600;
        line-height: 1.25;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .slot-notes {
        display: block;
        font-weight: 400;
        opacity: 0.85;
        font-size: 0.74rem;
        margin-top: 2px;
    }

    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.1);
        color: #cfcfe8;
        font-size: 0.75rem;
        margin-right: 6px;
    }

    /* Style Streamlit's native bordered containers to match the glass look */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 18px !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        margin-bottom: 1.2rem;
    }

    footer, #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────
# SECURE API KEY HANDLING
# ─────────────────────────────────────────────────────────────────────────
def get_api_key() -> str | None:
    """
    Resolve the Groq API key with the following priority, without ever
    hardcoding, printing, or persisting the key to disk:
      1. Streamlit secrets (st.secrets["GROQ_API_KEY"]) — recommended for
         deployed apps (.streamlit/secrets.toml, never committed to git).
      2. Environment variable GROQ_API_KEY — recommended for local runs.
      3. A password-masked input box — kept only in st.session_state for
         the current browser session, cleared when the session ends.
    """
    key = None

    # 1. Streamlit secrets
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None

    # 2. Environment variable
    if not key:
        key = os.environ.get("GROQ_API_KEY")

    # 3. Session-only manual entry
    if not key:
        key = st.session_state.get("manual_api_key")

    return key


def api_key_sidebar_widget():
    st.sidebar.markdown("### 🔑 Groq API Key")
    existing = None
    try:
        existing = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
    existing = existing or os.environ.get("GROQ_API_KEY")

    if existing:
        st.sidebar.success("API key loaded securely from server config ✅")
    else:
        st.sidebar.info(
            "No key found in environment/secrets. Enter one below — it is "
            "kept only in memory for this session and never saved to disk "
            "or logged."
        )
        manual = st.sidebar.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_************************",
            key="manual_api_key_input",
            help="Get a free key at https://console.groq.com/keys",
        )
        if manual:
            st.session_state["manual_api_key"] = manual

    with st.sidebar.expander("🔒 How your key is protected"):
        st.markdown(
            "- Never hardcoded in source code\n"
            "- Never written to disk or logs\n"
            "- Read from `st.secrets` or env var when available\n"
            "- Manually entered keys live only in this session's memory\n"
            "- Field is masked (`type=\"password\"`)\n"
            "- For deployment, use `.streamlit/secrets.toml` "
            "(add it to `.gitignore`!) or your host's secret manager."
        )


# ─────────────────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert scheduling assistant. Convert the user's
description into a structured weekly timetable.

Respond with ONLY a valid JSON object — no markdown fences, no commentary —
matching exactly this schema:

{
  "title": "Short descriptive title for this timetable",
  "entries": [
    {
      "day": "Monday",
      "start_time": "09:00",
      "end_time": "10:00",
      "activity": "Short activity name",
      "notes": "Optional short extra detail, or empty string"
    }
  ]
}

Rules:
- "day" must be one of: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.
- Times must be 24-hour "HH:MM" strings.
- Cover every day the user mentions (default to Monday–Friday if unspecified).
- Avoid overlapping time slots on the same day.
- Keep activity names concise (a few words).
- Output must be valid, parseable JSON and nothing else.
"""


def extract_json(text: str) -> dict:
    """Pull a JSON object out of the model response, tolerating stray fences/text."""
    text = text.strip()
    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    # Fallback: grab the first {...} block
    if not text.startswith("{"):
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)
    return json.loads(text)


def generate_timetable(api_key: str, model: str, user_prompt: str) -> dict:
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=3000,
    )
    raw = completion.choices[0].message.content
    return extract_json(raw)


# ─────────────────────────────────────────────────────────────────────────
# RENDERING HELPERS
# ─────────────────────────────────────────────────────────────────────────
def to_dataframe(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data.get("entries", []))
    if df.empty:
        return df
    for col in ["day", "start_time", "end_time", "activity", "notes"]:
        if col not in df.columns:
            df[col] = ""
    df["day"] = pd.Categorical(df["day"], categories=DAY_ORDER, ordered=True)
    df = df.sort_values(["day", "start_time"]).reset_index(drop=True)
    return df[["day", "start_time", "end_time", "activity", "notes"]]


def activity_colors(activities):
    unique = list(dict.fromkeys(activities))
    return {act: PALETTE[i % len(PALETTE)] for i, act in enumerate(unique)}


def render_grid_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p style='color:#ccc;'>No entries to display.</p>"

    days_present = [d for d in DAY_ORDER if d in df["day"].astype(str).unique()]
    colors = activity_colors(df["activity"].tolist())

    # Build a sorted list of unique time slots across all days
    slots = sorted(set(zip(df["start_time"], df["end_time"])), key=lambda x: x[0])

    lookup = {}
    for _, row in df.iterrows():
        lookup[(str(row["day"]), row["start_time"], row["end_time"])] = row

    html = ["<div class='timetable-wrapper'><table class='timetable'>"]
    html.append("<tr><th>Time</th>" + "".join(f"<th>{d}</th>" for d in days_present) + "</tr>")

    for start, end in slots:
        html.append("<tr>")
        html.append(f"<td class='time-label'>{start}&ndash;{end}</td>")
        for day in days_present:
            row = lookup.get((day, start, end))
            if row is not None:
                color = colors.get(row["activity"], "#6c5ce7")
                notes = f"<span class='slot-notes'>{row['notes']}</span>" if row["notes"] else ""
                html.append(
                    f"<td><div class='slot-card' style='background:{color};'>"
                    f"{row['activity']}{notes}</div></td>"
                )
            else:
                html.append("<td></td>")
        html.append("</tr>")

    html.append("</table></div>")
    return "".join(html)


def build_downloadable_html(title: str, df: pd.DataFrame) -> str:
    grid = render_grid_html(df)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: 'Poppins', Arial, sans-serif; background:#0f0c29; color:#eee; padding:24px; }}
h1 {{ background: linear-gradient(90deg,#a29bfe,#74b9ff,#55efc4); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
table.timetable {{ border-collapse: collapse; width:100%; }}
table.timetable th {{ background:rgba(255,255,255,0.08); padding:10px; }}
table.timetable td {{ padding:6px; border-bottom:1px solid rgba(255,255,255,0.08); }}
.slot-card {{ border-radius:10px; padding:8px 10px; color:white; font-weight:600; font-size:0.85rem; }}
.slot-notes {{ display:block; font-weight:400; opacity:0.85; font-size:0.75rem; margin-top:2px; }}
.time-label {{ color:#9d9dc0; font-size:0.8rem; font-weight:600; white-space:nowrap; }}
</style></head>
<body>
<h1>{title}</h1>
<p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
{grid}
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key_sidebar_widget()
    st.markdown("---")
    model_label = st.selectbox("🧠 Model", list(MODEL_OPTIONS.keys()))
    model = MODEL_OPTIONS[model_label]
    st.markdown("---")
    st.markdown(
        "<span class='badge'>Streamlit</span>"
        "<span class='badge'>Groq</span>"
        "<span class='badge'>Free API</span>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>🗓️ AI Timetable Generator</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-subtitle'>Describe your week in plain English — get a "
    "beautiful, downloadable timetable in seconds, powered by Groq's free LLMs.</div>",
    unsafe_allow_html=True,
)

with st.container(border=True):
    example = (
        "e.g. \"Make me a study timetable for Monday to Friday, 8am to 6pm. "
        "I need 2 hours of Math, 1.5 hours of Physics, and 1 hour of Chemistry "
        "daily, with a lunch break at 1pm and short breaks between subjects. "
        "Add gym on Tuesday and Thursday evenings.\""
    )
    user_prompt = st.text_area(
        "📝 Describe the timetable you want",
        height=150,
        placeholder=example,
    )
    col_a, col_b = st.columns([1, 3])
    with col_a:
        generate_clicked = st.button("✨ Generate Timetable", use_container_width=True)

if generate_clicked:
    api_key = get_api_key()
    if not api_key:
        st.error("Please provide a Groq API key in the sidebar first.")
    elif not user_prompt.strip():
        st.warning("Please describe the timetable you'd like to generate.")
    else:
        with st.spinner("Generating your timetable... 🪄"):
            try:
                data = generate_timetable(api_key, model, user_prompt)
                df = to_dataframe(data)
                if df.empty:
                    st.error("The model didn't return any timetable entries. Try rephrasing your prompt.")
                else:
                    st.session_state["timetable_data"] = data
                    st.session_state["timetable_df"] = df
            except json.JSONDecodeError:
                st.error("Couldn't parse the model's response. Please try again.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ─────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────
if "timetable_df" in st.session_state and not st.session_state["timetable_df"].empty:
    data = st.session_state["timetable_data"]
    df = st.session_state["timetable_df"]
    title = data.get("title", "My Timetable")

    with st.container(border=True):
        st.markdown(f"### 📌 {title}")

        view = st.radio("View as", ["Grid", "Table"], horizontal=True, label_visibility="collapsed")
        if view == "Grid":
            st.markdown(render_grid_html(df), unsafe_allow_html=True)
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    # Downloads
    with st.container(border=True):
        st.markdown("#### ⬇️ Download your timetable")
        dl1, dl2, dl3 = st.columns(3)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        with dl1:
            st.download_button(
                "Download CSV", data=csv_bytes, file_name=f"{title.replace(' ', '_')}.csv",
                mime="text/csv", use_container_width=True,
            )

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Timetable")
        with dl2:
            st.download_button(
                "Download Excel", data=excel_buffer.getvalue(),
                file_name=f"{title.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        html_doc = build_downloadable_html(title, df)
        with dl3:
            st.download_button(
                "Download HTML", data=html_doc.encode("utf-8"),
                file_name=f"{title.replace(' ', '_')}.html",
                mime="text/html", use_container_width=True,
            )
else:
    st.markdown(
        "<div class='glass-card' style='text-align:center; color:#a9a9c8;'>"
        "👆 Enter a prompt above and click <b>Generate Timetable</b> to get started."
        "</div>",
        unsafe_allow_html=True,
    )
