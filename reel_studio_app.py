"""Reel Studio — Instagram Reel multimedia production package generator.

Run with:  streamlit run reel_studio_app.py

Turns a topic + a few settings into a complete, multimedia-ready Reel package:
strategy, hook, script, storyboard, a vertical 9:16 preview, AI image/video
prompts, audio plan, subtitles (SRT/VTT), Canva & CapCut briefs, thumbnail,
asset checklist, and a full set of downloadable/exportable formats.
"""

from __future__ import annotations

import html

import streamlit as st

from reelstudio import generate_project
from reelstudio.models import CONTENT_TYPES, VOICE_STYLES
from reelstudio import exporters as ex
from reelstudio import integrations as integ

st.set_page_config(page_title="Reel Studio", page_icon="🎬", layout="wide")

TONES = ["Informative", "Inspirational", "Energetic", "Calm", "Urgent", "Playful"]
DURATIONS = [15, 20, 30, 45, 60, 90]

# Simple role -> accent color for the visual mockup / timeline.
_ROLE_COLOR = {
    "hook": "#ff3b6b",
    "context": "#3b82f6",
    "point": "#10b981",
    "payoff": "#a855f7",
    "cta": "#f59e0b",
}


# --------------------------------------------------------------------------- #
# Sidebar — inputs
# --------------------------------------------------------------------------- #
def sidebar_inputs() -> dict:
    st.sidebar.title("🎬 Reel Studio")
    st.sidebar.caption("Instagram Reel production package generator")

    with st.sidebar.form("inputs"):
        topic = st.text_area(
            "Topic / news content",
            value="A major breakthrough in affordable home solar batteries",
            height=90,
            help="Describe the subject of the reel. For news, paste the key facts.",
        )
        content_type = st.selectbox("Content type", CONTENT_TYPES, index=0)
        col1, col2 = st.columns(2)
        with col1:
            duration = st.selectbox("Duration (s)", DURATIONS, index=2)
        with col2:
            tone = st.selectbox("Tone", TONES, index=0)
        audience = st.text_input("Target audience", "Curious 18-35 social audience")
        voice_style = st.selectbox("Voice style", VOICE_STYLES, index=0)
        cta = st.text_input("Call to action", "Follow for more")
        color_palette = st.text_input("Color palette", "Deep navy, warm gold, soft white")
        key_points = st.text_area(
            "Key points (one per line, optional)",
            height=90,
            help="If provided, these become the body talking points.",
        )
        submitted = st.form_submit_button("✨ Generate package", use_container_width=True)

    return {
        "submitted": submitted,
        "topic": topic,
        "content_type": content_type,
        "duration": duration,
        "tone": tone,
        "audience": audience,
        "voice_style": voice_style,
        "cta": cta,
        "color_palette": color_palette,
        "key_points": key_points,
    }


# --------------------------------------------------------------------------- #
# Multimedia Studio — vertical reel mockup
# --------------------------------------------------------------------------- #
def render_reel_card(sc, palette_color: str) -> str:
    accent = _ROLE_COLOR.get(sc.role, "#6366f1")
    overlay = html.escape(sc.text_overlay)
    subtitle = html.escape(sc.subtitle.replace("\n", " "))
    vis = html.escape(sc.visual_type)
    return f"""
    <div style="width:200px;flex:0 0 auto;">
      <div style="position:relative;width:200px;height:356px;border-radius:18px;
           overflow:hidden;background:linear-gradient(160deg,{accent}33,{palette_color}cc,#0b1020);
           box-shadow:0 8px 24px rgba(0,0,0,.35);border:1px solid #ffffff22;">
        <div style="position:absolute;top:8px;left:8px;background:{accent};color:#fff;
             font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px;">
             #{sc.number} {html.escape(sc.role.upper())}</div>
        <div style="position:absolute;top:8px;right:8px;background:#00000088;color:#fff;
             font-size:10px;padding:2px 6px;border-radius:8px;">{sc.duration:g}s</div>
        <div style="position:absolute;top:46%;left:0;right:0;text-align:center;color:#fff;
             font-size:18px;font-weight:800;text-shadow:0 2px 6px #000;padding:0 10px;
             transform:translateY(-50%);line-height:1.15;">{overlay}</div>
        <div style="position:absolute;bottom:34px;left:8px;right:8px;text-align:center;color:#fff;
             font-size:11px;background:#00000066;border-radius:8px;padding:4px 6px;">{subtitle}</div>
        <div style="position:absolute;bottom:8px;left:8px;right:8px;display:flex;
             justify-content:space-between;color:#cbd5e1;font-size:9px;">
             <span>🎥 {vis}</span><span>🎵 {html.escape(sc.music_mood.split('/')[0].strip())}</span></div>
      </div>
      <div style="text-align:center;color:#94a3b8;font-size:10px;margin-top:4px;">
        ↓ {html.escape(sc.transition)}</div>
    </div>
    """


def multimedia_studio(project):
    st.subheader("📱 Vertical 9:16 Reel Preview")
    st.caption("Scene-by-scene mockup — background, overlay text, subtitle, timing, transition & music.")
    palette_color = "#1e293b"
    cards = "".join(render_reel_card(sc, palette_color) for sc in project.scenes)
    # CTA end screen card
    cta_card = f"""
      <div style="width:200px;flex:0 0 auto;">
        <div style="width:200px;height:356px;border-radius:18px;display:flex;flex-direction:column;
             align-items:center;justify-content:center;background:linear-gradient(160deg,#f59e0b,#0b1020);
             color:#fff;text-align:center;box-shadow:0 8px 24px rgba(0,0,0,.35);">
          <div style="font-size:30px;">👉</div>
          <div style="font-size:16px;font-weight:800;padding:0 12px;">{html.escape(project.cta.upper())}</div>
          <div style="font-size:10px;margin-top:6px;color:#fde68a;">CTA end screen</div>
        </div>
      </div>"""
    st.markdown(
        f'<div style="display:flex;gap:14px;overflow-x:auto;padding:10px 2px 18px;">{cards}{cta_card}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("🎞️ Reel Timeline")
    rows = []
    for sc in project.scenes:
        rows.append({
            "Scene": sc.number,
            "Time": f"{sc.start:g}-{sc.end:g}s",
            "Dur": f"{sc.duration:g}s",
            "Visual": sc.visual_type,
            "Overlay": sc.text_overlay,
            "Voiceover": sc.voiceover,
            "Audio (SFX)": sc.sound_effect,
            "Transition": sc.transition,
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Individual tab renderers
# --------------------------------------------------------------------------- #
def tab_strategy(project):
    s = project.strategy
    st.subheader("🎯 Strategy")
    st.write(f"**Goal:** {s.goal}")
    st.write(f"**Audience:** {s.audience}")
    st.write(f"**Angle:** {s.angle}")
    st.write(f"**Platform notes:** {s.platform_notes}")
    st.write(f"**Hashtags:** {' '.join(s.hashtags)}")
    st.info(f"💡 {s.posting_tip}")


def tab_hook(project):
    st.subheader("🪝 Hook")
    st.success(f"**Primary hook:** {project.primary_hook}")
    st.write("**Alternatives (A/B test these):**")
    for h in project.hook_options[1:]:
        st.write(f"- {h}")


def tab_script(project):
    st.subheader("📝 Script")
    st.code(ex.to_script_txt(project), language="text")


def tab_storyboard(project):
    st.subheader("🎬 Storyboard")
    for sc in project.scenes:
        with st.expander(f"Scene {sc.number} — {sc.role.upper()} ({sc.start:g}-{sc.end:g}s · {sc.duration:g}s)", expanded=sc.number == 1):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Visual type:** {sc.visual_type}")
                st.write(f"**Shot size:** {sc.shot_size}")
                st.write(f"**Camera:** {sc.camera_movement}")
                st.write(f"**Background:** {sc.background}")
                st.write(f"**Transition:** {sc.transition}")
                st.write(f"**Music mood:** {sc.music_mood}")
            with c2:
                st.write(f"**Text overlay:** {sc.text_overlay}")
                st.write(f"**Voiceover:** {sc.voiceover}")
                st.write(f"**Subtitle:** {sc.subtitle}")
                st.write(f"**Sound effect:** {sc.sound_effect}")
            st.caption(sc.visual_description)
            st.caption(f"Stock keywords — 🎥 {', '.join(sc.stock.video)} | 📷 {', '.join(sc.stock.photo)} | "
                       f"🔣 {', '.join(sc.stock.icon)} | 🖼 {', '.join(sc.stock.background)}")


def tab_image_prompts(project):
    st.subheader("🖼️ AI Image Prompts (9:16)")
    any_img = False
    for sc in project.scenes:
        if sc.image_prompt:
            any_img = True
            st.markdown(f"**Scene {sc.number} — {sc.role.upper()}**")
            st.code(sc.image_prompt.as_text(), language="text")
    if not any_img:
        st.info("No scenes flagged for AI images in this package.")


def tab_video_prompts(project):
    st.subheader("🎥 AI Video Prompts (Runway / Pika / Kling / Luma / Veo)")
    any_vid = False
    for sc in project.scenes:
        if sc.video_prompt:
            any_vid = True
            st.markdown(f"**Scene {sc.number} — {sc.role.upper()}**")
            st.code(sc.video_prompt.as_text(), language="text")
    if not any_vid:
        st.info("No scenes flagged for AI video. For news, the engine avoids AI/dramatized footage by design.")


def tab_audio(project):
    a = project.audio
    st.subheader("🔊 Audio Plan")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Voice style:** {a.voice_style}")
        st.write(f"**Emotional tone:** {a.emotional_tone}")
        st.write(f"**Voice speed:** {a.voice_speed}")
        st.write(f"**Music mood:** {a.music_mood}")
    with c2:
        st.write(f"**Music keywords:** {', '.join(a.music_keywords)}")
        st.write("**Silence for impact:**")
        for sm in a.silence_moments:
            st.write(f"- {sm}")
    st.write("**Sound effects per scene:**")
    st.dataframe(
        [{"Scene": i + 1, "SFX": s} for i, s in enumerate(a.sound_effects)],
        use_container_width=True, hide_index=True,
    )
    st.write("**Voiceover script:**")
    st.code(a.voiceover_script, language="text")


def tab_subtitles(project):
    st.subheader("💬 Subtitles")
    fmt = st.radio("Format", ["SRT", "VTT", "Plain"], horizontal=True)
    if fmt == "SRT":
        content, mime, name = ex.to_srt(project), "application/x-subrip", "reel.srt"
    elif fmt == "VTT":
        content, mime, name = ex.to_vtt(project), "text/vtt", "reel.vtt"
    else:
        content, mime, name = ex.to_plain_subtitles(project), "text/plain", "reel_subtitles.txt"
    st.code(content, language="text")
    st.download_button(f"Download {name}", content, file_name=name, mime=mime)


def tab_canva(project):
    st.subheader("🎨 Canva Brief")
    brief = ex.canva_brief(project)
    st.markdown(brief)
    st.download_button("Download Canva brief", brief, file_name="canva_brief.md")
    canva_avail = any(p.key == "canva" and p.available for p in integ.PROVIDERS)
    st.caption("✅ Canva integration configured" if canva_avail
               else "⚪ Connect Canva (API key or Canva MCP) to push this brief into a real design.")


def tab_capcut(project):
    st.subheader("✂️ CapCut Brief")
    brief = ex.capcut_brief(project)
    st.markdown(brief)
    st.download_button("Download CapCut brief", brief, file_name="capcut_brief.md")


def tab_thumbnail(project):
    t = project.thumbnail
    st.subheader("🖼️ Thumbnail")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f"""<div style="width:160px;height:285px;border-radius:14px;
            background:linear-gradient(160deg,#f59e0b,#0b1020);display:flex;
            align-items:flex-start;justify-content:center;padding-top:20px;
            color:#fff;font-weight:800;font-size:18px;text-align:center;
            box-shadow:0 8px 24px rgba(0,0,0,.35);">{html.escape(t.text)}</div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.write(f"**Text:** {t.text}")
        st.write(f"**Layout:** {t.layout}")
        st.write(f"**Background idea:** {t.background_idea}")
        st.write(f"**Subject idea:** {t.subject_idea}")
        st.write(f"**Color palette:** {t.color_palette}")
    st.write("**AI image prompt:**")
    st.code(t.ai_prompt, language="text")
    st.write("**Canva thumbnail brief:**")
    st.write(t.canva_brief)


def tab_assets(project):
    st.subheader("✅ Asset Checklist")
    st.markdown(ex.asset_checklist(project))
    mg = project.motion_graphics
    st.subheader("🎞️ Motion Graphics Instructions")
    st.write(f"**Text animation:** {mg['text_animation_style']}")
    st.write(f"**Entry:** {mg['entry_animation']} | **Exit:** {mg['exit_animation']}")
    st.write(f"**Emphasis words:** {', '.join(mg['emphasis_words'])}")
    st.write(f"**Highlight colors:** {', '.join(mg['highlight_color_suggestions'])}")
    st.write(f"**Background motion:** {mg['background_motion_idea']}")
    st.write(f"**Icon animation:** {mg['icon_animation_suggestions']}")


def tab_export(project):
    st.subheader("⬇️ Export")
    if not integ.any_renderer_available():
        st.warning(integ.LIMITATION_MESSAGE)

    st.markdown("**Text & data formats**")
    c = st.columns(3)
    c[0].download_button("TXT script", ex.to_script_txt(project), file_name="reel_script.txt")
    c[1].download_button("CSV storyboard", ex.to_csv_storyboard(project), file_name="reel_storyboard.csv")
    c[2].download_button("JSON project", ex.to_json(project), file_name="reel_project.json")

    c = st.columns(3)
    c[0].download_button("SRT subtitles", ex.to_srt(project), file_name="reel.srt")
    c[1].download_button("VTT subtitles", ex.to_vtt(project), file_name="reel.vtt")
    pdf = ex.to_pdf(project)
    if pdf:
        c[2].download_button("PDF brief", pdf, file_name="reel_brief.pdf", mime="application/pdf")
    else:
        c[2].caption("PDF: `pip install reportlab`")

    docx = ex.to_docx(project)
    st.markdown("**Editor-ready briefs**")
    c = st.columns(3)
    c[0].download_button("Canva brief", ex.canva_brief(project), file_name="canva_brief.md")
    c[1].download_button("CapCut brief", ex.capcut_brief(project), file_name="capcut_brief.md")
    c[2].download_button("Premiere brief", ex.premiere_brief(project), file_name="premiere_brief.md")
    c = st.columns(3)
    c[0].download_button("Final Cut brief", ex.finalcut_brief(project), file_name="finalcut_brief.md")
    c[1].download_button("Shot list", ex.shot_list(project), file_name="shot_list.md")
    c[2].download_button("Asset checklist", ex.asset_checklist(project), file_name="asset_checklist.md")
    if docx:
        st.download_button("DOCX production plan", docx, file_name="reel_plan.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        st.caption("DOCX: `pip install python-docx`")

    with st.expander("🔌 Media-rendering integration status"):
        st.dataframe(integ.status_table(), use_container_width=True, hide_index=True)
        st.caption(integ.LIMITATION_MESSAGE)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    inputs = sidebar_inputs()

    if inputs["submitted"] or "project" not in st.session_state:
        st.session_state.project = generate_project(
            topic=inputs["topic"],
            content_type=inputs["content_type"],
            duration=inputs["duration"],
            audience=inputs["audience"],
            tone=inputs["tone"],
            voice_style=inputs["voice_style"],
            cta=inputs["cta"],
            key_points=inputs["key_points"] or None,
            color_palette=inputs["color_palette"],
        )

    project = st.session_state.project

    st.title("🎬 " + project.title)
    m = st.columns(4)
    m[0].metric("Duration", f"{project.total_duration:g}s")
    m[1].metric("Scenes", len(project.scenes))
    m[2].metric("Type", project.content_type)
    m[3].metric("Voice", project.audio.voice_style)

    for note in project.safety_notes:
        if "NOT" in note or "do not" in note.lower():
            st.caption("⚠️ " + note)

    tabs = st.tabs([
        "Strategy", "Hook", "Script", "Storyboard", "Video Preview",
        "AI Image Prompts", "AI Video Prompts", "Audio Plan", "Subtitles",
        "Canva Brief", "CapCut Brief", "Thumbnail", "Asset Checklist", "Export",
    ])
    with tabs[0]:
        tab_strategy(project)
    with tabs[1]:
        tab_hook(project)
    with tabs[2]:
        tab_script(project)
    with tabs[3]:
        tab_storyboard(project)
    with tabs[4]:
        multimedia_studio(project)
    with tabs[5]:
        tab_image_prompts(project)
    with tabs[6]:
        tab_video_prompts(project)
    with tabs[7]:
        tab_audio(project)
    with tabs[8]:
        tab_subtitles(project)
    with tabs[9]:
        tab_canva(project)
    with tabs[10]:
        tab_capcut(project)
    with tabs[11]:
        tab_thumbnail(project)
    with tabs[12]:
        tab_assets(project)
    with tabs[13]:
        tab_export(project)


if __name__ == "__main__":
    main()
