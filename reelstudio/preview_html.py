"""Render a reel project to a self-contained, shareable HTML preview.

This produces the same vertical 9:16 storyboard the Streamlit "Video Preview"
tab shows, plus a timeline and the key content — as a single static .html file
that opens in any browser without running the server. Generated media from the
asset manifest is embedded by URL so the live images appear.
"""

from __future__ import annotations

import html

from .assets import AssetManifest
from .models import ReelProject

_ROLE_COLOR = {
    "hook": "#ff3b6b", "context": "#3b82f6", "point": "#10b981",
    "payoff": "#a855f7", "cta": "#f59e0b",
}


def _card(sc, media_url: str | None) -> str:
    accent = _ROLE_COLOR.get(sc.role, "#6366f1")
    overlay = html.escape(sc.text_overlay)
    subtitle = html.escape(sc.subtitle.replace("\n", " "))
    vis = html.escape(sc.visual_type)
    music = html.escape(sc.music_mood.split("/")[0].strip())
    if media_url:
        bg = (f'background:#0b1020;"><img src="{html.escape(media_url)}" '
              f'style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;"/>')
    else:
        bg = f'background:linear-gradient(160deg,{accent}33,#1e293bcc,#0b1020);">'
    return f"""
    <div class="cardwrap">
      <div class="card" style="{bg}
        <div class="badge" style="background:{accent};">#{sc.number} {html.escape(sc.role.upper())}</div>
        <div class="dur">{sc.duration:g}s</div>
        <div class="overlay">{overlay}</div>
        <div class="sub">{subtitle}</div>
        <div class="meta"><span>🎥 {vis}</span><span>🎵 {music}</span></div>
      </div>
      <div class="trans">↓ {html.escape(sc.transition)}</div>
    </div>"""


def build_preview_html(project: ReelProject, manifest: AssetManifest) -> str:
    cards = "".join(_card(sc, manifest.scene_media_url(sc.number)) for sc in project.scenes)
    cta = f"""
    <div class="cardwrap">
      <div class="card endscreen">
        <div style="font-size:34px;">👉</div>
        <div style="font-size:17px;font-weight:800;padding:0 12px;">{html.escape(project.cta.upper())}</div>
        <div style="font-size:11px;margin-top:6px;color:#fde68a;">CTA end screen</div>
      </div></div>"""

    rows = "".join(
        f"<tr><td>{sc.number}</td><td>{sc.start:g}-{sc.end:g}s</td><td>{sc.duration:g}s</td>"
        f"<td>{html.escape(sc.visual_type)}</td><td>{html.escape(sc.text_overlay)}</td>"
        f"<td>{html.escape(sc.voiceover)}</td><td>{html.escape(sc.transition)}</td></tr>"
        for sc in project.scenes
    )
    alt_hooks = "".join(f"<li>{html.escape(h)}</li>" for h in project.hook_options[1:])
    thumb = manifest.thumbnail_url()
    thumb_html = (f'<img src="{html.escape(thumb)}" class="thumb"/>' if thumb else
                  f'<div class="thumb placeholder">{html.escape(project.thumbnail.text)}</div>')
    canva = manifest.canva_edit_url()
    canva_html = (f'<p>🎨 Live Canva design: <a href="{html.escape(canva)}" target="_blank">open ↗</a></p>'
                  if canva else "")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(project.title)} — Preview</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       background:#0b1020;color:#e2e8f0;margin:0;padding:24px;}}
  h1{{font-size:22px;margin:0 0 4px;}} h2{{font-size:15px;color:#93c5fd;margin:26px 0 10px;
       text-transform:uppercase;letter-spacing:.05em;}}
  .metrics{{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0;}}
  .metric{{background:#1e293b;border-radius:10px;padding:8px 14px;}}
  .metric b{{display:block;font-size:18px;}} .metric span{{font-size:11px;color:#94a3b8;}}
  .reel{{display:flex;gap:16px;overflow-x:auto;padding:8px 2px 18px;}}
  .cardwrap{{flex:0 0 auto;}}
  .card{{position:relative;width:210px;height:374px;border-radius:18px;overflow:hidden;
        box-shadow:0 8px 24px rgba(0,0,0,.45);border:1px solid #ffffff22;}}
  .endscreen{{display:flex;flex-direction:column;align-items:center;justify-content:center;
        text-align:center;background:linear-gradient(160deg,#f59e0b,#0b1020);color:#fff;}}
  .badge{{position:absolute;top:8px;left:8px;color:#fff;font-size:11px;font-weight:700;
        padding:2px 8px;border-radius:10px;}}
  .dur{{position:absolute;top:8px;right:8px;background:#000a;color:#fff;font-size:10px;
        padding:2px 6px;border-radius:8px;}}
  .overlay{{position:absolute;top:46%;left:0;right:0;transform:translateY(-50%);text-align:center;
        color:#fff;font-size:19px;font-weight:800;text-shadow:0 2px 6px #000;padding:0 10px;line-height:1.15;}}
  .sub{{position:absolute;bottom:34px;left:8px;right:8px;text-align:center;color:#fff;font-size:11px;
        background:#0009;border-radius:8px;padding:4px 6px;}}
  .meta{{position:absolute;bottom:8px;left:8px;right:8px;display:flex;justify-content:space-between;
        color:#cbd5e1;font-size:9px;}}
  .trans{{text-align:center;color:#94a3b8;font-size:10px;margin-top:4px;}}
  table{{width:100%;border-collapse:collapse;font-size:12px;}}
  th,td{{border-bottom:1px solid #1e293b;padding:6px 8px;text-align:left;vertical-align:top;}}
  th{{color:#93c5fd;}} a{{color:#fbbf24;}}
  .thumb{{width:150px;height:266px;border-radius:12px;object-fit:cover;box-shadow:0 8px 24px #0007;}}
  .placeholder{{display:flex;align-items:flex-start;justify-content:center;padding-top:20px;
        background:linear-gradient(160deg,#f59e0b,#0b1020);color:#fff;font-weight:800;text-align:center;}}
  code{{background:#1e293b;padding:1px 5px;border-radius:4px;}}
  .note{{color:#94a3b8;font-size:12px;margin-top:30px;border-top:1px solid #1e293b;padding-top:12px;}}
</style></head><body>
  <h1>🎬 {html.escape(project.title)}</h1>
  <p style="color:#94a3b8;margin:0;">Static preview of the Reel Studio "Video Preview" tab · {html.escape(project.content_type)} · {html.escape(project.tone)}</p>
  <div class="metrics">
    <div class="metric"><b>{project.total_duration:g}s</b><span>DURATION</span></div>
    <div class="metric"><b>{len(project.scenes)}</b><span>SCENES</span></div>
    <div class="metric"><b>{html.escape(project.audio.voice_style)}</b><span>VOICE</span></div>
    <div class="metric"><b>{html.escape(project.audio.music_mood)}</b><span>MUSIC</span></div>
  </div>

  <h2>📱 Vertical 9:16 Reel Preview</h2>
  <div class="reel">{cards}{cta}</div>

  <h2>🪝 Hook</h2>
  <p><b>{html.escape(project.primary_hook)}</b></p>
  <ul>{alt_hooks}</ul>

  <h2>🎞️ Timeline</h2>
  <table><tr><th>#</th><th>Time</th><th>Dur</th><th>Visual</th><th>Overlay</th><th>Voiceover</th><th>Transition</th></tr>{rows}</table>

  <h2>🖼️ Thumbnail</h2>
  {thumb_html}
  {canva_html}

  <p class="note">This is a static snapshot. Run the full interactive app with all 14 tabs and exports via
  <code>streamlit run reel_studio_app.py</code>.</p>
</body></html>"""
