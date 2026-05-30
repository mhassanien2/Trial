# Trial

This repo contains two Streamlit apps.

## 🎬 Reel Studio — Instagram Reel production package generator

Turns a topic (or news content) plus a few settings into a **complete,
multimedia-ready Instagram Reel production package** — not just text.

```bash
pip install -r requirements.txt
streamlit run reel_studio_app.py
```

### What it generates

A single **Multimedia Studio** with a vertical 9:16 reel preview, a timeline
view, and 14 output tabs:

`Strategy · Hook · Script · Storyboard · Video Preview · AI Image Prompts ·
AI Video Prompts · Audio Plan · Subtitles · Canva Brief · CapCut Brief ·
Thumbnail · Asset Checklist · Export`

Each scene includes duration, visual type, shot size, camera movement,
background, on-screen text, subtitle, voiceover, sound effect, transition,
music mood, stock keywords, and AI image/video prompts (all 9:16).

### Exports

- **Text/data:** TXT script, CSV storyboard, JSON project file, SRT, VTT
- **Briefs:** Canva, CapCut, Premiere Pro, Final Cut, shot list, asset checklist
- **Documents:** PDF brief, DOCX plan (need `reportlab` / `python-docx`)

### Rendering integrations (modular)

The app ships *briefs and prompts*, not rendered video. The
`reelstudio/integrations.py` layer is designed to connect to video/image/voice
providers (Veo/Gemini, Runway, Pika, Luma, Stability AI, ElevenLabs, Google
TTS, Canva). Set the matching environment variable (e.g. `RUNWAY_API_KEY`) to
mark a provider as connected; `Provider.render()` is the single place to wire a
real API or MCP call. Until then the app shows:

> This prototype creates a complete video-ready production package. To generate
> the final MP4, copy the Canva or CapCut brief into your preferred editor, or
> connect this app to a video generation/rendering API.

### Safety / quality rules

All visual prompts are vertical 9:16 with mobile-readable captions, one idea
per scene, and royalty-free media suggestions. For **news**, the engine avoids
AI/dramatized footage and steers toward neutral B-roll, maps, and
headline-style graphics.

### Package layout

```
reel_studio_app.py        # Streamlit UI (Multimedia Studio + 14 tabs)
reelstudio/
  models.py               # dataclasses: ReelProject, Scene, AudioPlan, ...
  engine.py               # generate_project() — the content engine
  exporters.py            # TXT/CSV/JSON/SRT/VTT + editor briefs + PDF/DOCX
  integrations.py         # modular video/image/voice/design providers
```

## 📊 Student Dropout Predictor

The original EDA + model dashboard.

```bash
streamlit run eda_dropout_dashboard.py
```
