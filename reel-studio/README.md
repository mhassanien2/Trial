# 🎬 Reel Studio

Turn any topic (or news content) into a **complete, multimedia-ready Instagram
Reel production package** — not just text.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What it generates

A **Multimedia Studio** (vertical 9:16 reel preview + timeline) and 14 output tabs:

`Strategy · Hook · Script · Storyboard · Video Preview · AI Image Prompts ·
AI Video Prompts · Audio Plan · Subtitles · Canva Brief · CapCut Brief ·
Thumbnail · Asset Checklist · Export`

Each scene includes duration, visual type, shot size, camera movement,
background, on-screen text, subtitle, voiceover, sound effect, transition,
music mood, stock keywords, and 9:16 AI image/video prompts (with negative
prompts).

## Exports

- **Text/data:** TXT script, CSV storyboard, JSON project, SRT, VTT
- **Briefs:** Canva, CapCut, Premiere Pro, Final Cut, shot list, asset checklist
- **Documents:** PDF brief, DOCX plan (need `reportlab` / `python-docx`)

## Live media generation (modular)

`reelstudio/integrations.py` connects to media providers gated by environment
variables, with `Provider.render()` as the single integration point:

| Provider | Type | Enable with | Status |
|---|---|---|---|
| Stability AI | image | `STABILITY_API_KEY` | live REST |
| ElevenLabs | voice | `ELEVENLABS_API_KEY` | live REST |
| Google Veo / Gemini | video | `VEO_API_KEY` / `GEMINI_API_KEY` | hook |
| Runway | video | `RUNWAY_API_KEY` | hook |
| Pika | video | `PIKA_API_KEY` | hook |
| Luma | video | `LUMA_API_KEY` | hook |
| Google TTS | voice | `GOOGLE_TTS_API_KEY` | hook |
| Canva | design | `CANVA_API_KEY` / Canva MCP | hook |

The **Export → Live media generation** panel generates per-scene images and
voiceover from a configured provider, lets you attach any generated media URL
(e.g. a Canva/MCP output) into the 9:16 preview, and links a live Canva design.
Generated media is tracked in an `AssetManifest` (`samples/sample_assets.json`
ships with a real example) so binaries stay out of the repo.

When no renderer is connected the app shows:

> This prototype creates a complete video-ready production package. To generate
> the final MP4, copy the Canva or CapCut brief into your preferred editor, or
> connect this app to a video generation/rendering API.

## Safety / quality rules

All visual prompts are vertical 9:16 with mobile-readable captions, one idea per
scene, royalty-free media only. For **news**, the engine avoids AI/dramatized
footage and steers toward neutral B-roll, maps, and headline-style graphics.

## Layout

```
app.py                    # Streamlit UI (Multimedia Studio + 14 tabs)
reelstudio/
  models.py               # dataclasses: ReelProject, Scene, AudioPlan, ...
  engine.py               # generate_project() — content engine
  exporters.py            # TXT/CSV/JSON/SRT/VTT + editor briefs + PDF/DOCX
  integrations.py         # modular video/image/voice/design providers
  assets.py               # AssetManifest — bind generated media to a reel
samples/sample_assets.json
```
