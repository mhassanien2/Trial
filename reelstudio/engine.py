"""Reel content generation engine.

Deterministic, dependency-free generator that turns a small set of user inputs
(topic, content type, duration, tone, voice...) into a complete, professional
:class:`ReelProject`. It is intentionally template + heuristic driven so the
prototype works offline; the modular ``integrations`` layer can later swap in a
real LLM for richer copy without changing the rest of the app.
"""

from __future__ import annotations

import re
import textwrap

from .models import (
    AudioPlan,
    ImagePrompt,
    ReelProject,
    Scene,
    StockKeywords,
    Strategy,
    Thumbnail,
    VideoPrompt,
)

# Scene blueprints per duration. Each entry is an ordered list of scene roles.
_BLUEPRINTS = {
    15: ["hook", "point", "cta"],
    20: ["hook", "context", "point", "cta"],
    30: ["hook", "context", "point", "point", "cta"],
    45: ["hook", "context", "point", "point", "point", "cta"],
    60: ["hook", "context", "point", "point", "point", "payoff", "cta"],
    90: ["hook", "context", "point", "point", "point", "point", "payoff", "cta"],
}

# How long each role "wants" to be, used as relative weights for time budgeting.
_ROLE_WEIGHT = {
    "hook": 1.0,
    "context": 1.4,
    "point": 1.6,
    "payoff": 1.4,
    "cta": 1.2,
}

_TONE_MOODS = {
    "Informative": ("clean, trustworthy", "Neutral corporate / news bed"),
    "Inspirational": ("uplifting, cinematic", "Uplifting cinematic build"),
    "Energetic": ("high-energy, punchy", "Driving upbeat electronic"),
    "Calm": ("soft, reflective", "Soft ambient / lo-fi"),
    "Urgent": ("tense, important", "Tense pulse / news urgency"),
    "Playful": ("fun, light", "Quirky pop / playful"),
}

_VISUAL_PALETTE = {
    "News": ("AI-generated image", "Stock video", "Text-only motion graphic"),
    "Educational": ("Talking head", "Animation", "Text-only motion graphic"),
    "Story": ("AI-generated image", "B-roll", "Talking head"),
    "Promo": ("Stock video", "AI-generated image", "Talking head"),
    "Tips / How-to": ("Talking head", "Screen recording", "Text-only motion graphic"),
}

_VOICE_TONE = {
    "Warm female": "warm, friendly, reassuring",
    "Confident male": "confident, grounded, authoritative",
    "News anchor": "clear, measured, credible",
    "Calm educational": "calm, patient, explanatory",
    "Energetic creator": "excited, fast, conversational",
    "Luxury soft voice": "soft, premium, intimate",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "this", "that", "how", "why", "what", "your",
    "you", "we", "it", "as", "at", "by", "from", "about", "into", "be", "can",
}


def _keywords(text: str, limit: int = 6) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    seen: list[str] = []
    for w in words:
        if w in _STOPWORDS or len(w) < 3:
            continue
        if w not in seen:
            seen.append(w)
        if len(seen) >= limit:
            break
    return seen or ["topic"]


def _short_topic(topic: str, words: int = 6) -> str:
    parts = topic.strip().split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def _split_points(key_points: str | None, topic: str, n: int) -> list[str]:
    """Return ``n`` body talking points, using user lines when provided."""
    points: list[str] = []
    if key_points:
        for line in key_points.splitlines():
            line = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
            if line:
                points.append(line)
    kws = _keywords(topic, 8)
    fillers = [
        f"Here's what most people miss about {kws[0]}.",
        f"The key thing to understand is how {kws[0]} actually works.",
        f"This is why {kws[-1]} matters more than you think.",
        f"Let's break down the part nobody explains about {kws[0]}.",
    ]
    i = 0
    while len(points) < n:
        points.append(fillers[i % len(fillers)])
        i += 1
    return points[:n]


def _allocate_durations(roles: list[str], total: int) -> list[tuple[float, float]]:
    weights = [_ROLE_WEIGHT.get(r, 1.0) for r in roles]
    wsum = sum(weights)
    raw = [total * w / wsum for w in weights]
    # Round to 1 decimal and reconcile drift onto the last scene.
    durs = [round(x, 1) for x in raw]
    durs[-1] = round(durs[-1] + (total - sum(durs)), 1)
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for d in durs:
        spans.append((round(cursor, 1), round(cursor + d, 1)))
        cursor += d
    return spans


def _wrap_subtitle(text: str, width: int = 42) -> str:
    return "\n".join(textwrap.wrap(text, width=width)) or text


def _make_image_prompt(subject: str, setting: str, mood: str, palette: str) -> ImagePrompt:
    return ImagePrompt(
        subject=subject,
        setting=setting,
        mood=mood,
        lighting="soft natural light, gentle shadows",
        composition="rule-of-thirds, generous headroom for text overlay, shallow depth of field",
        style="cinematic realistic, premium social-media aesthetic",
        color_palette=palette,
        aspect_ratio="9:16",
        negative_prompt=(
            "no text, no logos, no watermarks, no distorted hands or faces, "
            "no extra fingers, no low-resolution artifacts"
        ),
    )


def _make_video_prompt(desc: str, mood: str, dur: float, motion: str) -> VideoPrompt:
    return VideoPrompt(
        scene_description=desc,
        movement="subtle, natural motion in subject and environment",
        camera_motion=motion,
        duration_seconds=round(dur, 1),
        lighting="soft, flattering, true-to-life",
        mood=mood,
        style="cinematic realistic, vertical social-media framing",
        aspect_ratio="9:16",
        avoid="logos, readable on-screen text, distorted fingers, brand marks, flicker",
    )


# Per-role camera + shot defaults that read well on a phone.
_ROLE_CAMERA = {
    "hook": ("Slow push-in", "Close-up"),
    "context": ("Slow pan", "Medium shot"),
    "point": ("Static / subtle handheld", "Medium shot"),
    "payoff": ("Slow push-in", "Close-up"),
    "cta": ("Static", "Medium shot"),
}


def generate_project(
    *,
    topic: str,
    content_type: str = "News",
    duration: int = 30,
    audience: str = "General social audience",
    tone: str = "Informative",
    voice_style: str = "Warm female",
    cta: str = "Follow for more",
    key_points: str | None = None,
    color_palette: str = "Deep navy, warm gold, soft white",
    title: str | None = None,
) -> ReelProject:
    """Build a fully populated :class:`ReelProject` from user inputs."""

    topic = topic.strip() or "Untitled topic"
    duration = min(_BLUEPRINTS.keys(), key=lambda d: abs(d - duration))
    roles = _BLUEPRINTS[duration]
    spans = _allocate_durations(roles, duration)
    kws = _keywords(topic, 8)
    mood_visual, music_bed = _TONE_MOODS.get(tone, _TONE_MOODS["Informative"])
    visuals = _VISUAL_PALETTE.get(content_type, _VISUAL_PALETTE["News"])
    n_points = roles.count("point")
    body_points = _split_points(key_points, topic, n_points)
    is_news = content_type == "News"

    project = ReelProject(
        title=title or f"Reel — {_short_topic(topic)}",
        topic=topic,
        content_type=content_type,
        duration=duration,
        audience=audience,
        tone=tone,
        cta=cta,
        color_palette=color_palette,
    )

    # --- Hooks ---------------------------------------------------------------
    project.hook_options = [
        f"Stop scrolling — here's what's really going on with {kws[0]}.",
        f"Most people get {kws[0]} completely wrong. Here's the truth.",
        f"In {duration} seconds, everything you need to know about {kws[0]}.",
        f"This {kws[0]} update could change how you think about {kws[-1]}.",
    ]
    project.primary_hook = project.hook_options[0]

    # --- Strategy ------------------------------------------------------------
    project.strategy = Strategy(
        goal=f"Deliver a fast, credible {content_type.lower()} reel on '{_short_topic(topic, 8)}' that earns saves and shares.",
        audience=audience,
        angle=(
            "Lead with a pattern-interrupt hook, deliver value in tight beats, "
            "close with a clear call to action."
        ),
        platform_notes=(
            "Vertical 9:16, captions always on (most viewers watch muted), "
            "front-load the payoff in the first 3 seconds, keep cuts under ~3s."
        ),
        hashtags=[f"#{k}" for k in kws[:5]] + ["#reels", "#explained"],
        posting_tip="Post when your audience is most active; pin a comment that restates the hook + CTA.",
    )

    # --- Scenes --------------------------------------------------------------
    point_i = 0
    for idx, (role, (start, end)) in enumerate(zip(roles, spans), start=1):
        dur = round(end - start, 1)
        camera, shot = _ROLE_CAMERA.get(role, ("Static", "Medium shot"))
        transition = "Hard cut" if role != "cta" else "Slide up"

        if role == "hook":
            voiceover = project.primary_hook
            overlay = _short_topic(topic, 5).upper()
            desc = f"An attention-grabbing opening visual representing {kws[0]}"
            visual_type = visuals[0]
            sfx = "Whoosh + subtle riser"
        elif role == "context":
            voiceover = f"Here's the context: {topic.strip().rstrip('.')}."
            overlay = "THE CONTEXT"
            desc = f"A scene-setting visual that grounds the viewer in the {kws[0]} story"
            visual_type = visuals[1 % len(visuals)]
            sfx = "Soft ambient tone"
        elif role == "point":
            voiceover = body_points[point_i]
            overlay = f"POINT {point_i + 1}"
            desc = f"A clear, supportive visual illustrating: {body_points[point_i]}"
            visual_type = visuals[(point_i + 1) % len(visuals)]
            sfx = "Light tick / pop on key word"
            point_i += 1
        elif role == "payoff":
            voiceover = f"Bottom line: this is why {kws[0]} matters for {audience.lower()}."
            overlay = "THE TAKEAWAY"
            desc = f"A satisfying, conclusive visual summarizing {kws[0]}"
            visual_type = visuals[0]
            sfx = "Resolve chord"
        else:  # cta
            voiceover = f"{cta}. Save this and share it with someone who needs it."
            overlay = cta.upper()
            desc = "A clean end-screen / CTA card with brand colors and a follow prompt"
            visual_type = "Text-only motion graphic"
            sfx = "Gentle confirm chime"

        subtitle = _wrap_subtitle(voiceover)
        needs_ai_image = visual_type == "AI-generated image"
        needs_ai_video = visual_type in ("Stock video", "B-roll") and not is_news
        # For sensitive news we steer AI video away from dramatized footage.
        if is_news and visual_type in ("Stock video", "B-roll"):
            needs_ai_video = False

        scene = Scene(
            number=idx,
            role=role,
            start=start,
            end=end,
            visual_type=visual_type,
            visual_description=desc + (
                " — use neutral, headline-style or abstract framing (no dramatized or fabricated footage)."
                if is_news else "."
            ),
            camera_movement=camera,
            shot_size=shot,
            background=f"{mood_visual} backdrop in {color_palette.split(',')[0].strip().lower()} tones",
            text_overlay=overlay,
            subtitle=subtitle,
            voiceover=voiceover,
            sound_effect=sfx,
            transition=transition,
            music_mood=music_bed,
            needs_ai_image=needs_ai_image,
            needs_ai_video=needs_ai_video,
            stock=StockKeywords(
                video=[f"{kws[0]} b-roll", f"{kws[min(1, len(kws)-1)]} vertical", "abstract motion background"],
                photo=[f"{kws[0]}", f"{kws[min(1, len(kws)-1)]} closeup", "lifestyle vertical"],
                icon=[kws[0], "arrow", "checkmark", "play button"],
                background=[f"{mood_visual} gradient", "bokeh", "soft texture vertical"],
            ),
        )

        if needs_ai_image or role in ("hook", "payoff"):
            scene.needs_ai_image = scene.needs_ai_image or role in ("hook", "payoff")
            scene.image_prompt = _make_image_prompt(
                subject=desc,
                setting=f"environment relevant to {kws[0]}, uncluttered, space at top/bottom for captions",
                mood=mood_visual,
                palette=color_palette,
            )
        if needs_ai_video:
            scene.video_prompt = _make_video_prompt(
                desc=desc + ".",
                mood=mood_visual,
                dur=dur,
                motion=camera,
            )

        project.scenes.append(scene)

    # --- Audio plan ----------------------------------------------------------
    project.audio = AudioPlan(
        voice_style=voice_style,
        voice_speed="Fast (≈170 wpm)" if tone in ("Energetic", "Urgent") else "Medium (≈150 wpm)",
        emotional_tone=_VOICE_TONE.get(voice_style, "clear and engaging"),
        voiceover_script=project.full_script,
        music_mood=music_bed,
        music_keywords=[
            music_bed.lower(),
            f"royalty-free {tone.lower()}",
            "no-copyright background music",
            "instagram reel music",
        ],
        sound_effects=[s.sound_effect for s in project.scenes],
        silence_moments=[
            "Beat of silence right after the hook (≈0.3s) to let it land.",
            "Brief pause before the CTA so the call to action stands out.",
        ],
    )

    # --- Thumbnail -----------------------------------------------------------
    thumb_text = _short_topic(topic, 4).upper()
    project.thumbnail = Thumbnail(
        text=thumb_text,
        layout="Bold 2-3 word headline top third, subject lower two-thirds, brand color block.",
        background_idea=f"{mood_visual} vertical image relating to {kws[0]}",
        subject_idea="Expressive face mid-reaction OR a single strong symbolic object",
        color_palette=color_palette,
        ai_prompt=_make_image_prompt(
            subject=f"eye-catching thumbnail subject for a reel about {kws[0]}",
            setting="clean vertical composition with empty space top third for headline text",
            mood="bold, high-contrast, scroll-stopping",
            palette=color_palette,
        ).as_text(),
        canva_brief=(
            f"Canva size 1080×1920. Headline '{thumb_text}' in heavy bold (e.g. Anton/Montserrat ExtraBold), "
            f"high contrast over a {mood_visual} background. Add a small brand logo bottom-left and a "
            "subtle color overlay (40-60% opacity) so white text stays readable."
        ),
    )

    # --- Motion graphics -----------------------------------------------------
    emphasis = kws[:3]
    project.motion_graphics = {
        "text_animation_style": "Word-by-word pop captions, centered, bottom-safe area",
        "entry_animation": "Fade + scale-up from 90% (200ms)",
        "exit_animation": "Fade + slide down (150ms)",
        "emphasis_words": emphasis,
        "highlight_color_suggestions": [c.strip() for c in color_palette.split(",")[:2]],
        "background_motion_idea": f"Slow {mood_visual} gradient drift or subtle parallax on stills",
        "icon_animation_suggestions": "Icons spring in on key points; checkmarks draw-on; arrows slide.",
    }

    # --- Safety / quality notes ---------------------------------------------
    project.safety_notes = [
        "All visual prompts are vertical 9:16 and leave safe space for mobile-readable captions.",
        "Use only royalty-free or platform-licensed music — do not use copyrighted tracks without rights.",
        "Keep on-screen text short (one idea per scene) for mobile readability.",
    ]
    if is_news:
        project.safety_notes += [
            "Do NOT generate fake or dramatized news footage or misleading visuals.",
            "For sensitive news use abstract visuals, neutral B-roll, maps, headline-style "
            "graphics, or presenter commentary instead of fabricated scenes.",
            "Attribute facts to credible sources and avoid implying unverified claims.",
        ]

    return project
