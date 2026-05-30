"""Data models for the Instagram Reel production package.

These dataclasses are the single source of truth for a reel project. Every
exporter, brief generator and UI view reads from a :class:`ReelProject`, so the
generation engine only has to populate this structure once.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# --- Controlled vocabularies (kept here so UI + engine stay in sync) ---------

VISUAL_TYPES = [
    "Talking head",
    "Stock video",
    "AI-generated image",
    "B-roll",
    "Screen recording",
    "Animation",
    "Text-only motion graphic",
]

SHOT_SIZES = [
    "Close-up",
    "Medium shot",
    "Wide shot",
    "Over-the-shoulder",
    "Screen capture",
]

VOICE_STYLES = [
    "Warm female",
    "Confident male",
    "News anchor",
    "Calm educational",
    "Energetic creator",
    "Luxury soft voice",
]

CONTENT_TYPES = [
    "News",
    "Educational",
    "Story",
    "Promo",
    "Tips / How-to",
]

TRANSITIONS = [
    "Hard cut",
    "Quick zoom",
    "Whip pan",
    "Cross dissolve",
    "Slide up",
    "Glitch",
    "Match cut",
]


@dataclass
class StockKeywords:
    """Per-scene search terms for licensed stock libraries."""

    video: list[str] = field(default_factory=list)
    photo: list[str] = field(default_factory=list)
    icon: list[str] = field(default_factory=list)
    background: list[str] = field(default_factory=list)


@dataclass
class ImagePrompt:
    subject: str = ""
    setting: str = ""
    mood: str = ""
    lighting: str = ""
    composition: str = ""
    style: str = ""
    color_palette: str = ""
    aspect_ratio: str = "9:16"
    negative_prompt: str = ""

    def as_text(self) -> str:
        """Flatten into a single ready-to-paste generation prompt."""
        parts = [
            f"Create a vertical {self.aspect_ratio} image.",
            f"Subject: {self.subject}.",
            f"Setting: {self.setting}.",
            f"Mood: {self.mood}.",
            f"Lighting: {self.lighting}.",
            f"Composition: {self.composition}.",
            f"Style: {self.style}.",
            f"Color palette: {self.color_palette}.",
        ]
        return " ".join(p for p in parts if not p.endswith(": .")) + (
            f" Negative prompt: {self.negative_prompt}." if self.negative_prompt else ""
        )


@dataclass
class VideoPrompt:
    scene_description: str = ""
    movement: str = ""
    camera_motion: str = ""
    duration_seconds: float = 5.0
    lighting: str = ""
    mood: str = ""
    style: str = ""
    aspect_ratio: str = "9:16"
    avoid: str = ""

    def as_text(self) -> str:
        return (
            f"Vertical {self.aspect_ratio} video, {self.duration_seconds:g} seconds. "
            f"{self.scene_description} Movement: {self.movement}. "
            f"Camera: {self.camera_motion}. Lighting: {self.lighting}. "
            f"Mood: {self.mood}. Style: {self.style}. "
            f"Do not include: {self.avoid}."
        )


@dataclass
class Scene:
    number: int
    role: str  # hook / context / point / payoff / cta
    start: float
    end: float
    visual_type: str
    visual_description: str
    camera_movement: str
    shot_size: str
    background: str
    text_overlay: str
    subtitle: str
    voiceover: str
    sound_effect: str
    transition: str
    music_mood: str
    needs_ai_image: bool = False
    needs_ai_video: bool = False
    image_prompt: ImagePrompt | None = None
    video_prompt: VideoPrompt | None = None
    stock: StockKeywords = field(default_factory=StockKeywords)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


@dataclass
class AudioPlan:
    voice_style: str = ""
    voice_speed: str = "Medium (≈150 wpm)"
    emotional_tone: str = ""
    voiceover_script: str = ""
    music_mood: str = ""
    music_keywords: list[str] = field(default_factory=list)
    sound_effects: list[str] = field(default_factory=list)  # one per scene
    silence_moments: list[str] = field(default_factory=list)


@dataclass
class Thumbnail:
    text: str = ""
    layout: str = ""
    background_idea: str = ""
    subject_idea: str = ""
    color_palette: str = ""
    ai_prompt: str = ""
    canva_brief: str = ""


@dataclass
class Strategy:
    goal: str = ""
    audience: str = ""
    angle: str = ""
    platform_notes: str = ""
    hashtags: list[str] = field(default_factory=list)
    posting_tip: str = ""


@dataclass
class ReelProject:
    title: str
    topic: str
    content_type: str
    duration: int
    audience: str
    tone: str
    cta: str
    color_palette: str
    hook_options: list[str] = field(default_factory=list)
    primary_hook: str = ""
    strategy: Strategy = field(default_factory=Strategy)
    scenes: list[Scene] = field(default_factory=list)
    audio: AudioPlan = field(default_factory=AudioPlan)
    thumbnail: Thumbnail = field(default_factory=Thumbnail)
    motion_graphics: dict[str, Any] = field(default_factory=dict)
    safety_notes: list[str] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return round(sum(s.duration for s in self.scenes), 2)

    @property
    def full_script(self) -> str:
        return "\n".join(s.voiceover for s in self.scenes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
