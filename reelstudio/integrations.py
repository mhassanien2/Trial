"""Modular media-generation integration layer.

The prototype ships *briefs and prompts*, not rendered media. This module
declares the providers the app is designed to connect to (video, image, voice,
design) and detects which are configured via environment variables. Each
provider exposes a uniform ``render(...)`` contract that today returns a
"not yet wired" stub — real implementations (or MCP calls) can be dropped in
later without touching the UI or the generation engine.

Detected here (env var -> provider):
    VEO_API_KEY / GEMINI_API_KEY  -> Google Veo / Gemini video
    RUNWAY_API_KEY                -> Runway
    PIKA_API_KEY                  -> Pika
    LUMA_API_KEY                  -> Luma Dream Machine
    STABILITY_API_KEY             -> Stability AI (image/video)
    ELEVENLABS_API_KEY            -> ElevenLabs (voiceover)
    GOOGLE_TTS_API_KEY            -> Google Text-to-Speech
    CANVA_API_KEY / Canva MCP     -> Canva export workflow
"""

from __future__ import annotations

import os
from dataclasses import dataclass


LIMITATION_MESSAGE = (
    "This prototype creates a complete video-ready production package. "
    "To generate the final MP4, copy the Canva or CapCut brief into your "
    "preferred editor, or connect this app to a video generation/rendering API."
)


@dataclass
class Provider:
    key: str
    name: str
    kind: str  # "video" | "image" | "voice" | "design"
    env_vars: tuple[str, ...]
    note: str = ""

    @property
    def available(self) -> bool:
        return any(os.environ.get(v) for v in self.env_vars)

    def render(self, prompt: str, **kwargs):
        """Uniform entry point. Returns a status dict; never raises for the UI."""
        if not self.available:
            return {
                "status": "unavailable",
                "provider": self.name,
                "message": (
                    f"{self.name} is not connected. Set one of "
                    f"{', '.join(self.env_vars)} (or wire the MCP tool) to enable "
                    f"live {self.kind} generation."
                ),
                "prompt": prompt,
            }
        # Placeholder for the real API/MCP call — intentionally not implemented
        # so the prototype never makes surprise external calls.
        return {
            "status": "not_implemented",
            "provider": self.name,
            "message": (
                f"{self.name} is configured. Implement Provider.render() to call "
                f"the live {self.kind} API/MCP here."
            ),
            "prompt": prompt,
        }


PROVIDERS: list[Provider] = [
    Provider("veo", "Google Veo / Gemini video", "video", ("VEO_API_KEY", "GEMINI_API_KEY")),
    Provider("runway", "Runway", "video", ("RUNWAY_API_KEY",)),
    Provider("pika", "Pika", "video", ("PIKA_API_KEY",)),
    Provider("luma", "Luma Dream Machine", "video", ("LUMA_API_KEY",)),
    Provider("stability", "Stability AI", "image", ("STABILITY_API_KEY",)),
    Provider("elevenlabs", "ElevenLabs", "voice", ("ELEVENLABS_API_KEY",)),
    Provider("google_tts", "Google Text-to-Speech", "voice", ("GOOGLE_TTS_API_KEY",)),
    Provider("canva", "Canva export", "design", ("CANVA_API_KEY",),
             note="Canva MCP tools may also be available in this environment."),
]


def providers_by_kind(kind: str) -> list[Provider]:
    return [p for p in PROVIDERS if p.kind == kind]


def status_table() -> list[dict[str, str]]:
    return [
        {
            "Provider": p.name,
            "Type": p.kind,
            "Status": "✅ Connected" if p.available else "⚪ Not connected",
            "Enable with": " / ".join(p.env_vars),
        }
        for p in PROVIDERS
    ]


def any_renderer_available() -> bool:
    return any(p.available for p in PROVIDERS if p.kind in ("video", "image"))
