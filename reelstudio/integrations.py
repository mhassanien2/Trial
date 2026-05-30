"""Modular media-generation integration layer.

The prototype ships *briefs and prompts*; this module turns them into real media
when a provider is configured. Each :class:`Provider` exposes a uniform
``render(prompt, out_path=...)`` contract:

* **Not configured** -> returns a structured ``unavailable`` dict (no network).
* **Configured** -> calls the provider's REST API and returns the saved file
  path (or bytes) on success, or a structured ``error`` dict on failure.

Implemented live paths: Stability AI (image) and ElevenLabs (text-to-speech),
which use simple API-key + REST and are wired end-to-end here. Video providers
(Veo/Gemini, Runway, Pika, Luma) and Canva use async or OAuth flows; their
``render`` is a clearly-marked hook returning ``not_implemented`` so the
prototype never makes a half-working call — drop the real request into the
matching ``_render_*`` function to enable them.

In this repo's agent environment the connected **MCP servers** (Canva +
image/video generation) are the live path used to produce the assets recorded in
``samples/sample_assets.json``; this REST layer is what powers a *deployed* app
where API keys are provided via environment variables.

Env var -> provider:
    VEO_API_KEY / GEMINI_API_KEY  -> Google Veo / Gemini video
    RUNWAY_API_KEY                -> Runway
    PIKA_API_KEY                  -> Pika
    LUMA_API_KEY                  -> Luma Dream Machine
    STABILITY_API_KEY             -> Stability AI (image)   [live]
    ELEVENLABS_API_KEY            -> ElevenLabs (voiceover)  [live]
    GOOGLE_TTS_API_KEY            -> Google Text-to-Speech
    CANVA_API_KEY                 -> Canva export workflow
"""

from __future__ import annotations

import os
from dataclasses import dataclass


LIMITATION_MESSAGE = (
    "This prototype creates a complete video-ready production package. "
    "To generate the final MP4, copy the Canva or CapCut brief into your "
    "preferred editor, or connect this app to a video generation/rendering API."
)

_TIMEOUT = 120


# --------------------------------------------------------------------------- #
# Live REST implementations
# --------------------------------------------------------------------------- #
def _save(out_path: str | None, data: bytes, suffix: str) -> str | None:
    if not out_path:
        return None
    if not os.path.splitext(out_path)[1]:
        out_path += suffix
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(data)
    return out_path


def _render_stability_image(prompt: str, *, out_path=None, aspect_ratio="9:16", **_):
    """Stability AI Stable Image — POST prompt, receive image bytes."""
    import requests

    resp = requests.post(
        "https://api.stability.ai/v2beta/stable-image/generate/core",
        headers={
            "Authorization": f"Bearer {os.environ['STABILITY_API_KEY']}",
            "Accept": "image/*",
        },
        files={"none": ""},
        data={"prompt": prompt, "aspect_ratio": aspect_ratio, "output_format": "png"},
        timeout=_TIMEOUT,
    )
    if resp.status_code != 200:
        return {"status": "error", "provider": "Stability AI",
                "message": f"HTTP {resp.status_code}: {resp.text[:300]}"}
    path = _save(out_path, resp.content, ".png")
    return {"status": "ok", "provider": "Stability AI", "kind": "image",
            "path": path, "bytes": None if path else resp.content}


def _render_elevenlabs_voice(prompt: str, *, out_path=None, voice_id="21m00Tcm4TlvDq8ikWAM", **_):
    """ElevenLabs text-to-speech — POST script text, receive mp3 bytes."""
    import requests

    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": os.environ["ELEVENLABS_API_KEY"],
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json={"text": prompt, "model_id": "eleven_multilingual_v2"},
        timeout=_TIMEOUT,
    )
    if resp.status_code != 200:
        return {"status": "error", "provider": "ElevenLabs",
                "message": f"HTTP {resp.status_code}: {resp.text[:300]}"}
    path = _save(out_path, resp.content, ".mp3")
    return {"status": "ok", "provider": "ElevenLabs", "kind": "audio",
            "path": path, "bytes": None if path else resp.content}


# Map provider key -> live implementation. Absent keys fall back to a documented
# "configured but render hook not implemented" response.
_LIVE = {
    "stability": _render_stability_image,
    "elevenlabs": _render_elevenlabs_voice,
}


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

    def render(self, prompt: str, *, out_path: str | None = None, **kwargs) -> dict:
        """Uniform entry point. Never raises for the UI; returns a status dict."""
        if not self.available:
            return {
                "status": "unavailable",
                "provider": self.name,
                "message": (
                    f"{self.name} is not connected. Set one of "
                    f"{', '.join(self.env_vars)} (or use the {self.kind} MCP tool) "
                    f"to enable live {self.kind} generation."
                ),
                "prompt": prompt,
            }
        impl = _LIVE.get(self.key)
        if impl is None:
            return {
                "status": "not_implemented",
                "provider": self.name,
                "message": (
                    f"{self.name} is configured. This provider uses an async/OAuth "
                    f"flow — add the request in integrations._render_* to enable "
                    f"live {self.kind} generation."
                ),
                "prompt": prompt,
            }
        try:
            return impl(prompt, out_path=out_path, **kwargs)
        except Exception as exc:  # network/SDK errors must not crash the UI
            return {"status": "error", "provider": self.name, "message": str(exc),
                    "prompt": prompt}


PROVIDERS: list[Provider] = [
    Provider("veo", "Google Veo / Gemini video", "video", ("VEO_API_KEY", "GEMINI_API_KEY")),
    Provider("runway", "Runway", "video", ("RUNWAY_API_KEY",)),
    Provider("pika", "Pika", "video", ("PIKA_API_KEY",)),
    Provider("luma", "Luma Dream Machine", "video", ("LUMA_API_KEY",)),
    Provider("stability", "Stability AI", "image", ("STABILITY_API_KEY",),
             note="Live REST implementation."),
    Provider("elevenlabs", "ElevenLabs", "voice", ("ELEVENLABS_API_KEY",),
             note="Live REST implementation."),
    Provider("google_tts", "Google Text-to-Speech", "voice", ("GOOGLE_TTS_API_KEY",)),
    Provider("canva", "Canva export", "design", ("CANVA_API_KEY",),
             note="Canva MCP tools are the live path in the agent environment."),
]


def get_provider(key: str) -> Provider | None:
    return next((p for p in PROVIDERS if p.key == key), None)


def providers_by_kind(kind: str) -> list[Provider]:
    return [p for p in PROVIDERS if p.kind == kind]


def first_available(kind: str) -> Provider | None:
    return next((p for p in providers_by_kind(kind) if p.available), None)


def status_table() -> list[dict[str, str]]:
    return [
        {
            "Provider": p.name,
            "Type": p.kind,
            "Status": "✅ Connected" if p.available else "⚪ Not connected",
            "Live render": "yes" if p.key in _LIVE else "hook",
            "Enable with": " / ".join(p.env_vars),
        }
        for p in PROVIDERS
    ]


def any_renderer_available() -> bool:
    return any(p.available for p in PROVIDERS if p.kind in ("video", "image"))
