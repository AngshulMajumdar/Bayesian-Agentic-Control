"""Deterministic stand-in tools for exercising the inference ladder offline.

These encode the failure modes the framework is meant to survive: a fast source
that is confidently wrong, an authoritative source that disagrees with it, and
checkers that only reveal the conflict a step later.

Retrieval tools report an `answer` plus a self-assessed `confidence`. Checkers
report a boolean `ok` and a `verdict`, deliberately never an `answer`, so a
checker's output can never be mistaken for the user-facing result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

JsonDict = Dict[str, Any]

STALE_SERBIA_CLAIM = "June to August is ideal for Serbia on both weather and budget."
VERIFIED_SERBIA_CLAIM = (
    "Late April to June and September to October are best for Serbia "
    "when balancing weather, crowds, and budget."
)


@dataclass
class FastSearchTool:
    """Cheap retrieval. High confidence, frequently stale."""

    name: str = "fast_search"

    def invoke(self, inp: JsonDict) -> JsonDict:
        q = str(inp.get("query", "")).lower()
        if "serbia" in q:
            return {
                "answer": STALE_SERBIA_CLAIM,
                "confidence": 0.72,
                "is_verified": False,
                "source": "fast",
            }
        return {
            "answer": f"Fast guess for: {inp.get('query', '')}",
            "confidence": 0.65,
            "is_verified": False,
            "source": "fast",
        }


@dataclass
class VerifiedSearchTool:
    """Slower retrieval against an authoritative source."""

    name: str = "verified_search"

    def invoke(self, inp: JsonDict) -> JsonDict:
        q = str(inp.get("query", "")).lower()
        if "serbia" in q:
            return {
                "answer": VERIFIED_SERBIA_CLAIM,
                "confidence": 0.96,
                "is_verified": True,
                "source": "official",
            }
        return {
            "answer": f"Verified answer for: {inp.get('query', '')}",
            "confidence": 0.94,
            "is_verified": True,
            "source": "official",
        }


@dataclass
class ConsistencyCheckTool:
    """Flags the known-stale claim. Emits a verdict, never an answer."""

    name: str = "consistency_check"

    def invoke(self, inp: JsonDict) -> JsonDict:
        text = str(inp.get("text", ""))
        ok = text != STALE_SERBIA_CLAIM
        return {"ok": ok, "verdict": "consistent" if ok else "inconsistent"}


@dataclass
class GeoDisambiguateTool:
    """Surfaces competing readings of an ambiguous place name."""

    name: str = "geo_disambiguate"

    def invoke(self, inp: JsonDict) -> JsonDict:
        return {
            "candidates": [
                {"label": "Salt Lake City, Utah, USA", "p": 0.31},
                {"label": "Salt Lake, Kolkata, India", "p": 0.69},
            ],
            "ambiguous": True,
            "confidence": 0.60,
        }


@dataclass
class AskClarificationTool:
    """Simulates the user resolving the ambiguity."""

    name: str = "ask_clarification"

    def invoke(self, inp: JsonDict) -> JsonDict:
        return {
            "user_choice": "saltlake_kolkata",
            "user_text": "I meant Salt Lake in Kolkata.",
            "confidence": 0.99,
        }


@dataclass
class SchoolSearchUS:
    name: str = "school_search_us"

    def invoke(self, inp: JsonDict) -> JsonDict:
        return {
            "location": "Salt Lake City, Utah, USA",
            "answer": "Top schools near Salt Lake City, Utah: East High School.",
            "confidence": 0.90,
            "is_verified": True,
        }


@dataclass
class SchoolSearchIN:
    name: str = "school_search_india"

    def invoke(self, inp: JsonDict) -> JsonDict:
        return {
            "location": "Salt Lake, Kolkata, India",
            "answer": (
                "Top schools near Salt Lake, Kolkata: Delhi Public School Newtown "
                "(CBSE) and Salt Lake School (WB board)."
            ),
            "confidence": 0.92,
            "is_verified": True,
        }


@dataclass
class NoisyWebSearchTool:
    """A widely repeated but incorrect date."""

    name: str = "noisy_web_search"

    def invoke(self, inp: JsonDict) -> JsonDict:
        return {
            "answer": "The scholarship deadline is May 30.",
            "confidence": 0.58,
            "is_verified": False,
            "source": "random_web",
        }


@dataclass
class OfficialNoticeTool:
    """The authoritative notice."""

    name: str = "official_notice"

    def invoke(self, inp: JsonDict) -> JsonDict:
        return {
            "answer": "The scholarship deadline is April 30.",
            "confidence": 0.97,
            "is_verified": True,
            "source": "official_notice",
        }


@dataclass
class NoticeCheckTool:
    """Validates a claimed deadline against the official notice."""

    name: str = "notice_check"

    def invoke(self, inp: JsonDict) -> JsonDict:
        text = str(inp.get("text", ""))
        ok = "April 30" in text
        return {"ok": ok, "verdict": "consistent" if ok else "inconsistent"}


def default_toolset() -> Dict[str, Any]:
    """Every mock tool, keyed by name."""
    tools = [
        FastSearchTool(),
        VerifiedSearchTool(),
        ConsistencyCheckTool(),
        GeoDisambiguateTool(),
        AskClarificationTool(),
        SchoolSearchUS(),
        SchoolSearchIN(),
        NoisyWebSearchTool(),
        OfficialNoticeTool(),
        NoticeCheckTool(),
    ]
    return {t.name: t for t in tools}
