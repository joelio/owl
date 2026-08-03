"""Council configuration management."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(os.environ.get("OWL_CONFIG_DIR", Path.home() / ".owl"))
CONFIG_FILE = CONFIG_DIR / "config.yaml"


@dataclass
class CouncilMember:
    name: str
    source: str  # "llm", "openai-deep", "perplexity", "google-deep", "deepseek", "xai"

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> CouncilMember:
        return cls(name=data["name"], source=data["source"])


@dataclass
class Config:
    council: list[CouncilMember] = field(default_factory=list)
    # Model that reconciles the council's answers when --synthesise is used.
    # Optional: without one, --synthesise falls back to the first council
    # member reached over the `llm` source.
    arbiter: CouncilMember | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"council": [m.to_dict() for m in self.council]}
        if self.arbiter is not None:
            data["arbiter"] = self.arbiter.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        members = [CouncilMember.from_dict(m) for m in data.get("council", [])]
        raw_arbiter = data.get("arbiter")
        arbiter = CouncilMember.from_dict(raw_arbiter) if raw_arbiter else None
        return cls(council=members, arbiter=arbiter)

    def resolve_arbiter(self) -> CouncilMember | None:
        """Pick the model that will reconcile the answers.

        Prefers an explicitly configured arbiter.  Otherwise falls back to
        the first plain `llm` council member, since deep research providers
        are slow and expensive for what is a summarising job.
        """
        if self.arbiter is not None:
            return self.arbiter
        for member in self.council:
            if member.source == "llm":
                return member
        return None


def load_config() -> Config:
    if not CONFIG_FILE.exists():
        return Config()
    with open(CONFIG_FILE) as f:
        data = yaml.safe_load(f) or {}
    return Config.from_dict(data)


def save_config(config: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
