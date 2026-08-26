from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (ROOT / "config.yaml")
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    processed: Path
    reports: Path
    sample: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> Paths:
        base = root or ROOT
        p = cls(
            root=base,
            raw=base / "data" / "raw",
            processed=base / "data" / "processed",
            reports=base / "reports",
            sample=base / "data" / "sample",
        )
        for d in (p.raw, p.processed, p.reports, p.sample):
            d.mkdir(parents=True, exist_ok=True)
        return p
