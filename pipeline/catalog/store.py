"""Tiny JSON-file store. Files under data/catalog are the pipeline's state: a stage is done
for a video when its output file exists, which makes every stage idempotent and resumable."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        text = data.model_dump_json(indent=1, exclude_none=False)
    else:
        text = json.dumps(data, ensure_ascii=False, indent=1, default=str)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_model[M: BaseModel](path: Path, model: type[M]) -> M:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def maybe_model[M: BaseModel](path: Path, model: type[M]) -> M | None:
    return read_model(path, model) if path.exists() else None
