from __future__ import annotations

import os
from typing import Optional


def resolve_task_id(explicit: Optional[int] = None) -> Optional[int]:
    if explicit is not None:
        return int(explicit)
    for key in ("TASK_ID", "TASK_WITHIN_JOB", "SLURM_ARRAY_TASK_ID"):
        raw = os.environ.get(key)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value:
            continue
        if value.isdigit():
            return int(value)
        raise ValueError(f"Environment variable {key} must be an integer, got: {raw!r}")
    return None


def apply_task_template(value: Optional[str], task_id: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    rendered = text
    if task_id is None:
        return rendered
    token = str(task_id)
    rendered = rendered.replace("{task_id}", token)
    rendered = rendered.replace("<TASK_WITHIN_JOB>", token)
    rendered = rendered.replace("$TASK_WITHIN_JOB", token)
    rendered = rendered.replace("${TASK_WITHIN_JOB}", token)
    rendered = rendered.replace("$TASK_ID", token)
    rendered = rendered.replace("${TASK_ID}", token)
    return rendered
