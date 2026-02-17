from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone


def main() -> int:
    parser = argparse.ArgumentParser(description="Array task test utility")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    args = parser.parse_args()

    task_id_raw = os.environ.get("TASK_ID")
    task_within_job_raw = os.environ.get("TASK_WITHIN_JOB")
    slurm_array_id_raw = os.environ.get("SLURM_ARRAY_TASK_ID")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "config_path": args.config_path,
        "output_path": args.output_path,
        "task_id": int(task_id_raw) if task_id_raw and task_id_raw.isdigit() else task_id_raw,
        "task_within_job": int(task_within_job_raw) if task_within_job_raw and task_within_job_raw.isdigit() else task_within_job_raw,
        "slurm_array_task_id": int(slurm_array_id_raw) if slurm_array_id_raw and slurm_array_id_raw.isdigit() else slurm_array_id_raw,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
