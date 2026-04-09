from __future__ import annotations

from typing import Any

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


class _SimpleProgress:
    def __init__(self, total: int | None, desc: str, unit: str = "it") -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.count = 0
        self._last_print = -1
        if self.total is None:
            print(f"{self.desc}: started")
        else:
            print(f"{self.desc}: 0/{self.total} {self.unit}")

    def update(self, n: int = 1) -> None:
        self.count += n
        if self.total is None:
            step = max(1, n)
            if self.count - self._last_print >= step * 20:
                self._last_print = self.count
                print(f"{self.desc}: {self.count} {self.unit}")
            return
        if self.count == self.total or self.count - self._last_print >= max(1, self.total // 20):
            self._last_print = self.count
            print(f"{self.desc}: {self.count}/{self.total} {self.unit}")

    def set_postfix_str(self, _: str) -> None:
        return

    def close(self) -> None:
        if self.total is None:
            print(f"{self.desc}: done ({self.count} {self.unit})")
        else:
            print(f"{self.desc}: done ({self.count}/{self.total} {self.unit})")


def new_progress(total: int | None, desc: str, unit: str = "it") -> Any:
    if _tqdm is not None:
        return _tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)
    return _SimpleProgress(total=total, desc=desc, unit=unit)
