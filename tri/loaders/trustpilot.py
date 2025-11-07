from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple
import json
from tqdm import tqdm

from dp.tri.loaders.base import AttackerDatasetAdapter, AttackerDatasetRecord

class TrustpilotAttackerDatasetAdapter(AttackerDatasetAdapter):
    def __init__(
        self,
        adapter: DatasetAdapter,
        max_background_tokens: int = 512,
        rewriter_max_length: int = 150,
        rewriter_min_length: int = 40,
    ) -> None:
        self.adapter = adapter
        self.max_background_tokens = max_background_tokens
        self.rewriter_max_length = rewriter_max_length
        self.rewriter_min_length = rewriter_min_length
        self.rewriter: Optional[RewriterProtocol] = None
        self._cache_map: Optional[Dict[str, Dict[str, Any]]] = None

    def set_rewriter(self, rewriter: RewriterProtocol)

    def extract_background_knowledge(self, record: DatasetRecord) -> List[Tuple[str, str]]:
        raise NotImplementedError