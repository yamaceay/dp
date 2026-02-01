from __future__ import annotations

from typing import List, Optional, Tuple, Union, Iterable
from tqdm import tqdm

from dp.loaders.base import DatasetRecord
from dp.loaders import get_adapter
from dp.tri.loaders.base import AttackerDatasetAdapter, AttackerDatasetRecord
from dp.utils.rewriter import BartRewriter
from dp.utils.device import resolve_device

class DBBioAttackerDatasetAdapter(AttackerDatasetAdapter):
    def __init__(
        self,
        rewriter_model_name: str = "facebook/bart-large-cnn",
        rewriter_device: Optional[Union[str, int]] = None,
        rewriter_max_length: int = 256,
        rewriter_min_length: int = 128,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 1.2,
        max_background_tokens: int = 512,
        n_train_samples: int = 3,
        n_eval_samples: int = 1,
        **data_kwargs,
    ) -> None:
        adapter = get_adapter("db_bio", **data_kwargs)
        super().__init__(
            adapter=adapter,
        )
        if rewriter_device is None:
            rewriter_device = resolve_device()
        rewriter = BartRewriter(
            model_name=rewriter_model_name,
            device=rewriter_device,
            max_input_tokens=max_background_tokens,
        )
        self.rewriter_kwargs = {
            "max_length": rewriter_max_length,
            "min_length": rewriter_min_length,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
        }
        self.set_rewriter(rewriter)
        self.n_train_samples = n_train_samples
        self.n_eval_samples = n_eval_samples

    def iter_records(self, progress: bool = False) -> Iterable[AttackerDatasetRecord]:
        records_list = list(self.adapter.iter_records())
        iterator = iter(records_list)
        if progress:
            iterator = tqdm(records_list, desc="Processing attacker records", total=len(records_list))
        for idx, record in enumerate(iterator):
            modified_record = record
            if self._starting_anonymizations_by_idx is not None and idx < len(self._starting_anonymizations_by_idx):
                anns = self._starting_anonymizations_by_idx[idx]
                if anns:
                    modified_record = record.copy()
                    modified_record.text = apply_annotations(record.text, anns, mask_text=self._starting_replacement)
            if self._cache_map is not None and modified_record.name in self._cache_map:
                ext = self._cache_map.get(modified_record.name, {})
                train_texts = _normalize_texts(ext.get("train_texts", []))
                eval_texts = _normalize_texts(ext.get("eval_texts", []))
            else:
                train_texts = [self.rewriter.rewrite(text=modified_record.text, **self.rewriter_kwargs) for _ in range(self.n_train_samples)]
                eval_texts = [self.rewriter.rewrite(text=modified_record.text, **self.rewriter_kwargs) for _ in range(self.n_eval_samples)]
            yield AttackerDatasetRecord(
                name=modified_record.name,
                train_texts=train_texts,
                eval_texts=eval_texts,
            )
__all__ = ["DBBioAttackerDatasetAdapter"]
