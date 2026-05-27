from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback

from dp.bert.common import (
    EarlyStoppingCallback,
    build_optimizer_and_scheduler,
    mask_stopword_tokens,
    pretrain_backbone_with_mlm,
)
from dp.utils.device import resolve_device


def load_backbone_with_optional_checkpoint(model_name: str, checkpoint_source: Optional[str]) -> Any:
    base_model = AutoModel.from_pretrained(model_name)
    if checkpoint_source:
        checkpoint_path = Path(checkpoint_source)
        safetensors_file = checkpoint_path / "model.safetensors"
        pytorch_file = checkpoint_path / "pytorch_model.bin"
        if safetensors_file.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_file))
            base_model.load_state_dict(state_dict, strict=False)
        elif pytorch_file.exists():
            state_dict = torch.load(str(pytorch_file), map_location="cpu")
            base_model.load_state_dict(state_dict, strict=False)
        else:
            try:
                checkpoint_model = AutoModel.from_pretrained(checkpoint_source)
                base_model.load_state_dict(checkpoint_model.state_dict(), strict=False)
            except Exception as exc:
                print(f"WARNING: failed to load init_checkpoint from {checkpoint_source}: {exc}")
    return base_model


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: Any, label_dtype: torch.dtype):
        self.encodings = encodings
        self.labels = labels
        self.label_dtype = label_dtype

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=self.label_dtype)
        return item


@dataclass
class HFTrainSpec:
    metric_name: str
    minimize_metric: bool
    weight_decay_effective: float
    compute_metrics: Callable


class BertHFPlumbing:
    def __init__(self, device: Optional[str]):
        self.device = resolve_device(device)
        self._tokenizer: Optional[AutoTokenizer] = None
        self._trainer: Optional[Trainer] = None
        self._stopwords: Set[str] = set()
        self._active_checkpoint: Optional[str] = None

    def _model_device(self, model: torch.nn.Module) -> torch.device:
        return next(model.parameters()).device

    def _maybe_pretrain(
        self,
        *,
        use_pretraining: bool,
        model_name: str,
        texts: Sequence[str],
        output_dir: str,
        init_checkpoint: Optional[str],
        pretraining_epochs: int,
        pretraining_batch_size: int,
        pretraining_learning_rate: float,
        pretraining_mlm_probability: float,
    ) -> None:
        self._active_checkpoint = None
        if not use_pretraining:
            return
        self._active_checkpoint = pretrain_backbone_with_mlm(
            model_name=model_name,
            texts=texts,
            output_dir=output_dir,
            init_checkpoint=init_checkpoint,
            epochs=pretraining_epochs,
            batch_size=pretraining_batch_size,
            learning_rate=pretraining_learning_rate,
            mlm_probability=pretraining_mlm_probability,
        )

    def _load_tokenizer_with_fallback(self, *, model_name: str, init_checkpoint: Optional[str]) -> None:
        tokenizer_source = self._active_checkpoint or init_checkpoint or model_name
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _maybe_enable_stopwords(self, mask_stopwords: bool) -> None:
        self._stopwords = set()
        if not mask_stopwords:
            return
        from dp.utils.stopwords import DEFAULT_STOPWORDS

        self._stopwords = DEFAULT_STOPWORDS
        print(f"Stopword masking enabled: {len(self._stopwords)} stopwords will be masked")

    def _encode_texts(self, texts: Sequence[str], *, mask_stopwords: bool) -> Dict[str, torch.Tensor]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        enc = self._tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        if mask_stopwords:
            enc = mask_stopword_tokens(self._tokenizer, self._stopwords, enc, list(texts))
        return enc

    def _make_trainer(
        self,
        *,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        spec: HFTrainSpec,
        checkpoint_dir: Optional[str],
        epochs: int,
        batch_size: int,
        warmup_steps: int,
        head_lr: float,
        gradient_clip: float,
        optimizer_type: str,
        scheduler_type: str,
        encoder_lr: float,
        weight_decay: float,
        warmup_ratio: Optional[float],
        early_stop_patience: int,
        early_stop_threshold: Optional[float],
        save_checkpoints: bool = True,
        early_stopping_callback: Optional[TrainerCallback] = None,
        extra_callbacks: Optional[Sequence[TrainerCallback]] = None,
    ) -> Tuple[Trainer, EarlyStoppingCallback]:
        output_dir = checkpoint_dir
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            learning_rate=head_lr,
            weight_decay=spec.weight_decay_effective,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch" if save_checkpoints else "no",
            load_best_model_at_end=bool(save_checkpoints),
            metric_for_best_model=spec.metric_name,
            greater_is_better=(not spec.minimize_metric),
            max_grad_norm=gradient_clip,
            report_to="none",
        )

        steps_per_epoch = max(1, int(np.ceil(len(train_dataset) / batch_size)))
        num_training_steps = steps_per_epoch * epochs

        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            optimizer_type=optimizer_type,
            scheduler_type=scheduler_type,
            encoder_lr=encoder_lr,
            head_lr=head_lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            num_training_steps=num_training_steps,
        )

        early_stopping = early_stopping_callback
        if early_stopping is None:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=early_stop_patience,
                early_stopping_threshold=early_stop_threshold,
                metric_name=spec.metric_name,
                minimize=spec.minimize_metric,
            )
        callbacks = [early_stopping]
        if extra_callbacks:
            callbacks.extend(list(extra_callbacks))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=spec.compute_metrics,
            callbacks=callbacks,
            optimizers=(optimizer, scheduler),
        )
        return trainer, early_stopping

    def _predict_in_batches(
        self,
        *,
        model: torch.nn.Module,
        texts: Sequence[str],
        batch_size: int,
        mask_stopwords: bool,
    ) -> Sequence[Any]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = list(texts[i : i + batch_size])
                enc = self._tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                if mask_stopwords:
                    enc = mask_stopword_tokens(self._tokenizer, self._stopwords, enc, batch_texts)
                enc = {k: v.to(self._model_device(model)) for k, v in enc.items()}
                out = model(**enc)
                outputs.append(out)
        return outputs

    def _training_status_path(self, checkpoint_dir: str, status_file: str) -> Path:
        return Path(checkpoint_dir) / str(status_file)

    def _training_lock_path(self, checkpoint_dir: str, status_file: str) -> Path:
        return Path(checkpoint_dir) / f"{status_file}.lock"

    def _list_checkpoint_dirs(self, checkpoint_dir: str) -> List[Path]:
        root = Path(checkpoint_dir)
        if not root.exists():
            return []
        out: List[tuple[int, Path]] = []
        for candidate in root.glob("checkpoint-*"):
            if not candidate.is_dir():
                continue
            suffix = candidate.name.replace("checkpoint-", "", 1)
            try:
                step = int(suffix)
            except Exception:
                continue
            out.append((step, candidate))
        out.sort(key=lambda item: item[0])
        return [p for _, p in out]

    def _latest_checkpoint_dir(self, checkpoint_dir: str) -> Optional[Path]:
        entries = self._list_checkpoint_dirs(checkpoint_dir)
        if not entries:
            return None
        return entries[-1]

    def _read_training_complete(self, path: Path) -> Optional[bool]:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            text = path.read_text(encoding="utf-8").strip().lower()
            if text in {"1", "true", "yes", "done", "complete", "completed"}:
                return True
            if text in {"0", "false", "no", "running", "incomplete"}:
                return False
            return None
        if isinstance(payload, bool):
            return payload
        if isinstance(payload, dict):
            for key in ("complete", "completed", "trained", "ready"):
                value = payload.get(key)
                if isinstance(value, bool):
                    return value
        return None

    def _write_training_complete(self, path: Path, complete: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"complete": bool(complete), "updated_at": int(time.time())}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _acquire_training_lock(self, path: Path) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True

    def _release_training_lock(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def _is_lock_stale(self, path: Path) -> bool:
        try:
            pid_str = path.read_text(encoding="utf-8").strip()
            if not pid_str:
                return False
            pid = int(pid_str)
            os.kill(pid, 0)
            return False
        except (ValueError, FileNotFoundError):
            return False
        except OSError:
            return True

    def _load_model_state_from_checkpoint(self, model: torch.nn.Module, checkpoint_dir: Path) -> bool:
        safetensors_file = checkpoint_dir / "model.safetensors"
        pytorch_file = checkpoint_dir / "pytorch_model.bin"
        state_dict: Optional[Dict[str, Any]] = None
        if safetensors_file.exists():
            from safetensors.torch import load_file

            state_dict = load_file(str(safetensors_file))
        elif pytorch_file.exists():
            state_dict = torch.load(str(pytorch_file), map_location="cpu")
        if state_dict is None:
            return False
        model.load_state_dict(state_dict, strict=False)
        return True

    def _prepare_training_or_reuse(
        self,
        *,
        model: torch.nn.Module,
        checkpoint_dir: Optional[str],
        status_file: str,
        wait_for_completion: bool,
        poll_interval_seconds: float,
        wait_timeout_seconds: Optional[float],
        mark_existing_complete: bool,
    ) -> Tuple[bool, bool]:
        if not checkpoint_dir:
            return False, False

        checkpoint_root = Path(checkpoint_dir)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        status_path = self._training_status_path(checkpoint_dir, status_file)
        lock_path = self._training_lock_path(checkpoint_dir, status_file)

        latest = self._latest_checkpoint_dir(checkpoint_dir)
        status = self._read_training_complete(status_path)

        if status is True and latest is not None and self._load_model_state_from_checkpoint(model, latest):
            self._active_checkpoint = str(latest)
            return True, False

        if status is not True and mark_existing_complete and latest is not None:
            self._write_training_complete(status_path, True)
            if self._load_model_state_from_checkpoint(model, latest):
                self._active_checkpoint = str(latest)
                return True, False

        owns_lock = self._acquire_training_lock(lock_path)
        if not owns_lock and self._is_lock_stale(lock_path):
            self._release_training_lock(lock_path)
            owns_lock = self._acquire_training_lock(lock_path)

        if owns_lock:
            self._write_training_complete(status_path, False)
            return False, True
        if not wait_for_completion:
            return False, False

        started = time.time()
        sleep_time = max(0.2, float(poll_interval_seconds))
        while True:
            latest = self._latest_checkpoint_dir(checkpoint_dir)
            status = self._read_training_complete(status_path)
            if status is True and latest is not None and self._load_model_state_from_checkpoint(model, latest):
                self._active_checkpoint = str(latest)
                return True, False
            if wait_timeout_seconds is not None and (time.time() - started) >= float(wait_timeout_seconds):
                raise TimeoutError(f"timed out waiting for completed model in {checkpoint_dir}")
            if self._is_lock_stale(lock_path):
                self._release_training_lock(lock_path)
                owns_lock = self._acquire_training_lock(lock_path)
                if owns_lock:
                    self._write_training_complete(status_path, False)
                    return False, True
            time.sleep(sleep_time)

    def _finalize_training_status(
        self,
        *,
        checkpoint_dir: Optional[str],
        status_file: str,
        lock_owned: bool,
        complete: bool,
    ) -> None:
        if not checkpoint_dir:
            return
        status_path = self._training_status_path(checkpoint_dir, status_file)
        lock_path = self._training_lock_path(checkpoint_dir, status_file)
        if lock_owned:
            self._write_training_complete(status_path, complete)
            self._release_training_lock(lock_path)

    def _run_training_with_reuse(
        self,
        *,
        model: torch.nn.Module,
        checkpoint_dir: Optional[str],
        status_file: str,
        wait_for_completion: bool,
        poll_interval_seconds: float,
        wait_timeout_seconds: Optional[float],
        mark_existing_complete: bool,
        train_fn: Callable[[], None],
    ) -> bool:
        reused, lock_owned = self._prepare_training_or_reuse(
            model=model,
            checkpoint_dir=checkpoint_dir,
            status_file=status_file,
            wait_for_completion=wait_for_completion,
            poll_interval_seconds=poll_interval_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
            mark_existing_complete=mark_existing_complete,
        )
        if reused:
            return True
        try:
            train_fn()
        except Exception:
            self._finalize_training_status(
                checkpoint_dir=checkpoint_dir,
                status_file=status_file,
                lock_owned=lock_owned,
                complete=False,
            )
            raise
        self._finalize_training_status(
            checkpoint_dir=checkpoint_dir,
            status_file=status_file,
            lock_owned=lock_owned,
            complete=True,
        )
        return False

    def cleanup_plumbing(self, model_attr_names: Sequence[str]) -> None:
        for name in model_attr_names:
            obj = getattr(self, name, None)
            if obj is not None:
                del obj
            setattr(self, name, None)
        if self._tokenizer is not None:
            del self._tokenizer
        if self._trainer is not None:
            del self._trainer
        self._tokenizer = None
        self._trainer = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
