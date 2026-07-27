import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from meta_scores import MetaScoreEngine, parse_meta_score_defs


def _compute_dummy_privacy(dataset: str) -> dict[str, float]:
    privacy_logs = Path("logs/privacy") / dataset
    tri_preproc = Path("data") / dataset / "tri_preprocessing"

    def _harmonic(n: int) -> float:
        return sum(1.0 / i for i in range(1, n + 1))

    def _for_attacker(attacker: str) -> tuple[float, float]:
        with (privacy_logs / f"{attacker}.jsonl").open() as f:
            N = int(json.loads(f.readline())["original_record_count"])
        preproc_path = tri_preproc / f"{attacker}.jsonl"
        if preproc_path.exists():
            with preproc_path.open() as f:
                records = [json.loads(line) for line in f if line.strip()]
            counts = Counter({r["name"]: len(r["train_texts"]) for r in records})
            top_name = max(counts, key=lambda name: counts[name])
            top_record = next(r for r in records if r["name"] == top_name)
            k = len(top_record.get("test_texts", []))
        else:
            k = 1
        if k == 0:
            return 0.0, 0.0
        return k / N, _harmonic(N) / N

    result: dict[str, float] = {}
    for attacker, mrr_key, acc_key in [
        ("exact", "exact_mean_reciprocal_rank", "exact_accuracy"),
        ("more", "more_mean_reciprocal_rank", "more_accuracy"),
        ("full", "full_mean_reciprocal_rank", "full_accuracy"),
    ]:
        if attacker in mrr_measures[dataset]:
            trir, mrr = _for_attacker(attacker)
            result[mrr_key] = mrr
            result[acc_key] = trir
    return result


INPUT_LOGS_PATH = Path("merged_logs.json")

_SCALE_BY_DUMMY: list[tuple[str, str]] = [
    ("P_exact",    "rel_P_exact"),
    ("P_more",     "rel_P_more"),
    ("P_full",     "rel_P_full"),
    ("TRIR_exact", "rel_TRIR_exact"),
    ("TRIR_more",  "rel_TRIR_more"),
    ("TRIR_full",  "rel_TRIR_full"),
    ("U_acc",      "rel_U_acc"),
    ("U_mae",      "rel_U_mae"),
    ("PU_acc",     "rel_PU_acc"),
    ("PU_mae",     "rel_PU_mae"),
]

_SCALE_BY_BASELINE: list[tuple[str, str]] = [
    ("D_bertscore",  "rel_D_bertscore"),
    ("D_cosine",     "rel_D_cosine"),
    ("D_pp",         "rel_D_pp"),
    ("T_anon_avg_s", "rel_T_anon_avg_s"),
]
OUTPUT_DIR = Path("mds")
REDDIT_UTILITY_OUTPUT_PATH = OUTPUT_DIR / "reddit_utility.json"
METHOD_SETS_CONFIG_PATH = Path("configs/visualize/methods.yaml")
DATASET_SETS_CONFIG_PATH = Path("configs/visualize/datasets.yaml")
PARAM_SETS_CONFIG_PATH = Path("configs/visualize/params.yaml")
PARAMS_MANIFEST_PATH = OUTPUT_DIR / "params_manifest.json"
TYPST_TABLES_MANIFEST_PATH = OUTPUT_DIR / "tables_manifest.json"
INCLUDE_GROUPED_ROWS = False
HIERARCHICAL_SPLIT_TABLES = True
ENABLED_DATASETS = {"db_bio", "tab", "rat_bench"}

dataset_lengths = {
    "db_bio": 2419,
    "tab": 1268,
    "rat_bench": 300,
}

mrr_measures = {
    "db_bio": ["full", "more", "exact"],
    "tab": ["full", "more", "exact"],
    "rat_bench": ["full"],
}

def read_logs(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


class LogMetricsEnricher:
    def __init__(self, data: list[dict[str, Any]]):
        self.data = data

    def aggregate(self) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for item in self.data:
            item_copy = dict(item)
            for section in ("utility", "pseudo_utility"):
                section_metrics = item_copy.get(section)
                if not isinstance(section_metrics, dict):
                    continue
                item_copy[section] = self._augment_utility_scores(
                    dict(section_metrics),
                    allow_missing_rmae=True,
                    dataset=item_copy.get("dataset"),
                )
            enriched.append(item_copy)
        return enriched

    def _extract_feature_counts(self, section_metrics: dict[str, Any], dataset: str) -> dict[str, int]:
        default_count = dataset_lengths.get(dataset)
        if default_count is None:
            raise ValueError(f"Unknown dataset: {dataset}")
        suffixes = ("_mae", "_rmae", "_acc", "_racc", "_count")
        features: set[str] = set()
        for metric_name in section_metrics.keys():
            if not isinstance(metric_name, str):
                continue
            for suffix in suffixes:
                if metric_name.endswith(suffix):
                    feature = metric_name[: -len(suffix)]
                    if feature and "_baseline" not in feature and "_dummy" not in feature:
                        features.add(feature)
                    break

        feature_counts: dict[str, int] = {}
        for feature in sorted(features):
            raw_count = section_metrics.get(f"{feature}_count")
            if isinstance(raw_count, (int, float)) and raw_count > 0:
                feature_counts[feature] = int(raw_count)
            else:
                feature_counts[feature] = default_count
        return feature_counts

    def _augment_utility_scores(
        self,
        utility_metrics: dict[str, Any],
        allow_missing_rmae: bool = False,
        dataset: str = None,
    ) -> dict[str, Any]:
        feature_counts = self._extract_feature_counts(utility_metrics, dataset)
        weighted_sum = 0.0
        weighted_count = 0
        ordinal_weighted_sum = 0.0
        ordinal_weighted_count = 0
        nominal_weighted_sum = 0.0
        nominal_weighted_count = 0
        ordinal_raw_sum = 0.0
        ordinal_raw_count = 0
        nominal_raw_sum = 0.0
        nominal_raw_count = 0
        for feature, count in feature_counts.items():
            if not feature:
                continue
            score: float | None = None
            mae_key = f"{feature}_mae"
            rmae_key = f"{feature}_rmae"
            acc_key = f"{feature}_acc"
            racc_key = f"{feature}_racc"
            if mae_key in utility_metrics:
                mae_value = float(utility_metrics[mae_key])
                ordinal_raw_sum += mae_value * count
                ordinal_raw_count += count
                if rmae_key in utility_metrics:
                    score = float(utility_metrics[rmae_key])
                elif not allow_missing_rmae:
                    raise ValueError(
                        f"Feature '{feature}' has mae but missing ordinal utility score '{rmae_key}'"
                    )
                if score is not None:
                    ordinal_weighted_sum += score * count
                    ordinal_weighted_count += count
            elif acc_key in utility_metrics:
                acc_value = float(utility_metrics[acc_key])
                nominal_raw_sum += acc_value * count
                nominal_raw_count += count
                if racc_key in utility_metrics:
                    score = float(utility_metrics[racc_key])
                elif not allow_missing_rmae:
                    raise ValueError(
                        f"Feature '{feature}' has acc but missing nominal utility score '{racc_key}'"
                    )
                if score is not None:
                    nominal_weighted_sum += score * count
                    nominal_weighted_count += count
            else:
                if not allow_missing_rmae:
                    raise ValueError(
                        f"Feature '{feature}' has neither ordinal mae nor nominal acc metric"
                    )
                continue
            if score is None:
                continue
            utility_metrics[f"{feature}_score"] = score
            weighted_sum += score * count
            weighted_count += count
        if weighted_count > 0:
            utility_metrics["utility_weighted_score"] = weighted_sum / weighted_count
            utility_metrics["utility_total_score"] = utility_metrics["utility_weighted_score"]
            utility_metrics["_utility_score_count"] = weighted_count
        if ordinal_weighted_count > 0:
            utility_metrics["utility_ordinal_weighted_score"] = ordinal_weighted_sum / ordinal_weighted_count
            utility_metrics["_utility_ordinal_score_count"] = ordinal_weighted_count
        if nominal_weighted_count > 0:
            utility_metrics["utility_nominal_weighted_score"] = nominal_weighted_sum / nominal_weighted_count
            utility_metrics["_utility_nominal_score_count"] = nominal_weighted_count
        if ordinal_raw_count > 0:
            utility_metrics["utility_ordinal_raw_mae"] = ordinal_raw_sum / ordinal_raw_count
            utility_metrics["_utility_ordinal_raw_count"] = ordinal_raw_count
        if nominal_raw_count > 0:
            utility_metrics["utility_nominal_raw_acc"] = nominal_raw_sum / nominal_raw_count
            utility_metrics["_utility_nominal_raw_count"] = nominal_raw_count
        return utility_metrics

class RedditUtilityByHardness:
    def __init__(self, data: list[dict[str, Any]]):
        self.data = data
        self.filtered_data = self.filter()
        self.grouped_data = self.group()

    def filter(self):
        filtered_data = []
        for item in self.data:
            if item['dataset'] == 'reddit' and 'group' in item and 'utility' in item:
                filtered_data.append(item)
        return filtered_data

    def group(self):
        groups = {}
        for item in self.filtered_data:
            group = item['group']
            metrics = item['utility']
            method_and_params = item["method"]
            if item["params"]:
                method_and_params += "?" + "&".join(f"{k}={v}" for k, v in item["params"].items())
            for feature, metric_value in metrics.items():
                if feature.startswith("_"):
                    feature = feature[1:]
                keys = ["macro_f1", "f1", "macro_mae", "mae", "acc", "count"]
                for key in keys:
                    if feature.endswith("_" + key):
                        feature = feature[:-len(key) - 1]
                        groups.setdefault(group, {}).setdefault(feature, {}).setdefault(key, {}).setdefault(method_and_params, metric_value)
                        break
        return groups


class FlatDatasetLogs:
    def __init__(self, data: list[dict[str, Any]], include_grouped: bool = False):
        self.data = data
        self.include_grouped = include_grouped
        self.filtered_data = self.filter()
        self.flattened_data = self.flatten()

    def filter(self) -> list[dict[str, Any]]:
        filtered_data: list[dict[str, Any]] = []
        for item in self.data:
            if item.get("dataset") not in ENABLED_DATASETS:
                continue
            has_group = "group" in item
            if self.include_grouped or not has_group:
                filtered_data.append(item)
        return filtered_data

    def flatten(self) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for item in self.filtered_data:
            params = item["params"] or {}
            if not isinstance(params, dict):
                params = {}
            flat_item: dict[str, Any] = {
                "dataset": item["dataset"],
                "method": item["method"],
                "params": order_params_dict(params),
                "split": item.get("split"),
            }
            for section in ("privacy", "utility", "pseudo_utility", "divergence", "runtime"):
                for metric, value in item.get(section, {}).items():
                    flat_item[f"{section}_{metric}"] = value
            if item["method"] == "baseline":
                flat_item["divergence_bertscore"] = 0.0
                flat_item["divergence_cosine"] = 0.0
                flat_item["runtime_avg_anon_time_s"] = 0.0
            if item["method"] == "manual":
                flat_item["runtime_avg_anon_time_s"] = float("nan")
            if item["method"] == "dummy":
                for metric, value in _compute_dummy_privacy(item["dataset"]).items():
                    flat_item[f"privacy_{metric}"] = value
                flat_item["divergence_bertscore"] = 1.0
                flat_item["divergence_cosine"] = 1.0
            flattened.append(flat_item)
        sorted_flattened = self.sort_by_method_order(flattened)
        return sorted_flattened

    def by_dataset(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in self.flattened_data:
            grouped.setdefault(item["dataset"], []).append(item)
        return grouped
    
    def sort_by_method_order(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        param_specs = read_param_specs_config(PARAM_SETS_CONFIG_PATH)
        param_print_order = {
            spec["name"]: spec.get("print_order", "low_to_high")
            for spec in param_specs
        }
        method_order = [
            "baseline",
            "dummy",
            "presidio",
            "spacy",
            "manual",
            "baroud",
            "petre_shap",
            "risk_shap",
            "dpbart",
            "dpparaphrase",
            "dpprompt",
            "dpmlm_uniform",
            "dpmlm_shap"
        ]
        for item in items:
            method = item["method"]
            params = item["params"] or {}
            method_index = method_order.index(method) if method in method_order else len(method_order)

            ordered_params = sorted(params.items(), key=lambda kv: kv[0].lower())
            normalized_params: list[tuple[str, tuple[int, int | float | str]]] = []
            for param_name, raw_value in ordered_params:
                value = sortable_param_value(raw_value)
                if isinstance(value, (int, float)) and param_print_order.get(param_name) == "high_to_low":
                    value = -value
                value_type_rank = 0 if isinstance(value, (int, float)) else 1
                normalized_params.append((param_name, (value_type_rank, value)))

            item["_sort_key"] = (method_index, tuple(normalized_params))

        sorted_items = sorted(items, key=lambda x: x["_sort_key"])
        for item in sorted_items:
            del item["_sort_key"]
        return sorted_items


class MarkdownDatasetLogsWriter:
    def __init__(
        self,
        data: list[dict[str, Any]],
        output_dir: str | Path = ".",
        include_grouped: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flat_logs = FlatDatasetLogs(data, include_grouped=include_grouped)
        self.dataset_projection_config = self._read_dataset_projection_config(DATASET_SETS_CONFIG_PATH)

    def _read_dataset_projection_config(self, path: str | Path) -> dict[str, list[dict[str, Any]]]:
        return read_dataset_projection_config(path)

    def _dataset_frames(self) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for dataset, logs in self.flat_logs.by_dataset().items():
            frames[dataset] = pd.DataFrame(logs)
        return frames

    def _write_frame(self, frame: pd.DataFrame, base_path: Path) -> list[Path]:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path = base_path.with_suffix(".md")
        csv_path = base_path.with_suffix(".csv")
        json_path = base_path.with_suffix(".json")
        with open(markdown_path, "w") as f:
            f.write(frame.to_markdown(index=False) + "\n")
        frame.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(
                {
                    "columns": list(frame.columns),
                    "rows": frame.to_dict(orient="records"),
                },
                f,
                indent=2,
            )
        return [markdown_path, csv_path, json_path]

    def _project_method_set_frame(self, dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
        scores = self.dataset_projection_config.get(dataset)
        required_columns = ["method", "params", "split"]
        columns: list[str] = list(required_columns)
        rename_map: dict[str, str] = {}
        for score in scores or []:
            score_name = score["name"]
            if score_name not in frame.columns:
                frame[score_name] = None
            columns.append(score_name)
            rename_map[score_name] = score.get("rename_as", score.get("print_as", score_name))

        missing_required = [column for column in required_columns if column not in frame.columns]
        if missing_required:
            raise ValueError(f"Missing required columns for dataset '{dataset}' projection: {missing_required}")

        projected = frame[columns].copy()
        projected = projected.rename(columns=rename_map)
        for score in scores or []:
            renamed_col = rename_map.get(score["name"], score["name"])
            exclude_methods = score.get("exclude", [])
            if exclude_methods and renamed_col in projected.columns:
                mask = projected["method"].isin(exclude_methods)
                projected.loc[mask, renamed_col] = None
        return projected

    def write_dataset_tables(self) -> list[Path]:
        written: list[Path] = []
        for dataset, frame in self._dataset_frames().items():
            written.extend(self._write_frame(frame, self.output_dir / dataset / "logs"))
        return written

    def write_method_set_tables(self, method_sets_file: str | Path) -> list[Path]:
        with open(method_sets_file, "r") as f:
            config = yaml.safe_load(f)
        method_sets = config.get("method_sets", [])

        written: list[Path] = []
        for dataset, logs in self.flat_logs.by_dataset().items():
            for method_set in method_sets:
                if not self._dataset_enabled_for_method_set(dataset, method_set):
                    continue
                set_name = method_set["name"]
                methods = method_set.get("methods", [])
                filtered_logs = [item for item in logs if self._matches_any_method_spec(item, methods)]
                if not filtered_logs:
                    continue
                filtered_logs = self._sort_by_method_set_order(filtered_logs, methods)
                frame = pd.DataFrame(filtered_logs)
                frame = self._project_method_set_frame(dataset, frame)
                written.extend(self._write_frame(frame, self.output_dir / dataset / set_name))
        return written

    def write_meta_tables(self, method_sets_file: str | Path) -> list[Path]:
        defs = parse_meta_score_defs(DATASET_SETS_CONFIG_PATH)
        engine = MetaScoreEngine(defs)
        meta_cols = [c for c in engine.meta_columns() if not c.startswith("rel_")]
        keep = ["method", "params", "split"] + meta_cols + ["GAIN", "relative_gain"]

        with open(method_sets_file, "r") as f:
            config = yaml.safe_load(f)
        method_sets = config.get("method_sets", [])

        written: list[Path] = []
        for dataset, raw_frame in self._dataset_frames().items():
            projected = self._project_method_set_frame(dataset, raw_frame)
            baseline_raw = self._extract_baseline(projected)
            dummy_raw = self._extract_dummy(projected)
            projected = self._add_relative_columns(projected, baseline_raw, dummy_raw)
            computed = engine.compute(projected)
            baseline_computed = self._extract_baseline(computed)
            meta_frame = self._add_relative_gain(self._add_gain(computed), baseline_computed)
            self._log_resolved_args(engine, dataset, "meta_logs")
            self._write_meta_args_manifest(engine, dataset)
            meta_only = meta_frame[[c for c in keep if c in meta_frame.columns]]
            written.extend(self._write_frame(meta_only, self.output_dir / dataset / "meta_logs"))
            mega = self._mean_per_method(meta_only)
            written.extend(self._write_frame(mega, self.output_dir / dataset / "mega_meta"))

            for method_set in method_sets:
                if not self._dataset_enabled_for_method_set(dataset, method_set):
                    continue
                set_name = method_set["name"]
                methods = method_set.get("methods", [])
                logs = self.flat_logs.by_dataset().get(dataset, [])
                filtered_logs = [item for item in logs if self._matches_any_method_spec(item, methods)]
                if not filtered_logs:
                    continue
                filtered_logs = self._sort_by_method_set_order(filtered_logs, methods)
                frame = pd.DataFrame(filtered_logs)
                projected_set = self._project_method_set_frame(dataset, frame)
                projected_set = self._add_relative_columns(projected_set, baseline_raw, dummy_raw)
                computed_set = engine.compute(projected_set)
                baseline_computed_set = self._extract_baseline(computed_set)
                meta_set = self._add_relative_gain(self._add_gain(computed_set), baseline_computed_set)
                self._log_resolved_args(engine, dataset, set_name)
                meta_set_only = meta_set[[c for c in keep if c in meta_set.columns]]
                written.extend(self._write_frame(meta_set_only, self.output_dir / dataset / f"meta_{set_name}"))
        return written

    @staticmethod
    def _extract_baseline(frame: pd.DataFrame) -> pd.Series | None:
        mask = (frame["method"] == "baseline") & frame["split"].isna()
        rows = frame[mask]
        return rows.iloc[0] if not rows.empty else None

    @staticmethod
    def _extract_dummy(frame: pd.DataFrame) -> pd.Series | None:
        mask = (frame["method"] == "dummy") & frame["split"].isna()
        rows = frame[mask]
        return rows.iloc[0] if not rows.empty else None

    @staticmethod
    def _add_relative_columns(
        frame: pd.DataFrame,
        baseline: pd.Series | None,
        dummy: pd.Series | None,
    ) -> pd.DataFrame:
        if baseline is None or dummy is None:
            return frame
        frame = frame.copy()

        def _fval(s: pd.Series, col: str) -> float:
            v = s.get(col)
            return float("nan") if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

        for raw, rel in _SCALE_BY_DUMMY:
            if raw not in frame.columns:
                continue
            b, d = _fval(baseline, raw), _fval(dummy, raw)
            denom = b - d
            if math.isnan(b) or math.isnan(d) or abs(denom) < 1e-9:
                frame[rel] = float("nan")
            else:
                frame[rel] = (frame[raw] - d) / denom

        for raw, rel in _SCALE_BY_BASELINE:
            if raw not in frame.columns:
                continue
            b, d = _fval(baseline, raw), _fval(dummy, raw)
            denom = d - b
            if math.isnan(b) or math.isnan(d) or abs(denom) < 1e-9:
                frame[rel] = float("nan")
            else:
                frame[rel] = (frame[raw] - b) / denom

        return frame

    @staticmethod
    def _add_gain(frame: pd.DataFrame) -> pd.DataFrame:
        def _val(row: pd.Series, col: str) -> float:
            v = row.get(col)
            return float("nan") if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

        def _gain_row(row: pd.Series) -> float:
            rel_p = _val(row, "rel_P")
            rel_u = _val(row, "rel_U")
            if any(math.isnan(v) for v in (rel_p, rel_u)):
                return float("nan")
            return rel_u - rel_p

        frame = frame.copy()
        frame["GAIN"] = frame.apply(_gain_row, axis=1)
        return frame

    @staticmethod
    def _add_relative_gain(frame: pd.DataFrame, baseline: pd.Series | None) -> pd.DataFrame:
        def _fval(s: pd.Series, col: str) -> float:
            v = s.get(col)
            return float("nan") if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

        frame = frame.copy()
        if baseline is None:
            frame["relative_gain"] = float("nan")
            return frame

        u0 = _fval(baseline, "U")
        p0 = _fval(baseline, "P")

        def _rg_row(row: pd.Series) -> float:
            if any(math.isnan(v) for v in (u0, p0)) or abs(u0) < 1e-9 or abs(p0) < 1e-9:
                return float("nan")
            u = _fval(row, "U")
            p = _fval(row, "P")
            if math.isnan(u) or math.isnan(p):
                return float("nan")
            return u / u0 - p / p0

        frame["relative_gain"] = frame.apply(_rg_row, axis=1)
        return frame

    @staticmethod
    def _mean_per_method(meta_frame: pd.DataFrame) -> pd.DataFrame:
        main = meta_frame[
            meta_frame["split"].isna() | (meta_frame["split"] == "")
        ].copy()
        valid = main.dropna(subset=["GAIN"])
        if valid.empty:
            return valid
        numeric = valid.select_dtypes(include="number").columns.tolist()
        agg = valid.groupby("method")[numeric].mean().reset_index()
        return agg.sort_values("GAIN", ascending=False).reset_index(drop=True)

    def _write_meta_args_manifest(self, engine: "MetaScoreEngine", dataset: str) -> None:
        non_exact: dict[str, dict[str, Any]] = {}
        defs_by_output: dict[str, list[Any]] = {}
        for d in engine._defs:
            defs_by_output.setdefault(d.output, []).append(d)

        for output, args in engine.resolved_args.items():
            defs_for_output = defs_by_output.get(output, [])
            if not any(d.strategy != "exact_set" for d in defs_for_output):
                continue
            formula = self._meta_formula_for_output(
                output=output,
                defs_by_output=defs_by_output,
                resolved_args=engine.resolved_args,
            )
            non_exact[output] = {"args": args, "formula": formula}

        path = self.output_dir / dataset / "meta_args_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(non_exact, f, indent=2)

    def _meta_formula_for_output(
        self,
        output: str,
        defs_by_output: dict[str, list[Any]],
        resolved_args: dict[str, list[str]],
    ) -> str:
        expr = self._meta_expr_for_output(output, defs_by_output, resolved_args, set())
        return f"{output} = {expr}"

    def _meta_expr_for_output(
        self,
        output: str,
        defs_by_output: dict[str, list[Any]],
        resolved_args: dict[str, list[str]],
        stack: set[str],
    ) -> str:
        if output in stack:
            return output
        defs_for_output = defs_by_output.get(output, [])
        if not defs_for_output:
            return output

        d = next((item for item in defs_for_output if item.split is None), defs_for_output[0])
        args = tuple(resolved_args.get(output, [])) or d.args
        stack.add(output)
        rendered_args: list[str] = []
        for arg in args:
            if arg in defs_by_output:
                rendered_args.append(f"({self._meta_expr_for_output(arg, defs_by_output, resolved_args, stack)})")
            else:
                rendered_args.append(arg)
        stack.remove(output)

        if d.op == "avg":
            return f"avg({', '.join(rendered_args)})"
        if d.op == "sum":
            return " + ".join(rendered_args)
        if d.op == "sub":
            return " - ".join(rendered_args)
        if d.op == "div":
            return " / ".join(rendered_args)
        return f"{d.op}({', '.join(rendered_args)})"

    def _log_resolved_args(
        self, engine: "MetaScoreEngine", dataset: str, table: str
    ) -> None:
        defs_by_output: dict[str, list[Any]] = {}
        for d in engine._defs:
            defs_by_output.setdefault(d.output, []).append(d)
        for output, args in engine.resolved_args.items():
            defs_for_output = [
                d for d in engine._defs if d.output == output and d.strategy != "exact_set"
            ]
            if defs_for_output:
                formula = self._meta_formula_for_output(
                    output=output,
                    defs_by_output=defs_by_output,
                    resolved_args=engine.resolved_args,
                )
                print(f"[meta] {dataset}/{table}: {formula}")

    def _sort_by_method_set_order(
        self, items: list[dict[str, Any]], method_specs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        def spec_index(item: dict[str, Any]) -> int:
            for i, spec in enumerate(method_specs):
                if self._matches_method_spec(item, spec):
                    return i
            return len(method_specs)

        return sorted(items, key=spec_index)

    def _matches_any_method_spec(
        self,
        item: dict[str, Any],
        method_specs: list[dict[str, Any]],
    ) -> bool:
        return any(self._matches_method_spec(item, method_spec) for method_spec in method_specs)

    def _dataset_enabled_for_method_set(self, dataset: str, method_set: dict[str, Any]) -> bool:
        return self._dataset_matches_scope(dataset, method_set.get("datasets"))

    def _dataset_enabled_for_method_spec(self, dataset: str, method_spec: dict[str, Any]) -> bool:
        return self._dataset_matches_scope(dataset, method_spec.get("datasets"))

    def _dataset_matches_scope(self, dataset: str, datasets_scope: Any) -> bool:
        if datasets_scope is None:
            return True
        if isinstance(datasets_scope, list):
            return dataset in datasets_scope
        raise ValueError(f"Unsupported datasets scope format: {type(datasets_scope)}")

    def _matches_method_spec(self, item: dict[str, Any], method_spec: dict[str, Any]) -> bool:
        if not self._dataset_enabled_for_method_spec(item["dataset"], method_spec):
            return False
        expected_method = method_spec["method"]
        if item["method"] != expected_method:
            return False

        split_spec = method_spec.get("split")
        item_split = item.get("split")
        if split_spec is not None and item_split is not None:
            if split_spec != item_split:
                return False
        elif split_spec is not None or item_split is not None:
            return False

        params = item["params"] or {}
        if not isinstance(params, dict):
            return False

        params_one_run = method_spec.get("params_one_run", [])
        params_one_run_keys = self._params_one_run_keys(params_one_run)
        expected_param_keys = set(method_spec.get("params", [])) | params_one_run_keys
        if set(params.keys()) != expected_param_keys:
            return False

        return self._matches_params_one_run_values(params, params_one_run)

    def _params_one_run_keys(self, params_one_run: Any) -> set[str]:
        if isinstance(params_one_run, dict):
            return set(params_one_run.keys())
        if isinstance(params_one_run, list):
            return set(params_one_run)
        raise ValueError(f"Unsupported params_one_run format: {type(params_one_run)}")

    def _matches_params_one_run_values(self, params: dict[str, Any], params_one_run: Any) -> bool:
        if isinstance(params_one_run, list):
            return True
        if not isinstance(params_one_run, dict):
            raise ValueError(f"Unsupported params_one_run format: {type(params_one_run)}")

        for param_name, allowed_values in params_one_run.items():
            if param_name not in params:
                return False
            if not isinstance(allowed_values, list):
                raise ValueError(
                    f"params_one_run values must be lists. Got {param_name}={type(allowed_values)}"
                )
            if not any(self._values_equal(params[param_name], allowed_value) for allowed_value in allowed_values):
                return False
        return True

    def _values_equal(self, left: Any, right: Any) -> bool:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return float(left) == float(right)
        return left == right


def build_typst_tables_manifest(output_dir: Path) -> list[dict[str, Any]]:
    dataset_projection_config = read_dataset_projection_config(DATASET_SETS_CONFIG_PATH)
    dataset_print_labels = read_dataset_print_labels_config(DATASET_SETS_CONFIG_PATH)
    entries: list[dict[str, str]] = []
    for path in sorted(output_dir.glob("*/*.md")):
        dataset = path.parent.name
        section = path.stem
        if dataset not in ENABLED_DATASETS:
            continue
        if section == "logs":
            continue
        heatmap_columns = []
        for score in dataset_projection_config.get(dataset, []):
            heatmap_columns.append(
                {
                    "name": score.get("rename_as", score.get("print_as", score["name"])),
                    "source_name": score["name"],
                    "best": score.get("best"),
                    "worst": score.get("worst"),
                }
            )
        json_path = path.with_suffix(".json")
        if json_path.exists():
            with open(json_path, "r") as f:
                table_json = json.load(f)
            if "relative_gain" in table_json.get("columns", []):
                heatmap_columns.append(
                    {
                        "name": "relative_gain",
                        "source_name": "relative_gain",
                        "best": 1,
                        "worst": -1,
                    }
                )
        entries.append(
            {
                "file": str(path.relative_to(output_dir)),
                "csv_file": str(path.with_suffix(".csv").relative_to(output_dir)),
                "json_file": str(path.with_suffix(".json").relative_to(output_dir)),
                "dataset": dataset,
                "section": section,
                "title": typst_table_title(dataset, section, dataset_print_labels),
                "heatmap_columns": heatmap_columns,
                "render_mode": "hierarchical" if HIERARCHICAL_SPLIT_TABLES else "flat",
            }
        )
    return entries


def typst_table_title(dataset: str, section: str, dataset_print_labels: dict[str, str] | None = None) -> str:
    dataset_label = (dataset_print_labels or {}).get(dataset, dataset.replace("_", "-").upper())
    if section == "all":
        return f"{dataset_label} results (all methods)"
    return f"{dataset_label} results ({section.replace('_', ' ')})"


def _dataset_name_entries(dataset_set: dict[str, Any]) -> list[tuple[str, str | None]]:
    entries = dataset_set.get("names")
    if isinstance(entries, list):
        result: list[tuple[str, str | None]] = []
        for entry in entries:
            if isinstance(entry, str):
                result.append((entry, None))
                continue
            if isinstance(entry, dict):
                dataset_name = entry.get("name")
                dataset_print_as = entry.get("print_as")
                if isinstance(dataset_name, str):
                    result.append((dataset_name, dataset_print_as if isinstance(dataset_print_as, str) else None))
        if result:
            return result
    dataset_name = dataset_set.get("name")
    dataset_print_as = dataset_set.get("print_as")
    if isinstance(dataset_name, str):
        return [(dataset_name, dataset_print_as if isinstance(dataset_print_as, str) else None)]
    return []


def read_dataset_print_labels_config(path: str | Path) -> dict[str, str]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    labels: dict[str, str] = {}
    for dataset_set in config.get("dataset_sets", []):
        for dataset_name, dataset_print_as in _dataset_name_entries(dataset_set):
            if dataset_print_as:
                labels[dataset_name] = dataset_print_as
    return labels


def read_dataset_projection_config(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    projections: dict[str, list[dict[str, Any]]] = {}
    for dataset_set in config.get("dataset_sets", []):
        for dataset_name, _ in _dataset_name_entries(dataset_set):
            projections[dataset_name] = dataset_set.get("scores", [])
    return projections


def read_param_specs_config(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("param_sets", [])


def order_params_dict(params: dict[str, Any]) -> dict[str, Any]:
    ordered_keys = sorted(params.keys(), key=lambda key: key.lower())
    return {key: params[key] for key in ordered_keys}


def sortable_param_value(value: Any) -> int | float | str:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    try:
        if "." in str(value):
            return float(value)
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = LogMetricsEnricher(read_logs(INPUT_LOGS_PATH)).aggregate()
    data = [item for item in data if item.get("dataset") in ENABLED_DATASETS]

    writer = MarkdownDatasetLogsWriter(
        data,
        output_dir=OUTPUT_DIR,
        include_grouped=INCLUDE_GROUPED_ROWS,
    )
    writer.write_dataset_tables()
    writer.write_method_set_tables(METHOD_SETS_CONFIG_PATH)
    writer.write_meta_tables(METHOD_SETS_CONFIG_PATH)
    with open(PARAMS_MANIFEST_PATH, "w") as f:
        json.dump(read_param_specs_config(PARAM_SETS_CONFIG_PATH), f, indent=2)
    with open(TYPST_TABLES_MANIFEST_PATH, "w") as f:
        json.dump(build_typst_tables_manifest(OUTPUT_DIR), f, indent=2)
