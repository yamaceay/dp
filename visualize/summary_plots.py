from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml
import json
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from visualize.config_loader import (
	load_methods_set as _ext_load_methods_set,
	load_results_set as _ext_load_results_set,
	validate_methods_config as _ext_validate_methods_config,
	validate_results_config as _ext_validate_results_config,
	list_method_sets as _ext_list_method_sets,
	list_result_sets as _ext_list_result_sets,
)


def _load_config(path: Path) -> Dict[str, Any]:
	with path.open("r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def _first_experiment(conf: Dict[str, Any]) -> Dict[str, Any]:
	experiments = conf.get("experiments") or []
	return experiments[0] if experiments else {}


def _select_experiment(conf: Dict[str, Any], name: str | None, dataset_hint: str | None) -> Dict[str, Any]:
	experiments = conf.get("experiments") or []
	if not experiments:
		return {}
	if name:
		for exp in experiments:
			if str(exp.get("name")) == str(name):
				return exp
	if dataset_hint:
		for exp in experiments:
			if str(exp.get("dataset")) == str(dataset_hint):
				return exp
	return experiments[0]


def _get_value_by_path(conf: Dict[str, Any], path_expr: str) -> Any:
	text = str(path_expr).strip()
	if not text.startswith("${") or not text.endswith("}"):
		return None
	body = text[2:-1]
	parts = [p for p in body.split(".") if p]
	cur: Any = conf
	for p in parts:
		if not isinstance(cur, dict) or p not in cur:
			return None
		cur = cur[p]
	return cur


def _expand_methods(methods: List[Any], exp: Dict[str, Any]) -> List[Any]:
	if not methods:
		return []
	conf_root = exp.get("__conf__") if isinstance(exp.get("__conf__"), dict) else None
	out: List[Any] = []
	for item in methods:
		if isinstance(item, str) and conf_root is not None:
			resolved = _get_value_by_path(conf_root, item)
			if isinstance(resolved, list) and resolved:
				out.extend(resolved)
				continue
		out.append(item)
	return out


def _method_specs(exp: Dict[str, Any]) -> List[Dict[str, Any]]:
	methods = exp.get("methods") or []
	methods = _expand_methods(methods, exp)
	out: List[Dict[str, Any]] = []
	for item in methods:
		if isinstance(item, dict) and item.get("method"):
			base = str(item.get("method"))
			across = [str(p) for p in (item.get("params") or [])]
			within = [str(p) for p in (item.get("params_one_run") or [])]
			print_as = str(item.get("print_as")) if item.get("print_as") else base
			expected_keys = sorted(set(across + within))
			spec_id = base + "(" + ",".join(expected_keys) + ")"
			out.append({"method": base, "params": across, "params_one_run": within, "expected_keys": expected_keys, "id": spec_id, "print_as": print_as})
		else:
			text = str(item)
			name = text.split("[")[0].split("(")[0].strip()
			across: List[str] = []
			within: List[str] = []
			if "[" in text and "]" in text:
				body = text[text.index("[") + 1 : text.index("]")]
				across.extend([p.strip() for p in body.split(",") if p.strip()])
			if "(" in text and ")" in text:
				body = text[text.index("(") + 1 : text.index(")")]
				within.extend([p.strip() for p in body.split(",") if p.strip()])
			expected_keys = sorted(set(across + within))
			spec_id = name + "(" + ",".join(expected_keys) + ")"
			out.append({"method": name, "params": across, "params_one_run": within, "expected_keys": expected_keys, "id": spec_id, "print_as": name})
	return out


def _params_scope(exp: Dict[str, Any]) -> Tuple[List[str], List[str]]:
	methods = _method_specs(exp)
	across: List[str] = []
	within: List[str] = []
	for m in methods:
		for p in m.get("params", []):
			if p not in across:
				across.append(p)
		for p in m.get("params_one_run", []):
			if p not in within:
				within.append(p)
	return within, across

def _load_methods_set(path: Path, set_name: str) -> List[Any]:
	return _ext_load_methods_set(path, set_name)

def _load_results_set(path: Path, set_name: str) -> Tuple[str, str]:
	return _ext_load_results_set(path, set_name)

	def _load_rows_from_config(config_path: Path) -> List[Dict[str, Any]]:
		conf = _load_config(config_path)
		exp = _first_experiment(conf)
		files = exp.get("files") or []
		rows: List[Dict[str, Any]] = []
		for f in files:
			ftype = str(f.get("type"))
			if ftype not in {"privacy", "utility"} or not f.get("file"):
				continue
			p = Path("visualize") / str(f.get("file")) if not Path(str(f.get("file"))).exists() else Path(str(f.get("file")))
			with p.open("r", encoding="utf-8") as fp:
				data = json.load(fp)
			if isinstance(data, list):
				rows.extend(data)
			elif isinstance(data, dict):
				for method_name, entries in data.items():
					for e in entries:
						row: Dict[str, Any] = {"method": method_name, "params": e.get("params") or {}, "dataset": exp.get("dataset")}
						for k, v in e.items():
							if k not in {"params"}:
								row[k] = v
						rows.append(row)
		return rows

def _method_alias(name: str) -> str:
	if name == "petre":
		return "petre_shap"
	return name

def _match_exact(row: Dict[str, Any], spec: Dict[str, Any]) -> bool:
	mname = _method_alias(spec["method"]) 
	if str(row.get("method")) != mname:
		return False
	keys = set((row.get("params") or {}).keys())
	expected = set(spec.get("expected_keys") or ((spec.get("params") or []) + (spec.get("params_one_run") or [])))
	return keys == expected


def _load_flat(path: Path) -> List[Dict[str, Any]]:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f) or []


def _filter_dataset(rows: Sequence[Dict[str, Any]], dataset: str, metric: str) -> List[Dict[str, Any]]:
	return [r for r in rows if str(r.get("dataset")) == dataset and metric in r]


def _unique_sorted(values: Sequence[Any]) -> List[Any]:
	seen = set()
	out: List[Any] = []
	for v in values:
		key = str(v)
		if key in seen:
			continue
		seen.add(key)
		out.append(v)
	def key_fn(x: Any) -> Tuple[int, float | str]:
		s = str(x)
		if s.replace(".", "", 1).isdigit():
			return (0, float(s))
		return (1, s)
	return sorted(out, key=key_fn)


def _param_label(params: Dict[str, Any], include: Sequence[str], exclude: Sequence[str]) -> str:
	names = [n for n in include if n in params and n not in exclude]
	if not names:
		return "baseline"
	pairs = [f"{n}={params[n]}" for n in names]
	return ",".join(pairs)


def _dedup_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
	seen: set[tuple] = set()
	out: List[Dict[str, Any]] = []
	for r in rows:
		method = _method_alias(str(r.get("method")))
		params = r.get("params") or {}
		key = (method, tuple(sorted((str(k), str(v)) for k, v in params.items())))
		if key in seen:
			continue
		seen.add(key)
		# normalize method name to alias for consistency
		rr = dict(r)
		rr["method"] = method
		out.append(rr)
	return out
def _ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def _method_order(methods: List[Dict[str, Any]]) -> List[str]:
	return [m["id"] for m in methods]


def _method_print_as(methods: List[Dict[str, Any]]) -> Dict[str, str]:
	return {m["id"]: m["print_as"] for m in methods}


def _method_palette(names: Sequence[str]) -> Dict[str, Any]:
	cmap = matplotlib.colormaps.get_cmap("tab20")
	size = max(1, len(names))
	return {n: cmap(i / size) for i, n in enumerate(names)}


def _value_palette(values: List[float], family: str) -> Dict[float, Any]:
	cmap_name = {"epsilon": "Blues", "k": "Oranges"}.get(family, "Greens")
	cmap = matplotlib.colormaps.get_cmap(cmap_name)
	uniq = _unique_sorted(values)
	if not uniq:
		return {}
	positions = [i / max(1, len(uniq) - 1) for i in range(len(uniq))]
	return {float(v): cmap(0.25 + 0.6 * p) for v, p in zip(uniq, positions)}


def _auto_method_specs_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	by_method: Dict[str, List[Dict[str, Any]]] = {}
	for r in rows:
		name = _method_alias(str(r.get("method")))
		by_method.setdefault(name, []).append(r)
	specs: List[Dict[str, Any]] = []
	for name, entries in by_method.items():
		keys = set()
		for e in entries:
			for k in (e.get("params") or {}).keys():
				keys.add(str(k))
		expected_keys = sorted(keys)
		spec_id = name + "(" + ",".join(expected_keys) + ")"
		specs.append({
			"method": name,
			"params": expected_keys,
			"params_one_run": [],
			"expected_keys": expected_keys,
			"id": spec_id,
			"print_as": name,
		})
	return specs


def plot_bars(config_path: Path, flat_path: Path | None, dataset: str, metric: str, experiment: str, output_dir: Path, debug: bool = False, methods_config: Path | None = None, methods_set: str | None = None, results_config: Path | None = None, results_set: str | None = None) -> None:
	if debug:
		print(f"[plot-bars] config={config_path} flat={flat_path} dataset={dataset} experiment={experiment} metric={metric} out={output_dir}")
	conf = _load_config(config_path)
	exp = _select_experiment(conf, experiment, dataset)
	if exp:
		exp = dict(exp)
		exp["__conf__"] = conf
	# Decoupled mode: override methods and results if provided
	decoupled_methods: List[Any] = []
	decoupled_flat: str = ""
	decoupled_dataset = ""
	if methods_config and methods_set:
		decoupled_methods = _load_methods_set(methods_config, methods_set)
	if results_config and results_set:
		decoupled_dataset, decoupled_flat = _load_results_set(results_config, results_set)
	if debug:
		print(f"[plot-bars] selected_exp dataset={exp.get('dataset')} name={experiment} decoupled_dataset={decoupled_dataset} methods_set={methods_set} results_set={results_set}")
	methods = _method_specs({"methods": decoupled_methods if decoupled_methods else (exp.get("methods") or []), "__conf__": exp.get("__conf__")})
	within, across = _params_scope({"methods": methods})
	if debug:
		print(f"[plot-bars] methods={len(methods)} ids={[m['id'] for m in methods]} within={within} across={across}")
	dataset_sel = dataset if dataset else (decoupled_dataset if decoupled_dataset else (str(exp.get("dataset")) if exp.get("dataset") else ""))
	flat_source = flat_path if flat_path is not None else None
	if decoupled_flat:
		candidate = Path("visualize/pretty") / decoupled_flat if not Path(decoupled_flat).exists() else Path(decoupled_flat)
		flat_source = candidate
	elif flat_source is None:
		raise ValueError("flat source not provided; supply --flat or --results-config/--results-set with a flat file")
	if debug:
		print(f"[plot-bars] dataset_sel={dataset_sel} flat_source={flat_source}")
	base_rows = _filter_dataset(_load_flat(Path(flat_source)), dataset_sel, metric)
	if debug:
		print(f"[plot-bars] base_rows={len(base_rows)} from flat")
	_ensure_dir(output_dir / f"{dataset_sel}_{experiment}")
	files: List[Dict[str, Any]] = [] if decoupled_flat else (exp.get("files") or [])
	alt_rows: List[Dict[str, Any]] = []
	for f in files:
		ftype = str(f.get("type"))
		if ftype not in {"privacy", "utility"} or not f.get("file"):
			continue
		p = Path("visualize/pretty") / str(f.get("file")) if not Path(str(f.get("file"))).exists() else Path(str(f.get("file")))
		if debug:
			print(f"[plot-bars] loading alt file type={ftype} path={p}")
		with p.open("r", encoding="utf-8") as fp:
			data = json.load(fp)
		if isinstance(data, list):
			alt_rows.extend(data)
		elif isinstance(data, dict):
			for method_name, entries in data.items():
				for e in entries:
					row_dataset = str(e.get("dataset")) if e.get("dataset") else dataset_sel
					row: Dict[str, Any] = {"method": method_name, "params": e.get("params") or {}, "dataset": row_dataset}
					for k, v in e.items():
						if k not in {"params"}:
							row[k] = v
					alt_rows.append(row)
	rows = _dedup_rows(base_rows + _filter_dataset(alt_rows, dataset_sel, metric))
	if debug:
		print(f"[plot-bars] alt_rows_raw={len(alt_rows)} alt_rows_filtered={len(_filter_dataset(alt_rows, dataset_sel, metric))} rows_dedup={len(rows)}")
	# Fallback: auto-generate method specs if configured methods do not match any rows
	has_matches = False
	if rows and methods:
		for spec in methods:
			if any(_match_exact(r, spec) for r in rows):
				has_matches = True
				break
	if debug:
		print(f"[plot-bars] has_matches={has_matches} configured_methods_present={bool(methods)}")
	if rows and (not methods or not has_matches):
		methods = _auto_method_specs_from_rows(rows)
		within, across = _params_scope({"methods": methods})
		if debug:
			print(f"[plot-bars] fallback_methods ids={[m['id'] for m in methods]} within={within} across={across}")
	order = _method_order(methods)
	print_as = _method_print_as(methods)
	eps_name = next((p for p in across if p == "epsilon"), None)
	k_palette = _value_palette([r.get("params", {}).get("k") for r in rows if "k" in (r.get("params") or {})], "k")
	method_colors = _method_palette(order)
	no_eps_rows: List[Dict[str, Any]] = []
	for spec in methods:
		if eps_name in (spec.get("params") or []):
			continue
		m_rows = [r for r in rows if _match_exact(r, spec)]
		for r in m_rows:
			r["__spec_id"] = spec["id"]
		no_eps_rows.extend(m_rows)
	if debug:
		print(f"[plot-bars] no_eps_rows={len(no_eps_rows)} output_dir={output_dir / f'{dataset_sel}_{experiment}'}")
	if no_eps_rows:
		grouped: Dict[str, List[Dict[str, Any]]] = {m["id"]: [] for m in methods}
		for r in no_eps_rows:
			grouped[str(r.get("__spec_id"))].append(r)
		names = [m["id"] for m in methods if grouped[m["id"]]]
		centers = {m: float(i) for i, m in enumerate(names)}
		width_total = 0.82
		figure, axis = plt.subplots(figsize=(max(12, int(0.55 * sum(max(1, len(grouped[m])) for m in names))), 7))
		xs: List[float] = []
		ys: List[float] = []
		cs: List[Any] = []
		ws: List[float] = []
		labs: List[str] = []
		for m in names:
			entries = grouped[m]
			count = max(1, len(entries))
			step = width_total / count
			start_x = centers[m] - width_total / 2 + step / 2
			for idx, r in enumerate(entries):
				params = r.get("params") or {}
				kval = params.get("k")
				color = k_palette.get(float(kval)) if kval is not None else method_colors[m]
				xs.append(start_x + idx * step)
				ys.append(float(r.get(metric)))
				cs.append(color)
				ws.append(step * 0.85)
				labs.append(_param_label(params, within, []))
		axis.bar(xs, ys, color=cs, width=ws, edgecolor="black", linewidth=0.9)
		axis.set_xticks([centers[m] for m in names])
		axis.set_xticklabels([print_as[m] for m in names], rotation=30, ha="right", fontsize=11)
		axis.set_ylabel(metric, fontsize=12)
		axis.set_title(f"{dataset_sel}_{experiment}: no_eps", fontsize=14)
		if ys:
			axis.set_ylim(0, max(ys) * 1.22)
		for xv, yv, t in zip(xs, ys, labs):
			axis.annotate(t, xy=(xv, yv), xytext=(0, 5), textcoords="offset points", rotation=90, ha="center", va="bottom", fontsize=9, clip_on=False)
		_ensure_dir(output_dir)
		figure.tight_layout()
		out_path = output_dir / f"{dataset_sel}_{experiment}" / "no_eps.png"
		figure.savefig(out_path, dpi=220)
		if debug:
			print(f"[plot-bars] wrote {out_path}")
		plt.close(figure)
	if eps_name:
		eps_values = _unique_sorted([r.get("params", {}).get(eps_name) for r in rows if eps_name in (r.get("params") or {})])
		if debug:
			print(f"[plot-bars] eps_name={eps_name} eps_values={eps_values}")
		for eps in eps_values:
			subset: List[Dict[str, Any]] = []
			for spec in methods:
				if eps_name not in (spec.get("params") or []):
					continue
				candidates = [r for r in rows if float((r.get("params") or {}).get(eps_name, float("nan"))) == float(eps)]
				candidates = [r for r in candidates if _match_exact(r, spec)]
				for r in candidates:
					r["__spec_id"] = spec["id"]
				subset.extend(candidates)
			if debug:
				print(f"[plot-bars] eps={eps} subset_rows={len(subset)}")
			if not subset:
				continue
			grouped: Dict[str, List[Dict[str, Any]]] = {m["id"]: [] for m in methods}
			for r in subset:
				grouped[str(r.get("__spec_id"))].append(r)
			names = [m["id"] for m in methods if grouped[m["id"]]]
			centers = {m: float(i) for i, m in enumerate(names)}
			width_total = 0.82
			figure, axis = plt.subplots(figsize=(max(12, int(0.55 * sum(max(1, len(grouped[m])) for m in names))), 7))
			xs: List[float] = []
			ys: List[float] = []
			cs: List[Any] = []
			ws: List[float] = []
			labs: List[str] = []
			for m in names:
				entries = grouped[m]
				count = max(1, len(entries))
				step = width_total / count
				start_x = centers[m] - width_total / 2 + step / 2
				for idx, r in enumerate(entries):
					params = r.get("params") or {}
					kval = params.get("k")
					color = k_palette.get(float(kval)) if kval is not None else method_colors[m]
					xs.append(start_x + idx * step)
					ys.append(float(r.get(metric)))
					cs.append(color)
					ws.append(step * 0.85)
					labs.append(_param_label(params, within, [eps_name]))
			axis.bar(xs, ys, color=cs, width=ws, edgecolor="black", linewidth=0.9)
			axis.set_xticks([centers[m] for m in names])
			axis.set_xticklabels([print_as[m] for m in names], rotation=30, ha="right", fontsize=11)
			axis.set_ylabel(metric, fontsize=12)
			axis.set_title(f"{dataset_sel}_{experiment}: {eps_name}={eps}", fontsize=14)
			if ys:
				axis.set_ylim(0, max(ys) * 1.22)
			for xv, yv, t in zip(xs, ys, labs):
				axis.annotate(t, xy=(xv, yv), xytext=(0, 5), textcoords="offset points", rotation=90, ha="center", va="bottom", fontsize=9, clip_on=False)
			_ensure_dir(output_dir)
			figure.tight_layout()
			safe_eps = str(eps).replace(".", "_")
			out_path = output_dir / f"{dataset_sel}_{experiment}" / f"eps_{safe_eps}.png"
			figure.savefig(out_path, dpi=220)
			if debug:
				print(f"[plot-bars] wrote {out_path}")
			plt.close(figure)


def build_results_string(config_path: Path, flat_path: Path, dataset: str, metric: str) -> str:
	conf = _load_config(config_path)
	exp = _select_experiment(conf, None, dataset)
	if exp:
		exp = dict(exp)
		exp["__conf__"] = conf
	methods = _method_specs(exp)
	within, across = _params_scope(exp)
	rows = _filter_dataset(_load_flat(flat_path), dataset, metric)
	eps_name = across[0] if across else None
	eps_values = _unique_sorted([r.get("params", {}).get(eps_name) for r in rows if eps_name and eps_name in (r.get("params") or {})]) if eps_name else []
	parts: List[str] = []
	no_eps_bits: List[str] = []
	for m in methods:
		mrows = [r for r in rows if str(r.get("method")) == m["method"] and eps_name not in (r.get("params") or {})]
		if not mrows:
			continue
		entries = []
		for r in mrows:
			params = r.get("params") or {}
			label = _param_label(params, within, [])
			entries.append(f"{label}=>{r.get(metric)}")
		no_eps_bits.append(f"{m['print_as']}:[{';'.join(entries)}]")
	if no_eps_bits:
		parts.append(f"no_eps={','.join(no_eps_bits)}")
	for eps in eps_values:
		eps_bits: List[str] = []
		for m in methods:
			mrows = [r for r in rows if str(r.get("method")) == m["method"] and float((r.get("params") or {}).get(eps_name, float('nan'))) == float(eps)]
			if not mrows:
				continue
			entries = []
			for r in mrows:
				params = r.get("params") or {}
				label = _param_label(params, within, [eps_name] if eps_name else [])
				entries.append(f"{label}=>{r.get(metric)}")
			eps_bits.append(f"{m['print_as']}:[{';'.join(entries)}]")
		parts.append(f"{eps_name}={eps}:{','.join(eps_bits)}")
	header = build_demo_string(config_path)
	return header + " | results=" + " | ".join(parts) if parts else header + " | results=none"


def build_demo_string(config_path: Path) -> str:
	conf = _load_config(config_path)
	exp = _first_experiment(conf)
	if exp:
		exp = dict(exp)
		exp["__conf__"] = conf
	dataset = str(exp.get("dataset")) if exp.get("dataset") else ""
	method_class = str(exp.get("method_class")) if exp.get("method_class") else ""
	methods = _method_specs(exp)
	within, across = _params_scope(exp)
	parts: List[str] = []
	parts.append(f"dataset={dataset}")
	parts.append(f"method_class={method_class}")
	parts.append(f"across_plots=[{','.join(across)}]")
	parts.append(f"within_plot=[{','.join(within)}]")
	method_bits = []
	for m in methods:
		params = ",".join(m.get("params") or [])
		method_bits.append(f"{m['method']}({params}) as '{m['print_as']}'")
	parts.append(f"methods=[{'; '.join(method_bits)}]")
	parts.append("labels=params only; colors=within_plot families; ordering=config order")
	return " | ".join(parts)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("action", nargs="?", choices=["demo", "demo-results", "plot-bars", "noop"], default="noop")
	parser.add_argument("--config")
	parser.add_argument("--flat")
	parser.add_argument("--dataset")
	parser.add_argument("--experiment")
	parser.add_argument("--metric")
	parser.add_argument("--out", default="visualize/plots")
	parser.add_argument("--debug", action="store_true")
	parser.add_argument("--methods-config")
	parser.add_argument("--methods-set")
	parser.add_argument("--results-config")
	parser.add_argument("--results-set")
	# Utility commands for config hygiene
	parser.add_argument("--list-method-sets", action="store_true")
	parser.add_argument("--list-result-sets", action="store_true")
	parser.add_argument("--validate-configs", action="store_true")
	args = parser.parse_args()
	# Optional utility operations
	if args.list_method_sets and args.methods_config:
		for name in _ext_list_method_sets(Path(args.methods_config)):
			print(name)
		return
	if args.list_result_sets and args.results_config:
		for name in _ext_list_result_sets(Path(args.results_config)):
			print(name)
		return
	if args.validate_configs:
		if args.methods_config:
			_ext_validate_methods_config(Path(args.methods_config))
			print("methods: ok")
		if args.results_config:
			_ext_validate_results_config(Path(args.results_config))
			print("results: ok")
		return

	if args.action == "demo":
		if not args.config:
			raise SystemExit("--config is required for demo")
		text = build_demo_string(Path(args.config))
		print(text)
		return
	if args.action == "demo-results":
		if not (args.config and args.flat and args.dataset and args.metric):
			raise SystemExit("--config, --flat, --dataset and --metric are required for demo-results")
		text = build_results_string(Path(args.config), Path(args.flat), str(args.dataset), str(args.metric))
		print(text)
		return
	if args.action == "plot-bars":
		if not (args.config and args.dataset and args.experiment and args.metric):
			raise SystemExit("--config, --dataset, --experiment and --metric are required for plot-bars")
		plot_bars(
			Path(args.config),
			Path(args.flat) if args.flat else None,
			str(args.dataset),
			str(args.metric),
			str(args.experiment),
			Path(args.out),
			debug=bool(args.debug),
			methods_config=Path(args.methods_config) if args.methods_config else None,
			methods_set=str(args.methods_set) if args.methods_set else None,
			results_config=Path(args.results_config) if args.results_config else None,
			results_set=str(args.results_set) if args.results_set else None,
		)
		return
	if args.action == "noop":
		if not (args.list_method_sets or args.list_result_sets or args.validate_configs):
			raise SystemExit("no action provided; use an action or utility flags")


if __name__ == "__main__":
	main()

