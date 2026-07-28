"""POC: verify the RAT-Bench adapter (flattened metadata, identifier spans as
TextAnnotations, cross-difficulty identity clustering, deterministic names)
and the derived persona utility targets.
"""

from collections import defaultdict

from dp.loaders._ratbench import RatBenchDatasetAdapter
from dp.loaders.derive import DERIVE_REGISTRY, get_ordinal_label_order

adapter = RatBenchDatasetAdapter(data="rat_bench", data_in="data/rat_bench")
records = list(adapter.iter_records())
print(f"loaded {len(records)} records")

# --- structural sanity: uid uniqueness, metadata flattening, spans ---
uids = [r.uid for r in records]
assert len(set(uids)) == len(uids), "uid collision"
print(f"uid unique: {len(set(uids))}/{len(uids)}")

for r in records[:3]:
    print("\n---", r.uid, "---")
    print("name:", r.name)
    print("metadata:", r.metadata)
    print("spans:")
    for span in r.spans or []:
        print(" ", span)

# a record with no direct identifier (falls back to profile descriptor for name)
no_direct = next(r for r in records if not any(s.metadata.get("category") == "direct" for s in (r.spans or [])))
print("\n--- no direct identifier example ---")
print("name:", no_direct.name)

# --- identity clustering: same id, split into >1 name (e.g. difficulty 3 mismatch) ---
by_root_id = defaultdict(list)
for r in records:
    root_id = r.uid.rsplit("_", 1)[0]
    by_root_id[root_id].append(r)

split_examples = [(rid, rs) for rid, rs in by_root_id.items() if len({r.name for r in rs}) > 1]
merged_examples = [(rid, rs) for rid, rs in by_root_id.items() if len({r.name for r in rs}) == 1 and len(rs) > 1]
print(f"\nids with a genuine identity split: {len(split_examples)}")
print(f"ids correctly merged into one identity: {len(merged_examples)}")

rid, rs = split_examples[0]
print(f"\n--- split example (raw id={rid}) ---")
for r in rs:
    print(f"  uid={r.uid} difficulty={r.metadata['difficulty']} name={r.name!r}")

rid, rs = merged_examples[0]
print(f"\n--- merged example (raw id={rid}) ---")
for r in rs:
    print(f"  uid={r.uid} difficulty={r.metadata['difficulty']} name={r.name!r}")

# --- utility getters: run every registered rat_bench getter over all records ---
print("\n--- utility target coverage ---")
for key, getter in DERIVE_REGISTRY["rat_bench"].items():
    values = [getter(r) for r in records]
    non_null = [v for v in values if v is not None]
    distinct = sorted(set(non_null), key=str)
    order = get_ordinal_label_order("rat_bench", key)
    print(f"{key:28s} coverage={len(non_null):3d}/{len(records)}  classes={len(distinct)}"
          + (f"  order={order}" if order else ""))
    if order:
        missing = set(non_null) - set(order)
        assert not missing, f"{key}: values outside declared order: {missing}"
