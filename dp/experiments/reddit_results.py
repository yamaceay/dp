from typing import Iterable
import json
import pandas as pd
from dp.loaders import get_adapter
from dp.loaders.base import DatasetRecord
from dp.loaders.derive import get_getter, ORDINAL_GROUPERS, NOMINAL_GROUPERS

def get_level(feature: str, label: str) -> int:
    if feature in ORDINAL_GROUPERS:
        _, levels = ORDINAL_GROUPERS[feature]
        if label in levels:
            return levels.index(label)
    raise ValueError(f"Unknown feature or label: {feature}, {label}")

def make_inference_on_reddit(records: Iterable[DatasetRecord]) -> pd.DataFrame:
    df = pd.DataFrame([{
        "feature": get_getter("reddit", "feature")(record),
        "label": get_getter("reddit", "label")(record),
        "hardness": get_getter("reddit", "hardness")(record),
    } for record in records])

    if not df["feature"].isin(set(ORDINAL_GROUPERS.keys()).union(set(NOMINAL_GROUPERS.keys()))).all():
        raise ValueError("Unknown features found in the dataset.")
    
    ordinal_df = df[df["feature"].isin(ORDINAL_GROUPERS.keys())]
    ordinal_df["label_group"] = ordinal_df.apply(lambda row: ORDINAL_GROUPERS[row["feature"]][0](row["label"]), axis=1)
    ordinal_df["absolute_error"] = ordinal_df.apply(lambda row: abs(get_level(row["feature"], row["label_group"]) - get_level(row["feature"], row["prediction"])), axis=1)

    nominal_df = df[df["feature"].isin(NOMINAL_GROUPERS.keys())]
    nominal_df["label_group"] = nominal_df.apply(lambda row: NOMINAL_GROUPERS.get(row["feature"], lambda x: x)(row["label"]), axis=1)

    nominal_df["is_correct"] = nominal_df["label_group"] == nominal_df["prediction"]
    ordinal_df_results = ordinal_df.groupby(["feature", "hardness"]).agg(
        count=("label", "size"),
        mae=("absolute_error", "mean"),
    ).reset_index()
    nominal_df_results = nominal_df.groupby(["feature", "hardness"]).agg(
        count=("label", "size"),
        accuracy=("is_correct", "mean"),
    ).reset_index()

    df_results = pd.concat([ordinal_df_results, nominal_df_results], ignore_index=True)
    return df_results

if __name__ == "__main__":
    adapter = get_adapter("reddit")
    records = list(adapter.load_records())
    getter = get_getter("reddit", "feature_label")

    df_results = make_inference_on_reddit(records)
    print(df, df_results)

# import json
# import pandas as pd
# from dp.loaders.derive import ORDINAL_GROUPERS, NOMINAL_GROUPERS

# def get_level(feature: str, label: str) -> int:
#     if feature in ORDINAL_GROUPERS:
#         _, levels = ORDINAL_GROUPERS[feature]
#         if label in levels:
#             return levels.index(label)
#     raise ValueError(f"Unknown feature or label: {feature}, {label}")

# def make_inference_on_reddit(df: pd.DataFrame) -> pd.DataFrame:
#     if not df["feature"].isin(set(ORDINAL_GROUPERS.keys()).union(set(NOMINAL_GROUPERS.keys()))).all():
#         raise ValueError("Unknown features found in the dataset.")
    
#     ordinal_df = df[df["feature"].isin(ORDINAL_GROUPERS.keys())]
#     ordinal_df["label_group"] = ordinal_df.apply(lambda row: ORDINAL_GROUPERS[row["feature"]][0](row["label"]), axis=1)
#     ordinal_df["absolute_error"] = ordinal_df.apply(lambda row: abs(get_level(row["feature"], row["label_group"]) - get_level(row["feature"], row["prediction"])), axis=1)

#     nominal_df = df[df["feature"].isin(NOMINAL_GROUPERS.keys())]
#     nominal_df["label_group"] = nominal_df.apply(lambda row: NOMINAL_GROUPERS.get(row["feature"], lambda x: x)(row["label"]), axis=1)

#     nominal_df["is_correct"] = nominal_df["label_group"] == nominal_df["prediction"]
#     ordinal_df_results = ordinal_df.groupby(["feature", "hardness"]).agg(
#         count=("label", "size"),
#         mae=("absolute_error", "mean"),
#     ).reset_index()
#     nominal_df_results = nominal_df.groupby(["feature", "hardness"]).agg(
#         count=("label", "size"),
#         accuracy=("is_correct", "mean"),
#     ).reset_index()

#     df_results = pd.concat([ordinal_df_results, nominal_df_results], ignore_index=True)
#     return df_results

# if __name__ == "__main__":
#     with open("logs/reddit_utility.jsonl", "r") as f:
#         results = [json.loads(line) for line in f]
    
#     df = pd.DataFrame(results)
#     df_results = make_inference_on_reddit(df)
#     print(df.index, df_results.index, df, df_results)
#     # df_results.to_csv("logs/reddit_utility_processed.csv", index=False)