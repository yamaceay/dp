import re
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

from dp.loaders import get_adapter
from dp.loaders.derive import get_getter

AVAILABLE_FILES = [
    "logs/a2_reddit_debug_tri/2519179_0.out",
    "logs/a2_reddit_debug_tri/2519180_1.out",
    "logs/a2_reddit_debug_tri/2519181_2.out",
    "logs/a2_reddit_debug_tri/2519178_3.out",
]

def get_all_predictions(text):
    pattern = r"Top 10 Predictions for UID reddit_\d+: \[(.*?)\]"
    matches = re.findall(pattern, text)
    all_predictions = []
    for match in matches:
        predictions = match.split(", ")
        predictions = [pred.strip("'") for pred in predictions]
        all_predictions.append(predictions)
    return all_predictions

def get_all_persona_votes(records, getter, all_predictions, only_feature):
    all_features = set()
    persona_by_name = {}
    for record in records:
        feature, label = getter(record).split(": ")
        all_features.add(feature)
        persona_by_name.setdefault(record.name, {})[feature] = label

    all_persona_votes = {}
    for i, record in enumerate(records):
        true_values = persona_by_name[record.name]
        prediction = all_predictions[i]
        requested_feature, _ = getter(record).split(": ")
        voted_predictions_for_features = {}
        for pred in prediction:
            for feature, label in persona_by_name[pred].items():
                if feature not in true_values:
                    continue
                if only_feature and not feature == requested_feature:
                    continue
                counts = voted_predictions_for_features.setdefault(feature, {}).setdefault(label, 0)
                voted_predictions_for_features[feature][label] = counts + 1
        
        major_vote_for_features = {}
        for feature, predictions in voted_predictions_for_features.items():
            major_vote = max(predictions.items(), key=lambda x: x[1])[0]
            major_vote_for_features[feature] = int(major_vote == true_values[feature])
                
        all_persona_votes.setdefault(record.name, []).append(major_vote_for_features)

    return all_persona_votes

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Reddit predictions")
    parser.add_argument("input_file", type=str, help="Path to the input text file containing predictions")
    parser.add_argument("--only_feature", action="store_true", help="Only analyze feature-level accuracy")
    args = parser.parse_args()
    with open(args.input_file) as f:
        text = f.read()

    all_predictions = get_all_predictions(text)

    getter = get_getter("reddit", "feature_label")
    records = list(get_adapter("reddit", data_in="data/reddit/reddit.jsonl").iter_records())

    all_persona_votes = get_all_persona_votes(records, getter, all_predictions, args.only_feature)

    persona_accuracy_by_feature = {}
    for name, votes in all_persona_votes.items():
        list_of_votes_by_feature = {}
        for vote_dict in votes:
            for feature, accuracy in vote_dict.items():
                list_of_votes_by_feature.setdefault(feature, []).append(accuracy)
        for feature, votes in list_of_votes_by_feature.items():
            persona_accuracy_by_feature.setdefault(feature, []).append(sum(votes) / len(votes))
    
    df = pd.DataFrame(persona_accuracy_by_feature.values(), index=persona_accuracy_by_feature.keys()).transpose()

    fig, axes = plt.subplots(nrows=2, ncols=len(df.columns) // 2, figsize=(12, 8))

    for feature in df.columns:
        ax = axes.flatten()[df.columns.get_loc(feature)]
        ax.hist(df[feature], bins=10, alpha=0.7, density=True)
        kde = gaussian_kde(df[feature].dropna())
        x_range = np.linspace(df[feature].min(), df[feature].max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2)
        ax.set_title(f"Accuracy for {feature}")
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Density")
    plt.tight_layout()
    plt.show()