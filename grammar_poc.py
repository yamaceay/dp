"""POC: pure-Python text quality / grammatical-acceptability check.

dp/experiments/divergence/grammar.py uses language_tool_python, which needs a
local Java runtime (unavailable in the target NVIDIA PyTorch container). This
POC checks whether a RoBERTa classifier fine-tuned on CoLA (Corpus of
Linguistic Acceptability) gives a usable substitute: a single forward pass
per sentence, torch/transformers only, no Java, no external server.

Model: cointegrated/roberta-large-cola-krishna2020 (ships safetensors weights;
textattack/roberta-base-CoLA only ships an unsafe pytorch_model.bin and is
blocked by transformers' torch.load safety gate on torch<2.6).

Label convention (verified empirically below, not documented on the model
card): LABEL_0 = acceptable, LABEL_1 = unacceptable. We report
p(acceptable) = LABEL_0 probability as the "grammatical correctness" score.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "cointegrated/roberta-large-cola-krishna2020"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()


def acceptability(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return float(torch.softmax(logits, dim=-1)[0, 0])


TESTS = [
    ("clean", "The quick brown fox jumps over the lazy dog."),
    ("clean", "She has been working at the hospital since last spring."),
    ("subject_verb_agreement_error", "The dog run fast in the park every morning."),
    ("tense_error", "Yesterday I go to the store and buy some milk."),
    ("article_and_preposition_errors", "He is going to store buy a bread for family."),
    ("word_salad", "Store the buy milk fast dog quick brown."),
    ("dp_mlm_style_degradation", "The Male patient state Oregon born email connor gmail com."),
    ("informal_double_negative", "I don't have no time for this nonsense."),
    ("run_on", "I went to the store I bought milk I came home."),
]

if __name__ == "__main__":
    for tag, text in TESTS:
        score = acceptability(text)
        print(f"{tag:32s} p(acceptable)={score:.3f}  | {text}")
