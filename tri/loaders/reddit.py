from __future__ import annotations

import hashlib
import random
import re
from tqdm import tqdm
from typing import Dict, List, Tuple, cast

from dp.loaders.base import DatasetRecord
from dp.loaders.reddit import RedditDatasetAdapter
from dp.tri.loaders.base import AttackerDatasetAdapter, AttackerDatasetRecord

SECTION_PATTERN = re.compile(r"(Type|Inference|Guess):\s*(.*?)(?=(?:Type|Inference|Guess):|\Z)", re.S)

REQUIRED_KEYS = [
    "age",
    "sex",
    "city_country",
    "birth_city_country",
    "occupation",
    "income",
    "income_level",
    "education",
    "relationship_status",
]

A_TPL: List[str] = [
    "{SUBJ_CAP} {is} {age} years old and {was} born {sex}.",
    "At {age} years of age, {subj} {was} born {sex}.",
    "{SUBJ_CAP} {is} {age}, born {sex}.",
    "Currently {age}, {subj} {was} born {sex}.",
    "{SUBJ_CAP}: {age} years old; born {sex}.",
    "Age {age}; birth sex: {sex}. {SUBJ_CAP} {is} so noted.",
]

B_TPL: List[str] = [
    "{SUBJ_CAP} {lives} in {city_country} after being born in {birth_city_country}.",
    "Originally from {birth_city_country}, {subj} now {live} in {city_country}.",
    "{SUBJ_CAP} currently {live} in {city_country}. {SUBJ_CAP} {was} born in {birth_city_country}.",
    "Residence: {city_country}. Birthplace: {birth_city_country}.",
    "{SUBJ_CAP} calls {city_country} home, with origins in {birth_city_country}.",
]

C_TPL: List[str] = [
    "{SUBJ_CAP} {work} as {occupation} and {earn} {income} ({income_level}).",
    "{SUBJ_CAP} {is} employed as {occupation}, earning {income} ({income_level}).",
    "Professionally: {occupation}. Compensation: {income} ({income_level}).",
    "{SUBJ_CAP} {work} as {occupation}. Income: {income} ({income_level}).",
    "Role: {occupation}; {subj} {earn} {income} ({income_level}).",
]

D_TPL_COMPLETED: List[str] = [
    "{SUBJ_CAP} completed {education} and {is} {relationship_status}.",
    "Education completed: {education}. {SUBJ_CAP} {is} {relationship_status}.",
    "Having completed {education}, {subj} {is} {relationship_status}.",
    "{SUBJ_CAP}: {education} complete; status: {relationship_status}.",
]


def normalize_text(value: Optional[Any]) -> str:
    if not value:
        return ""
    elif isinstance(value, int):
        return str(value)
    return " ".join(value.split())


def enforce_terminal_punctuation(value: str) -> str:
    if not value:
        return value
    return value if value[-1] in ".!?" else f"{value}."


def ensure_period(value: str) -> str:
    stripped = value.rstrip()
    if not stripped:
        return stripped
    return stripped if stripped[-1] in ".!?" else f"{stripped}."


def strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def normalize_space(value: str | int | None) -> str:
    if isinstance(value, int):
        return str(value)
    if value is None:
        return ""
    return " ".join(str(value).split())


def resolve_pronouns(sex: str | None) -> Tuple[str, str, str]:
    if not sex:
        return "they", "them", "their"
    normalized = sex.strip().lower()
    if normalized in {"male", "m", "man", "boy"}:
        return "he", "him", "his"
    if normalized in {"female", "f", "woman", "girl"}:
        return "she", "her", "her"
    return "they", "them", "their"


def verb_form(subject: str, singular: str, plural: str) -> str:
    return plural if subject.lower() == "they" else singular


def parse_guess_sections(guess: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    for label, raw_value in SECTION_PATTERN.findall(guess):
        cleaned = strip_matching_quotes(normalize_text(raw_value))
        sections[label.lower()] = cleaned
    return sections


def build_sentence_blocks(persona: Dict[str, str], rng: random.Random) -> List[str]:
    subject, _, _ = resolve_pronouns(persona.get("sex"))
    is_ = verb_form(subject, "is", "are")
    was_ = verb_form(subject, "was", "were")
    live = verb_form(subject, "lives", "live")
    work = verb_form(subject, "works", "work")
    earn = verb_form(subject, "earns", "earn")
    values = {key: normalize_space(persona.get(key, "")) for key in REQUIRED_KEYS}
    fmt = {
        "subj": subject,
        "SUBJ_CAP": subject.capitalize(),
        "is": is_,
        "was": was_,
        "live": live,
        "lives": live,
        "work": work,
        "earn": earn,
        "age": values["age"],
        "sex": values["sex"],
        "city_country": values["city_country"],
        "birth_city_country": values["birth_city_country"],
        "occupation": values["occupation"],
        "income": values["income"],
        "income_level": values["income_level"],
        "education": values["education"],
        "relationship_status": values["relationship_status"],
    }
    first = ensure_period(rng.choice(A_TPL).format(**fmt))
    second = ensure_period(rng.choice(B_TPL).format(**fmt))
    third = ensure_period(rng.choice(C_TPL).format(**fmt))
    fourth = ensure_period(rng.choice(D_TPL_COMPLETED).format(**fmt))
    blocks = [first, second, third, fourth]
    rng.shuffle(blocks)
    return blocks


def render_persona(persona: Dict[str, str], rng: random.Random) -> str:
    return " ".join(build_sentence_blocks(persona, rng)).strip()


def format_inference(metadata: Dict[str, str], persona: Dict[str, str], pronouns: Tuple[str, str, str]) -> str:
    _, _, possessive = pronouns
    if "feature" not in metadata or "guess" not in metadata:
        raise ValueError("feature and guess must be present in metadata")
    feature = normalize_text(metadata["feature"])
    if not feature:
        raise ValueError("feature value is empty")
    label = normalize_text(persona.get(feature) or metadata.get("label"))
    if not label:
        raise ValueError(f"label for feature '{feature}' is missing")
    sections = parse_guess_sections(metadata["guess"])
    feature_from_guess = normalize_text(sections.get("type"))
    if feature_from_guess and feature_from_guess != feature:
        raise ValueError(f"Mismatch in feature! Expected: {feature}, Found: {feature_from_guess}")
    inference_raw = sections.get("inference")
    if not inference_raw:
        raise ValueError("No inference provided in the inference section")
    inference_clean = enforce_terminal_punctuation(normalize_text(inference_raw))
    guesses_raw = sections.get("guess")
    if not guesses_raw:
        raise ValueError("No guesses provided in the guess section")
    guesses_list = [
        strip_matching_quotes(candidate).rstrip("., ").strip()
        for candidate in guesses_raw.split(";")
        if candidate.strip()
    ]
    if not guesses_list:
        raise ValueError("Parsed guess list is empty")
    guesses_clean = ensure_period(", ".join(guesses_list))
    return " ".join(
        [
            f"An agent is asked to predict {possessive} {feature}.",
            f"The agent reasons: '{inference_clean}'",
            f"Then it predicts: {guesses_clean}",
            f"The correct answer is {label}.",
        ]
    )


class RedditAttackerDatasetAdapter(AttackerDatasetAdapter):
    def __init__(
        self,
        *args,
        data: str | None = None,
        data_in: str | None = None,
        max_records: int | None = None,
        seed: int = 42,
        **kwargs,
    ) -> None:
        adapter = RedditDatasetAdapter(data=data, data_in=data_in, max_records=max_records)
        super().__init__(adapter=adapter, *args, **kwargs)
        self._seed = seed
        self._persona_records: List[AttackerDatasetRecord] = self._build_persona_records()

    def _persona_seed(self, persona: Dict[str, str]) -> int:
        entries = "|".join(f"{key}={normalize_space(persona.get(key, ''))}" for key in sorted(persona.keys()))
        seed_source = f"{self._seed}|{entries}"
        digest = hashlib.sha256(seed_source.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _persona_text(self, persona: Dict[str, str]) -> str:
        rng = random.Random(self._persona_seed(persona))
        return render_persona(persona, rng)

    def _build_background_entry(
        self,
        record: DatasetRecord,
        persona: Dict[str, str],
        pronouns: Tuple[str, str, str],
    ) -> str:
        metadata = dict(record.metadata or {})
        question_raw = strip_matching_quotes(normalize_text(metadata.get("question")))
        question_clean = enforce_terminal_punctuation(question_raw) if question_raw else "No question provided."

        answer_raw = strip_matching_quotes(normalize_text(record.text))
        answer_clean = enforce_terminal_punctuation(answer_raw) if answer_raw else "No answer provided."

        inference_payload = {
            "feature": metadata.get("feature"),
            "guess": metadata.get("guess"),
            "label": metadata.get("label"),
        }
        inference_text = format_inference(inference_payload, persona, pronouns)

        subject = pronouns[0]
        subject_cap = subject.capitalize()
        interaction = ensure_period(
            f"The question was asked in a forum: '{question_clean}' {subject_cap} answered with: '{answer_clean}'"
        )
        return ensure_period(f"{interaction} {inference_text}")

    def _build_persona_records(self) -> List[AttackerDatasetRecord]:
        grouped: Dict[str, Dict[str, object]] = {}
        for record in self.adapter.iter_records():
            metadata = dict(record.metadata or {})
            persona = {key[len("persona_"):]: value for key, value in metadata.items() if key.startswith("persona_")}
            if not persona:
                raise ValueError("Persona metadata is required for Reddit attacker adapter")
            persona_text = self._persona_text(persona)
            pronouns = resolve_pronouns(persona.get("sex"))
            background_entry = self._build_background_entry(record, persona, pronouns)

            bucket = grouped.setdefault(
                record.name,
                {
                    "persona_text": persona_text,
                    "metadata": persona,
                    "background": {},
                },
            )
            bucket["background"][record.uid] = background_entry

        persona_records: List[AttackerDatasetRecord] = []
        for persona_hash, payload in grouped.items():
            persona_text = payload["persona_text"]
            background_dict = cast(Dict[str, str], payload["background"])
            background_items = [(uid, background_dict[uid]) for uid in sorted(background_dict.keys())]
            persona_records.append(
                AttackerDatasetRecord(
                    text=persona_text,
                    uid=persona_hash,
                    name=persona_hash,
                    metadata={"persona": payload["metadata"]},
                    background_knowledge=background_items,
                    rewrited_text=persona_text,
                )
            )
        return persona_records

    def extract_background_knowledge(self, record: DatasetRecord) -> List[Tuple[str, str]]:
        raise NotImplementedError("RedditAttackerDatasetAdapter aggregates records by persona; use iter_records()")

    def rewrite_original_text(self, record: DatasetRecord) -> str:
        raise NotImplementedError("RedditAttackerDatasetAdapter provides persona summaries directly; use iter_records()")

    def iter_records(self, progress: bool = False):
        records = self._persona_records
        iterator = records
        if progress:
            iterator = tqdm(records, desc="Processing attacker records", total=len(records))
        for record in iterator:
            yield record


__all__ = ["RedditAttackerDatasetAdapter"]
