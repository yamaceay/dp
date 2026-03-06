from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, Iterable, List, Optional

from dp.loaders import get_adapter
from dp.tri.loaders.base import AttackerDatasetAdapter, AttackerDatasetRecord, merge_records, presidio_anonymize
from dp.loaders.base import DatasetRecord


def _normalize_text(value: Optional[Any]) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return " ".join(str(value).split())


def _ensure_period(value: str) -> str:
    s = value.rstrip()
    if not s:
        return s
    return s if s[-1] in ".!?" else f"{s}."


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


class YelpAttackerDatasetAdapter(AttackerDatasetAdapter):
    def __init__(self, seed: int = 42, need_to_deidentify: bool = False, *args, **kwargs) -> None:
        adapter = get_adapter("yelp", *args, **kwargs)
        super().__init__(adapter=adapter)
        self._seed = seed
        self._need_to_deidentify = need_to_deidentify
        self._business_by_id: Dict[str, Dict[str, Any]] = {}
        self._user_by_id: Dict[str, Dict[str, Any]] = {}
        with open(self.adapter.data_in, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "review_id" in obj:
                    continue
                if "business_id" in obj:
                    bid = str(obj.get("business_id"))
                    self._business_by_id[bid] = obj
                    continue
                if "user_id" in obj:
                    uid = str(obj.get("user_id"))
                    self._user_by_id[uid] = obj

    def __len__(self) -> int:
        return len(list(self.adapter.iter_records()))

    def _business_seed(self, business: Dict[str, Any], variant: str = "") -> int:
        bid = _normalize_text(business.get("business_id"))
        name = _normalize_text(business.get("name"))
        city = _normalize_text(business.get("city"))
        state = _normalize_text(business.get("state"))
        categories = _normalize_text(business.get("categories"))
        payload = f"{self._seed}|{bid}|{name}|{city}|{state}|{categories}|{variant}"
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int(h[:16], 16)

    def _format_business_core(self, business: Dict[str, Any]) -> Dict[str, str]:
        attrs = business.get("attributes") or {}
        return {
            "name": _normalize_text(business.get("name")),
            "address": _normalize_text(business.get("address")),
            "city": _normalize_text(business.get("city")),
            "state": _normalize_text(business.get("state")),
            "postal_code": _normalize_text(business.get("postal_code")),
            "latitude": _normalize_text(business.get("latitude")),
            "longitude": _normalize_text(business.get("longitude")),
            "stars": _normalize_text(business.get("stars")),
            "review_count": _normalize_text(business.get("review_count")),
            "is_open": "open" if bool(business.get("is_open")) else "closed",
            "categories": _normalize_text(business.get("categories")),
            "price": _normalize_text(attrs.get("RestaurantsPriceRange2")),
            "cards": _normalize_text(attrs.get("BusinessAcceptsCreditCards")),
            "kids": _normalize_text(attrs.get("GoodForKids")),
            "bike": _normalize_text(attrs.get("BikeParking")),
        }

    def _business_blocks(self, business: Dict[str, Any], rng: random.Random) -> List[str]:
        fmt = self._format_business_core(business)
        cats = [c for c in (fmt["categories"].split(", ") if fmt["categories"] else []) if c]
        cat_text = ", ".join(cats) if cats else "unspecified"
        price = fmt["price"] or "unknown"
        cards = fmt["cards"].lower()
        cards_text = "accepts credit cards" if cards in {"true", "1", "yes"} else "does not accept credit cards"
        kids = fmt["kids"].lower()
        kids_text = "good for kids" if kids in {"true", "1", "yes"} else "not marked good for kids"
        bike = fmt["bike"].lower()
        bike_text = "has bike parking" if bike in {"true", "1", "yes"} else "no bike parking noted"
        a = _ensure_period(rng.choice([
            f"{fmt['name']} is a {fmt['is_open']} business in {fmt['city']}, {fmt['state']}",
            f"In {fmt['city']}, {fmt['state']}, {fmt['name']} is currently {fmt['is_open']}",
            f"{fmt['name']} operates in {fmt['city']}, {fmt['state']} and is {fmt['is_open']}",
        ]))
        b = _ensure_period(rng.choice([
            f"It is located at {fmt['address']} {fmt['postal_code']}",
            f"Address: {fmt['address']} {fmt['postal_code']}",
        ]))
        c = _ensure_period(rng.choice([
            f"Categories include {cat_text}",
            f"Tagged as {cat_text}",
        ]))
        d = _ensure_period(rng.choice([
            f"It has {fmt['stars']} stars based on {fmt['review_count']} reviews",
            f"Rated {fmt['stars']} stars with {fmt['review_count']} reviews",
        ]))
        e = _ensure_period(rng.choice([
            f"Price level is {price}",
            f"Listed price range: {price}",
        ]))
        f = _ensure_period(rng.choice([
            cards_text,
            kids_text,
            bike_text,
        ]))
        blocks = [a, b, c, d, e, f]
        rng.shuffle(blocks)
        return blocks

    def _business_summary(self, business: Dict[str, Any], variant: str) -> str:
        rng = random.Random(self._business_seed(business, variant))
        return " ".join(self._business_blocks(business, rng)).strip()

    def _user_seed(self, user: Dict[str, Any], variant: str = "") -> int:
        uid = _normalize_text(user.get("user_id"))
        name = _normalize_text(user.get("name"))
        since = _normalize_text(user.get("yelping_since"))
        reviews = _normalize_text(user.get("review_count"))
        payload = f"{self._seed}|{uid}|{name}|{since}|{reviews}|{variant}"
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int(h[:16], 16)

    def _format_user_core(self, user: Dict[str, Any]) -> Dict[str, str]:
        elite = user.get("elite") or []
        friends = user.get("friends") or []
        elite_count = len(elite) if isinstance(elite, list) else 0
        friends_count = len(friends) if isinstance(friends, list) else 0
        return {
            "name": _normalize_text(user.get("name")),
            "yelping_since": _normalize_text(user.get("yelping_since")),
            "review_count": _normalize_text(user.get("review_count")),
            "average_stars": _normalize_text(user.get("average_stars")),
            "fans": _normalize_text(user.get("fans")),
            "useful": _normalize_text(user.get("useful")),
            "funny": _normalize_text(user.get("funny")),
            "cool": _normalize_text(user.get("cool")),
            "elite_count": _normalize_text(elite_count),
            "friends_count": _normalize_text(friends_count),
        }

    def _user_blocks(self, user: Dict[str, Any], rng: random.Random) -> List[str]:
        fmt = self._format_user_core(user)
        a = _ensure_period(rng.choice([
            f"{fmt['name']} has written {fmt['review_count']} reviews since {fmt['yelping_since']}",
            f"Since {fmt['yelping_since']}, {fmt['name']} posted {fmt['review_count']} reviews",
        ]))
        b = _ensure_period(rng.choice([
            f"Average rating is {fmt['average_stars']} stars",
            f"Gives on average {fmt['average_stars']} stars",
        ]))
        c = _ensure_period(rng.choice([
            f"Elite years: {fmt['elite_count']}",
            "No elite years" if fmt["elite_count"] in {"", "0"} else f"Elite years: {fmt['elite_count']}",
        ]))
        d = _ensure_period(rng.choice([
            f"Friends: {fmt['friends_count']}",
            f"Has {fmt['friends_count']} friends",
        ]))
        e = _ensure_period(rng.choice([
            f"Votes: {fmt['useful']} useful, {fmt['funny']} funny, {fmt['cool']} cool",
            f"Received {fmt['useful']} useful, {fmt['funny']} funny, and {fmt['cool']} cool votes",
        ]))
        blocks = [a, b, c, d, e]
        rng.shuffle(blocks)
        return blocks

    def _user_summary(self, user: Dict[str, Any], variant: str) -> str:
        rng = random.Random(self._user_seed(user, variant))
        return " ".join(self._user_blocks(user, rng)).strip()

    def _build_background_entry(self, review: DatasetRecord, business: Dict[str, Any], user: Dict[str, Any], variant: str) -> str:
        review_text = _strip_quotes(_normalize_text(review.text))
        review_clean = _ensure_period(review_text) if review_text else "No review text provided."
        business_summary = self._business_summary(business, variant)
        user_summary = self._user_summary(user, variant)
        parts = [
            _ensure_period(f"A reviewer wrote: '{review_clean}'"),
            _ensure_period(f"The review concerns the business: {business_summary}"),
            _ensure_period(f"The reviewer profile: {user_summary}"),
        ]
        seed_payload = f"{self._seed}|{review.uid}|{business.get('business_id')}|{user.get('user_id')}|{variant}|order"
        rng = random.Random(int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:16], 16))
        rng.shuffle(parts)
        return _ensure_period(" ".join(parts))

    def iter_records(self, progress: bool = False) -> Iterable[AttackerDatasetRecord]:
        grouped_train: Dict[str, List[str]] = {}
        grouped_eval: Dict[str, List[str]] = {}
        grouped_test: Dict[str, List[str]] = {}
        base_iter = list(self.adapter.iter_records())
        for idx, record in enumerate(base_iter):
            name = record.name
            bid = str(record.metadata.get("business_id"))
            business = self._business_by_id.get(bid)
            user = self._user_by_id.get(name)
            if not business or not user:
                continue
            modified_record = record
            background_entry = self._build_background_entry(modified_record, business, user, variant="train")
            if self._need_to_deidentify:
                background_entry = presidio_anonymize(background_entry)
            user_summary = self._user_summary(user, variant="eval")
            grouped_train.setdefault(name, []).append(background_entry)
            grouped_eval.setdefault(name, []).append(user_summary)
            grouped_test.setdefault(name, []).append(record.text)
        return merge_records(grouped_train, grouped_eval, grouped_test)

__all__ = ["YelpAttackerDatasetAdapter"]
