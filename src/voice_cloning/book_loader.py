"""Load sentences from open-source fairy tales / children's books (Project Gutenberg)."""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import requests

_LOGGER = logging.getLogger(__name__)

DEFAULT_BOOK_SOURCE = "https://www.gutenberg.org/cache/epub/19163/pg19163.txt"
MAX_BOOK_SENTENCE_WORDS = 50


@dataclass(frozen=True)
class BookPreset:
    """Project Gutenberg plain-text book shortcut for dataset sentences."""

    key: str
    lang: str
    title: str
    author: str
    description: str
    url: str


BOOK_PRESETS: dict[str, BookPreset] = {
    "de-andersen-maerchen": BookPreset(
        key="de-andersen-maerchen",
        lang="de",
        title="Märchen",
        author="Hans Christian Andersen",
        description="German fairy tales — narrative, expressive sentences.",
        url="https://www.gutenberg.org/cache/epub/19163/pg19163.txt",
    ),
    "en-grimm-fairy-tales": BookPreset(
        key="en-grimm-fairy-tales",
        lang="en",
        title="Household Tales",
        author="Brothers Grimm",
        description="English Grimm fairy tales — classic short-story cadence.",
        url="https://www.gutenberg.org/cache/epub/2591/pg2591.txt",
    ),
    "fr-perrault-contes": BookPreset(
        key="fr-perrault-contes",
        lang="fr",
        title="Contes de ma mère l'Oye",
        author="Charles Perrault",
        description=(
            "French fairy tales (Perrault) from Le Cabinet des Fées — "
            "clear, literary prose."
        ),
        url="https://www.gutenberg.org/cache/epub/28891/pg28891.txt",
    ),
}


CUSTOM_BOOK_KEY = "custom"


def resolve_book_preset(key: str) -> BookPreset:
    preset = BOOK_PRESETS.get(key.strip().lower())
    if preset is None:
        known = ", ".join(sorted(BOOK_PRESETS))
        raise ValueError(f"Unknown book preset {key!r}. Known presets: {known}")
    return preset


def resolve_book_source(*, preset: str, custom_url: str = "") -> tuple[str, str | None]:
    """Return (gutenberg_url, preset_label). preset_label is None for custom URLs."""
    key = preset.strip().lower()
    if key == CUSTOM_BOOK_KEY:
        url = custom_url.strip()
        if not url:
            raise ValueError(
                "BOOK_PRESET is 'custom' — set CUSTOM_BOOK_URL to a Project Gutenberg "
                "plain-text URL (https://www.gutenberg.org/cache/epub/.../pg....txt)"
            )
        _resolve_book_url(url)
        return url, None
    book = resolve_book_preset(key)
    return book.url, book.key


def _book_cache_name(url: str) -> str:
    return re.sub(r"[^\w.-]+", "_", url.rstrip("/").split("/")[-1].replace(".txt", ""))


def _resolve_book_url(source: str) -> str:
    source = source.strip()
    if source.startswith("http"):
        return source
    raise ValueError(
        "book_source must be a Project Gutenberg plain-text URL "
        f"(https://www.gutenberg.org/cache/epub/.../pg....txt), got: {source!r}"
    )


def download_book(url: str, output_path: Path) -> Path:
    """Download a Project Gutenberg plain-text book."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    response.encoding = "utf-8"
    output_path.write_text(response.text, encoding="utf-8")
    return output_path


def _strip_gutenberg_boilerplate(text: str) -> str:
    start = re.search(
        r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    end = re.search(
        r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if start and end:
        return text[start.end() : end.start()]
    return text


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_FAIRY_TALE_START_MARKERS: dict[str, tuple[str, ...]] = {
    "de": (
        r"Hilfe suchend kam",
        r"Es war einmal",
        r"War einmal",
        r"Da war einmal",
        r"Einst war",
        r"In alten Zeiten",
    ),
    "en": (
        r"Once upon a time",
        r"There was once",
        r"In old times when",
        r"Long ago",
    ),
    "fr": (
        r"Il était une fois",
        r"Il etait une fois",
        r"C'était une fois",
        r"C'etait une fois",
    ),
}


def _strip_fairy_tale_front_matter(text: str) -> str:
    """Skip publisher pages, poems, and tables of contents in fairy-tale editions."""
    all_markers = [marker for markers in _FAIRY_TALE_START_MARKERS.values() for marker in markers]
    pattern = re.compile(
        r"(?:^|\n\n)(" + "|".join(all_markers) + r")",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    if match:
        return text[match.start() :].lstrip()
    return text


_SKIP_LINE_PATTERNS = (
    re.compile(r"^inhaltsverzeichnis$", re.I),
    re.compile(r"^table of contents$", re.I),
    re.compile(r"^kapitel\b", re.I),
    re.compile(r"^chapter\b", re.I),
    re.compile(r"^chapitre\b", re.I),
    re.compile(r"^buch\b", re.I),
    re.compile(r"^book\b", re.I),
    re.compile(r"^von\s+", re.I),
    re.compile(r"^by\s+", re.I),
    re.compile(r"^par\s+", re.I),
    re.compile(r"^ein buch\b", re.I),
    re.compile(r"^project gutenberg", re.I),
    re.compile(r"^also sprach zarathustra$", re.I),
    re.compile(r"^zarathustra", re.I),
    re.compile(r"^friedrich", re.I),
    re.compile(r"^jacob grimm", re.I),
    re.compile(r"^wilhelm grimm", re.I),
    re.compile(r"^translated by", re.I),
    re.compile(r"^household tales", re.I),
    re.compile(r"^märchengruß", re.I),
    re.compile(r"^frei nach der", re.I),
    re.compile(r"^(stuttgart|verlag|druck von|neunte auflage)\b", re.I),
    re.compile(r"^\[illustration|\[abbildung", re.I),
)

_BAD_SENTENCE_PATTERNS = (
    re.compile(r"\[Illustration|\[Abbildung", re.I),
    re.compile(r"Frei nach der", re.I),
    re.compile(r"^(Stuttgart|Verlag|Druck von|Loewes Verlag)", re.I),
    re.compile(r"\s+\d{1,3}\s*$"),  # table-of-contents page numbers
)


def _word_count(sentence: str) -> int:
    return len(sentence.split())


def _is_valid_sentence(
    sentence: str,
    *,
    max_words: int | None = None,
) -> bool:
    words = _word_count(sentence)
    min_words = 3 if sentence.rstrip().endswith(("!", "?")) else 5
    if words < min_words:
        return False
    if max_words is not None:
        if words > max_words:
            return False
    elif len(sentence) > 220:
        return False
    if sentence.isupper():
        return False
    for pattern in _SKIP_LINE_PATTERNS:
        if pattern.search(sentence):
            return False
    for pattern in _BAD_SENTENCE_PATTERNS:
        if pattern.search(sentence):
            return False
    if sentence.count("_") > 2:
        return False
    if sum(1 for ch in sentence if ch.islower()) < 8:
        return False
    return True


def filter_sentences_by_max_words(
    sentences: list[str],
    max_words: int,
    *,
    log_label: str = "",
) -> list[str]:
    """Drop sentences longer than `max_words` (used for cached manifests too)."""
    kept: list[str] = []
    skipped = 0
    for sentence in sentences:
        if _word_count(sentence) > max_words:
            skipped += 1
            continue
        kept.append(sentence)
    if skipped:
        suffix = f" ({log_label})" if log_label else ""
        _LOGGER.info(
            "Skipped %d sentence(s) over %d words%s",
            skipped,
            max_words,
            suffix,
        )
    return kept


def _fairy_tale_style_priority(sentence: str) -> tuple[int, int, int]:
    """Lower sorts first: prefer short lines ending with ! or ?."""
    stripped = sentence.rstrip()
    ends_excited = stripped.endswith("!") or stripped.endswith("?")
    has_marker = "!" in sentence or "?" in sentence
    return (
        0 if ends_excited else (1 if has_marker else 2),
        _word_count(sentence),
        len(sentence),
    )


def _select_book_sentences(
    sentences: list[str], limit: int, seed: int
) -> list[str]:
    """Prefer short exclamations and questions, then fill with other short lines."""
    rng = random.Random(seed)
    tiers: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for sentence in sentences:
        tiers[_fairy_tale_style_priority(sentence)[0]].append(sentence)

    for tier_sentences in tiers.values():
        rng.shuffle(tier_sentences)

    selected: list[str] = []
    seen: set[str] = set()
    for tier in (0, 1, 2):
        for sentence in tiers[tier]:
            key = sentence.casefold()
            if key in seen:
                continue
            seen.add(key)
            selected.append(sentence)
            if len(selected) >= limit:
                return selected
    return selected


def _split_sentences(text: str, *, max_words: int | None = None) -> list[str]:
    """Split prose into utterance-sized sentences."""
    word_limit = max_words or MAX_BOOK_SENTENCE_WORDS
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences: list[str] = []
    skipped_long = 0

    for paragraph in paragraphs:
        if len(paragraph) < 20 and paragraph.isupper():
            continue
        if paragraph.startswith(("INHALT", "KAPITEL", "CHAPTER", "CHAPITRE")):
            continue
        if paragraph.startswith("[") and paragraph.endswith("]"):
            continue

        parts = re.split(
            r"(?<=[.!?…])\s+(?=[„\"'»(A-ZÀÂÄÉÈÊËÎÏÔŒÙÛÜŸÇ])",
            paragraph,
        )
        for part in parts:
            sentence = part.strip()
            sentence = re.sub(r"\s+", " ", sentence)
            sentence = sentence.strip("«»\"' _")
            if not sentence:
                continue

            if _word_count(sentence) > word_limit:
                skipped_long += 1
                continue
            if not _is_valid_sentence(sentence, max_words=word_limit):
                continue
            sentences.append(sentence)

    if skipped_long:
        _LOGGER.info(
            "Dismissed %d sentence(s) over %d words while parsing book text",
            skipped_long,
            word_limit,
        )

    return sentences


def load_sentences(
    book_path: Path | None = None,
    book_url: str | None = None,
    book_source: str | None = None,
    cache_dir: Path | None = None,
    limit: int = 5000,
    seed: int = 42,
    max_words: int | None = None,
) -> list[str]:
    """Load up to `limit` unique sentences from a fairy-tale / children's book source."""
    word_limit = max_words or MAX_BOOK_SENTENCE_WORDS
    source = (book_source or book_url or DEFAULT_BOOK_SOURCE).strip()
    url = _resolve_book_url(source)

    cache_dir = cache_dir or Path("data/books")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if book_path is None:
        book_path = cache_dir / f"{_book_cache_name(url)}.txt"
        if not book_path.exists():
            download_book(url, book_path)

    raw = book_path.read_text(encoding="utf-8", errors="replace")
    body = _strip_gutenberg_boilerplate(raw)
    body = _strip_fairy_tale_front_matter(body)
    body = _normalize_whitespace(body)
    sentences = _split_sentences(body, max_words=word_limit)

    unique = _select_book_sentences(sentences, limit=limit * 3, seed=seed)
    if len(unique) < limit:
        raise RuntimeError(
            f"Only extracted {len(unique)} book sentences (max {word_limit} words) "
            f"from {book_path}, but {limit} were requested."
        )

    return unique[:limit]
