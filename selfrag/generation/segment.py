"""Generate one candidate next segment from one passage (or from no passage)."""
import re
from functools import cache
from pathlib import Path

from selfrag.models import Candidate, Chunk
from selfrag.providers.interfaces import LLMClient
from selfrag.scoring.scores import END

PROMPT_VERSION = "v1"
PROMPTS_DIR = Path(__file__).parent / "prompts"

# Thinking models spend part of the output budget on hidden reasoning.
MAX_TOKENS = {"sentence": 512, "answer": 1024}

_CITATION_MARK = re.compile(r"\s*\[\d+\]")

# A sentence ends at . ! ? (or …), optionally followed by closing quotes/brackets, then
# whitespace and the start of a new sentence (capital letter, digit, quote or bracket).
_SENTENCE_END = re.compile(r"""[.!?\u2026]+["'\u201d\u2019)\]]*(?=\s+["'\u201c\u2018(\[]?[A-Z0-9])""")
# Words whose trailing period does not end a sentence.
_ABBREVIATIONS = {"e.g", "i.e", "etc", "vs", "cf", "al", "fig", "figs", "eq", "eqs", "sec",
                  "vol", "pp", "dr", "mr", "mrs", "ms", "prof", "st", "approx", "inc",
                  "ltd", "co", "jr", "sr", "u.s", "u.k", "tab", "ch", "ref", "refs"}


@cache
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.{PROMPT_VERSION}.txt").read_text(encoding="utf-8").strip()


def first_sentence(text: str) -> str:
    """The first sentence of `text`, skipping abbreviations (e.g., et al.) and decimals."""
    for m in _SENTENCE_END.finditer(text):
        if text[m.start()] == ".":
            word = re.search(r"(\S+)$", text[:m.start()])
            if word and (word.group(1).lower().lstrip("(\"'") in _ABBREVIATIONS
                         or len(word.group(1)) == 1 and word.group(1).isupper()):  # initials
                continue
        return text[:m.end()].strip()
    return text.strip()


def clean(text: str, unit: str) -> str:
    text = text.strip()
    if END in text:
        return END
    text = _CITATION_MARK.sub("", text).strip()
    if unit == "sentence":      # models often write several sentences: keep only the first
        text = next((line.strip() for line in text.splitlines() if line.strip()), "")
        text = first_sentence(text)
    return text


async def generate(llm: LLMClient, question: str, prefix: list[str], passage: Chunk | None,
                   n: int, unit: str = "sentence", temperature: float = 0.0) -> Candidate:
    name = ("answer" if unit == "answer" else "segment") + ("" if passage else "_no_evidence")
    fields = dict(question=question, prefix=" ".join(prefix) or "(none)", n=n)
    if passage:
        fields.update(title=passage.title, passage=passage.text)
    out = await llm.complete(load_prompt(name).format(**fields), max_tokens=MAX_TOKENS[unit],
                             temperature=temperature, logprobs=llm.supports_logprobs)
    lps = out.logprobs or []
    return Candidate(clean(out.text, unit), passage, sum(lps) / len(lps) if lps else None)
