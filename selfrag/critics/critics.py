"""The four reflection-token critics (paper Appendix D prompts, versioned files)."""
from functools import cache
from pathlib import Path

from selfrag.critics.critic import Critic

PROMPT_VERSION = "v1"
PROMPTS_DIR = Path(__file__).parent / "prompts"

RETRIEVE_LABELS = ["Yes", "No", "Continue"]
ISREL_LABELS = ["Relevant", "Irrelevant"]
ISSUP_LABELS = ["Fully", "Partially", "No"]
ISUSE_LABELS = ["1", "2", "3", "4", "5"]


@cache
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.{PROMPT_VERSION}.txt").read_text(encoding="utf-8").strip()


async def retrieve(critic: Critic, question: str, prefix: str, evidence: str) -> dict[str, float]:
    return await critic.label_probs(load_prompt("retrieve").format(
        question=question, prefix=prefix or "(none)", evidence=evidence or "none"), RETRIEVE_LABELS)


async def isrel(critic: Critic, question: str, prefix: str, passage: str) -> dict[str, float]:
    return await critic.label_probs(load_prompt("isrel").format(
        question=question, prefix=prefix or "(none)", passage=passage), ISREL_LABELS)


async def issup(critic: Critic, question: str, passage: str, segment: str) -> dict[str, float]:
    return await critic.label_probs(load_prompt("issup").format(
        question=question, passage=passage, segment=segment), ISSUP_LABELS)


async def isuse(critic: Critic, question: str, answer: str) -> dict[str, float]:
    return await critic.label_probs(load_prompt("isuse").format(
        question=question, answer=answer), ISUSE_LABELS)
