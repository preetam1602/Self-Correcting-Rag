import pytest

from selfrag.generation.segment import clean, first_sentence
from selfrag.scoring.scores import END

SELF_RAG = ("Self-RAG is a retrieval-augmented generation framework that critiques its own "
            "output with reflection tokens.")


@pytest.mark.parametrize("text, expected", [
    (f"{SELF_RAG} During training the model learns to generate these tokens.", SELF_RAG),
    ("It works well! But it is slow.", "It works well!"),
    ("Does it retrieve? Only when needed.", "Does it retrieve?"),
    ("It uses critics, e.g. IsSup, to check support. Then it ranks.",
     "It uses critics, e.g. IsSup, to check support."),
    ("This follows Asai et al. (2023) closely. It needs no training.",
     "This follows Asai et al. (2023) closely."),
    ("The threshold is 0.2 by default. Lower values retrieve more.",
     "The threshold is 0.2 by default."),
    ("See Fig. 1 for the overview. It has four critics.", "See Fig. 1 for the overview."),
    ('It is called "Self-RAG." The paper is from 2023.', 'It is called "Self-RAG."'),
    ("Written by J. Smith in 2023. It is short.", "Written by J. Smith in 2023."),
    ("A single sentence without a final period", "A single sentence without a final period"),
])
def test_first_sentence(text, expected):
    assert first_sentence(text) == expected


def test_sentence_mode_keeps_only_the_first_sentence():
    raw = f"{SELF_RAG} [1] During training the model learns to generate these tokens [1]."
    assert clean(raw, "sentence") == SELF_RAG


def test_citation_before_the_period_is_removed_before_splitting():
    assert clean("It retrieves on demand [2]. It then critiques.", "sentence") == \
        "It retrieves on demand."


def test_sentence_mode_still_uses_the_first_line():
    assert clean("\nFirst line here.\nSecond line here.", "sentence") == "First line here."


def test_answer_mode_keeps_every_sentence():
    raw = "First sentence [1]. Second sentence [2]."
    assert clean(raw, "answer") == "First sentence. Second sentence."


@pytest.mark.parametrize("unit", ["sentence", "answer"])
def test_end_marker(unit):
    assert clean("The answer is complete. <END>", unit) == END
