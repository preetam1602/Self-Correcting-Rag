"""Self-RAG scoring (paper Eq. 3-4, Appendix A.3) and segment-level beam selection. Pure functions."""
import math

from selfrag.models import Beam, Candidate

END = "<END>"
USE_W = {"1": -1.0, "2": -0.5, "3": 0.0, "4": 0.5, "5": 1.0}


def s_rel(p: dict[str, float]) -> float:
    return p["Relevant"] / (p["Relevant"] + p["Irrelevant"] + 1e-9)


def s_sup(p: dict[str, float]) -> float:
    z = p["Fully"] + p["Partially"] + p["No"] + 1e-9
    return p["Fully"] / z + 0.5 * p["Partially"] / z


def s_use(p: dict[str, float]) -> float:
    z = sum(p.values()) + 1e-9
    return sum(USE_W[k] * v / z for k, v in p.items())


def segment_score(c: Candidate, w_rel: float = 1.0, w_sup: float = 1.0, w_use: float = 0.5,
                  hard_sup: bool = False) -> float:
    """f(y_t, d) = p(y_t | x, d, y_<t) + sum of weighted critique scores."""
    if hard_sup and c.sup and max(c.sup, key=c.sup.get) == "No":
        return float("-inf")                        # paper's hard-constraint option
    score = math.exp(c.gen_logprob) if c.gen_logprob is not None else 0.0
    if c.rel:
        score += w_rel * s_rel(c.rel)
    if c.sup:
        score += w_sup * s_sup(c.sup)
    if c.use:
        score += w_use * s_use(c.use)
    return score


def is_end(c: Candidate) -> bool:
    return not c.text or c.text == END


def support_label(c: Candidate) -> str | None:
    """Most likely IsSup label, lowercased ("fully" | "partially" | "no"); None if not judged."""
    return max(c.sup, key=c.sup.get).lower() if c.sup else None


def select_beams(beams: list[Beam], expansions: list[list[Candidate]], B: int = 2,
                 **weights) -> list[Beam]:
    """Extend every beam with each of its candidates and keep the best B.

    Beams are ranked by their mean segment score, so longer answers are not favoured.
    """
    pool = []
    for beam, candidates in zip(beams, expansions):
        for c in candidates:
            if is_end(c):
                pool.append(Beam(list(beam.segments), list(beam.citations), list(beam.scores),
                                 list(beam.supports), beam.last_passage, done=True))
            else:
                pool.append(Beam(beam.segments + [c.text], beam.citations + [c.passage],
                                 beam.scores + [segment_score(c, **weights)],
                                 beam.supports + [support_label(c)],
                                 c.passage or beam.last_passage))
    return sorted(pool, key=lambda b: b.score, reverse=True)[:B]
