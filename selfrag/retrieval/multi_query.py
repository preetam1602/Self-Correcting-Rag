from selfrag.providers.interfaces import LLMClient

PARAPHRASE_PROMPT = """Write {n} different search queries that ask for the same information as the
query below, using different wording. One query per line, no numbering.
Query: {query}"""


async def make_paraphrases(query: str, n: int, llm: LLMClient) -> list[str]:
    if n <= 0:
        return []
    out = await llm.complete(PARAPHRASE_PROMPT.format(n=n, query=query), max_tokens=200,
                             temperature=0.0)
    lines = [line.strip(" -*0123456789.)\t") for line in out.text.splitlines()]
    return [line for line in lines if line and line != query][:n]
