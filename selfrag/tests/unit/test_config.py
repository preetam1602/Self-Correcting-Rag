import asyncio

import pytest
from pydantic import ValidationError

from selfrag.config.overrides import ConfigOverrides, OverridesNotAllowed, apply_overrides
from selfrag.config.settings import CONFIG_FILE_ENV, Settings
from selfrag.critics.critic import Critic
from selfrag.models import Filter
from selfrag.orchestrator.graph import InvalidOverride, SelfRAG, SelfRAGConfig
from selfrag.tests.unit.test_orchestrator import GOOD, FakeLLM, FakeRetriever

SECRETS = {"DATABASE_URL": "postgresql://x", "GEMINI_API_KEY": "k",
           "GENERATOR_MODEL": "gen", "CRITIC_MODEL": "crit"}


@pytest.fixture
def env(monkeypatch):
    monkeypatch.delenv(CONFIG_FILE_ENV, raising=False)
    for key, value in SECRETS.items():
        monkeypatch.setenv(key, value)
    return monkeypatch


def settings() -> Settings:
    return Settings(_env_file=None)         # ignore the developer's .env


def test_yaml_defaults_are_loaded(env):
    s = settings()
    assert (s.rrf_k, s.retrieve_threshold_delta, s.beam_width_b, s.w_is_use) == (60, 0.2, 2, 0.5)
    assert (s.critic_prob_mode, s.segment_unit) == ("verbal", "answer")


def test_deployment_yaml_overrides_defaults_and_env_overrides_both(env, tmp_path):
    deployment = tmp_path / "prod.yaml"
    deployment.write_text("retrieve_threshold_delta: 0.0\nbeam_width_b: 1\n", encoding="utf-8")
    env.setenv(CONFIG_FILE_ENV, str(deployment))
    env.setenv("BEAM_WIDTH_B", "3")
    s = settings()
    assert s.retrieve_threshold_delta == 0.0        # from the deployment YAML
    assert s.beam_width_b == 3                      # environment wins
    assert s.rrf_k == 60                            # untouched default


def test_invalid_setting_is_rejected(env):
    env.setenv("CRITIC_PROB_MODE", "guess")
    with pytest.raises(ValidationError):
        settings()


def test_config_from_settings(env):
    cfg = SelfRAGConfig.from_settings(settings())
    assert (cfg.delta, cfg.k, cfg.B, cfg.critic_mode) == (0.2, 5, 2, "verbal")


# ---------------------------------------------------------------- overrides

def test_overrides_in_range_are_applied_to_a_copy():
    base = SelfRAGConfig()
    cfg = apply_overrides(base, ConfigOverrides(delta=0.1, k=3, w_sup=2.0), is_admin=True)
    assert (cfg.delta, cfg.k, cfg.w_sup, cfg.B) == (0.1, 3, 2.0, base.B)
    assert base.delta == 0.2                        # deployment config untouched


@pytest.mark.parametrize("bad", [{"delta": 1.5}, {"k": 0}, {"B": 9}, {"w_use": -1},
                                 {"critic_mode": "guess"}, {"max_segments": 2}])
def test_out_of_range_or_unknown_overrides_are_rejected(bad):
    with pytest.raises(ValidationError):
        ConfigOverrides(**bad)


def test_overrides_are_admin_only():
    with pytest.raises(OverridesNotAllowed):
        apply_overrides(SelfRAGConfig(), ConfigOverrides(delta=0.1), is_admin=False)
    # no overrides is fine for anyone
    assert apply_overrides(SelfRAGConfig(), ConfigOverrides(), is_admin=False) == SelfRAGConfig()
    assert apply_overrides(SelfRAGConfig(), None, is_admin=False) == SelfRAGConfig()


# ---------------------------------------------------------------- per request in the loop

class RecordingRetriever(FakeRetriever):
    async def retrieve(self, query, filters, top_n, variants=()):
        self.top_ns = getattr(self, "top_ns", []) + [top_n]
        return await super().retrieve(query, filters, top_n, variants)


def make_rag():
    llm, retriever = FakeLLM(), RecordingRetriever([GOOD])
    return SelfRAG(Critic(llm, "verbal"), llm, llm, retriever, SelfRAGConfig(k=5)), llm, retriever


def ask(rag, overrides=None, is_admin=True):
    return asyncio.run(rag.respond("Q?", Filter(tenant_id="t"), overrides, is_admin=is_admin))


def test_request_override_changes_this_request_only():
    rag, _, retriever = make_rag()
    assert ask(rag, ConfigOverrides(k=2)).status == "ok"
    assert ask(rag).status == "ok"
    assert retriever.top_ns == [2, 5]
    assert rag.cfg.k == 5


def test_request_can_switch_critic_mode():
    rag, llm, _ = make_rag()
    ask(rag, ConfigOverrides(critic_mode="sample"))
    assert any("Reply with one word" in p for p in llm.prompts)
    assert not any("Reply with JSON only" in p for p in llm.prompts)


def test_logprobs_override_is_invalid_without_logprob_support():
    rag, *_ = make_rag()
    with pytest.raises(InvalidOverride):
        ask(rag, ConfigOverrides(critic_mode="logprobs"))


def test_non_admin_overrides_rejected_before_streaming():
    rag, llm, _ = make_rag()

    async def first_event():
        return await rag.stream("Q?", Filter(tenant_id="t"), ConfigOverrides(B=1)).__anext__()
    with pytest.raises(OverridesNotAllowed):
        asyncio.run(first_event())
    assert llm.prompts == []                        # nothing ran


def test_config_snapshot_is_kept_in_state():
    rag, *_ = make_rag()
    cfg = rag.resolve_config(ConfigOverrides(delta=0.4), is_admin=True)
    state = asyncio.run(rag.answer("Q?", Filter(tenant_id="t"), cfg=cfg))
    assert state["config"].delta == 0.4
