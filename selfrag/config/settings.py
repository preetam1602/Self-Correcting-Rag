"""Deployment settings: YAML defaults, overridden by a deployment YAML, .env and environment."""
import os
from pathlib import Path
from typing import Literal

from pydantic_settings import (BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict,
                               YamlConfigSettingsSource)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_FILE = Path(__file__).parent / "defaults.yaml"
CONFIG_FILE_ENV = "SELFRAG_CONFIG_FILE"     # optional per-deployment YAML, layered on the defaults


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # Secrets and connections (environment only)
    database_url: str
    qdrant_api_key: str | None = None
    llm_provider: Literal["gemini", "groq", "ollama"] = "groq"
    gemini_api_key: str | None = None       # required when llm_provider is gemini
    groq_api_key: str | None = None         # required when llm_provider is groq
    ollama_url: str = "http://localhost:11434"  # local server; no key needed
    generator_model: str
    critic_model: str
    generator_fallback_models: str = ""     # comma-separated, tried in order on overload/quota
    critic_fallback_models: str = ""

    # Stores
    qdrant_url: str
    qdrant_collection: str

    # Embedding and reranker models (local)
    embedding_model: str
    sparse_model: str
    reranker_model: str

    # Chunking
    chunk_min_tokens: int
    chunk_max_tokens: int
    chunk_overlap_ratio: float

    # Retrieval
    dense_top_k: int
    sparse_top_k: int
    rrf_k: int
    fused_top_n: int
    multi_query_variants: int

    # LLMs
    critic_prob_mode: Literal["logprobs", "sample", "verbal"]
    critic_n_samples: int
    generator_temperature: float
    llm_max_concurrency: int
    llm_max_retries: int
    llm_timeout_s: float
    llm_thinking_level: Literal["minimal", "low", "medium", "high"] | None
    llm_fallback_cooldown_s: float
    llm_reasoning_headroom: int             # Groq: extra completion tokens for reasoning

    # Local LLMs (Ollama)
    ollama_max_concurrency: int
    ollama_timeout_s: float
    ollama_num_thread: int | None           # None: half the logical cores
    ollama_num_ctx: int | None              # None: the model's default
    ollama_keep_alive: str

    # Self-RAG
    passages_per_step_k: int
    retrieve_threshold_delta: float
    force_retrieve_first: bool
    beam_width_b: int
    max_segments: int
    segment_unit: Literal["sentence", "answer"]
    w_is_rel: float
    w_is_sup: float
    w_is_use: float
    hard_constraint_no_support: bool
    min_rel_to_generate: float
    max_query_rewrites: int

    @classmethod
    def settings_customise_sources(cls, settings_cls: type[BaseSettings],
                                   init_settings: PydanticBaseSettingsSource,
                                   env_settings: PydanticBaseSettingsSource,
                                   dotenv_settings: PydanticBaseSettingsSource,
                                   file_secret_settings: PydanticBaseSettingsSource):
        yaml_files = [DEFAULTS_FILE]
        if os.environ.get(CONFIG_FILE_ENV):
            yaml_files.append(Path(os.environ[CONFIG_FILE_ENV]))
        return (init_settings, env_settings, dotenv_settings,
                YamlConfigSettingsSource(settings_cls, yaml_file=yaml_files),
                file_secret_settings)
