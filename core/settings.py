"""Typed, env-aware configuration -- validated at boot, overridable via env vars."""
from __future__ import annotations

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(str, Enum):
    dev = "dev"
    staging = "staging"
    prod = "prod"


class EncoderSettings(BaseModel):
    # local HuggingFace Matryoshka model; override SC_ENCODER__MODEL for OpenAI/API-backed models
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    octaves: tuple[int, ...] = (64, 128, 256, 512, 1024)
    batch_size: int = Field(128, ge=1)

    @field_validator("octaves")
    @classmethod
    def _ascending_and_nested(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if list(v) != sorted(v):
            raise ValueError("octaves must be ascending (nested prefixes)")
        return v


class ConeSettings(BaseModel):
    curvature: float = Field(1.0, gt=0)
    dim: int = Field(8, ge=1)
    epochs: int = Field(200, ge=1)
    lr: float = Field(1e-2, gt=0)
    margin: float = Field(0.01, gt=0)
    neg_samples: int = Field(5, ge=1)
    seed: int = 0


class StoreSettings(BaseModel):
    backend: Literal["memory", "qdrant", "pgvector"] = "memory"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    lakefs_repo: str = "semantic-cones"


class MemorySettings(BaseModel):
    budget_tokens: int = Field(2048, ge=0)
    digest_min_members: int = Field(2, ge=1)
    digest_tau: float = 0.0
    recency_lambda: float = Field(0.05, ge=0.0)
    max_pinned: int = Field(64, ge=1)
    summary_max_chars: int = Field(160, ge=8)


class ContextSettings(BaseModel):
    max_tokens: int = Field(2048, ge=0)
    overlap_threshold: float = 0.5
    distance_summary_threshold: float = 0.0
    max_members_per_node: int = Field(4, ge=1)
    reserve_tokens: int = Field(64, ge=0)
    max_dedup_candidates: int = Field(256, ge=1)


class RecursiveSettings(BaseModel):
    max_depth: int = Field(4, ge=1)
    max_breadth: int = Field(8, ge=1)
    beam_k: int = Field(3, ge=1)
    min_aperture_stop: float = Field(0.1, ge=0.0)


class AgentSettings(BaseModel):
    usage_weight: float = Field(0.0, ge=0.0)        # blend of usage feedback into ranking; 0 == pure relevance
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)  # 1.0 == pure relevance, lower == more diversity
    octave_fusion: bool = False                     # fuse rankings across octaves (RRF)
    incremental_ingest: bool = True                 # reuse cached embeddings on ingest
    consolidate_tension: float = 0.3                # tension threshold above which consolidate acts
    max_query_chars: int = Field(2048, ge=1)        # serving-side query length cap


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SC_", env_nested_delimiter="__")
    env: Env = Env.dev
    encoder: EncoderSettings = EncoderSettings()
    cone: ConeSettings = ConeSettings()
    store: StoreSettings = StoreSettings()
    memory: MemorySettings = MemorySettings()
    context: ContextSettings = ContextSettings()
    recursive: RecursiveSettings = RecursiveSettings()
    agent: AgentSettings = AgentSettings()
    enable_nla_labels: bool = False            # optional, decoupled
