"""Typed, env-aware configuration -- validated at boot, overridable via env vars."""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(str, Enum):
    dev = "dev"
    staging = "staging"
    prod = "prod"


class EncoderSettings(BaseModel):
    # real Matryoshka model (native dim 768); override SC_ENCODER__MODEL for OpenAI/API-backed models
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    native_dim: int = Field(768, ge=1)  # native output dim of `model`; octaves must not exceed this
    octaves: tuple[int, ...] = (64, 128, 256, 512, 768)
    batch_size: int = Field(128, ge=1)
    device: str | None = None   # 'cpu'|'cuda'|None(auto); set cpu for zero-VRAM
    fp16: bool = False          # half-precision weights; halves encoder VRAM on GPU

    @field_validator("octaves")
    @classmethod
    def _ascending_and_nested(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if list(v) != sorted(v):
            raise ValueError("octaves must be ascending (nested prefixes)")
        if len(set(v)) != len(v):
            raise ValueError("octaves must be distinct")
        return v

    @field_validator("octaves")
    @classmethod
    def _within_native_dim(cls, v: tuple[int, ...], info) -> tuple[int, ...]:
        native = info.data.get("native_dim")
        if native is not None and any(o > native for o in v):
            raise ValueError(f"octaves {v} exceed native_dim={native}; slicing past it duplicates the last real octave")
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
    hilbert_partitions: int = Field(16, ge=1)   # Hilbert-bucket count per octave; VStream-style partition template
    catapult_cache_size: int = Field(512, ge=1)  # LRU-bounded query-locality shortcut cache (CatapultDB)
    bm25_k1: float = Field(1.5, gt=0)
    bm25_b: float = Field(0.75, ge=0.0, le=1.0)


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
    entropy_weight: float = Field(0.0, ge=0.0, le=1.0)  # 0 = no entropy weighting; 1 = aggressive


class RecursiveSettings(BaseModel):
    max_depth: int = Field(4, ge=1)
    max_breadth: int = Field(8, ge=1)
    beam_k: int = Field(3, ge=1)
    min_aperture_stop: float = Field(0.1, ge=0.0)


class ResearchSettings(BaseModel):
    max_cycles: int = Field(8, ge=1)                       # hard cap; fail loud past it
    convergence_energy_delta: float = Field(0.01, ge=0.0)  # stop when energy improves less than this
    frontier_top_k: int = Field(5, ge=1)                   # candidate regions per propose stage
    min_support_score: float = Field(0.5, ge=0.0, le=1.0)  # observation success threshold
    max_no_observation: int = Field(3, ge=1)               # bound on empty-observation skips
    instruction_persist_path: str = ""                     # sidecar for refined instructions; empty == no persist


class AgentSettings(BaseModel):
    usage_weight: float = Field(0.0, ge=0.0)        # blend of usage feedback into ranking; 0 == pure relevance
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)  # 1.0 == pure relevance, lower == more diversity
    octave_fusion: bool = False                     # fuse rankings across octaves (RRF)
    hybrid_lexical: bool = False                    # fuse BM25 lexical rank into the RRF accumulator
    incremental_ingest: bool = True                 # reuse cached embeddings on ingest
    consolidate_tension: float = 0.3                # tension threshold above which consolidate acts
    max_query_chars: int = Field(2048, ge=1)        # serving-side query length cap
    hybrid_score_cosine_weight: float = Field(0.7, ge=0.0, le=1.0)   # hybrid_score: SBERT cosine share
    activation_blend_encoder_weight: float = Field(0.8, ge=0.0, le=1.0)  # activation_embed: encoder share


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SC_", env_nested_delimiter="__")
    env: Env = Env.dev
    encoder: EncoderSettings = EncoderSettings()
    cone: ConeSettings = ConeSettings()
    store: StoreSettings = StoreSettings()
    memory: MemorySettings = MemorySettings()
    context: ContextSettings = ContextSettings()
    recursive: RecursiveSettings = RecursiveSettings()
    research: ResearchSettings = ResearchSettings()
    agent: AgentSettings = AgentSettings()
    enable_nla_labels: bool = False            # optional, decoupled
