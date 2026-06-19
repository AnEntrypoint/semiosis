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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SC_", env_nested_delimiter="__")
    env: Env = Env.dev
    encoder: EncoderSettings = EncoderSettings()
    cone: ConeSettings = ConeSettings()
    store: StoreSettings = StoreSettings()
    enable_nla_labels: bool = False            # optional, decoupled
