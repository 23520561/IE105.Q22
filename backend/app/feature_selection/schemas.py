from typing import Any, Dict, List, Literal

from pydantic import BaseModel

from .library import FeatureInfo, FeaturesSummary


class FilterRequest(BaseModel):
    data: list[dict]
    target: str


class RfeRequest(BaseModel):
    n_features: int = 10


class RfeResponse(BaseModel):
    recommended_to_keep: list[str]
    feature_ranking: dict[str, int]
    n_features_kept: int
    estimator_used: str
    feature_importances: dict[str, float]


class BackwardRequest(BaseModel):
    min_features: int = 10


class ReductionRequest(BaseModel):
    data: List[Dict]
    method: Literal["pca", "umap"] = "pca"
    n_components: int = 2


class AnalyzedFeatures(BaseModel):
    select: List[str]
    summary: FeaturesSummary
    detail: Dict[str, FeatureInfo]


class ReductionResponse(BaseModel):
    method: Literal["pca", "umap"]
    data: Any
