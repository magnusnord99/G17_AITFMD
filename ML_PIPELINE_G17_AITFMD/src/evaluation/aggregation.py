"""ROI-level and patient-level aggregation of patch inference results."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.inference import InferenceResult


def mean_prob_aggregate(
    result: InferenceResult,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Aggreger patch-sannsynligheter til ett ROI-estimat via gjennomsnitt.

    Gjennomsnitt av P(class=1) over alle patcher terskles til ROI-prediksjon.
    """
    if result.n_patches == 0:
        return {
            "roi_id": result.roi_id,
            "mean_prob": float("nan"),
            "pred_label": -1,
            "n_patches": 0,
        }
    mean_p = float(np.mean(result.probs))
    return {
        "roi_id": result.roi_id,
        "mean_prob": mean_p,
        "pred_label": int(mean_p >= threshold),
        "n_patches": result.n_patches,
    }


def majority_vote_aggregate(
    result: InferenceResult,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Aggreger patch-prediksjoner til ett ROI-estimat via majoritetsavstemning.

    threshold brukes ikke direkte i selve votering, men lagres for sporbarhet.
    """
    if result.n_patches == 0:
        return {
            "roi_id": result.roi_id,
            "vote_frac_class1": float("nan"),
            "pred_label": -1,
            "n_patches": 0,
        }
    counts = np.bincount(result.preds.astype(int), minlength=2)
    maj_label = int(np.argmax(counts))
    vote_frac = float(counts[1]) / result.n_patches
    return {
        "roi_id": result.roi_id,
        "vote_frac_class1": vote_frac,
        "pred_label": maj_label,
        "n_patches": result.n_patches,
    }


def patient_level_aggregate(
    roi_results: list[dict[str, Any]],
    strategy: str = "mean_prob",
) -> list[dict[str, Any]]:
    """Aggreger ROI-prediksjoner til pasientnivå via majoritetsavstemning over ROI-er.

    strategy: "mean_prob" bruker pred_mean_prob, "majority_vote" bruker pred_majority.
    """
    from collections import defaultdict

    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in roi_results:
        by_patient[r["patient_id"]].append(r)

    pred_key = "pred_mean_prob" if strategy == "mean_prob" else "pred_majority"
    patient_results: list[dict[str, Any]] = []

    for pid, rows in sorted(by_patient.items()):
        labels = {int(r["true_label"]) for r in rows}
        true_label = int(rows[0]["true_label"])

        preds = [int(r[pred_key]) for r in rows]
        pred_label = 1 if sum(preds) >= len(preds) / 2 else 0

        patient_results.append(
            {
                "patient_id": pid,
                "true_label": true_label,
                "pred_label": pred_label,
                "n_rois": len(rows),
                "inconsistent_labels": len(labels) > 1,
            }
        )

    return patient_results
