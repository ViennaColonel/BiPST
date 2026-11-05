"""Public signed revision and progressive negative-prompt construction.

All inputs are predictions, masks and prompt states; manual references are absent.
The constants TARGET, INCREMENTS and paired replacement budgets define the paper
policy. They are not a round-by-round point-count schedule.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import copy, json
import cv2
import numpy as np

TARGET = 100
INCREMENTS = [5, 10, 20]
VARIANTS = ["sam_ranked"]


@dataclass
class PromptConfig:
    """Scoring and anchor settings of the reported public prompt method."""

    foreground_confidence: float = 0.85
    background_confidence: float = 0.15
    min_positive_prompts: int = 3
    max_positive_prompts: int = 10
    min_negative_prompts: int = 3
    max_negative_prompts: int = 10
    prompt_min_distance: int = 20
    max_retained_positive_prompts: int = 100
    max_retained_negative_prompts: int = 100
    max_total_positive_prompts: int = 110
    max_total_negative_prompts: int = 110
    prompt_probability_weight: float = 0.5
    prompt_distance_weight: float = 0.3
    prompt_stability_weight: float = 0.2


@dataclass
class PromptState:
    """Native-image [x,y] coordinates, separated by positive and negative signs."""

    positive_points: list[list[int]]
    negative_points: list[list[int]]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        positives = np.asarray(self.positive_points, dtype=np.int32).reshape(-1, 2)
        negatives = np.asarray(self.negative_points, dtype=np.int32).reshape(-1, 2)
        points = np.concatenate([positives, negatives], axis=0)
        labels = np.concatenate(
            [np.ones(len(positives), np.int32), np.zeros(len(negatives), np.int32)]
        )
        return (points, labels)

    def as_dict(self) -> dict:
        return {"positive_points": self.positive_points, "negative_points": self.negative_points}


def _prompt_candidate_records(
    region: np.ndarray,
    class_confidence: np.ndarray,
    variance: np.ndarray,
    source: str,
    max_points: int,
    config: PromptConfig,
) -> list[dict]:
    """Rank confidence, normalized interior distance and view stability, then spatially suppress nearby candidates."""
    binary = region.astype(np.uint8)
    if not binary.any() or max_points <= 0:
        return []
    bordered = binary.copy()
    bordered[[0, -1], :] = 0
    bordered[:, [0, -1]] = 0
    distance = cv2.distanceTransform(bordered, cv2.DIST_L2, 5)
    distance_normalized = distance / max(float(distance.max()), 1e-07)
    stability = np.clip(1.0 - 4.0 * variance, 0.0, 1.0)
    score_map = (
        config.prompt_probability_weight * class_confidence
        + config.prompt_distance_weight * distance_normalized
        + config.prompt_stability_weight * stability
    )
    score_map = np.where(binary > 0, score_map, -1.0)
    working = score_map.copy()
    yy, xx = np.ogrid[: binary.shape[0], : binary.shape[1]]
    records: list[dict] = []
    for _ in range(max_points):
        flat_index = int(np.argmax(working))
        score = float(working.flat[flat_index])
        if score < 0:
            break
        y, x = np.unravel_index(flat_index, working.shape)
        records.append(
            {
                "point": [int(x), int(y)],
                "confidence": score,
                "class_probability": float(class_confidence[y, x]),
                "distance_score": float(distance_normalized[y, x]),
                "tta_stability": float(stability[y, x]),
                "source": source,
            }
        )
        suppression = (xx - x) ** 2 + (yy - y) ** 2 <= config.prompt_min_distance**2
        working[suppression] = -1.0
    return records


def _retained_prompt_records(
    points: Sequence[Sequence[int]],
    probability: np.ndarray,
    variance: np.ndarray,
    positive: bool,
    limit: int,
    config: PromptConfig,
) -> list[dict]:
    """Rank historical anchors by model support; support is not reference correctness."""
    h, w = probability.shape
    records: list[dict] = []
    for point in points:
        x = int(np.clip(point[0], 0, w - 1))
        y = int(np.clip(point[1], 0, h - 1))
        foreground_probability = float(probability[y, x])
        valid = (
            foreground_probability >= config.background_confidence
            if positive
            else foreground_probability <= config.foreground_confidence
        )
        if not valid:
            continue
        class_probability = foreground_probability if positive else 1.0 - foreground_probability
        stability = float(np.clip(1.0 - 4.0 * variance[y, x], 0.0, 1.0))
        confidence = (
            config.prompt_probability_weight + config.prompt_distance_weight
        ) * class_probability + config.prompt_stability_weight * stability
        records.append(
            {
                "point": [x, y],
                "confidence": float(confidence),
                "class_probability": class_probability,
                "distance_score": 0.0,
                "tta_stability": stability,
                "source": "retained_positive" if positive else "retained_negative",
            }
        )
    records.sort(key=lambda record: record["confidence"], reverse=True)
    return records[:limit]


def _compose_budgeted_prompt_records(
    retained: Sequence[dict],
    corrections: Sequence[dict],
    fillers: Sequence[dict],
    previous_count: int,
    minimum_points: int,
    maximum_points: int,
) -> list[dict]:
    """Fill each sign using disagreement points, retained anchors, then agreement fillers."""
    target_count = min(maximum_points, max(minimum_points, previous_count))
    selected: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for group in (corrections, retained, fillers):
        for record in sorted(group, key=lambda item: item["confidence"], reverse=True):
            point = tuple(record["point"])
            if point in seen:
                continue
            selected.append(record)
            seen.add(point)
            if len(selected) >= target_count:
                return selected
    return selected


def propose_bidirectional_prompts(
    probability: np.ndarray,
    variance: np.ndarray,
    previous_teacher_mask: np.ndarray,
    previous_state: PromptState,
    config: PromptConfig,
) -> tuple[PromptState, dict]:
    """Construct the broader anchor-revision branch from the main student."""
    teacher_foreground = previous_teacher_mask.astype(bool)
    confident_foreground = probability > config.foreground_confidence
    confident_background = probability < config.background_confidence
    positive_disagreement = np.logical_and(confident_foreground, ~teacher_foreground)
    negative_disagreement = np.logical_and(confident_background, teacher_foreground)
    positive_agreement = np.logical_and(confident_foreground, teacher_foreground)
    negative_agreement = np.logical_and(confident_background, ~teacher_foreground)
    retained_positive = _retained_prompt_records(
        previous_state.positive_points,
        probability,
        variance,
        True,
        config.max_retained_positive_prompts,
        config,
    )
    retained_negative = _retained_prompt_records(
        previous_state.negative_points,
        probability,
        variance,
        False,
        config.max_retained_negative_prompts,
        config,
    )
    positive_corrections = _prompt_candidate_records(
        positive_disagreement,
        probability,
        variance,
        "positive_disagreement",
        config.max_positive_prompts,
        config,
    )
    negative_corrections = _prompt_candidate_records(
        negative_disagreement,
        1.0 - probability,
        variance,
        "negative_disagreement",
        config.max_negative_prompts,
        config,
    )
    positive_fillers = _prompt_candidate_records(
        positive_agreement,
        probability,
        variance,
        "positive_agreement_anchor",
        config.max_positive_prompts,
        config,
    )
    negative_fillers = _prompt_candidate_records(
        negative_agreement,
        1.0 - probability,
        variance,
        "negative_agreement_anchor",
        config.max_negative_prompts,
        config,
    )
    positive_records = _compose_budgeted_prompt_records(
        retained_positive,
        positive_corrections,
        positive_fillers,
        len(previous_state.positive_points),
        config.min_positive_prompts,
        config.max_total_positive_prompts,
    )
    negative_records = _compose_budgeted_prompt_records(
        retained_negative,
        negative_corrections,
        negative_fillers,
        len(previous_state.negative_points),
        config.min_negative_prompts,
        config.max_total_negative_prompts,
    )
    if not positive_records:
        y, x = np.unravel_index(int(np.argmax(probability)), probability.shape)
        positive_records = [
            {
                "point": [int(x), int(y)],
                "confidence": float(probability[y, x]),
                "class_probability": float(probability[y, x]),
                "distance_score": 0.0,
                "tta_stability": float(np.clip(1.0 - 4.0 * variance[y, x], 0.0, 1.0)),
                "source": "positive_probability_fallback",
            }
        ]
    state = PromptState(
        positive_points=[record["point"] for record in positive_records],
        negative_points=[record["point"] for record in negative_records],
    )
    details = {
        "positive_records": positive_records,
        "negative_records": negative_records,
        "positive_disagreement_pixels": int(positive_disagreement.sum()),
        "negative_disagreement_pixels": int(negative_disagreement.sum()),
        "positive_disagreement_ratio": float(positive_disagreement.mean()),
        "negative_disagreement_ratio": float(negative_disagreement.mean()),
    }
    return (state, details)


def prompt_state_equal(a: PromptState, b: PromptState) -> bool:
    return {tuple(point) for point in a.positive_points} == {
        tuple(point) for point in b.positive_points
    } and {tuple(point) for point in a.negative_points} == {
        tuple(point) for point in b.negative_points
    }


def records(region, confidence, variance, limit):
    return _prompt_candidate_records(
        region, confidence, variance, "cross_image", limit, PromptConfig()
    )


def replace_points(old, corrections, confidence, budget):
    """Replace weakly supported same-sign slots, preserving other coordinates. Sparse sets can grow toward three points."""
    result = [list(q) for q in old]
    original = len(result)
    edits = []
    ranked = sorted(range(len(result)), key=lambda j: float(confidence[result[j][1], result[j][0]]))
    replacements = []
    for rec in corrections:
        q = rec["point"]
        if q in result or q in replacements:
            continue
        replacements.append(q)
        if len(replacements) >= budget:
            break
    for t, q in enumerate(replacements):
        if t < len(ranked):
            j = ranked[t]
            edits.append(dict(remove=result[j], insert=q))
            result[j] = q
        elif len(result) < max(3, original):
            result.append(q)
            edits.append(dict(remove=None, insert=q))
    return (result, edits)


def suppress(available, x, y, radius):
    h, w = available.shape
    x0, x1 = (max(0, x - radius), min(w, x + radius + 1))
    y0, y1 = (max(0, y - radius), min(h, y + radius + 1))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    patch = available[y0:y1, x0:x1]
    patch[(xx - x) ** 2 + (yy - y) ** 2 <= radius**2] = False


def negative_pool(views, oldmask, excluded_positive):
    """Build up to 100 ranked locations: confident foreground disagreement, confident exterior background, then low-probability fallback. A pool is not an adopted prompt state."""
    prob = views.mean(0)
    var = views.var(0)
    confident = views.max(0) < 0.15
    available = np.ones(prob.shape, bool)
    excluded = np.zeros(prob.shape, bool)
    for x, y in excluded_positive:
        excluded[y, x] = True
    available[excluded] = False
    assert int(available.sum()) >= TARGET
    stability = np.clip(1 - 4 * var, 0, 1)
    selected = []
    for label, region in [
        ("confident_disagreement", confident & oldmask),
        ("confident_agreement", confident & ~oldmask),
        ("probability_fallback", ~excluded),
    ]:
        if label == "probability_fallback":
            score = 1 - prob
        else:
            bordered = region.astype(np.uint8)
            bordered[[0, -1], :] = 0
            bordered[:, [0, -1]] = 0
            distance = cv2.distanceTransform(bordered, cv2.DIST_L2, 5)
            distance /= max(float(distance.max()), 1e-07)
            score = 0.5 * (1 - prob) + 0.3 * distance + 0.2 * stability
        while len(selected) < TARGET:
            candidates = region & available
            if not candidates.any():
                break
            index = int(np.argmax(np.where(candidates, score, -np.inf)))
            y, x = np.unravel_index(index, prob.shape)
            selected.append(
                dict(
                    point=[int(x), int(y)],
                    source=label,
                    mean_foreground_probability=float(prob[y, x]),
                    max_view_foreground_probability=float(views[:, y, x].max()),
                    confident=bool(confident[y, x]),
                    spacing_relaxed=False,
                )
            )
            suppress(available, int(x), int(y), 20)
        if len(selected) == TARGET:
            break
    if len(selected) < TARGET:
        available = ~excluded
        for rec in selected:
            x, y = rec["point"]
            available[y, x] = False
        while len(selected) < TARGET:
            index = int(np.argmin(np.where(available, prob, np.inf)))
            y, x = np.unravel_index(index, prob.shape)
            assert available[y, x]
            selected.append(
                dict(
                    point=[int(x), int(y)],
                    source="spacing_fallback",
                    mean_foreground_probability=float(prob[y, x]),
                    max_view_foreground_probability=float(views[:, y, x].max()),
                    confident=bool(confident[y, x]),
                    spacing_relaxed=True,
                )
            )
            available[y, x] = False
    assert len({tuple(r["point"]) for r in selected}) == TARGET
    return selected


def grow_negative(state, pool, target):
    """Add eligible coordinates to a revised state subject to its target and 20-pixel negative-point spacing."""
    result = copy.deepcopy(state.negative_points)
    assert len(result) == len({tuple(q) for q in result}) <= target
    positive = {tuple(q) for q in state.positive_points}
    added = []
    for record in pool:
        if len(result) >= target:
            break
        q = record["point"]
        if tuple(q) in positive or any(
            ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 <= 20**2 for p in result)
        ):
            continue
        result.append(list(q))
        added.append(record)
    return (
        PromptState(copy.deepcopy(state.positive_points), result),
        dict(
            target_count=target,
            before_fill=len(state.negative_points),
            added_count=len(added),
            low_confidence_added=sum((not r["confident"] for r in added)),
            points=added,
            filled_target=len(result) == target,
        ),
    )


def candidate_bank(old, oldmask, v, vc, config, pool_kind="structured"):
    """Pair replacement/growth budgets (1,5), (3,10), (6,20); do not form a Cartesian product. Both prediction sources and three anchor branches yield at most 21 distinct states."""
    proposed, _ = propose_bidirectional_prompts(v.mean(0), v.var(0), oldmask, old, config)
    n = len(old.negative_points)
    raw = [
        ("anchor_positive", PromptState(proposed.positive_points, old.negative_points), "self", 0),
        ("anchor_negative", PromptState(old.positive_points, proposed.negative_points), "self", 20),
        ("anchor_joint", proposed, "self", 20),
    ]
    for label, evidence in [("self", v), ("cross", vc)]:
        prob, var = (evidence.mean(0), evidence.var(0))
        ps = records((evidence.min(0) > 0.85) & ~oldmask, prob, var, 6)
        ns = records((evidence.max(0) < 0.15) & oldmask, 1 - prob, var, 6)
        for b, increment in zip((1, 3, 6), INCREMENTS):
            pp, _ = replace_points(old.positive_points, ps, prob, b)
            nn, _ = replace_points(old.negative_points, ns, 1 - prob, b)
            raw += [
                (f"{label}_positive_{b}", PromptState(pp, old.negative_points), label, 0),
                (
                    f"{label}_negative_{b}_grow{increment}",
                    PromptState(old.positive_points, nn),
                    label,
                    increment,
                ),
                (f"{label}_joint_{b}_grow{increment}", PromptState(pp, nn), label, increment),
            ]
    excluded = {tuple(q) for _, st, _, _ in raw for q in st.positive_points}
    pool_builder = negative_pool if pool_kind == "structured" else probability_pool
    pools = {
        label: pool_builder(evidence, oldmask, excluded)
        for label, evidence in [("self", v), ("cross", vc)]
    }
    bank, seen = ([], set())
    for name, st, label, increment in raw:
        if increment:
            target = min(100, n + increment)
            if len(st.negative_points) > target:
                prob = (v if label == "self" else vc).mean(0)
                nn = sorted(st.negative_points, key=lambda q: float(prob[q[1], q[0]]))[:target]
                st = PromptState(st.positive_points, nn)
            st, details = grow_negative(st, pools[label], target)
        else:
            details = dict(
                target_count=n,
                before_fill=n,
                added_count=0,
                low_confidence_added=0,
                points=[],
                filled_target=True,
            )
        details.update(
            source=label,
            requested_increment=increment,
            predecessor_count=n,
            actual_increment=len(st.negative_points) - n,
        )
        signature = json.dumps(st.as_dict(), sort_keys=True)
        if signature in seen or prompt_state_equal(old, st):
            continue
        seen.add(signature)
        assert n <= len(st.negative_points) <= min(100, n + 20)
        bank.append((name, st, details))
    assert 0 <= len(bank) <= 21
    return bank


def choose(records, variant, step):
    """Require actual negative-count growth, rank SAM-predicted IoU, prefer fewer negatives on ties, and retain first-generated state on complete ties. No score comparison to the predecessor."""
    if variant != "sam_ranked":
        raise ValueError("This release exposes the reported SAM ranking policy.")
    for r in records:
        r["eligible"] = r["negative_growth"]["actual_increment"] > 0
    eligible = [r for r in records if r["eligible"]]
    return (
        max(eligible, key=lambda r: (r["sam_predicted_iou"], -len(r["state"]["negative_points"])))
        if eligible
        else None
    )


def probability_pool(views, oldmask, excluded_positive):
    """Control that changes only negative-growth-pool ranking to 1-mean foreground probability."""
    prob = views.mean(0)
    available = np.ones(prob.shape, bool)
    excluded = np.zeros_like(available)
    for x, y in excluded_positive:
        excluded[y, x] = True
    available[excluded] = False
    selected = []
    for relaxed in [False, True]:
        if relaxed:
            available = ~excluded
            for rec in selected:
                x, y = rec["point"]
                available[y, x] = False
        while len(selected) < 100 and available.any():
            index = int(np.argmax(np.where(available, 1 - prob, -np.inf)))
            y, x = np.unravel_index(index, prob.shape)
            selected.append(
                dict(
                    point=[int(x), int(y)],
                    source="probability_only",
                    mean_foreground_probability=float(prob[y, x]),
                    max_view_foreground_probability=float(views[:, y, x].max()),
                    confident=bool(views[:, y, x].max() < 0.15),
                    spacing_relaxed=relaxed,
                )
            )
            if relaxed:
                available[y, x] = False
            else:
                suppress(available, int(x), int(y), 20)
        if len(selected) == 100:
            break
    assert len(selected) == 100
    return selected
