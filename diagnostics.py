"""Internal aggregation helpers for Lyapunov and activity diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True, slots=True)
class AggregateRecord:
    network_size: int
    alpha: float
    gain: float
    gain_index: int
    count: int
    lambda_max_mean: float
    lambda_max_std: float


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    network_size: int
    alpha: float
    critical_gain: float | None
    critical_gain_index: int | None


class DiagnosticIntegrityError(ValueError):
    """Raised when incomplete diagnostics would make an aggregate misleading."""


def aggregate_trials(records: Iterable[Mapping[str, object]]) -> list[AggregateRecord]:
    """Aggregate raw per-condition MLE values with population standard deviation."""

    grouped: dict[tuple[int, float, int, float], list[float]] = defaultdict(list)
    for record in records:
        if not bool(record["finite"]):
            raise DiagnosticIntegrityError(
                f"condition {record.get('condition_id', '<unknown>')} is non-finite"
            )
        key = (
            int(record["network_size"]),
            float(record["alpha"]),
            int(record["gain_index"]),
            float(record["gain"]),
        )
        value = float(record["lambda_max"])
        if not math.isfinite(value):
            raise DiagnosticIntegrityError(
                f"condition {record.get('condition_id', '<unknown>')} has non-finite lambda_max"
            )
        grouped[key].append(value)

    aggregates: list[AggregateRecord] = []
    for (network_size, alpha, gain_index, gain), values in grouped.items():
        aggregates.append(
            AggregateRecord(
                network_size=network_size,
                alpha=alpha,
                gain=gain,
                gain_index=gain_index,
                count=len(values),
                lambda_max_mean=statistics.fmean(values),
                lambda_max_std=statistics.pstdev(values),
            )
        )
    return sorted(aggregates, key=lambda row: (row.network_size, row.alpha, row.gain_index))


def locate_transitions(aggregates: Iterable[AggregateRecord]) -> list[TransitionRecord]:
    """Find the first discrete negative-to-nonnegative mean-MLE crossing."""

    grouped: dict[tuple[int, float], list[AggregateRecord]] = defaultdict(list)
    for aggregate in aggregates:
        grouped[(aggregate.network_size, aggregate.alpha)].append(aggregate)

    transitions: list[TransitionRecord] = []
    for (network_size, alpha), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row.gain_index)
        indices = [row.gain_index for row in rows]
        if indices != list(range(len(rows))):
            raise DiagnosticIntegrityError(
                f"N={network_size}, alpha={alpha:g} has an incomplete gain grid: {indices}"
            )
        counts = {row.count for row in rows}
        if len(counts) != 1:
            raise DiagnosticIntegrityError(
                f"N={network_size}, alpha={alpha:g} has inconsistent trial counts: "
                f"{sorted(counts)}"
            )
        crossing: AggregateRecord | None = None
        for previous, current in zip(rows, rows[1:], strict=False):
            if previous.lambda_max_mean < 0.0 <= current.lambda_max_mean:
                crossing = current
                break
        transitions.append(
            TransitionRecord(
                network_size=network_size,
                alpha=alpha,
                critical_gain=None if crossing is None else crossing.gain,
                critical_gain_index=None if crossing is None else crossing.gain_index,
            )
        )
    return transitions
