import pytest

from heavyrnn.diagnostics import (
    AggregateRecord,
    DiagnosticIntegrityError,
    aggregate_trials,
    locate_transitions,
)


def test_aggregate_trials_uses_population_standard_deviation() -> None:
    records = [
        {
            "network_size": 8,
            "alpha": 1.5,
            "gain_index": 0,
            "gain": 0.5,
            "lambda_max": 1.0,
            "finite": True,
        },
        {
            "network_size": 8,
            "alpha": 1.5,
            "gain_index": 0,
            "gain": 0.5,
            "lambda_max": 3.0,
            "finite": True,
        },
        {
            "network_size": 8,
            "alpha": 1.5,
            "gain_index": 1,
            "gain": 1.0,
            "lambda_max": -0.25,
            "finite": True,
        },
    ]

    aggregates = aggregate_trials(records)

    assert len(aggregates) == 2
    assert aggregates[0].count == 2
    assert aggregates[0].lambda_max_mean == pytest.approx(2.0)
    assert aggregates[0].lambda_max_std == pytest.approx(1.0)
    assert aggregates[1].count == 1
    assert aggregates[1].lambda_max_std == 0.0


def test_aggregate_rejects_nonfinite_trials() -> None:
    with pytest.raises(DiagnosticIntegrityError, match="non-finite"):
        aggregate_trials(
            [
                {
                    "condition_id": "bad-condition",
                    "network_size": 8,
                    "alpha": 1.5,
                    "gain_index": 0,
                    "gain": 0.5,
                    "lambda_max": 1.0,
                    "finite": False,
                }
            ]
        )


def test_transition_uses_first_discrete_negative_to_nonnegative_crossing() -> None:
    rows = [
        AggregateRecord(8, 1.0, 0.25, 0, 3, -0.4, 0.1),
        AggregateRecord(8, 1.0, 0.50, 1, 3, 0.0, 0.1),
        AggregateRecord(8, 1.0, 1.00, 2, 3, -0.1, 0.1),
        AggregateRecord(8, 1.0, 2.00, 3, 3, 0.3, 0.1),
        AggregateRecord(8, 2.0, 0.25, 0, 3, -0.5, 0.1),
        AggregateRecord(8, 2.0, 0.50, 1, 3, -0.2, 0.1),
    ]

    transitions = locate_transitions(reversed(rows))

    assert transitions[0].network_size == 8
    assert transitions[0].alpha == 1.0
    assert transitions[0].critical_gain == 0.50
    assert transitions[0].critical_gain_index == 1
    assert transitions[1].alpha == 2.0
    assert transitions[1].critical_gain is None
    assert transitions[1].critical_gain_index is None


def test_transition_rejects_missing_gain_rows() -> None:
    with pytest.raises(DiagnosticIntegrityError, match="incomplete gain grid"):
        locate_transitions(
            [
                AggregateRecord(8, 1.0, 0.25, 0, 3, -0.4, 0.1),
                AggregateRecord(8, 1.0, 1.00, 2, 3, 0.2, 0.1),
            ]
        )
