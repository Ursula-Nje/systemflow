"""Boundary tests for the TEWS bands and level thresholds (no MCP, no model)."""

import pytest

from systemflow import sats


def pts(**kw):
    base = dict(respiratory_rate=12, heart_rate=70, systolic_bp=120, temperature_c=36.8, avpu="A", mobility="walking")
    base.update(kw)
    return sats.score(**base)["tews_total"]


@pytest.mark.parametrize(
    "rr,expected", [(8, 2), (9, 0), (14, 0), (15, 1), (20, 1), (21, 2), (29, 2), (30, 3)]
)
def test_respiratory_rate_bands(rr, expected):
    assert pts(respiratory_rate=rr) == expected


@pytest.mark.parametrize(
    "hr,expected", [(40, 2), (41, 1), (50, 1), (51, 0), (100, 0), (101, 1), (110, 1), (111, 2), (129, 2), (130, 3)]
)
def test_heart_rate_bands(hr, expected):
    assert pts(heart_rate=hr) == expected


@pytest.mark.parametrize(
    "sbp,expected", [(70, 3), (71, 2), (80, 2), (81, 1), (100, 1), (101, 0), (199, 0), (200, 2)]
)
def test_systolic_bands(sbp, expected):
    assert pts(systolic_bp=sbp) == expected


@pytest.mark.parametrize("temp,expected", [(34.9, 2), (35.0, 0), (38.4, 0), (38.5, 2)])
def test_temperature_bands(temp, expected):
    assert pts(temperature_c=temp) == expected


@pytest.mark.parametrize("total,level", [(0, "GREEN"), (2, "GREEN"), (3, "YELLOW"), (4, "YELLOW"), (5, "ORANGE"), (6, "ORANGE"), (7, "RED")])
def test_level_thresholds(total, level):
    assert sats.level_from_tews(total) == level


def test_missing_values_are_flagged_not_hidden():
    result = sats.score(complaint="headache", systolic_bp=120)
    assert result["incomplete"] and "INCOMPLETE" in result["warnings"][0]


def test_implausible_value_is_refused():
    with pytest.raises(ValueError):
        sats.score(respiratory_rate=140)


def test_discriminator_can_raise_but_never_lower():
    high = sats.score(complaint="chest pain", respiratory_rate=12, heart_rate=70, systolic_bp=120, temperature_c=36.8, avpu="A", mobility="walking")
    assert high["level"] == "ORANGE" and high["basis"] == "discriminator"
    shock = sats.score(complaint="sore throat", respiratory_rate=32, heart_rate=134, systolic_bp=68, temperature_c=38.9, avpu="V", mobility="immobile")
    assert shock["level"] == "RED"
