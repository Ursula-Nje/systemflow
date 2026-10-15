"""Triage scoring based on the South African Triage Scale (SATS).

Implements the Triage Early Warning Score (TEWS) from vital signs plus a
*subset* of SATS clinical discriminators matched against the presenting
complaint. This module is plain deterministic Python: the language model never
computes a priority, it only extracts values from the nurse's note and calls
the tool that wraps this code.

IMPORTANT: This is a research prototype built on synthetic data. The tables
below were transcribed from the published SATS TEWS and must be verified
against the current official SATS manual before any clinical use. The
discriminator list is a deliberately small subset, not the full scale.
"""

from __future__ import annotations

import math
import re
from typing import Any

SOURCE = (
    "South African Triage Scale (SATS): TEWS + clinical discriminators (subset). "
    "Verify against the current official manual before clinical use."
)

LEVELS = ["GREEN", "YELLOW", "ORANGE", "RED"]
RANK = {name: i for i, name in enumerate(LEVELS)}
MEANING = {
    "RED": "emergency",
    "ORANGE": "very urgent",
    "YELLOW": "urgent",
    "GREEN": "routine",
}
# Published SATS target times to be seen by a clinician (verify against manual).
TARGET = {"RED": "immediate", "ORANGE": "10 min", "YELLOW": "60 min", "GREEN": "240 min"}

INF = math.inf

# (exclusive upper bound, points, label). A value takes the first band it is below.
RR_BANDS = [(9, 2, "<9"), (15, 0, "9-14"), (21, 1, "15-20"), (30, 2, "21-29"), (INF, 3, ">=30")]
HR_BANDS = [
    (41, 2, "<41"),
    (51, 1, "41-50"),
    (101, 0, "51-100"),
    (111, 1, "101-110"),
    (130, 2, "111-129"),
    (INF, 3, ">=130"),
]
SBP_BANDS = [(71, 3, "<71"), (81, 2, "71-80"), (101, 1, "81-100"), (200, 0, "101-199"), (INF, 2, ">=200")]
AVPU_POINTS = {"A": 0, "V": 1, "P": 2, "U": 3}
AVPU_LABEL = {"A": "alert", "V": "reacts to voice", "P": "reacts to pain", "U": "unresponsive"}
MOBILITY_POINTS = {"walking": 0, "with_help": 1, "immobile": 2}

# (level, rule id, regex on the complaint text, label)
DISCRIMINATORS: list[tuple[str, str, str, str]] = [
    ("RED", "D-AIRWAY", r"airway (?:obstruct\w*|compromis\w*)|choking|stridor", "Airway compromised"),
    ("RED", "D-NOT-BREATHING", r"not breathing|stopped breathing|apno?ea", "Not breathing"),
    ("RED", "D-ARREST", r"cardiac arrest|no pulse", "Cardiac arrest"),
    ("RED", "D-SEIZURE-NOW", r"\bseizing\b|fitting now|convulsing|active seizure|currently (?:seizing|fitting)", "Seizure in progress"),
    ("RED", "D-INHALATION-BURN", r"inhalation (?:burn|injury)|facial burn", "Facial or inhalation burn"),
    ("ORANGE", "D-CHEST-PAIN", r"chest pain", "Chest pain"),
    ("ORANGE", "D-SOB", r"short(?:ness)? of breath|difficulty breathing|can'?t breathe|breathless", "Shortness of breath"),
    ("ORANGE", "D-HAEMOPTYSIS", r"coughing (?:up )?blood|ha?emoptysis", "Coughing blood"),
    ("ORANGE", "D-HAEMORRHAGE", r"uncontrolled bleeding|bleeding heavily|heavy bleeding|ha?emorrhag\w*", "Significant haemorrhage"),
    ("ORANGE", "D-STROKE", r"facial droop|slurred speech|one[- ]sided weakness|sudden weakness", "Possible stroke signs"),
    ("ORANGE", "D-POST-ICTAL", r"post[- ]ictal|had a seizure|seizure (?:earlier|today)", "Recent seizure"),
    ("ORANGE", "D-OVERDOSE", r"overdose|poisoning|ingested", "Overdose or poisoning"),
    ("ORANGE", "D-SEVERE-PAIN", r"severe pain|worst pain", "Severe pain"),
    ("YELLOW", "D-VOMITING", r"vomiting", "Vomiting"),
    ("YELLOW", "D-ABD-PAIN", r"abdominal pain|stomach pain", "Abdominal pain"),
    ("YELLOW", "D-FRACTURE", r"fracture|dislocat\w*|deformity", "Possible fracture or dislocation"),
    ("YELLOW", "D-MOD-PAIN", r"moderate pain", "Moderate pain"),
]

# Plausibility limits: values outside these are almost certainly typos or
# extraction errors, so the tool refuses them instead of scoring them.
LIMITS = {
    "respiratory_rate": (0, 80),
    "heart_rate": (0, 300),
    "systolic_bp": (0, 300),
    "temperature_c": (25.0, 45.0),
    "glucose_mmol": (0.0, 60.0),
}


def _lookup(bands: list[tuple[float, int, str]], value: float) -> tuple[int, str]:
    for upper, points, label in bands:
        if value < upper:
            return points, label
    raise AssertionError("bands must end with an infinite upper bound")


def normalise_avpu(value: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    first = str(value).strip().upper()[0]
    if first not in AVPU_POINTS:
        raise ValueError(f"avpu must be one of A, V, P, U (or alert/voice/pain/unresponsive); got {value!r}")
    return first


def normalise_mobility(value: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"stretcher": "immobile", "bedbound": "immobile", "assisted": "with_help"}
    v = aliases.get(v, v)
    if v not in MOBILITY_POINTS:
        raise ValueError(f"mobility must be walking, with_help or immobile; got {value!r}")
    return v


def _check_range(name: str, value: float | None) -> None:
    if value is None:
        return
    low, high = LIMITS[name]
    if not (low <= value <= high):
        raise ValueError(
            f"{name}={value} is outside the plausible range {low}-{high}; "
            "re-check the measurement with the nurse instead of scoring it"
        )


def level_from_tews(total: int) -> str:
    if total >= 7:
        return "RED"
    if total >= 5:
        return "ORANGE"
    if total >= 3:
        return "YELLOW"
    return "GREEN"


def score(
    *,
    complaint: str = "",
    respiratory_rate: float | None = None,
    heart_rate: float | None = None,
    systolic_bp: float | None = None,
    temperature_c: float | None = None,
    avpu: str | None = None,
    mobility: str | None = None,
    trauma: bool = False,
    glucose_mmol: float | None = None,
) -> dict[str, Any]:
    """Return a sourced priority. Never raises on missing vitals (flags them)."""
    for name, value in (
        ("respiratory_rate", respiratory_rate),
        ("heart_rate", heart_rate),
        ("systolic_bp", systolic_bp),
        ("temperature_c", temperature_c),
        ("glucose_mmol", glucose_mmol),
    ):
        _check_range(name, value)
    avpu_n = normalise_avpu(avpu)
    mobility_n = normalise_mobility(mobility)

    components: list[dict[str, Any]] = []
    citations: list[str] = []
    missing: list[str] = []

    def add(name: str, unit: str, value: Any, points: int | None, rule: str) -> None:
        components.append({"name": name, "value": value, "points": points, "rule": rule})
        if points is None:
            missing.append(name)
        else:
            if isinstance(value, float) and value == int(value):
                value = int(value)
            shown = f"{value}{unit}"
            citations.append(f"{name} {shown} -> {points} pt (SATS TEWS band {rule})")

    for name, unit, value, bands in (
        ("respiratory rate", "/min", respiratory_rate, RR_BANDS),
        ("heart rate", "/min", heart_rate, HR_BANDS),
        ("systolic BP", " mmHg", systolic_bp, SBP_BANDS),
    ):
        if value is None:
            add(name, unit, None, None, "not recorded")
        else:
            pts, label = _lookup(bands, value)
            add(name, unit, value, pts, label)

    if temperature_c is None:
        add("temperature", " C", None, None, "not recorded")
    elif temperature_c < 35 or temperature_c > 38.4:
        add("temperature", " C", temperature_c, 2, "<35 or >38.4")
    else:
        add("temperature", " C", temperature_c, 0, "35-38.4")

    if avpu_n is None:
        add("AVPU", "", None, None, "not recorded")
    else:
        add("AVPU", "", AVPU_LABEL[avpu_n], AVPU_POINTS[avpu_n], avpu_n)

    if mobility_n is None:
        add("mobility", "", None, None, "not recorded")
    else:
        add("mobility", "", mobility_n, MOBILITY_POINTS[mobility_n], mobility_n)

    trauma_pts = 1 if trauma else 0
    add("trauma", "", "yes" if trauma else "no", trauma_pts, "yes=1, no=0")

    tews = sum(c["points"] for c in components if c["points"] is not None)
    tews_level = level_from_tews(tews)

    fired: list[dict[str, str]] = []
    text = complaint or ""
    for level, rule_id, pattern, label in DISCRIMINATORS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fired.append({"rule": rule_id, "level": level, "label": label, "matched": match.group(0)})
            citations.append(f"Discriminator {rule_id} ({label}): matched {match.group(0)!r} in complaint -> {level}")
    if glucose_mmol is not None and glucose_mmol < 3:
        fired.append(
            {"rule": "D-HYPOGLYCAEMIA", "level": "RED", "label": "Low glucose", "matched": f"glucose {glucose_mmol}"}
        )
        citations.append(f"Discriminator D-HYPOGLYCAEMIA: glucose {glucose_mmol} mmol/L (<3) -> RED")

    discriminator_level = max((d["level"] for d in fired), key=RANK.get, default="GREEN")
    level = max(tews_level, discriminator_level, key=RANK.get)
    if fired and RANK[discriminator_level] > RANK[tews_level]:
        basis = "discriminator"
    elif fired and RANK[discriminator_level] == RANK[tews_level] and level != "GREEN":
        basis = "tews+discriminator"
    else:
        basis = "tews"

    warnings: list[str] = []
    if missing:
        warnings.append(
            "INCOMPLETE: not recorded -> " + ", ".join(missing) + ". "
            "Missing values count as 0 points here, so this level may be too LOW; "
            "complete the observations before relying on it."
        )

    return {
        "level": level,
        "meaning": MEANING[level],
        "target_time_to_clinician": TARGET[level],
        "basis": basis,
        "tews_total": tews,
        "tews_level": tews_level,
        "components": components,
        "discriminators_fired": fired,
        "citations": citations,
        "incomplete": bool(missing),
        "missing": missing,
        "warnings": warnings,
        "source": SOURCE,
        "disclaimer": "Decision support only. Not a diagnosis. A named clinician must confirm before queue changes.",
    }


def rules_doc() -> dict[str, Any]:
    """Machine-readable copy of the rules, served as an MCP resource."""
    return {
        "source": SOURCE,
        "tews": {
            "respiratory_rate": [[label, pts] for _, pts, label in RR_BANDS],
            "heart_rate": [[label, pts] for _, pts, label in HR_BANDS],
            "systolic_bp": [[label, pts] for _, pts, label in SBP_BANDS],
            "temperature_c": [["<35 or >38.4", 2], ["35-38.4", 0]],
            "avpu": {AVPU_LABEL[k]: v for k, v in AVPU_POINTS.items()},
            "mobility": MOBILITY_POINTS,
            "trauma": {"yes": 1, "no": 0},
            "levels": {"0-2": "GREEN", "3-4": "YELLOW", "5-6": "ORANGE", ">=7": "RED"},
        },
        "target_time_to_clinician": TARGET,
        "discriminators": [{"rule": r, "level": lv, "label": lab} for lv, r, _, lab in DISCRIMINATORS],
        "limitations": [
            "Discriminators are a subset and use keyword matching on the complaint text.",
            "Keyword matching does not understand negation (for example 'denies chest pain').",
            "Verify all tables against the current official SATS manual before clinical use.",
        ],
    }
