import sys
import re

import logging
from typing import Optional
import jpamb

from abstract import analyze_method_no_inputs
from abstract import get_abstract_warnings

# Always present
methodid = jpamb.parse_methodid(sys.argv[1])

classname, methodname, args = re.match(r"(.*)\.(.*):(.*)", sys.argv[1]).groups()

logging.error(args)

# --- scoring policy & pretty printing ---------------------------------------

LABELS = ["ok", "divide by zero", "assertion error", "out of bounds", "null pointer", "*"]

def label_from_warning(w: str) -> Optional[str]:
    w = (w or "").lower()
    if "divide" in w and "zero" in w: return "divide by zero"
    if "assert" in w: return "assertion error"
    if "bound" in w: return "out of bounds"
    if "null" in w and "pointer" in w: return "null pointer"
    return None

def distro_from_outcome_and_warnings(outcome: str, warns: list[str]) -> dict[str, float]:
    outcome = (outcome or "").strip().lower()
    # Start with empty distribution
    dist: dict[str, float] = {k: 0.0 for k in LABELS}

    # If outcome is a known terminal error, make it 100%
    if outcome in {"assertion error", "divide by zero", "out of bounds", "null pointer"}:
        dist[outcome] = 100.0
        return dist

    # If outcome is "ok"
    if outcome == "ok":
        mapped = [label for w in warns if (label := label_from_warning(w)) is not None]
        mapped = list(dict.fromkeys(mapped))  # dedupe, preserve order

        if not mapped:
            dist["ok"] = 100.0
            return dist

        # Assign most weight to ok, distribute the rest to warned outcomes.
        # Example policy: ok gets 70%, remainder split evenly among warnings.
        ok_weight = 70.0
        per_warning = (100.0 - ok_weight) / len(mapped)
        dist["ok"] = ok_weight
        for lab in mapped:
            dist[lab] += per_warning
        return dist

    # Unknown outcome → put all mass on "*"
    dist["*"] = 100.0
    return dist

def normalize_percentages(dist: dict[str, float]) -> dict[str, int]:
    """Round to integers that sum to 100 using largest-remainder method."""
    # keep only nonzero entries
    items = [(k, v) for k, v in dist.items() if v > 0]
    if not items:
        return {"*": 100}

    total = sum(v for _, v in items)
    if total <= 0:
        return {"*": 100}

    # scale to 100
    scaled = [(k, v * 100.0 / total) for k, v in items]
    floors = [(k, int(v)) for k, v in scaled]
    remainders = sorted(
        ((k, v - int(v)) for k, v in scaled),
        key=lambda kv: kv[1],
        reverse=True,
    )
    acc = sum(v for _, v in floors)
    need = 100 - acc
    result = dict(floors)
    for i in range(need):
        k, _ = remainders[i]
        result[k] += 1
    return result

# --- run analysis and print results -----------------------------------------

abstract_outcome = analyze_method_no_inputs(methodid)
warnings = get_abstract_warnings()

# Print a nice readable header
print("Abstract Outcome:", abstract_outcome)
if warnings:
    print("Warnings:")
    for w in warnings:
        print(f"  - {w}")
else:
    print("Warnings: none")

# Compute probability distribution
dist = distro_from_outcome_and_warnings(abstract_outcome, warnings)
perc = normalize_percentages(dist)

for label, p in perc.items():
    print(f"{label}: {p}")
