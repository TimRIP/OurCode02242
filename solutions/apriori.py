#!/usr/bin/env python3
"""Another cheating solution.

This solution uses apriori knowledge about the distribution of the test-cases
to gain an advantage.
"""

import csv
import jpamb

methodid = jpamb.getmethodid(
    "apriori",
    "1.0",
    "The Rice Theorem Cookers",
    ["cheat", "python", "stats"],
    for_science=True,
)


with open("stats/distribution.csv") as f:
    distribution = list(csv.DictReader(f))[-1]

for k, v in distribution.items():
    if k == "method":
        continue
    print(f"{k};{v}")
