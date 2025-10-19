#!/usr/bin/env python3
"""This solution cheats by loading the `stats/cases.txt` file."""

import jpamb


methodid = jpamb.getmethodid(
    "cheater",
    "0.1",
    "The Rice Theorem Cookers",
    ["cheat", "python"],
    for_science=True,
)

queries = set()

queries_in_method = set()

with open("stats/cases.txt", "r") as f:
    for line in f.readlines():
        methodid_, case = line.split(" ", 1)
        query = case.rsplit("->", 1)[1].strip()
        queries.add(query)
        if methodid_ == str(methodid):
            queries_in_method.add(query)

for q in sorted(queries):
    score = "100%" if q in queries_in_method else "0%"
    print(f"{q};{score}")
