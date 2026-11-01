"""Prepare deep occupancy inputs for a declared spin-learning task.

This imports board occupancy and provenance only, never old labels or evaluation
scores. It drops first-bag boundaries, assigns whole-run splits when absent and
purges identical/reflected fields with test/validation precedence.
"""
import argparse
import hashlib
import json
from pathlib import Path
from spin_contract import TASK, NEXT_BAG_TASK, canonical_rows, reject_constant, unique_keys, validate_sample

PROVENANCE = {"kind", "family", "reference", "bagBoundary", "group", "split",
              "seed", "runSeed", "profile", "worker", "runId", "verified", "qualified",
              "probability", "firstBagProbability", "modelMode", "appVersion",
              "sourceFileSha256", "scope", "originInputId", "originCertificateId",
              "originObservationId", "originWitnessId", "rootCandidateId", "rootBagBoundary",
              "rootSeed", "buildVersion", "rules", "trust", "measurementTrust"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--observation-version", type=int, choices=(2, 3), default=2,
                   help="Declare the intended collector task; existing labels are never imported")
    a = p.parse_args()
    if a.output.resolve() in {x.resolve() for x in a.input}:
        p.error("Input and output paths must differ")
    records, inputs, shallow = [], [], 0
    for path in a.input:
        data = path.read_bytes()
        if len(data) > 128 * 1024 * 1024:
            raise ValueError("Input file too large")
        inputs.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
        for line in data.decode().splitlines():
            if not line.strip():
                continue
            if len(line.encode()) > 8 * 1024 * 1024:
                raise ValueError("Input record too large")
            original = json.loads(line, object_pairs_hook=unique_keys, parse_constant=reject_constant)
            occurrence = original if "canonicalBoardId" in original else None
            original = original.get("input", original)
            source = {key: value for key, value in original["source"].items() if key in PROVENANCE}
            if occurrence is not None:
                # An exported source occurrence keeps its immutable assigned
                # group/split even when the physical input predates assignment.
                group, split = occurrence.get("sourceGroup"), occurrence.get("split")
                if not isinstance(group, str) or not group or split not in {"train", "validation", "test"}:
                    raise ValueError("Source occurrence requires its assigned group and split")
                if ("group" in source and source["group"] != group) or ("split" in source and source["split"] != split):
                    raise ValueError("Source occurrence conflicts with its original assignment")
                source["group"], source["split"] = group, split
            if type(source.get("bagBoundary")) is int and source["bagBoundary"] < 2:
                shallow += 1
                continue
            group = source.setdefault("group", source.get("family"))
            if not isinstance(group, str) or not group:
                raise ValueError("Missing source family/run group")
            if "split" not in source:
                bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
                source["split"] = "train" if bucket < 7 else "validation" if bucket == 7 else "test"
            if source["split"] not in {"train", "validation", "test"}:
                raise ValueError("Invalid split")
            sample = {"id": original["id"], "rows": original["rows"], "source": source}
            validate_sample(sample)
            records.append(sample)
    priority = {"test": 0, "validation": 1, "train": 2}
    selected, duplicate = {}, 0
    for record in sorted(records, key=lambda x: (priority[x["source"]["split"]], x["id"])):
        key = canonical_rows(record["rows"])
        if key in selected:
            duplicate += 1
        else:
            selected[key] = record
    groups, families = {}, {}
    for record in selected.values():
        source = record["source"]
        if groups.setdefault(source["group"], source["split"]) != source["split"] or families.setdefault(source["family"], source["split"]) != source["split"]:
            raise ValueError("A source family/run occurs in multiple splits")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    output = sorted(selected.values(), key=lambda x: x["id"])
    a.output.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in output))
    manifest = {"schemaVersion": a.observation_version, "task": NEXT_BAG_TASK if a.observation_version == 3 else TASK,
                "inputFiles": inputs, "boards": len(output), "droppedShallow": shallow,
                "droppedDuplicateOrReflection": duplicate,
                "splitCounts": {split: sum(r["source"]["split"] == split for r in output) for split in priority},
                "sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
                "scope": "Only occupancy/provenance imported; all old evaluation targets and prior fields discarded. New labels for the declared task must be collected separately."}
    if a.observation_version == 3:
        manifest["targetHorizonBags"] = 1
    a.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
