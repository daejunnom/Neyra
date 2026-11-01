"""Upload observations or retrieve complete private learning snapshots.

Set NEYRA_SITE_URL to the exact private Site HTTPS origin and NEYRA_SITE_TOKEN
to its existing service credential. Tokens never enter command arguments, files,
URLs, log output or public browser code. Redirects are refused.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
import uuid

from spin_contract import (EXPLORATION_POLICY, MAX_RECORD_BYTES, POLICY, NEXT_BAG_POLICY, CONCENTRATED_POLICY,
                           canonical_rows, integer, refined_selection_key, refined_rank,
                           read_jsonl, reject_constant, unique_keys, validate_label,
                           validate_model, validate_sample, validate_refined_record, validate_refined_evidence)

EXPORT_KINDS = ("boards", "labels", "models", "runs", "traces")
SOURCE_EXPORT_KINDS = (*EXPORT_KINDS, "sources")
REFINED_EXPORT_KINDS = (*SOURCE_EXPORT_KINDS, "refinedTargets", "validationRoster", "replayAttempts")
MAX_EXCLUDED_GROUPS = 10000


class ResponseTooLarge(ValueError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise ValueError("Private Site redirects are refused to protect the service credential")


def connection():
    origin = os.environ.get("NEYRA_SITE_URL", "").rstrip("/")
    token = os.environ.get("NEYRA_SITE_TOKEN", "")
    parsed = urllib.parse.urlsplit(origin)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path:
        raise ValueError("NEYRA_SITE_URL must be the exact HTTPS Site origin")
    if not token or "\n" in token or "\r" in token:
        raise ValueError("A private Site service credential is required in NEYRA_SITE_TOKEN")
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    return origin, token, opener


def read_json(path):
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_RECORD_BYTES + 1)
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError("Upload file exceeds the bounded record limit")
    return json.loads(data, object_pairs_hook=unique_keys, parse_constant=reject_constant)


def post(origin, token, opener, path, body):
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError("Upload body exceeds the record limit")
    request = urllib.request.Request(origin + path, data=encoded, method="POST",
        headers={"Content-Type": "application/json", "OAI-Sites-Authorization": "Bearer " + token})
    try:
        with opener.open(request, timeout=60) as response:
            if response.geturl().split("?", 1)[0] != origin + path:
                raise ValueError("Unexpected response origin")
            payload = response.read(MAX_RECORD_BYTES + 1)
            if len(payload) > MAX_RECORD_BYTES:
                raise ValueError("Unexpectedly large Site response")
            result = json.loads(payload, object_pairs_hook=unique_keys, parse_constant=reject_constant)
            if not isinstance(result, dict) or result.get("error"):
                raise ValueError("Site did not accept the upload")
            return result
    except urllib.error.HTTPError as error:
        raise ValueError(f"Private Site rejected the upload (HTTP {error.code})") from None
    except urllib.error.URLError:
        raise ValueError("Private Site connection failed; check origin and verified TLS configuration") from None


def get(origin, token, opener, path):
    request = urllib.request.Request(origin + path,
        headers={"OAI-Sites-Authorization": "Bearer " + token})
    try:
        with opener.open(request, timeout=60) as response:
            payload = response.read(MAX_RECORD_BYTES + 1)
            if response.geturl().split("?", 1)[0] != origin + path or len(payload) > MAX_RECORD_BYTES:
                raise ValueError("Unexpected private Site readback")
            return json.loads(payload, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    except urllib.error.HTTPError as error:
        raise ValueError(f"Private Site readback failed (HTTP {error.code})") from None
    except urllib.error.URLError:
        raise ValueError("Private Site readback connection failed") from None


def get_verified(origin, token, opener, path):
    """Verify exact UTF-8 response bytes, not a language-specific JSON rendering."""
    request = urllib.request.Request(origin + path,
        headers={"OAI-Sites-Authorization": "Bearer " + token})
    try:
        with opener.open(request, timeout=60) as response:
            if response.geturl() != origin + path:
                raise ValueError("Unexpected private Site export location")
            payload = response.read(MAX_RECORD_BYTES + 1)
            if len(payload) > MAX_RECORD_BYTES:
                raise ResponseTooLarge("Export page exceeds the bounded response limit")
            expected = response.headers.get("X-Neyra-Content-SHA256", "")
            actual = hashlib.sha256(payload).hexdigest()
            if not re.fullmatch("[0-9a-f]{64}", expected) or expected != actual:
                raise ValueError("Private export response SHA-256 does not match its bytes")
            value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_keys, parse_constant=reject_constant)
            if not isinstance(value, dict) or value.get("error"):
                raise ValueError("Private export did not return an object")
            return value, payload, actual
    except urllib.error.HTTPError as error:
        raise ValueError(f"Private Site export failed (HTTP {error.code})") from None
    except urllib.error.URLError:
        raise ValueError("Private Site export connection failed") from None


def snapshot_id(value):
    if not isinstance(value, str):
        raise ValueError("Missing snapshot ID")
    try:
        normalized = str(uuid.UUID(value))
    except (ValueError, AttributeError):
        raise ValueError("Snapshot ID must be a UUID") from None
    if normalized != value:
        raise ValueError("Snapshot ID must use its canonical UUID spelling")
    return value


def record_bytes(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def bounded_text(value, name, maximum=256):
    if (not isinstance(value, str) or not value.strip() or len(value) > maximum
            or re.search(r"[\x00-\x1f\x7f]", value)):
        raise ValueError(f"Invalid bounded {name}")
    return value


def manifest_exclusions(manifest):
    """Validate the immutable exclusion list before creating any local files."""
    corpus = manifest.get("corpus")
    if "corpus" in manifest:
        bounded_text(corpus, "manifest corpus", 128)
    entries = manifest.get("excludedTrainingGroups", [])
    if not isinstance(entries, list) or len(entries) > MAX_EXCLUDED_GROUPS:
        raise ValueError("Manifest excluded training groups exceed their bounded contract")
    groups = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"group", "reason", "excludedAt"}:
            raise ValueError("Malformed manifest training exclusion")
        group = bounded_text(entry["group"], "excluded training group")
        if group in groups:
            raise ValueError("Duplicate manifest training exclusion group")
        groups.add(group)
        timestamp = integer(entry["excludedAt"], "exclusion timestamp", 0, 2**53 - 1)
        if "createdAt" in manifest and timestamp > integer(manifest["createdAt"], "manifest timestamp", 0, 2**53 - 1):
            raise ValueError("Training exclusion postdates its immutable snapshot")
        reason = entry["reason"]
        if not isinstance(reason, dict) or not reason:
            raise ValueError("Training exclusion needs a structured reason")
        bounded_text(reason.get("kind"), "exclusion reason kind", 128)
        if "group" in reason and reason["group"] != group:
            raise ValueError("Exclusion reason names a different training group")
        if "trainingTarget" in reason and reason["trainingTarget"] is not False:
            raise ValueError("Training exclusion cannot be a supervised target")
        # Reasons are retained metadata, never labels. Bound their shape without
        # requiring one particular exclusion cause or silently ignoring one.
        pending, visited = [(reason, 0)], 0
        while pending:
            value, depth = pending.pop()
            visited += 1
            if depth > 12 or visited > 512:
                raise ValueError("Exclusion reason exceeds its structural bounds")
            if isinstance(value, dict):
                for key, child in value.items():
                    bounded_text(key, "exclusion reason key", 128)
                    pending.append((child, depth + 1))
            elif isinstance(value, list):
                pending.extend((child, depth + 1) for child in value)
            elif isinstance(value, str):
                if len(value) > 4096 or "\0" in value:
                    raise ValueError("Exclusion reason text is too large or invalid")
            elif value is not None and type(value) is not bool:
                if (type(value) not in (int, float)
                        or (type(value) is int and abs(value) > 2**53 - 1)
                        or (type(value) is float and not math.isfinite(value))):
                    raise ValueError("Exclusion reason must contain finite JSON values")
        if len(record_bytes(reason)) > 8192:
            raise ValueError("Exclusion reason exceeds its byte limit")
    # Explicit algorithm avoids depending on incoming JSON key order. The exact
    # original exclusion metadata also remains in the verified manifest.json.
    encoded = (json.dumps(sorted(entries, key=lambda entry: entry["group"]), sort_keys=True,
                          ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return groups, hashlib.sha256(encoded).hexdigest()


def excluded_label_groups(record, exclusions):
    if not exclusions:
        return set()  # Old snapshots retain exactly their previous behavior.
    source = record["input"]["source"]
    candidates = {bounded_text(source["family"], "source family")}
    for container, key in ((source, "group"), (record, "measurementGroup")):
        if key in container:
            candidates.add(bounded_text(container[key], key))
    if isinstance(record.get("evidence"), dict) and "measurementGroup" in record["evidence"]:
        candidates.add(bounded_text(record["evidence"]["measurementGroup"], "evidence measurementGroup"))
    if "runId" in source:
        run = bounded_text(source["runId"], "source runId", 128)
        candidates.update((run, "search:" + run))
    return candidates & exclusions


def manifest_source_groups(manifest, maximum):
    entries = manifest.get("trainingSourceGroups", [])
    if not isinstance(entries, list) or len(entries) > maximum:
        raise ValueError("Manifest source-group registry exceeds its bounded contract")
    registry, components = {}, {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"group", "component", "split", "assignment", "createdAt"}:
            raise ValueError("Malformed immutable source-group membership")
        group = bounded_text(entry["group"], "source group")
        component = bounded_text(entry["component"], "source component")
        split = entry["split"]
        if group in registry or split not in {"train", "validation", "test"}:
            raise ValueError("Duplicate source group or invalid fixed split")
        if components.setdefault(component, split) != split:
            raise ValueError("One retained source component cannot span training splits")
        if not isinstance(entry["assignment"], dict) or not entry["assignment"] or len(record_bytes(entry["assignment"])) > 8192:
            raise ValueError("Source membership requires bounded assignment provenance")
        created = integer(entry["createdAt"], "source group timestamp", 0, 2**53 - 1)
        if "createdAt" in manifest and created > manifest["createdAt"]:
            raise ValueError("Source membership postdates its immutable snapshot")
        registry[group] = entry
    return registry


def validate_source_occurrence(record, record_id, registry, corpus):
    if set(record) != {"input", "canonicalBoardId", "sourceGroup", "split", "createdAt"}:
        raise ValueError("Source occurrences are provenance records, never labels")
    validate_sample(record["input"])
    if not re.fullmatch("[0-9a-f]{64}", record_id):
        raise ValueError("Invalid source occurrence identity")
    bounded_text(corpus, "source corpus", 128)
    canonical = ",".join(map(str, canonical_rows(record["input"]["rows"])))
    if record["canonicalBoardId"] != hashlib.sha256((corpus + ":" + canonical).encode()).hexdigest():
        raise ValueError("Source occurrence canonical board identity mismatch")
    group = bounded_text(record["sourceGroup"], "occurrence source group")
    if group not in registry or record["split"] != registry[group]["split"]:
        raise ValueError("Source occurrence is outside its immutable group membership")
    source = record["input"]["source"]
    if source.get("group", group) != group or source.get("split", record["split"]) != record["split"]:
        raise ValueError("Source occurrence disagrees with its input provenance")
    integer(record["createdAt"], "source occurrence timestamp", registry[group]["createdAt"], 2**53 - 1)
    return record


def download_snapshot(origin, token, opener, identity, destination, *, manifest_hash=None,
                      max_records=250000, max_bytes=512 * 1024 * 1024):
    identity = snapshot_id(identity)
    base = "/api/learning/exports/" + identity
    manifest, encoded, digest = get_verified(origin, token, opener, base + "/manifest")
    if manifest_hash is not None and manifest_hash != digest:
        raise ValueError("Snapshot creation receipt disagrees with its immutable manifest")
    if manifest.get("schemaVersion") != 1 or manifest.get("snapshotId") != identity:
        raise ValueError("Export manifest identity or schema is invalid")
    counts, bounds = manifest.get("counts"), manifest.get("bounds")
    if (not isinstance(counts, dict) or not isinstance(bounds, dict)
            or set(counts) not in (set(EXPORT_KINDS), set(SOURCE_EXPORT_KINDS), set(REFINED_EXPORT_KINDS)) or set(bounds) != set(counts)):
        raise ValueError("Export manifest must count every retained data kind")
    kinds = REFINED_EXPORT_KINDS if "refinedTargets" in counts else SOURCE_EXPORT_KINDS if "sources" in counts else EXPORT_KINDS
    if "pagination" in manifest and (not isinstance(manifest["pagination"], dict)
            or manifest["pagination"].get("kinds") != list(kinds)):
        raise ValueError("Export pagination disagrees with its retained data kinds")
    for kind in kinds:
        integer(counts[kind], "manifest count", 0, max_records)
        integer(bounds[kind], "manifest rowid bound", counts[kind], 2**53 - 1)
    if sum(counts.values()) > max_records:
        raise ValueError("Snapshot exceeds --max-records; raise the explicit local download bound")
    exclusions, exclusion_hash = manifest_exclusions(manifest)
    registry = manifest_source_groups(manifest, max_records)
    if "sources" in counts and "trainingSourceGroups" not in manifest:
        raise ValueError("Source occurrences require the immutable source-group registry")
    folder = destination / identity
    if folder.exists():
        raise ValueError("Snapshot destination already exists; retained snapshots are never overwritten")
    folder.mkdir(parents=True)
    (folder / "pages").mkdir()
    (folder / "manifest.json").write_bytes(encoded)
    downloaded_bytes, observed, file_hashes = len(encoded), {}, {}
    policy_counts = {POLICY: 0, EXPLORATION_POLICY: 0, NEXT_BAG_POLICY: 0, CONCENTRATED_POLICY: 0}
    eligible_counts = dict(policy_counts)
    excluded_counts = dict(policy_counts)
    selected_counts = dict(policy_counts)
    duplicate_counts = dict(policy_counts)
    next_bag_representatives = {}
    applied_exclusions = set()
    fixed_path, exploration_path = folder / "fixed-labels.jsonl", folder / "exploration-labels.jsonl"
    next_bag_path = folder / "next-bag-fixed-labels.jsonl"
    concentrated_path, refined_path = folder / "refined-observations.jsonl", folder / "refined-targets.jsonl"
    proof_path = folder / "observations-index.jsonl"
    raw_proofs, refined_records = {}, []

    def remember_proof(identity, label):
        if not isinstance(identity, str) or not re.fullmatch("[0-9a-f]{64}", identity):
            raise ValueError("A v4 proof requires its immutable observation digest")
        validate_label(label)
        if label["schemaVersion"] != 4:
            raise ValueError("A refinement evidence artifact contains another observation task")
        if identity in raw_proofs and raw_proofs[identity] != label:
            raise ValueError("An immutable observation identity has conflicting raw evidence")
        raw_proofs[identity] = label

    try:
        with fixed_path.open("xb") as fixed_file, exploration_path.open("xb") as exploration_file, next_bag_path.open("xb") as next_bag_file, concentrated_path.open("xb") as concentrated_file:
            policy_files = {POLICY: fixed_file, EXPLORATION_POLICY: exploration_file, NEXT_BAG_POLICY: next_bag_file,
                            CONCENTRATED_POLICY: concentrated_file}
            for kind in kinds:
                cursor, page_number, seen = "0", 0, set()
                page_limit = 1 if kind == "models" else 16
                stream_hash = hashlib.sha256()
                with (folder / (kind + ".jsonl")).open("xb") as stream:
                    while True:
                        query = urllib.parse.urlencode({"kind": kind, "cursor": cursor, "limit": page_limit})
                        try:
                            page, raw, _ = get_verified(origin, token, opener, base + "/page?" + query)
                        except ResponseTooLarge:
                            if page_limit == 1:
                                raise
                            page_limit = max(1, page_limit // 2)
                            continue
                        downloaded_bytes += len(raw)
                        if downloaded_bytes > max_bytes:
                            raise ValueError("Snapshot exceeds --max-bytes; increase the explicit local download bound")
                        if page.get("schemaVersion") != 1 or page.get("snapshotId") != identity or page.get("kind") != kind or page.get("cursor") != cursor:
                            raise ValueError("Export page does not match its requested snapshot cursor")
                        records, ids = page.get("records"), page.get("recordIds")
                        if not isinstance(records, list) or not isinstance(ids, list) or len(records) != len(ids) or len(records) > page_limit:
                            raise ValueError("Export records and identities are incoherent")
                        for record, record_id in zip(records, ids):
                            if not isinstance(record_id, str) or not record_id or len(record_id) > 256 or record_id in seen:
                                raise ValueError("Export contains an invalid or duplicate record identity")
                            if not isinstance(record, dict):
                                raise ValueError("Export records must be objects")
                            if kind == "boards":
                                validate_sample(record)
                            elif kind == "labels":
                                validate_label(record)
                                if record["schemaVersion"] == 4:
                                    remember_proof(record_id, record)
                            elif kind == "models":
                                validate_model(record)
                                if record["modelId"] != record_id:
                                    raise ValueError("Export model identity mismatch")
                            elif kind == "sources":
                                validate_source_occurrence(record, record_id, registry, manifest.get("corpus"))
                                if record["createdAt"] > manifest["createdAt"]:
                                    raise ValueError("Source occurrence postdates its snapshot")
                            elif kind == "refinedTargets":
                                validate_refined_record(record)
                                if record["id"] != record_id:
                                    raise ValueError("Refined target identity mismatch")
                                refined_records.append(record)
                            elif record.get("id") != record_id:
                                raise ValueError("Export metadata identity mismatch")
                            if kind == "traces" and record.get("kind") == "v4-refinement-evidence":
                                payload = record.get("payload")
                                if (not isinstance(payload, dict) or payload.get("artifactKey") != "observations/" + str(record.get("observationId")) + ".json"):
                                    raise ValueError("Refinement evidence trace has an incoherent artifact reference")
                                remember_proof(record.get("observationId"), payload.get("observation"))
                            data = record_bytes(record)
                            stream.write(data)
                            stream_hash.update(data)
                            if kind == "labels":
                                policy_counts[record["policy"]] += 1
                                excluded = excluded_label_groups(record, exclusions)
                                if excluded:
                                    excluded_counts[record["policy"]] += 1
                                    applied_exclusions.update(excluded)
                                else:
                                    eligible_counts[record["policy"]] += 1
                                    selected = True
                                    if record["policy"] == NEXT_BAG_POLICY:
                                        # Export pages follow immutable insertion (rowid) order,
                                        # matching the private dataset's earliest eligible proof.
                                        # Quarantine is applied first: an excluded earlier alias
                                        # must never shadow an independent eligible measurement.
                                        key = (canonical_rows(record["input"]["rows"]), record["policy"],
                                               json.dumps(record["config"], sort_keys=True, separators=(",", ":")))
                                        split = record["input"]["source"].get("split")
                                        if key in next_bag_representatives:
                                            previous_split = next_bag_representatives[key]
                                            if (split is not None and previous_split is not None
                                                    and split != previous_split):
                                                raise ValueError("Canonical v3 measurements disagree on their fixed split")
                                            if previous_split is None:
                                                next_bag_representatives[key] = split
                                            duplicate_counts[record["policy"]] += 1
                                            selected = False
                                        else:
                                            next_bag_representatives[key] = split
                                    if selected:
                                        selected_counts[record["policy"]] += 1
                                        policy_files[record["policy"]].write(data)
                            seen.add(record_id)
                        if len(seen) > counts[kind]:
                            raise ValueError("Export exceeded its immutable manifest count")
                        (folder / "pages" / f"{kind}-{page_number:06d}.json").write_bytes(raw)
                        page_number += 1
                        following = page.get("nextCursor")
                        if following is None:
                            break
                        if (not isinstance(following, str) or not re.fullmatch("[1-9][0-9]*", following)
                                or not records or not int(cursor) < int(following) <= bounds[kind]):
                            raise ValueError("Export cursor did not make bounded forward progress")
                        cursor = following
                if len(seen) != counts[kind]:
                    raise ValueError("Export ended before its manifest count was retrieved")
                observed[kind] = len(seen)
                file_hashes[kind + ".jsonl"] = stream_hash.hexdigest()
        # Every derived target is checked against the exact admitted physical
        # proof, including history that is now excluded from fitting.
        refined_best, refined_splits, refined_excluded = {}, {}, 0
        for position, record in enumerate(refined_records):
            identity = record["evidence"]["observationId"]
            if identity not in raw_proofs:
                raise ValueError("A refined target's original proof is missing from the complete snapshot")
            validate_refined_evidence(record, raw_proofs[identity])
            excluded = excluded_label_groups(record, exclusions)
            if excluded:
                refined_excluded += 1; applied_exclusions.update(excluded)
                continue
            group, split = record["input"]["source"]["group"], record["input"]["source"]["split"]
            if group not in registry or registry[group]["split"] != split:
                raise ValueError("Refined target disagrees with its immutable measurement-group assignment")
            key = refined_selection_key(record)
            if refined_splits.setdefault(key, split) != split:
                raise ValueError("Refined canonical measurements disagree on their fixed source split")
            previous = refined_best.get(key)
            # Strict greater-than keeps the earliest immutable row for exact ties.
            if previous is None or refined_rank(record) > refined_rank(previous[1]):
                refined_best[key] = position, record
        selected_refined = [record for _, record in sorted(refined_best.values(), key=lambda item: item[0])]
        with refined_path.open("xb") as stream:
            for record in selected_refined:
                stream.write(record_bytes(record))
        with proof_path.open("xb") as stream:
            for identity, label in raw_proofs.items():
                stream.write(record_bytes({"id": identity, "label": label}))
        for path in (fixed_path, exploration_path, next_bag_path, concentrated_path, refined_path, proof_path):
            digest_file = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest_file.update(chunk)
            file_hashes[path.name] = digest_file.hexdigest()
        receipt = {"schemaVersion": 1, "snapshotId": identity, "manifestSha256": digest,
                   "counts": observed, "labelPolicies": policy_counts, "filesSha256": file_hashes,
                   "trainingSelection": {
                       "schemaVersion": 1, "corpus": manifest.get("corpus"),
                       "exclusionManifestPresent": "excludedTrainingGroups" in manifest,
                       "excludedTrainingGroupsDeclared": len(exclusions),
                       "excludedTrainingGroupsApplied": sorted(applied_exclusions),
                       "sourceGroupRegistryPresent": "trainingSourceGroups" in manifest,
                       "sourceGroupCount": len(registry),
                       "excludedTrainingGroupsSha256": exclusion_hash,
                       "exclusionHashAlgorithm": "sha256-utf8-json-sorted-group-and-keys-newline-v1",
                       "eligibleLabelPolicies": eligible_counts, "excludedLabelPolicies": excluded_counts,
                       "selectedLabelPolicies": selected_counts, "duplicateLabelPolicies": duplicate_counts,
                       "refinedTargets": {"retained": len(refined_records), "excluded": refined_excluded,
                                          "selected": len(selected_refined), "proofs": len(raw_proofs),
                                          "selection": "Best eligible local spin count per canonical board/exact config/measurement scope; ties prefer reached, completed, distinct evidence, then earliest insertion."},
                       "files": {fixed_path.name: {"records": selected_counts[POLICY], "sha256": file_hashes[fixed_path.name]},
                                 exploration_path.name: {"records": selected_counts[EXPLORATION_POLICY], "sha256": file_hashes[exploration_path.name]},
                                 next_bag_path.name: {"records": selected_counts[NEXT_BAG_POLICY], "sha256": file_hashes[next_bag_path.name]},
                                 concentrated_path.name: {"records": selected_counts[CONCENTRATED_POLICY], "sha256": file_hashes[concentrated_path.name]},
                                 refined_path.name: {"records": len(selected_refined), "sha256": file_hashes[refined_path.name]},
                                 proof_path.name: {"records": len(raw_proofs), "sha256": file_hashes[proof_path.name]}},
                       "rule": "Exclude labels whose measurement or source group/family/run is explicitly quarantined. Preserve historical source-component membership without excluding previously valid peers. For v3, choose the first remaining insertion-order proof per canonical board, policy and exact config; preserve its actual orientation. Keep v2 fixed, v2 exploration and v3 one-bag labels separate; raw evidence remains complete.",
                   },
                   "responseBytesVerified": downloaded_bytes, "complete": True,
                   "scope": "Complete retained snapshot; raw proofs, source occurrences and received traces remain unchanged. Fit v2 with fixed-labels.jsonl, v3 with next-bag-fixed-labels.jsonl, or v4 with refined-targets.jsonl plus observations-index.jsonl. Never mix tasks or fit raw v4 observations. V4 winners are selected only after exclusions; historical replay attempts and validation rosters are retained."}
        (folder / "download-receipt.json").write_bytes(record_bytes(receipt))
        # Only complete, verified snapshots can become the local latest pointer.
        temporary = destination / ("latest-" + str(uuid.uuid4()) + ".tmp")
        temporary.write_bytes(record_bytes({"snapshotId": identity, "manifestSha256": digest,
                                            "directory": identity, "complete": True}))
        temporary.replace(destination / "latest.json")
        return receipt
    except Exception:
        # Keep partial pages for inspection, never advertise them as complete.
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="kind", required=True)
    labels = sub.add_parser("labels")
    labels.add_argument("--input", type=Path, required=True)
    model = sub.add_parser("model")
    model.add_argument("--model", type=Path, required=True)
    model.add_argument("--report", type=Path, required=True)
    for name in ("sync", "pull"):
        command = sub.add_parser(name, help="Create and download a complete snapshot" if name == "sync" else "Download an existing immutable snapshot")
        command.add_argument("--output", type=Path, required=True,
                             help="Snapshot root; previous UUID directories remain immutable")
        command.add_argument("--max-records", type=int, default=250000)
        command.add_argument("--max-bytes", type=int, default=512 * 1024 * 1024)
        if name == "pull":
            command.add_argument("--snapshot", required=True)
    a = p.parse_args()
    if a.kind == "labels":
        records = read_jsonl(a.input)
        origin, token, opener = connection()
        for record in records:
            receipt = post(origin, token, opener, "/api/learning/labels", {"label": record})
            identity = receipt.get("id")
            if not isinstance(identity, str) or not re.fullmatch("[0-9a-f]{64}", identity) or type(receipt.get("duplicate")) is not bool:
                raise ValueError("Site did not return a valid label receipt")
            stored = get(origin, token, opener, "/api/learning/labels/" + identity)
            validate_label(stored)
            for key in ("schemaVersion", "task", "policy", "exploration", "config", "targets", "measured",
                        "certifiedWitnesses", "reachableWitness", "targetHorizonBags", "targetWitness", "reachableWholeBagWitnesses", "refinement"):
                if stored.get(key) != record.get(key):
                    raise ValueError(f"Stored label {key} differs from the submitted observation")
            if stored["input"]["rows"] != record["input"]["rows"]:
                raise ValueError("Stored physical board orientation differs from its proof")
        tasks = sorted({record["task"] for record in records})
        print(json.dumps({"acceptedLabels": len(records), "readbackVerified": True,
                          "task": tasks[0] if len(tasks) == 1 else None, "tasks": tasks}))
    elif a.kind == "model":
        value = validate_model(read_json(a.model))
        if value["schemaVersion"] == 4:
            raise ValueError("V4 uploads require the private Site's durable training/prepare and training/complete replay lease. Offline checkpoints cannot bypass source/ancestor eligibility checks through the generic model endpoint; retain this local artifact for review.")
        report = read_json(a.report)
        if (report.get("modelId") != value["modelId"] or report.get("datasetSha256") != value["datasetSha256"]
                or report.get("task") != value["task"] or report.get("schemaVersion") != value["schemaVersion"]):
            raise ValueError("Model report does not match the checkpoint")
        if not isinstance(report.get("metrics"), dict) or not isinstance(report.get("training"), dict):
            raise ValueError("The model needs measured metrics and training provenance")
        origin, token, opener = connection()
        result = post(origin, token, opener, "/api/learning/models",
                      {"model": value, "metrics": report["metrics"], "training": report["training"]})
        if result.get("stored") is not True or result.get("modelId") != value["modelId"] or result.get("weightsSha256") != value["weightsSha256"]:
            raise ValueError("Site model receipt does not match the submitted checkpoint")
        stored = get(origin, token, opener, "/api/learning/models/" + urllib.parse.quote(value["modelId"], safe=""))
        stored = validate_model(stored.get("model") if isinstance(stored, dict) else None)
        if (stored["modelId"] != value["modelId"] or stored["weightsSha256"] != value["weightsSha256"]
                or stored["task"] != value["task"] or stored.get("targetHorizonBags") != value.get("targetHorizonBags")
                or stored["labelConfig"] != value["labelConfig"]):
            raise ValueError("Stored model differs from the uploaded checkpoint")
        print(json.dumps({"acceptedModel": value["modelId"], "readbackVerified": True}))
    else:
        if not 1 <= a.max_records <= 10000000 or not MAX_RECORD_BYTES <= a.max_bytes <= 64 * 1024**3:
            p.error("Invalid bounded snapshot download limits")
        origin, token, opener = connection()
        if a.kind == "sync":
            created = post(origin, token, opener, "/api/learning/exports", {})
            identity = snapshot_id(created.get("snapshotId"))
            expected = created.get("manifestSha256")
            if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
                raise ValueError("Snapshot creation did not return a manifest hash")
        else:
            identity, expected = snapshot_id(a.snapshot), None
        result = download_snapshot(origin, token, opener, identity, a.output,
                                   manifest_hash=expected, max_records=a.max_records, max_bytes=a.max_bytes)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
