"""Upload observations or retrieve complete private learning snapshots.

Set NEYRA_SITE_URL to the exact private Site HTTPS origin and NEYRA_SITE_TOKEN
to its existing service credential. Tokens never enter command arguments, files,
URLs, log output or public browser code. Redirects are refused.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
import uuid

from spin_contract import (EXPLORATION_POLICY, MAX_RECORD_BYTES, POLICY, integer,
                           read_jsonl, reject_constant, unique_keys, validate_label,
                           validate_model, validate_sample)

EXPORT_KINDS = ("boards", "labels", "models", "runs", "traces")


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
    if not isinstance(counts, dict) or not isinstance(bounds, dict) or set(counts) != set(EXPORT_KINDS) or set(bounds) != set(EXPORT_KINDS):
        raise ValueError("Export manifest must count every retained data kind")
    for kind in EXPORT_KINDS:
        integer(counts[kind], "manifest count", 0, max_records)
        integer(bounds[kind], "manifest rowid bound", counts[kind], 2**53 - 1)
    if sum(counts.values()) > max_records:
        raise ValueError("Snapshot exceeds --max-records; raise the explicit local download bound")
    folder = destination / identity
    if folder.exists():
        raise ValueError("Snapshot destination already exists; retained snapshots are never overwritten")
    folder.mkdir(parents=True)
    (folder / "pages").mkdir()
    (folder / "manifest.json").write_bytes(encoded)
    downloaded_bytes, observed, file_hashes = len(encoded), {}, {}
    policy_counts = {POLICY: 0, EXPLORATION_POLICY: 0}
    fixed_path, exploration_path = folder / "fixed-labels.jsonl", folder / "exploration-labels.jsonl"
    try:
        with fixed_path.open("xb") as fixed_file, exploration_path.open("xb") as exploration_file:
            for kind in EXPORT_KINDS:
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
                            elif kind == "models":
                                validate_model(record)
                                if record["modelId"] != record_id:
                                    raise ValueError("Export model identity mismatch")
                            elif record.get("id") != record_id:
                                raise ValueError("Export metadata identity mismatch")
                            data = record_bytes(record)
                            stream.write(data)
                            stream_hash.update(data)
                            if kind == "labels":
                                policy_counts[record["policy"]] += 1
                                (fixed_file if record["policy"] == POLICY else exploration_file).write(data)
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
        for path in (fixed_path, exploration_path):
            digest_file = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest_file.update(chunk)
            file_hashes[path.name] = digest_file.hexdigest()
        receipt = {"schemaVersion": 1, "snapshotId": identity, "manifestSha256": digest,
                   "counts": observed, "labelPolicies": policy_counts, "filesSha256": file_hashes,
                   "responseBytesVerified": downloaded_bytes, "complete": True,
                   "scope": "Complete retained snapshot; full proofs and traces preserved. Training uses fixed-labels.jsonl only."}
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
            for key in ("policy", "exploration", "config", "targets", "measured", "certifiedWitnesses", "reachableWitness"):
                if stored.get(key) != record.get(key):
                    raise ValueError(f"Stored label {key} differs from the submitted observation")
            if stored["input"]["rows"] != record["input"]["rows"]:
                raise ValueError("Stored physical board orientation differs from its proof")
        print(json.dumps({"acceptedLabels": len(records), "readbackVerified": True, "task": "naga-spin-observations-v2"}))
    elif a.kind == "model":
        value = validate_model(read_json(a.model))
        report = read_json(a.report)
        if report.get("modelId") != value["modelId"] or report.get("datasetSha256") != value["datasetSha256"]:
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
        if stored["modelId"] != value["modelId"] or stored["weightsSha256"] != value["weightsSha256"]:
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
