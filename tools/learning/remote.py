"""Upload measured spin labels and CPU checkpoints to the private learning Site.

Set NEYRA_SITE_URL to the exact private Site HTTPS origin and NEYRA_SITE_TOKEN
to its existing service credential. Tokens never enter command arguments, files,
URLs, log output or public browser code. Redirects are refused.
"""
import argparse
import json
import os
from pathlib import Path
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request

from spin_contract import (MAX_RECORD_BYTES, read_jsonl, reject_constant,
                           unique_keys, validate_label, validate_model)


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
    data = Path(path).read_bytes()
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="kind", required=True)
    labels = sub.add_parser("labels")
    labels.add_argument("--input", type=Path, required=True)
    model = sub.add_parser("model")
    model.add_argument("--model", type=Path, required=True)
    model.add_argument("--report", type=Path, required=True)
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
            for key in ("config", "targets", "measured", "certifiedWitnesses", "reachableWitness"):
                if stored.get(key) != record.get(key):
                    raise ValueError(f"Stored label {key} differs from the submitted observation")
            if stored["input"]["rows"] != record["input"]["rows"]:
                raise ValueError("Stored physical board orientation differs from its proof")
        print(json.dumps({"acceptedLabels": len(records), "readbackVerified": True, "task": "naga-spin-observations-v2"}))
    else:
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


if __name__ == "__main__":
    main()
