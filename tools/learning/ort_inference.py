"""Optional ONNX Runtime inference, with CPU defaults and drained GPU switches.

This does not move Rust search or gradient training into ONNX Runtime. Each run
uses one immutable exported checkpoint. A GPU request is applied before the next
inference; an in-flight run finishes before a subsequent CPU run starts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import threading
import sys

import numpy as np
import onnxruntime as ort
from spin_contract import (ARCHITECTURE, HASH, MAX_RECORD_BYTES, TASK, NEXT_BAG_TASK, REFINED_TASK, TARGET_DERIVATION,
                           canonical_rows, integer, reject_constant, unique_keys, validate_config)

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
FORMAT = "neyra-spin-onnx-v1"


def read_json(path):
    with Path(path).open("rb") as source:
        encoded = source.read(MAX_RECORD_BYTES + 1)
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError("ONNX metadata exceeds the bounded artifact limit")
    return json.loads(encoded, object_pairs_hook=unique_keys, parse_constant=reject_constant)


def encode_boards(boards):
    if not isinstance(boards, list) or not 1 <= len(boards) <= 256:
        raise ValueError("An inference batch requires 1 through 256 boards")
    data = np.zeros((len(boards), 1, 24, 10), dtype=np.float32)
    for n, rows in enumerate(boards):
        for y, row in enumerate(canonical_rows(rows)):
            for x in range(10):
                data[n, 0, y, x] = (row >> x) & 1
    return data


class SpinInference:
    """Thread-safe serial submissions, with multithreaded CPU operators.

    set_gpu_enabled() changes only the requested backend, so another thread may
    call it during run(). Session creation and replacement happen at run
    boundaries. Returned NumPy outputs synchronize the ordinary ORT execution;
    no custom asynchronous CUDA stream or GPU-owned output is exposed.
    """

    def __init__(self, model_path, *, manifest_path=None, threads=2, gpu=False):
        integer(threads, "threads", 1, 16)
        if type(gpu) is not bool:
            raise ValueError("gpu must be an explicit boolean")
        self.model_path = Path(model_path)
        self.manifest = read_json(manifest_path or self.model_path.with_suffix(".manifest.json"))
        m = self.manifest
        if (m.get("format") != FORMAT or m.get("task") not in {TASK, NEXT_BAG_TASK, REFINED_TASK} or m.get("architecture") != ARCHITECTURE
                or m.get("input") != {"name": "occupancy", "shape": ["batch", 1, 24, 10], "encoding": "canonical-reflection-bottom-up-f32"}
                or m.get("output") != {"name": "raw_scores", "shape": ["batch", 2], "countScale": 6}):
            raise ValueError("Unexpected ONNX model contract")
        for key in ("onnxSha256", "weightsSha256", "sourceModelSha256"):
            if not isinstance(m.get(key), str) or not HASH.fullmatch(m[key]):
                raise ValueError("Missing ONNX model provenance")
        integer(m.get("horizonBags"), "horizonBags", 2, 3)
        refined = m["task"] == REFINED_TASK
        next_bag = refined or m["task"] == NEXT_BAG_TASK
        config = validate_config(m.get("labelConfig"), 4 if refined else 3 if next_bag else 2)
        if refined and m.get("targetDerivation") != TARGET_DERIVATION:
            raise ValueError("ONNX v4 must retain its measured target derivation")
        self.target_horizon = 1 if next_bag else m["horizonBags"]
        if (m["horizonBags"] != config["horizonBags"]
                or next_bag and m.get("targetHorizonBags") != 1
                or not next_bag and m.get("targetHorizonBags", m["horizonBags"]) != m["horizonBags"]):
            raise ValueError("ONNX prediction horizon disagrees with its measurement policy")
        if not isinstance(m.get("modelId"), str) or not m["modelId"]:
            raise ValueError("Missing source model identity")
        with self.model_path.open("rb") as source:
            self._model_bytes = source.read(MAX_RECORD_BYTES + 1)
        if (len(self._model_bytes) > MAX_RECORD_BYTES
                or hashlib.sha256(self._model_bytes).hexdigest() != m["onnxSha256"]):
            raise ValueError("ONNX bytes do not match the verified export manifest")
        self.threads = threads
        self._state = threading.Lock()
        self._runs = threading.Lock()
        self._requested_gpu = gpu
        self._generation = 0
        self._failed_generation = None
        self._active = CPU
        self._fallback = None
        self._cpu = self._new_session([CPU])
        self._cuda = None

    def _new_session(self, providers):
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = ort.InferenceSession(self._model_bytes, sess_options=options, providers=providers)
        session.disable_fallback()  # The explicit retry below records fallback.
        ins, outs = session.get_inputs(), session.get_outputs()
        metadata = session.get_modelmeta().custom_metadata_map
        if (len(ins) != 1 or ins[0].name != "occupancy" or ins[0].type != "tensor(float)"
                or len(outs) != 1 or outs[0].name != "raw_scores" or outs[0].type != "tensor(float)"
                or metadata.get("neyra.modelId") != self.manifest["modelId"]
                or metadata.get("neyra.weightsSha256") != self.manifest["weightsSha256"]
                or metadata.get("neyra.format") != FORMAT):
            raise ValueError("ONNX session disagrees with its checkpoint manifest")
        if (metadata.get("neyra.task", TASK) != self.manifest["task"]
                or metadata.get("neyra.targetHorizonBags", str(self.manifest["horizonBags"])) != str(self.target_horizon)):
            raise ValueError("ONNX task metadata cannot reinterpret retained output heads")
        if self.manifest["task"] == REFINED_TASK:
            try:
                derivation = json.loads(metadata.get("neyra.targetDerivation", "null"), object_pairs_hook=unique_keys, parse_constant=reject_constant)
            except (ValueError, TypeError):
                raise ValueError("ONNX v4 has invalid target derivation metadata") from None
            if derivation != TARGET_DERIVATION:
                raise ValueError("ONNX graph and manifest disagree on the v4 target derivation")
        return session

    def set_gpu_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("GPU control requires true or false")
        with self._state:
            if enabled != self._requested_gpu:
                self._generation += 1
                self._requested_gpu = enabled
                self._fallback = None
            return self._status()

    def _status(self):
        return {"gpuRequested": self._requested_gpu, "configuredProvider": self._active,
                "transitionPending": (self._active == CUDA) != self._requested_gpu
                and self._failed_generation != self._generation,
                "fallbackReason": self._fallback, "cpuThreads": self.threads}

    def status(self):
        with self._state:
            return self._status()

    def _session(self):
        with self._state:
            requested, generation = self._requested_gpu, self._generation
            blocked = self._failed_generation == generation
        if not requested or blocked:
            if not requested:
                self._cuda = None  # Release the drained GPU session at the boundary.
            with self._state:
                self._active = CPU
            return self._cpu, CPU
        failure = None
        if CUDA not in ort.get_available_providers():
            failure = "cuda-provider-unavailable"
        else:
            try:
                if self._cuda is None:
                    # Optional wheels may locate CUDA/cuDNN in NVIDIA site
                    # packages. This is never attempted while GPU is disabled.
                    if hasattr(ort, "preload_dlls"):
                        ort.preload_dlls()
                    self._cuda = self._new_session([CUDA, CPU])
                if CUDA not in self._cuda.get_providers():
                    failure = "cuda-provider-not-initialized"
            except Exception:
                failure = "cuda-initialization-failed"
                self._cuda = None
        with self._state:
            # A control change during provider initialization takes precedence
            # before the run begins. It never interrupts an existing run.
            if self._generation != generation:
                self._active = CPU
                return self._cpu, CPU
            if failure:
                self._failed_generation = generation
                self._fallback = failure
                self._active = CPU
                return self._cpu, CPU
            self._active = CUDA
            self._fallback = None
        return self._cuda, CUDA

    def predict(self, boards):
        data = encode_boards(boards)

        def run_checked(session):
            raw = session.run(["raw_scores"], {"occupancy": data})[0]
            if raw.shape != (len(boards), 2) or not np.isfinite(raw).all():
                raise ValueError("Inference produced invalid spin predictions")
            return raw

        with self._runs:
            session, provider = self._session()
            try:
                raw = run_checked(session)
            except Exception:
                if provider != CUDA:
                    raise
                with self._state:
                    self._active = CPU
                    self._fallback = "cuda-run-failed-retried-on-cpu"
                    self._failed_generation = self._generation
                self._cuda = None
                provider = CPU
                raw = run_checked(self._cpu)
            scores = raw.astype(np.float64)
            found = 1 / (1 + np.exp(-np.clip(scores[:, 0], -80, 80)))
            counts = np.clip(scores[:, 1] * 6, 0, self.target_horizon * 6)
            return {"modelId": self.manifest["modelId"], "weightsSha256": self.manifest["weightsSha256"],
                    "configuredProviderForRun": provider, "status": self.status(),
                    "predictions": [{"foundSpinLikelihood": float(p), "observedSpins": float(n),
                                     "rawOutput": [float(v) for v in r]}
                                    for p, n, r in zip(found, counts, raw)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--gpu", action="store_true", help="Request CUDA; CPU remains available if unavailable or a GPU run fails")
    args = parser.parse_args()
    runtime = SpinInference(args.model, manifest_path=args.manifest, threads=args.threads, gpu=args.gpu)
    # JSONL accepts {rows:[24]} or {boards:[[24],...]}. Controls are applied
    # between completed runs: {"control":{"gpu":false}}.
    for line in sys.stdin:
        if not line.strip():
            continue
        if len(line.encode()) > MAX_RECORD_BYTES:
            raise ValueError("Inference request exceeds the bounded input limit")
        value = json.loads(line, object_pairs_hook=unique_keys, parse_constant=reject_constant)
        if isinstance(value, dict) and set(value) == {"control"}:
            control = value["control"]
            if not isinstance(control, dict) or set(control) != {"gpu"}:
                raise ValueError("Unknown runtime control")
            result = {"control": runtime.set_gpu_enabled(control["gpu"])}
        elif isinstance(value, dict) and set(value) == {"rows"}:
            result = runtime.predict([value["rows"]])
        elif isinstance(value, dict) and set(value) == {"boards"}:
            result = runtime.predict(value["boards"])
        else:
            raise ValueError("Expected rows, boards, or an explicit GPU control")
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
