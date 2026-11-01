"""Export a retained JSON CNN to ONNX and verify real-board CPU inference.

The portable JSON checkpoint remains authoritative and unchanged. ONNX is an
optional inference artifact, not a different target or automatic training path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from spin_contract import ARCHITECTURE, MAX_RECORD_BYTES, TASK, read_jsonl, validate_model
from train import BoardCNN
from ort_inference import FORMAT, CPU, SpinInference, encode_boards, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Retained spin-v2 JSON checkpoint")
    parser.add_argument("--boards", type=Path, required=True, help="Actual deep-board input JSONL used only for export parity")
    parser.add_argument("--output", type=Path, required=True, help="New .onnx artifact outside the source repository")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--checks", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16 or not 2 <= args.checks <= 256:
        parser.error("Invalid CPU thread or parity sample bound")
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.suffix != ".onnx" or args.output.exists() or manifest_path.exists():
        parser.error("Use a new .onnx output and manifest path; input or retained artifacts are never overwritten")
    if args.output.resolve() in {args.model.resolve(), args.boards.resolve()}:
        parser.error("The ONNX output must not replace an input")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint = validate_model(read_json(args.model))
    boards = [record["rows"] for record in read_jsonl(args.boards, kind="boards")[:args.checks]]
    if len(boards) < 2:
        parser.error("Provide at least two actual fields for dynamic-batch parity")
    model = BoardCNN().eval()
    model.load_state_dict({key: torch.tensor(checkpoint["weights"][key], dtype=torch.float32).reshape(value.shape)
                           for key, value in model.state_dict().items()}, strict=True)
    inputs = encode_boards(boards)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="neyra-onnx-", dir=args.output.parent) as temporary:
        staged = Path(temporary) / args.output.name
        staged_manifest = staged.with_suffix(".manifest.json")
        program = torch.onnx.export(model, (torch.from_numpy(inputs[:2]),), dynamo=True,
                                    opset_version=18, input_names=["occupancy"], output_names=["raw_scores"],
                                    dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=256)}},
                                    external_data=False, optimize=True)
        program.save(staged, external_data=False)
        graph = onnx.load(staged, load_external_data=False)
        onnx.helper.set_model_props(graph, {"neyra.format": FORMAT, "neyra.modelId": checkpoint["modelId"],
                                           "neyra.weightsSha256": checkpoint["weightsSha256"],
                                           "neyra.inputEncoding": "canonical-reflection-bottom-up-f32"})
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, staged)
        if staged.stat().st_size > MAX_RECORD_BYTES:
            raise ValueError("Export exceeded the bounded ONNX artifact limit")
        manifest = {"format": FORMAT, "task": TASK, "architecture": ARCHITECTURE,
                    "modelId": checkpoint["modelId"], "weightsSha256": checkpoint["weightsSha256"],
                    "sourceModelSha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    "onnxSha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
                    "horizonBags": checkpoint["horizonBags"], "labelConfig": checkpoint["labelConfig"],
                    "input": {"name": "occupancy", "shape": ["batch", 1, 24, 10], "encoding": "canonical-reflection-bottom-up-f32"},
                    "output": {"name": "raw_scores", "shape": ["batch", 2], "countScale": 6},
                    "runtime": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                                "opset": 18, "exporter": "torch.onnx dynamo", "cpuThreads": args.threads},
                    "scope": "Optional fixed-model inference. Rust search and training are unchanged; no public promotion."}
        staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        runtime = SpinInference(staged, threads=args.threads)
        errors = []
        for count in sorted({1, 2, len(boards)}):
            with torch.inference_mode():
                expected = model(torch.from_numpy(inputs[:count])).numpy()
            output = runtime.predict(boards[:count])
            assert output["configuredProviderForRun"] == CPU
            observed = np.asarray([r["rawOutput"] for r in output["predictions"]], dtype=np.float32)
            np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-5)
            errors.append({"batch": count, "maxAbsoluteRawError": float(np.max(np.abs(observed - expected)))})
        manifest["verification"] = {"realBoardParity": True, "checks": errors,
                                    "boardsFileSha256": hashlib.sha256(args.boards.read_bytes()).hexdigest(),
                                    "cpuOnlyVerified": True, "cudaHardwareVerified": False}
        staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        # Both final paths were required to be absent. Publication follows parity.
        with args.output.open("xb") as target:
            target.write(staged.read_bytes())
        with manifest_path.open("x") as target:
            target.write(staged_manifest.read_text())
    print(json.dumps({"onnx": str(args.output), "manifest": str(manifest_path), "modelId": checkpoint["modelId"],
                      "onnxSha256": manifest["onnxSha256"], "verification": manifest["verification"]}))


if __name__ == "__main__":
    main()
