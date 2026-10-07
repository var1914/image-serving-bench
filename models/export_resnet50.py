#!/usr/bin/env python3
"""Export torchvision ResNet-50 to ONNX, in Triton model-repository layout.

What this teaches (ONNX rung A1-A3):
- ONNX is a *graph* (operators + weights) frozen out of PyTorch, so any runtime
  (ONNX Runtime, TensorRT, Triton) can execute it without Python/PyTorch.
- opset = the version of the operator vocabulary the graph is written in. The
  torch.export-based ("dynamo") exporter starts at opset 18.
- dynamic batch axis: we mark dim 0 as variable ("batch"), otherwise the graph is
  hard-wired to one batch size and Triton's dynamic batcher could never group requests.
  torch.export treats a size-1 example dim as a constant, so the example batch is 2.
- We verify the export: same input -> PyTorch vs ONNX Runtime, compare outputs.

Needs torch >= 2.6 (dynamo exporter with dynamic_shapes and external_data).
Output: models/repo/resnet50_onnx/1/model.onnx (one self-contained file, ~102 MB)
"""
import os, time
import numpy as np
import torch, torchvision
import onnx, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "repo", "resnet50_onnx", "1", "model.onnx")

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)       # the exporter won't create it
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights).eval()
    example = torch.randn(2, 3, 224, 224)                  # batch 2: a size-1 dim gets baked in

    t = time.perf_counter()
    torch.onnx.export(model, (example,), OUT, dynamo=True, opset_version=18,
                      input_names=["input"], output_names=["logits"],
                      dynamic_shapes=({0: torch.export.Dim("batch")},),
                      external_data=False)                 # weights inside model.onnx, no .data sidecar
    print(f"exported in {time.perf_counter()-t:.1f}s -> {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB)")

    m = onnx.load(OUT)
    onnx.checker.check_model(m)
    inp = m.graph.input[0]
    dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"graph: {len(m.graph.node)} nodes, opset {m.opset_import[0].version}, input {inp.name} {dims}")

    # verify: PyTorch vs ONNX Runtime at batch 1 and 4 (neither is the example size,
    # so this also proves the batch axis is really dynamic)
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    for bs in (1, 4):
        x = torch.randn(bs, 3, 224, 224)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"input": x.numpy()})[0]
        print(f"verify batch={bs}: max |torch - ort| = {np.abs(ref-got).max():.2e}  "
              f"top-1 agree: {(ref.argmax(1)==got.argmax(1)).all()}")

if __name__ == "__main__":
    main()
