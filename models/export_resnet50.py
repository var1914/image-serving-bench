#!/usr/bin/env python3
"""Export torchvision ResNet-50 to ONNX, in Triton model-repository layout.

What this teaches (ONNX rung A1-A3):
- ONNX is a *graph* (operators + weights) frozen out of PyTorch, so any runtime
  (ONNX Runtime, TensorRT, Triton) can execute it without Python/PyTorch.
- opset = the version of the operator vocabulary the graph is written in.
- dynamic batch axis: we mark dim 0 as variable ("batch"), otherwise the graph is
  hard-wired to batch=1 and Triton's dynamic batcher could never group requests.
- We verify the export: same input -> PyTorch vs ONNX Runtime, compare outputs.

Output: models/repo/resnet50_onnx/1/model.onnx
"""
import os, time
import numpy as np
import torch, torchvision
import onnx, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "repo", "resnet50_onnx", "1", "model.onnx")

def main():
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights).eval()
    dummy = torch.randn(1, 3, 224, 224)

    t = time.perf_counter()
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=17,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:                                   # classic TorchScript-based exporter
        torch.onnx.export(model, dummy, OUT, dynamo=False, **kw)
    except TypeError:                      # older torch without the dynamo flag
        torch.onnx.export(model, dummy, OUT, **kw)
    print(f"exported in {time.perf_counter()-t:.1f}s -> {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB)")

    m = onnx.load(OUT)
    onnx.checker.check_model(m)
    inp = m.graph.input[0]
    dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"graph: {len(m.graph.node)} nodes, opset {m.opset_import[0].version}, input {inp.name} {dims}")

    # verify: PyTorch vs ONNX Runtime on the same batch of 4 (also proves batch is dynamic)
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})[0]
    print(f"verify batch=4: max |torch - ort| = {np.abs(ref-got).max():.2e}  "
          f"top-1 agree: {(ref.argmax(1)==got.argmax(1)).all()}")

if __name__ == "__main__":
    main()
