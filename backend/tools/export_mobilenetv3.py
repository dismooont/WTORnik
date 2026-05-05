"""One-time ONNX export for MobileNetV3-Large (ImageNet1K_V2 weights).

Run from backend/ directory:
    pip install torch torchvision
    python tools/export_mobilenetv3.py

Generates:
    ../model.onnx           — MobileNetV3-Large weights, NCHW, 1000 ImageNet logits
    ../imagenet_classes.py  — Python list of 1000 human-readable class names
"""

from pathlib import Path

import torch
import torchvision

OUT_ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = OUT_ROOT / "model.onnx"
CLASSES_PATH = OUT_ROOT / "imagenet_classes.py"


def main() -> None:
    weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = torchvision.models.mobilenet_v3_large(weights=weights).eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"wrote {ONNX_PATH} ({ONNX_PATH.stat().st_size // 1024} KB)")

    categories = list(weights.meta["categories"])
    assert len(categories) == 1000, f"expected 1000 classes, got {len(categories)}"

    with CLASSES_PATH.open("w", encoding="utf-8") as f:
        f.write('"""ImageNet1K class names — auto-generated, do not edit."""\n\n')
        f.write("IMAGENET_CLASSES = [\n")
        for name in categories:
            escaped = name.replace("\\", "\\\\").replace('"', '\\"')
            f.write(f'    "{escaped}",\n')
        f.write("]\n")
    print(f"wrote {CLASSES_PATH} ({len(categories)} classes)")


if __name__ == "__main__":
    main()
