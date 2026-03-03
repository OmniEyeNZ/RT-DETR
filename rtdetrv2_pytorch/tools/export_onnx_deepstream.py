"""Export RT-DETRv2 model to ONNX with DeepStream-compatible output format.

Produces an ONNX model with a single output tensor of shape [batch, num_detections, 6]
where each detection is [x1, y1, x2, y2, confidence, label]. This format is compatible
with the NvDsInferParseYolo custom parser used by DeepStream.

For the native RT-DETR ONNX export (separate labels/boxes/scores outputs), use
tools/export_onnx.py instead.

Example:
    cd rtdetrv2_pytorch

    python tools/export_onnx_deepstream.py \
        -c configs/rtdetrv2/rtdetrv2_r18vd_120e_cow.yml \
        -r path/to/checkpoint.pth \
        -o model_deepstream.onnx \
        --img-size 640 \
        --dynamic

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import YAMLConfig


class DeepStreamOutput(nn.Module):
    """Wraps RT-DETR output into DeepStream-Yolo compatible format.

    Converts model output from {pred_boxes: [cx,cy,w,h], pred_logits} to a single
    tensor of [x1, y1, x2, y2, confidence, label] per detection, with coordinates
    scaled to pixel values.
    """

    def __init__(self, img_size, use_focal_loss):
        super().__init__()
        self.img_size = img_size
        self.use_focal_loss = use_focal_loss

    def forward(self, x):
        boxes = x["pred_boxes"]
        convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=boxes.dtype,
            device=boxes.device,
        )
        boxes @= convert_matrix
        boxes *= (
            torch.as_tensor([[*self.img_size]])
            .flip(1)
            .tile([1, 2])
            .unsqueeze(1)
        )
        scores = (
            F.sigmoid(x["pred_logits"])
            if self.use_focal_loss
            else F.softmax(x["pred_logits"])[:, :, :-1]
        )
        scores, labels = torch.max(scores, dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


def load_model(config_path, weights_path):
    """Load RT-DETR model from config and checkpoint."""
    cfg = YAMLConfig(config_path, resume=weights_path)

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    cfg.model.load_state_dict(state)
    model = cfg.model.deploy()
    use_focal_loss = cfg.postprocessor.use_focal_loss

    return model, use_focal_loss


def main(args):
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    device = torch.device("cpu")
    img_size = [args.img_size, args.img_size]

    print(f"Loading model: {args.resume}")
    print(f"Config: {args.config}")
    model, use_focal_loss = load_model(args.config, args.resume)
    model = model.to(device)

    model = nn.Sequential(model, DeepStreamOutput(img_size, use_focal_loss))
    model.eval()

    dummy_input = torch.zeros(args.batch, 3, *img_size).to(device)
    output_file = args.output_file

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    print(
        f"Exporting to ONNX (opset {args.opset}, img_size {args.img_size}, "
        f"{'dynamic batch' if args.dynamic else f'batch {args.batch}'})"
    )
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    if args.simplify:
        try:
            import onnxslim
            import onnx

            print("Simplifying ONNX model...")
            model_onnx = onnx.load(output_file)
            model_onnx = onnxslim.slim(model_onnx)
            onnx.save(model_onnx, output_file)
            print("ONNX model simplified")
        except ImportError:
            print(
                "WARNING: onnxslim not installed, skipping simplification. "
                "Install with: pip install onnxslim"
            )

    if args.check:
        import onnx

        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed")

    print(f"Done: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export RT-DETRv2 to ONNX with DeepStream-compatible output"
    )
    parser.add_argument(
        "--config", "-c", required=True, type=str, help="Model config YAML file"
    )
    parser.add_argument(
        "--resume", "-r", required=True, type=str, help="Checkpoint (.pth) file"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default="model_deepstream.onnx",
        help="Output ONNX file (default: model_deepstream.onnx)",
    )
    parser.add_argument(
        "--img-size",
        "-s",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="Static batch size (default: 1)"
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Enable dynamic batch size"
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--simplify", action="store_true", help="Simplify ONNX model with onnxslim"
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate exported ONNX model"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.resume):
        parser.error(f"Checkpoint file not found: {args.resume}")

    if args.dynamic and args.batch > 1:
        parser.error(
            "Cannot use --dynamic with --batch > 1. "
            "Use static batch or dynamic, not both."
        )

    main(args)
