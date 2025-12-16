import os
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import YAMLConfig


class DeepStreamOutput(nn.Module):
    def __init__(self, img_size, use_focal_loss):
        super().__init__()
        self.img_size = img_size
        self.use_focal_loss = use_focal_loss

    def forward(self, x):
        boxes = x['pred_boxes']
        
        print("\nBox format debug info:")
        print("Original boxes shape:", boxes.shape)
        print("Sample box (first prediction):", boxes[0, 0])
        print("Box value ranges:")
        print("  Min values:", torch.min(boxes, dim=1)[0])
        print("  Max values:", torch.max(boxes, dim=1)[0])
        
        is_normalized = (boxes >= -2).all() and (boxes <= 2).all()
        print("Appears to be normalized coordinates:", is_normalized)
        
        is_center_format = (boxes[..., 2:] >= 0).all()
        print("Appears to be center format:", is_center_format)
        
        convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], 
            dtype=boxes.dtype, device=boxes.device
        )
        
    
        print("\nBefore conversion:")
        print("Sample box:", boxes[0, 0])
        
        boxes @= convert_matrix
        
        print("After format conversion:")
        print("Sample box:", boxes[0, 0])
        
        print("\nBefore scaling:")
        print("Sample box:", boxes[0, 0])
        
        boxes *= torch.as_tensor([[*self.img_size]]).flip(1).tile([1, 2]).unsqueeze(1)
        
        print("After scaling to absolute coordinates:")
        print("Sample box:", boxes[0, 0])
        print("Image size:", self.img_size)
        
        scores = F.sigmoid(x['pred_logits']) if self.use_focal_loss else F.softmax(x['pred_logits'])[:, :, :-1]
        scores, labels = torch.max(scores, dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)

def rtdetr_pytorch_export(weights, cfg_file, device):
    cfg = YAMLConfig(cfg_file, resume=weights)
    checkpoint = torch.load(weights, map_location=device)
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    cfg.model.load_state_dict(state)
    return cfg.model.deploy(), cfg.postprocessor.use_focal_loss


def suppress_warnings():
    import warnings
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=ResourceWarning)


def main(args):
    suppress_warnings()

    print(f'\nStarting: {args.weights}')
    print('Opening RT-DETR PyTorch model')

    device = torch.device('cpu')
    model, use_focal_loss = rtdetr_pytorch_export(args.weights, args.config, device)
    print(f"Using focal loss: {use_focal_loss}")

    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    with torch.no_grad():
        outputs = model(dummy_input)
        
    print("\n=== Model Output Debug Info ===")
    print(f"Output type: {type(outputs)}")
    
    if isinstance(outputs, dict):
        print("\nDictionary keys:")
        for k in outputs.keys():
            print(f"- {k}: {outputs[k].shape}")
    elif isinstance(outputs, (list, tuple)):
        print("\nList/Tuple elements:")
        for i, out in enumerate(outputs):
            print(f"- Output {i}: {out.shape}")
    elif isinstance(outputs, torch.Tensor):
        print("\nSingle tensor output:")
        print(f"Shape: {outputs.shape}")
    
    print("============================\n")

    # The model returns a dictionary, so we need to process it through DeepStreamOutput
    # to get the final tensor format
    print("Raw model output (dictionary format):")
    if isinstance(outputs, dict):
        print("- pred_logits shape:", outputs['pred_logits'].shape)
        print("- pred_boxes shape:", outputs['pred_boxes'].shape)

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    model = nn.Sequential(model, DeepStreamOutput(img_size, use_focal_loss))
    
    # Test the final model output after DeepStreamOutput wrapper
    with torch.no_grad():
        final_output = model(dummy_input)
        print(f"\nFinal output shape: {final_output.shape}")  # Should be [1, 300, 6]
        
        # Check some example values
        print("\nExample detection:")
        print("Box coordinates:", final_output[0, 0, :4])
        print("Score:", final_output[0, 0, 4])
        print("Label:", final_output[0, 0, 5])

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = f'{args.weights}.onnx'

    dynamic_axes = {
        'input': {
            0: 'batch'
        },
        'output': {
            0: 'batch'
        }
    }

    print('Exporting the model to ONNX')
    torch.onnx.export(
        model, onnx_input_im, onnx_output_file, verbose=False, opset_version=args.opset, do_constant_folding=True,
        input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes if args.dynamic else None
    )

    if args.simplify:
        print('Simplifying the ONNX model')
        import onnxslim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print(f'Done: {onnx_output_file}\n')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DeepStream RT-DETR PyTorch conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pth) file path (required)')
    parser.add_argument('-c', '--config', required=True, help='Input YAML (.yml) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if not os.path.isfile(args.config):
        raise SystemExit('Invalid config file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
