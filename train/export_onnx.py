
import argparse, torch
from occlusion_model import OcclusionNet

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', default='models/occlusion.onnx')
    ap.add_argument('--img', type=int, default=256)
    args = ap.parse_args()

    net = OcclusionNet(img=args.img)
    net.load_state_dict(torch.load(args.weights, map_location='cpu'))
    net.eval()

    x = torch.randn(1,3,args.img,args.img)
    with torch.no_grad():
        torch.onnx.export(
            net, x, args.out,
            input_names=['images'],
            output_names=['class_probs','part_masks'],
            opset_version=13,
            dynamic_axes={'images': {0:'batch'}, 'class_probs': {0:'batch'}, 'part_masks': {0:'batch'}}
        )
    print(f"[+] Exported to {args.out}")
