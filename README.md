
# Occlusion-Aware Face Recognition (Pro Kit)

This adds **trainable occlusion handling** + optional **liveness** to the starter kit.

## Modules
- `src/occlusion_infer.py` — ONNXRuntime inference for occlusion model (multi-label + part segmentation). Falls back to heuristic stub if model file missing.
- `train/train_occlusion.py` — PyTorch training script for occlusion model (classifier + segmentation head).
- `train/export_onnx.py` — Export trained `.pt` to `.onnx` for use at runtime.
- `src/liveness.py` — Simple frequency-based anti-spoof stub; can be replaced with a learned model later.
- `Dockerfile` — CPU base. Switch to GPU by changing the base image and installing CUDA toolchains.
- `config.yaml` — Centralized configuration.

## Quick Start (Runtime)
```bash
pip install -r requirements.txt
# (optional) If you plan to train: pip install -r requirements-train.txt

# Build gallery from images (put images under data/gallery/<ID>/*.jpg)
python app.py --build-gallery

# Run webcam
python app.py --webcam 0
# or video
python app.py --video path/to/video.mp4
```

If `models/occlusion.onnx` exists, the pipeline will use it; otherwise it falls back to a heuristic.

## Training the Occlusion Model
1. Prepare a CSV manifest (train/occlusion_manifest.csv):
```
img_path,labels,mask_path
/path/to/img1.jpg,"mask;glasses","/path/to/mask1.png"
/path/to/img2.jpg,"hand",""
```
- `labels` is a semicolon list from: `mask,hand,phone,glasses,scarf,hat,image,unknown`.
- `mask_path` optional segmentation label (same size as image). Omit or leave empty if not available.

2. (Optional) Drop synthetic occluders into `train/occluders/{hands,phones,masks,scarves,glasses}` (transparent PNGs).

3. Train:
```bash
python train/train_occlusion.py   --manifest train/occlusion_manifest.csv   --out models/occlusion.pt   --epochs 30 --batch 16 --img 256
```

4. Export to ONNX:
```bash
python train/export_onnx.py --weights models/occlusion.pt --out models/occlusion.onnx --img 256
```

5. Run the app again; it will pick up `models/occlusion.onnx` automatically.

## Liveness (Anti-Spoof)
The stub computes simple spectral features; for production, replace with a trained model and wire it in `src/liveness.py`.

## Docker
```bash
docker build -t occlu-face:cpu .
docker run --rm -it --device=/dev/video0:/dev/video0 occlu-face:cpu
```
