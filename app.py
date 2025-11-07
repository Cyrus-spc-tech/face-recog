
import os, argparse, yaml
from insightface.app import FaceAnalysis
from src.gallery import Gallery
from src.pipeline import run_stream

def load_cfg(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_gallery(cfg):
    print("[*] Building gallery...")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=tuple(cfg['runtime']['det_size']))
    from src.gallery import Gallery
    gal = Gallery(index_path=cfg['gallery']['index'], meta_path=cfg['gallery']['meta'])
    n = gal.build(app, gallery_root=cfg['gallery']['root'])
    print(f"[+] Built gallery with {n} entries")

def main():
    cfg = load_cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-gallery', action='store_true')
    parser.add_argument('--webcam', type=int, default=None)
    parser.add_argument('--video', type=str, default=None)
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs(cfg['gallery']['root'], exist_ok=True)

    if args.build_gallery:
        build_gallery(cfg)
        return

    source = args.video if args.video else (args.webcam if args.webcam is not None else 0)

    gal = Gallery(index_path=cfg['gallery']['index'], meta_path=cfg['gallery']['meta'])
    if not gal.load():
        print("[!] No gallery found. Run with --build-gallery after placing images in data/gallery/<id>")
        gal = None

    run_stream(
        source=source,
        cfg=cfg,
        gallery=gal
    )

if __name__ == '__main__':
    main()
