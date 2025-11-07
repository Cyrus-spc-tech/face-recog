
import os, argparse, random, cv2, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from occlusion_model import OcclusionNet, OCCLUSION_CLASSES, PARTS

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class OccDataset(Dataset):
    def __init__(self, manifest, img_size=256, occluder_root=None):
        self.df = pd.read_csv(manifest)
        self.size = img_size
        self.occ_root = occluder_root
        self.cls2idx = {c:i for i,c in enumerate(OCCLUSION_CLASSES)}
        self.augs = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_REFLECT_101),
            A.ColorJitter(0.2,0.2,0.2,0.1,p=0.5),
            A.MotionBlur(p=0.2), A.GaussNoise(p=0.2),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ])

    def __len__(self): return len(self.df)

    def _apply_synth_occluder(self, img):
        if not self.occ_root or random.random() < 0.5:
            return img
        cat = random.choice(['hands','phones','masks','scarves','glasses'])
        cat_dir = os.path.join(self.occ_root, cat)
        if not os.path.isdir(cat_dir): return img
        cands = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.lower().endswith(('.png','.webp'))]
        if not cands: return img
        occ = cv2.imread(random.choice(cands), cv2.IMREAD_UNCHANGED)
        if occ is None or occ.shape[2] < 4: return img
        H,W = img.shape[:2]
        scale = random.uniform(0.3, 0.8)
        oh, ow = int(H*scale), int(W*scale)
        occ = cv2.resize(occ, (ow, oh))
        oy = random.randint(0, H-oh); ox = random.randint(0, W-ow)
        roi = img[oy:oy+oh, ox:ox+ow]
        alpha = occ[:,:,3:4]/255.0
        img[oy:oy+oh, ox:ox+ow] = (alpha*occ[:,:,:3] + (1-alpha)*roi).astype(np.uint8)
        return img

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = cv2.imread(r['img_path'])
        if img is None:
            raise FileNotFoundError(r['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._apply_synth_occluder(img)
        labels = str(r['labels']).split(';') if isinstance(r['labels'], str) else []
        y = np.zeros(len(OCCLUSION_CLASSES), dtype=np.float32)
        for l in labels:
            if l in self.cls2idx: y[self.cls2idx[l]] = 1.0
        mask = None
        if 'mask_path' in r and isinstance(r['mask_path'], str) and r['mask_path']:
            m = cv2.imread(r['mask_path'], cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                # Here we keep a single-channel mask; a production setup would store 5-part masks.
                mask = np.stack([m/255.0]*5, axis=0).astype(np.float32)
        out = self.augs(image=img)
        x = out['image']
        if mask is None:
            mask = np.zeros((5, self.size, self.size), dtype=np.float32)
        return x/255.0, torch.from_numpy(y), torch.from_numpy(mask)

def train(args):
    seed_all(42)
    ds = OccDataset(args.manifest, img_size=args.img, occluder_root=args.occluders)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = OcclusionNet(img=args.img).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    for epoch in range(args.epochs):
        net.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        losses = []
        for x, y, m in pbar:
            x, y, m = x.to(device), y.to(device), m.to(device)
            cls, seg = net(x)
            loss_cls = bce(cls, y)
            loss_seg = mse(seg, m)
            loss = loss_cls + args.lambda_seg * loss_seg
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=sum(losses)/len(losses))
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save(net.state_dict(), args.out)
    print(f"[+] Saved weights to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', default='models/occlusion.pt')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--img', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lambda-seg', dest='lambda_seg', type=float, default=0.5)
    ap.add_argument('--occluders', type=str, default=None, help='folder with transparent PNG occluders')
    args = ap.parse_args(); train(args)
