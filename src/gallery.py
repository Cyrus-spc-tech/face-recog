
import os, glob, json
import numpy as np
import faiss
import cv2

class Gallery:
    def __init__(self, index_path='data/gallery.index', meta_path='data/gallery_meta.json'):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.ids = []

    def build(self, analyzer, gallery_root='data/gallery'):
        embs = []
        self.ids = []
        for pid in sorted(os.listdir(gallery_root)):
            person_dir = os.path.join(gallery_root, pid)
            if not os.path.isdir(person_dir): 
                continue
            for imgp in glob.glob(os.path.join(person_dir, "*.*")):
                img = cv2.imread(imgp)
                if img is None: 
                    continue
                faces = analyzer.get(img)
                if not faces:
                    continue
                f = max(faces, key=lambda x: x.det_score)
                emb = getattr(f, 'normed_embedding', None)
                if emb is None:
                    continue
                embs.append(emb.astype(np.float32))
                self.ids.append(pid)
        if not embs:
            raise RuntimeError("No embeddings built. Place images under data/gallery/<id>/*.jpg")
        embs = np.vstack(embs).astype('float32')
        d = embs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embs)
        self.index.add(embs)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'w') as f:
            json.dump(self.ids, f)
        return len(self.ids)

    def load(self):
        if not os.path.exists(self.index_path): 
            return False
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, 'r') as f:
            self.ids = json.load(f)
        return True

    def search(self, emb, topk=1):
        if self.index is None:
            raise RuntimeError("Gallery not loaded")
        import numpy as np, faiss
        v = emb.astype('float32')[None, :]
        faiss.normalize_L2(v)
        D, I = self.index.search(v, topk)
        sims = D[0]
        idxs = I[0]
        results = []
        for sim, idx in zip(sims, idxs):
            if idx == -1: 
                continue
            pid = self.ids[idx]
            results.append((pid, float(sim)))
        return results
