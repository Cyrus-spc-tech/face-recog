
import os, numpy as np, cv2

OCCLUSION_CLASSES = ["mask","hand","phone","glasses","scarf","hat","image","unknown"]

class HeuristicOcclusion:
    def _part_boxes(self, bbox, kps):
        x, y, w, h = bbox
        if kps is None:
            fh = int(0.25*h); eye_h = int(0.2*h); mouth_h = int(0.22*h)
            return {
                "forehead": (x, y, w, fh),
                "eyes": (x, y+fh, w, eye_h),
                "nose": (x, y+fh+eye_h, w, int(0.2*h)),
                "cheeks": (x, y+int(0.5*h), w, int(0.2*h)),
                "mouth": (x, y+int(0.72*h), w, mouth_h),
            }
        else:
            le, re, nose, lm, rm = kps
            x0, y0, w0, h0 = x, y, w, h
            eye_y = int(min(le[1], re[1]))
            mouth_y = int(max(lm[1], rm[1]))
            nose_y = int(nose[1])
            forehead_h = max(1, eye_y - y0)
            eyes_h = max(1, nose_y - eye_y)
            mouth_h = max(1, y0 + h0 - mouth_y)
            return {
                "forehead": (x0, y0, w0, forehead_h),
                "eyes": (x0, eye_y, w0, eyes_h),
                "nose": (x0, nose_y, w0, max(1, int(0.18*h0))),
                "cheeks": (x0, int(y0+0.5*h0), w0, int(0.2*h0)),
                "mouth": (x0, mouth_y, w0, mouth_h),
            }

    def _coverage_ratio(self, img, box):
        x,y,w,h = box
        H, W = img.shape[:2]
        x = max(0, min(W-1, x)); y = max(0, min(H-1, y))
        w = max(1, min(W - x, w)); h = max(1, min(H - y, h))
        roi = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        var = float(gray.var())
        norm = var / (1.0 + np.log1p(w*h))
        cov = float(np.clip(1.0 - norm/20.0, 0.0, 1.0))
        return cov

    def predict(self, frame, face):
        bbox = face["bbox"]
        kps = face.get("kps", None)
        parts = self._part_boxes(bbox, kps)
        part_cov = {p: self._coverage_ratio(frame, box) for p, box in parts.items()}
        classes = ["unknown"]
        if part_cov.get("mouth",0) > 0.7 and part_cov.get("nose",0) > 0.6:
            classes = ["mask"]
        return classes, part_cov

class OcclusionInfer:
    def __init__(self, onnx_path=None):
        self.onnx_path = onnx_path
        self.rt = None
        self.heur = HeuristicOcclusion()
        if onnx_path and os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                self.rt = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                self.inp_name = self.rt.get_inputs()[0].name
                self.out_names = [o.name for o in self.rt.get_outputs()]  # [class_probs, part_mask]
            except Exception as e:
                print(f"[!] Failed to load ONNX occlusion model: {e}. Falling back to heuristic.")

    def _pre(self, frame, face, img_size=256):
        x,y,w,h = face["bbox"]
        H,W = frame.shape[:2]
        x = max(0, min(W-1, x)); y = max(0, min(H-1, y))
        w = max(1, min(W-x, w)); h = max(1, min(H-y, h))
        crop = frame[y:y+h, x:x+w]
        img = cv2.resize(crop, (img_size, img_size))
        img = img[:,:,::-1].astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2,0,1))[None, ...]
        return img

    def predict(self, frame, face):
        if self.rt is None:
            return self.heur.predict(frame, face)
        x = self._pre(frame, face)
        outs = self.rt.run(self.out_names, {self.inp_name: x})
        probs = outs[0][0]  # (C,)
        part_mask = outs[1][0]  # (5,H,W) -> reduce to ratios
        classes = [OCCLUSION_CLASSES[i] for i, p in enumerate(probs) if p > 0.4]
        if not classes:
            classes = ["unknown"]
        ratios = part_mask.mean(axis=(1,2)).tolist()  # assume already in [0,1] coverage
        parts = ["eyes","nose","mouth","cheeks","forehead"]
        part_cov = {k: float(v) for k, v in zip(parts, ratios)}
        return classes, part_cov
