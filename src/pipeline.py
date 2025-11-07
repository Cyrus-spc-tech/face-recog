
import cv2, time, json, numpy as np
from .detector import FaceDetector
from .occlusion_infer import OcclusionInfer
from .embedder import Embedder
from .liveness import Liveness

def draw_face(frame, face, classes, part_cov, identity, liveness):
    x,y,w,h = face["bbox"]
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    label = f"{identity['id']}:{identity['score']:.2f}" if identity['id'] else "unknown"
    occ = ",".join(classes)
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"occ:{occ}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(frame, f"live:{liveness}", (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

def crop_from_bbox(frame, bbox):
    x,y,w,h = bbox
    H,W = frame.shape[:2]
    x = max(0, min(W-1, x)); y = max(0, min(H-1, y))
    w = max(1, min(W-x, w)); h = max(1, min(H-y, h))
    return frame[y:y+h, x:x+w]

def run_stream(source=0, cfg=None, gallery=None):
    det = FaceDetector(det_size=tuple(cfg['runtime']['det_size']))
    occ = OcclusionInfer(cfg['runtime']['occlusion_onnx'])
    embd = Embedder(det.app)
    live = Liveness(enabled=bool(cfg['runtime']['liveness_enable']))

    base_tau = float(cfg['runtime']['threshold'])
    alpha = float(cfg['runtime']['alpha'])

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = det.detect(frame)
        outputs = []
        for f in faces:
            classes, part_cov = occ.predict(frame, f)
            face_obj = f["insightface_obj"]
            emb = embd.get_embedding(face_obj)

            face_crop = crop_from_bbox(frame, f["bbox"])
            lv, lvscore = live.predict(face_crop)

            best = {"id": None, "score": 0.0, "threshold": base_tau}
            if emb is not None and gallery is not None and lv == "live":
                visible_ratio = 1.0 - float(np.mean(list(part_cov.values()))) if part_cov else 1.0
                tau = base_tau + alpha * (1.0 - visible_ratio)
                best["threshold"] = tau
                top = gallery.search(emb, topk=1)
                if top:
                    pid, sim = top[0]
                    if sim >= tau:
                        best["id"] = pid
                        best["score"] = sim

            outputs.append({
                "bbox": f["bbox"],
                "occlusions": {"classes": classes, "part_coverage": part_cov},
                "liveness": {"state": lv, "score": lvscore},
                "recognition": {"top1_id": best["id"], "top1_score": best["score"], "threshold_used": best["threshold"]}
            })
            draw_face(frame, f, classes, part_cov, {"id": best["id"] or "unknown", "score": best["score"]}, lv)
        cv2.imshow("Occlusion-Aware Face Recognition (Pro)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
