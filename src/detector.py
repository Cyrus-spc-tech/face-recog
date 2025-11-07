
import numpy as np
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, providers=None, det_size=(640,640)):
        self.app = FaceAnalysis(name='buffalo_l', providers=providers or ['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0 if 'CUDAExecutionProvider' in (providers or []) else -1, det_size=det_size)

    def detect(self, bgr_image):
        faces = self.app.get(bgr_image)
        results = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            kps = f.kps.astype(float) if hasattr(f, 'kps') else None
            results.append({
                "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                "kps": kps,
                "score": float(getattr(f, 'det_score', 1.0)),
                "insightface_obj": f
            })
        return results
