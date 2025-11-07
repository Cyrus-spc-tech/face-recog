
import numpy as np

class Embedder:
    def __init__(self, insightface_app):
        self.app = insightface_app

    def get_embedding(self, face_obj):
        if hasattr(face_obj, 'normed_embedding') and face_obj.normed_embedding is not None:
            return np.asarray(face_obj.normed_embedding, dtype=np.float32)
        if hasattr(face_obj, 'embedding') and face_obj.embedding is not None:
            v = face_obj.embedding
            v = v / (np.linalg.norm(v) + 1e-9)
            return v.astype(np.float32)
        return None
