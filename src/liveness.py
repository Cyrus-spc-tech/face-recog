
import numpy as np, cv2

class Liveness:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def predict(self, face_crop):
        if not self.enabled:
            return "unknown", 0.5
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.log(1 + np.abs(fshift))
        high = mag[int(0.4*mag.shape[0]):, int(0.4*mag.shape[1]):].mean()
        low = mag[:int(0.2*mag.shape[0]), :int(0.2*mag.shape[1])].mean()
        score = float(high / (low + 1e-6))
        live = "live" if score > 1.05 else "spoof"
        return live, score
