import joblib
from sklearn.svm import SVC
import os
import cv2
import numpy as np
def calculate_optical_flow_farneback(prev_frame, curr_frame):
    """
    Вычисляет оптический поток методом Farneback.

    Args:
        prev_frame: Предыдущий кадр.
        curr_frame: Текущий кадр.

    Returns:
        Векторное поле оптического потока (u, v) в виде NumPy массива.
    """

    # Преобразуем кадры в оттенки серого
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Вычисляем оптический поток методом Farneback
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    return flow
class VideoClassifier():
    def __init__(self):
        self.classifier = joblib.load('svc_model.pkl')

    def predict(self, cap):
        """
        Args:  cap: cv2.VideoCapture(vide_path)
        Returns: 1 - fake, 1 - real
        """


        ret, prev_frame = cap.read()

        u, v = np.array([]), np.array([])

        mx_frames = 30
        frame = 0

        while(True):
            # Получение текущего кадра
            ret, curr_frame = cap.read()
            if not ret:
                break

            flow = calculate_optical_flow_farneback(prev_frame, curr_frame)

            u = np.concatenate((u, flow[..., 0].ravel()))  # Horizontal flow (u component)
            v = np.concatenate((v, flow[..., 1].ravel()))

            prev_frame = curr_frame
            if frame >= mx_frames:
                break
            frame += 1

        u_mean = np.mean(u)
        v_mean = np.mean(v)
        u_variance = np.var(u)
        v_variance = np.var(v)
        u_max = np.max(u)
        v_max = np.max(v)

        features = [[u_mean, v_mean, u_variance, v_variance, u_max, v_max]]
        cap.release()


        return self.classifier.predict(features)
