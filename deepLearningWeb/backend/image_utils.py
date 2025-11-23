import cv2
import numpy as np

def apply_enhancement(frame, enhancement_type="none"):
    """
    manipulasi gambar / enchancement
    """
    if enhancement_type == "contrast":
        # Menggunakan CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Ini lebih bagus daripada equalizer biasa untuk CCTV/Jalan raya
        img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    elif enhancement_type == "brightness":
        # Meningkatkan brightness (alpha=1, beta=50)
        return cv2.convertScaleAbs(frame, alpha=1, beta=50)
    
    elif enhancement_type == "grayscale":
        # Ubah ke hitam putih (opsional, kadang bagus untuk deteksi malam)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return frame