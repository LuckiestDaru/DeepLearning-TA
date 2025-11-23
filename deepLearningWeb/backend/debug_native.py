from ultralytics import YOLO
import os

# Path
MODEL_PATH = "models/best.pt"
VIDEO_PATH = "models/video.mp4" # Pastikan video ini ada

def main():
    print("🚀 Menjalankan YOLO Native Mode (Tanpa Supervision)...")
    
    # Load model
    model = YOLO(MODEL_PATH)
    
    # Jalankan prediksi
    # save=True: YOLO akan otomatis menggambar kotak dan simpan videonya
    # imgsz=640: Memaksa input gambar di-resize ke ukuran standar training (PENTING!)
    # conf=0.25: Standar confidence threshold
    results = model.predict(
        source=VIDEO_PATH, 
        save=True, 
        imgsz=640, 
        conf=0.25,
        iou=0.45,
        project="runs/detect", # Hasil akan disimpan di folder backend/runs/detect
        name="debug_result"    # Nama subfolder
    )
    
    print(f"✅ Selesai! Cek hasil video di folder: backend/runs/detect/debug_result/")

if __name__ == "__main__":
    main()