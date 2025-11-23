import os
import cv2
from ultralytics import YOLO
import supervision as sv

# --- KONFIGURASI PATH ---
VIDEO_PATH = os.path.join("models", "video.mp4")
OUTPUT_PATH = os.path.join("models", "output_official.mp4")

# GANTI INI: Jangan pakai best.pt dulu, pakai yolo11m.pt (akan auto-download)
# yolo11m.pt biasanya ada di root folder backend setelah command langkah 1, 
# atau kamu bisa copy file yolo11m.pt ke folder models juga biar rapi.
MODEL_PATH = "yolo11m.pt"

def main():
    # 1. Cek keberadaan file
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model tidak ditemukan di {MODEL_PATH}")
        return
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video input tidak ditemukan di {VIDEO_PATH}")
        print("👉 Tolong masukkan video ke folder 'backend/models/' dan rename menjadi 'video.mp4'")
        return

    print(f"✅ Model found: {MODEL_PATH}")
    print(f"✅ Input video: {VIDEO_PATH}")
    print("⏳ Sedang memuat model...")

    # 2. Load Model
    model = YOLO(MODEL_PATH)

    # 3. Setup Video Info & Output
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
    print(f"📹 Info Video: {video_info.width}x{video_info.height} @ {video_info.fps} FPS")
    
    # 4. Setup Tracker & Annotators
    # ByteTrack digunakan untuk menjaga ID objek tetap sama antar frame
    tracker = sv.ByteTrack()
    
    # Annotator untuk gambar kotak dan teks
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    print("🚀 Mulai memproses video... (Tekan Ctrl+C untuk stop paksa)")

    # 5. Proses Loop Video
    # VideoSink akan otomatis menyimpan hasil frame ke file output
    with sv.VideoSink(target_path=OUTPUT_PATH, video_info=video_info) as sink:
        
        # Generator frame demi frame
        for frame_idx, frame in enumerate(sv.get_video_frames_generator(source_path=VIDEO_PATH)):
            
            # A. Deteksi dengan YOLO
            # conf=0.5 -> Hanya deteksi jika yakin > 50% (Mengurangi noise/kotak hantu)
            # iou=0.5  -> Mengurangi kotak tumpang tindih
            results = model(frame, verbose=False, conf=0.5, iou=0.5)[0]
            detections = sv.Detections.from_ultralytics(results)

            # B. Update Tracker
            detections = tracker.update_with_detections(detections)

            # C. Buat Label
            # Format: #ID NamaKelas Confidence
            labels = [
                f"#{tracker_id} {model.names[class_id]} {confidence:.2f}"
                for tracker_id, class_id, confidence
                in zip(detections.tracker_id, detections.class_id, detections.confidence)
            ]

            # D. Gambar Anotasi ke Frame
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

            # E. Simpan ke Video Output
            sink.write_frame(frame)

            # Tampilkan progress setiap 20 frame agar terminal tidak penuh
            if frame_idx % 20 == 0:
                print(f"Processing frame {frame_idx}...", end='\r')

    print(f"\n✅ Selesai! Hasil tersimpan di: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()