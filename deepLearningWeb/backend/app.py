import os
import time 
print("MEMULAI FLASK.....!!", flush=True)

import cv2
import numpy as np
import base64
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import supervision as sv
from werkzeug.utils import secure_filename
from image_utils import apply_enhancement

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Gunakan Nano agar ringan
model = YOLO("yolo11m.pt") 

CLASS_ID_TO_NAME = { 2: "car", 3: "motorbike", 5: "bus", 7: "truck" }
VALID_CLASSES = [2, 3, 5, 7]

# Tools
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator(thickness=2) 
label_annotator = sv.LabelAnnotator(text_scale=0.5)
trace_annotator = sv.TraceAnnotator()
line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=0, text_scale=0, color=sv.Color.GREEN)

current_config = {
    "mode": "detect", "enhancement": "none", "direction": "top-down", 
    "source": None, "is_image": False, "active": False 
}
stats = { "car": 0, "motorbike": 0, "bus": 0, "truck": 0, "total": 0 }
line_zone = None
last_detections = None

def setup_line_zone(frame_shape):
    global line_zone
    h, w, _ = frame_shape
    if current_config['direction'] == 'top-down':
        start, end = sv.Point(0, h//2), sv.Point(w, h//2)
    else: 
        start, end = sv.Point(w//2, 0), sv.Point(w//2, h)
    line_zone = sv.LineZone(start=start, end=end)

def process_frame_optimized(frame, run_ai=True):
    global line_zone, stats, last_detections
    
    height, width = frame.shape[:2]
    target_width = 640
    if width > target_width:
        scale = target_width / width
        new_height = int(height * scale)
        frame = cv2.resize(frame, (target_width, new_height))

    if current_config['enhancement'] != 'none':
        frame = apply_enhancement(frame, current_config['enhancement'])

    if current_config['is_image']:
        results = model(frame, verbose=False, conf=0.25, iou=0.5, classes=VALID_CLASSES)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        temp_stats = { "car": 0, "motorbike": 0, "bus": 0, "truck": 0, "total": 0 }
        for class_id in detections.class_id:
            if class_id in CLASS_ID_TO_NAME:
                temp_stats[CLASS_ID_TO_NAME[class_id]] += 1
                temp_stats['total'] += 1
        stats = temp_stats
        
        frame = box_annotator.annotate(scene=frame, detections=detections)
        # Label hanya nama kelas (tanpa angka ID)
        labels = [f"{CLASS_ID_TO_NAME[c]}" for c in detections.class_id]
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        return frame

    if current_config['active']:
        detections = None
        
        if run_ai:
            results = model(frame, verbose=False, conf=0.25, iou=0.5, classes=VALID_CLASSES)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            if current_config['mode'] == 'count-video':
                detections = tracker.update_with_detections(detections)
                if line_zone is None: setup_line_zone(frame.shape)
                cross_in, cross_out = line_zone.trigger(detections)
                crossed = cross_in | cross_out
                if np.any(crossed):
                    for c_id in detections.class_id[crossed]:
                        if c_id in CLASS_ID_TO_NAME:
                            stats['total'] += 1
                            stats[CLASS_ID_TO_NAME[c_id]] += 1
            
            last_detections = detections
        else:
            detections = last_detections

        if detections:
            labels = [f"{CLASS_ID_TO_NAME[c_id]}" for c_id in detections.class_id]
            
            if current_config['mode'] == 'count-video':
                frame = trace_annotator.annotate(scene=frame, detections=detections)
                line_zone_annotator.annotate(frame, line_counter=line_zone)

            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        text = f"TOTAL: {stats['total']}"
        cv2.rectangle(frame, (10, frame.shape[0] - 40), (150, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

def get_video_frames():
    if not current_config['active'] or current_config['source'] is None: return 
    cap = cv2.VideoCapture(current_config['source'])
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0: video_fps = 30 
    frame_duration = 1.0 / video_fps 

    frame_counter = 0
    SKIP_FRAMES = 2
    
    while current_config['active']:
        start_time = time.time()

        success, frame = cap.read()
        if not success:
            if isinstance(current_config['source'], str): 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else: break 
        
        frame_counter += 1
        should_run_ai = (frame_counter % SKIP_FRAMES == 0)
        
        processed_frame = process_frame_optimized(frame, run_ai=should_run_ai)
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_duration:
            time.sleep(frame_duration - elapsed_time)

    cap.release()

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global current_config
    current_config['active'] = False; current_config['source'] = None
    return jsonify({"status": "stopped"})

@app.route('/set_webcam', methods=['POST'])
def set_webcam():
    global current_config, tracker
    tracker = sv.ByteTrack() 
    current_config['active'] = True; current_config['source'] = 0; current_config['is_image'] = False
    return jsonify({"status": "switched to webcam"})

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_config, stats, line_zone, tracker
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    tracker = sv.ByteTrack()
    current_config['active'] = True; current_config['source'] = filepath
    stats = { "car": 0, "motorbike": 0, "bus": 0, "truck": 0, "total": 0 }
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in ['jpg', 'jpeg', 'png', 'webp']:
        current_config['is_image'] = True
    else:
        current_config['is_image'] = False; line_zone = None 
    return jsonify({"status": "success", "type": "image" if current_config['is_image'] else "video"})

@app.route('/video_feed')
def video_feed():
    return Response(get_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_image')
def get_processed_image():
    if not current_config['is_image']: return jsonify({"error": "No image"}), 400
    frame = cv2.imread(current_config['source'])
    processed = process_frame_optimized(frame) 
    ret, buffer = cv2.imencode('.jpg', processed)
    return jsonify({"image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"})

@app.route('/update_config', methods=['POST'])
def update_config():
    global current_config, line_zone
    data = request.json
    if 'direction' in data and data['direction'] != current_config['direction']: line_zone = None 
    current_config.update(data)
    return jsonify({"status": "success"})

@app.route('/stats')
def get_stats(): return jsonify(stats)

if __name__ == "__main__":
    railway_port = os.environ.get("PORT")
    if railway_port: app.run(host='0.0.0.0', port=int(railway_port), debug=False)
    else: app.run(debug=True, port=5000)