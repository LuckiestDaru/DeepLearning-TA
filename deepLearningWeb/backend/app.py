import os
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

#load model trained
model = YOLO("models/yolo11m.pt") 

#class mapping
CLASS_ID_TO_NAME = {
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck"
}
VALID_CLASSES = [2, 3, 5, 7]

#tools
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4, text_thickness=0, text_scale=0, color=sv.Color.GREEN 
)

#global state
current_config = {
    "mode": "detect",           
    "enhancement": "none",      
    "direction": "top-down",    
    "source": None,             
    "is_image": False,
    "active": False             
}

stats = { "car": 0, "motorbike": 0, "bus": 0, "truck": 0, "total": 0 }
line_zone = None

def setup_line_zone(frame_shape):
    global line_zone
    h, w, _ = frame_shape
    if current_config['direction'] == 'top-down':
        start, end = sv.Point(0, h//2), sv.Point(w, h//2)
    else: 
        start, end = sv.Point(w//2, 0), sv.Point(w//2, h)
    line_zone = sv.LineZone(start=start, end=end)

def process_frame(frame):
    global line_zone, stats
    
    # 1. Enhancement
    frame = apply_enhancement(frame, current_config['enhancement'])

    # 2. YOLO Inference
    results = model(frame, verbose=False, conf=0.4, iou=0.5, classes=VALID_CLASSES)[0]
    detections = sv.Detections.from_ultralytics(results)

    # image or video check
    # image pross
    if current_config['is_image']:
        temp_stats = { "car": 0, "motorbike": 0, "bus": 0, "truck": 0, "total": 0 }
        
        for class_id in detections.class_id:
            if class_id in CLASS_ID_TO_NAME:
                name = CLASS_ID_TO_NAME[class_id]
                temp_stats[name] += 1
                temp_stats['total'] += 1
        
        stats = temp_stats

        labels = [f"{CLASS_ID_TO_NAME[c_id]} {conf:.2f}" for c_id, conf in zip(detections.class_id, detections.confidence)]
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # video or webcam check
    elif not current_config['is_image'] and current_config['active']:
        
        if current_config['mode'] == 'detect':
            labels = [f"{CLASS_ID_TO_NAME[c_id]} {conf:.2f}" for c_id, conf in zip(detections.class_id, detections.confidence)]
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        
        elif current_config['mode'] == 'count-video':
            detections = tracker.update_with_detections(detections)
            if line_zone is None: setup_line_zone(frame.shape)

            cross_in, cross_out = line_zone.trigger(detections)
            crossed = cross_in | cross_out 
            
            if np.any(crossed):
                crossed_class_ids = detections.class_id[crossed]
                for class_id in crossed_class_ids:
                    if class_id in CLASS_ID_TO_NAME:
                        stats['total'] += 1
                        stats[CLASS_ID_TO_NAME[class_id]] += 1
            
            labels = [f"#{t_id} {CLASS_ID_TO_NAME[c_id]}" for t_id, c_id in zip(detections.tracker_id, detections.class_id)]
            frame = trace_annotator.annotate(scene=frame, detections=detections)
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
            line_zone_annotator.annotate(frame, line_counter=line_zone)
            
            text = f"TOTAL: {stats['total']}"
            cv2.rectangle(frame, (20, frame.shape[0] - 80), (250, frame.shape[0] - 20), (0, 0, 0), -1)
            cv2.putText(frame, text, (30, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def get_video_frames():
    if not current_config['active'] or current_config['source'] is None: return 
    cap = cv2.VideoCapture(current_config['source'])
    while current_config['active']:
        success, frame = cap.read()
        if not success:
            if isinstance(current_config['source'], str): 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else: break 
        
        processed_frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# camera routes
@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global current_config
    current_config['active'] = False
    current_config['source'] = None
    return jsonify({"status": "stopped"})

@app.route('/set_webcam', methods=['POST'])
def set_webcam():
    global current_config, tracker
    tracker = sv.ByteTrack() 
    current_config['active'] = True
    current_config['source'] = 0
    current_config['is_image'] = False
    return jsonify({"status": "switched to webcam"})

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_config, stats, line_zone, tracker
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    ext = filename.rsplit('.', 1)[1].lower()
    
    tracker = sv.ByteTrack()
    current_config['active'] = True
    current_config['source'] = filepath
    stats = { "car": 0, "motorbike": 0, "bus": 0, "truck": 0, "total": 0 }

    if ext in ['jpg', 'jpeg', 'png', 'webp']:
        current_config['is_image'] = True
    else:
        current_config['is_image'] = False
        line_zone = None 
    
    return jsonify({"status": "success", "type": "image" if current_config['is_image'] else "video"})

@app.route('/video_feed')
def video_feed():
    return Response(get_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_image')
def get_processed_image():
    if not current_config['is_image'] or not current_config['active']: return jsonify({"error": "No image"}), 400
    frame = cv2.imread(current_config['source'])
    processed = process_frame(frame)
    ret, buffer = cv2.imencode('.jpg', processed)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"image": f"data:image/jpeg;base64,{img_str}"})

@app.route('/update_config', methods=['POST'])
def update_config():
    global current_config, line_zone
    data = request.json
    if 'direction' in data and data['direction'] != current_config['direction']:
        line_zone = None 
    current_config.update(data)
    return jsonify({"status": "success", "config": current_config})

@app.route('/stats')
def get_stats():
    return jsonify(stats)

if __name__ == "__main__":
    #ceck if running on railway
    railway_port = os.environ.get("PORT")
    if railway_port:
        #host 0.0.0.0 untuk cloud deployment
        print(f"🚀 Running in Production Mode on Port {railway_port}")
        app.run(host='0.0.0.0', port=int(railway_port), debug=False)
    else:
        #host localhost untuk development lokal
        print("🚀 Running in Local Development Mode")
        app.run(debug=True, port=5000)