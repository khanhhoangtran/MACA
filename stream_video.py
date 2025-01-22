from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
from pathlib import Path
import time
import io

app = FastAPI()

# Function to draw bounding boxes
def draw_bbox(frame, line):
    h, w = frame.shape[:2]
    values = list(map(float, line.strip().split()))
    class_id, x_center, y_center, width, height = values

    color = get_color_for_id(int(class_id))

    x = int(x_center * w)
    y = int(y_center * h) 
    box_w = int(width * w)
    box_h = int(height * h)

    x1 = max(0, int(x - box_w / 2))
    y1 = max(0, int(y - box_h / 2))
    x2 = min(w, int(x + box_w / 2))
    y2 = min(h, int(y + box_h / 2))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"Class {int(class_id)}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)

def get_color_for_id(obj_id):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab20')
    color_rgb = cmap(obj_id % 20)[:3]
    return tuple(int(c * 255) for c in color_rgb[::-1])

# Stream video with bounding boxes
def stream_video(video_path: str, labels_dir: str, output_path: str):
    cap = cv2.VideoCapture(video_path)
    labels_dir = Path(labels_dir)
    output_path = Path(output_path)
    frame_count = 0

    # Prepare video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label_file = labels_dir / f"frame_{frame_count:06d}.txt"
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    draw_bbox(frame, line)

        # Write the frame to the output file
        writer.write(frame)

        # Encode frame to JPEG for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = io.BytesIO(buffer.tobytes())

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes.read() + b"\r\n"

        frame_count += 1
        time.sleep(0.03)  # Adjust sleep for frame rate

    cap.release()
    writer.release()

@app.get("/video")
def video_stream():
    video_path = "/Users/giaosusoi/Project Local/MACA tools/10AM-Magnolia-NE_chunk_2.mp4"   #source video path
    labels_dir = "/Users/giaosusoi/Project Local/MACA tools/labels"                         #labels path
    output_path = "/Users/giaosusoi/Project Local/MACA tools/output.mp4"                    #output video path
    return StreamingResponse(stream_video(video_path, labels_dir, output_path), media_type="multipart/x-mixed-replace; boundary=frame")