
# YOLO-World's Zero-Shot Detection

This project leverages the **YOLO-World architecture**, a variation of the YOLO object detection model, to automate video annotation. It processes video frames, detects objects, and annotates them with bounding boxes and labels. Additionally, the project incorporates **zero-shot detection**, enabling the identification of objects the model hasn't explicitly been trained on. The result is a fully annotated video, offering a streamlined solution for object detection and annotation tasks.

---

## Features

- **YOLO-World Architecture**: A robust object detection model used for video annotation.
- **Zero-Shot Detection**: Allows detection of unseen object categories without retraining.
- **Annotation Filtering**: Filters out smaller objects (<10% of frame area) to enhance output relevance.
- **Automated Video Processing**: Annotates video frames efficiently and generates an annotated video.

---

## Requirements

Ensure you have the following installed:

- Python 3.8+
- Required libraries:
  - `ultralytics` (for YOLOv8)
  - `segment-anything` (if SAM integration is added in the future)
  - `opencv-python`
  - `numpy`
  - `tqdm`

Install dependencies using:
```bash
pip install ultralytics opencv-python numpy tqdm
```

---

## How It Works

1. **Video Frame Extraction**:
   - Frames are extracted sequentially from the source video.

2. **Object Detection**:
   - YOLO-World detects objects in each frame with bounding boxes and labels.

3. **Annotation Filtering**:
   - Smaller objects occupying less than 10% of the frame area are removed.

4. **Zero-Shot Detection**:
   - YOLO-World identifies objects it hasn't explicitly been trained on, adding adaptability to new object categories.

5. **Annotated Video Output**:
   - The annotated frames are written into a new video file.

---

## Usage

1. **Prepare Your Video**:
   - Place your source video in the working directory and update the `SOURCE_VIDEO_PATH` and `TARGET_VIDEO_PATH` variables in the script.

2. **Run the Script**:
   ```bash
   python yolo_world_script.py
   ```

3. **Output**:
   - The annotated video will be saved to the specified target path.

---

## Example Code

```python
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
import some_video_processing_library as sv  # Replace with the actual library

# Paths
SOURCE_VIDEO_PATH = "path/to/source_video.mp4"
TARGET_VIDEO_PATH = "path/to/annotated_video.mp4"

# Load YOLO-World model
yolo_model = YOLO("yolo_world_model.pt")

# Video setup
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
sink = sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info)

frame_area = video_info.width * video_info.height

for frame in tqdm(frame_generator, total=video_info.total_frames):
    # YOLO detection
    results = yolo_model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()

    # Filter small detections
    areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
    detections = detections[areas / frame_area < 0.10]

    # Annotate frame
    annotated_frame = frame.copy()
    for box in detections:
        cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    # Write to output video
    sink.write_frame(annotated_frame)

sink.close()
```

---

## Future Improvements

- **SAM Integration**:
  - Use the **Segment Anything Model (SAM)** for pixel-perfect segmentation instead of bounding boxes.
- **Enhanced Filtering**:
  - Add advanced filtering techniques for more refined object detection.
- **Multi-Class Analysis**:
  - Improve handling of overlapping objects with different classes.

---

## Acknowledgments

- **YOLOv8**: Object detection framework by Ultralytics.
- **Zero-Shot Learning**: Enabling detection of unseen object categories.

---

## License

This project is licensed under the MIT License. Feel free to use and modify the code as needed.
