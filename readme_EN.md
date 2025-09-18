# CCTV System Based on YOLO and FFmpeg

This is a CCTV system that uses YOLO for object detection and FFmpeg for video capture and processing. It supports motion detection, object detection, Telegram notifications, and video recording.

---

## **Key Features**

1. **Motion Detection:**
   - Analyzes frames for motion using a background subtraction algorithm.
   - Logs the zone where motion is detected (e.g., "top left").
   - Visualizes the motion area on screenshots.

2. **Object Detection:**
   - Uses YOLO to detect objects (people, animals, etc.).
   - Logs the object type, confidence, and its location on the frame.
   - Supports filtering objects by types (e.g., only people and cats).

3. **Telegram Notifications:**
   - Sends photos and videos to Telegram when motion or objects are detected.
   - Supports individual chat configuration for each camera.

4. **Video Recording:**
   - Records video when motion or objects are detected.
   - Supports 10-second buffering.

5. **Test Functions:**
   - Captures a test frame and video on system startup.

---

## **Requirements**

1. **Hardware:**
   - CPU: Intel i7-2700K or higher.
   - RAM: 8 GB or more.
   - Hard Drive: for storing buffer and recordings.

2. **Software:**
   - Python 3.8 or higher.
   - FFmpeg.
   - Libraries: `opencv-python`, `ultralytics`, `aiohttp`, `numpy`, `yaml`.

---

## **Installation**

1. Install dependencies:
   ```bash
   pip install opencv-python ultralytics aiohttp numpy yaml# Surveillance System Based on YOLO and FFmpeg

This is a surveillance system that uses YOLO for object detection and FFmpeg for video capture and processing. It supports motion detection, object detection, sending notifications to Telegram, and video recording.

---

## **Core Features**

1.  **Motion Detection:**
    *   Analyzes frames for motion using a background subtraction algorithm.
    *   Logs the zone where motion is detected (e.g., "top left").
    *   Visualizes the motion area on screenshots.

2.  **Object Detection:**
    *   Uses YOLO to detect objects (people, animals, etc.).
    *   Logs the object type, confidence score, and its location in the frame.
    *   Supports filtering objects by type (e.g., only people and cats).

3.  **Telegram Notifications:**
    *   Sends photos and videos to Telegram upon motion or object detection.
    *   Supports configuring a specific chat for each camera.

4.  **Video Recording:**
    *   Records video when motion or objects are detected.
    *   Supports 10-second buffering.

5.  **Test Functions:**
    *   Captures a test frame and video upon system startup.

---

## **Requirements**

1.  **Hardware:**
    *   CPU: Intel i7-2700K or higher.
    *   RAM: 8 GB or more.
    *   Hard Drive: For storing buffer and recordings.

2.  **Software:**
    *   Python 3.8 or higher.
    *   FFmpeg.
    *   Libraries: `opencv-python`, `ultralytics`, `aiohttp`, `numpy`, `yaml`.

---

## **Installation**

1.  Install dependencies:
    ```bash
    pip install opencv-python ultralytics aiohttp numpy pyyaml
    ```

2.  Install FFmpeg:
    *   For Ubuntu:
        ```bash
        sudo apt install ffmpeg
        ```
    *   For Windows: Download from the [official website](https://ffmpeg.org/download.html).

3.  Clone the repository:
    ```bash
    git clone clone evollved/MotionWatch-CCTV_SYSTEM_TELEGRAM_BOT
    cd MotionWatch-CCTV_SYSTEM_TELEGRAM_BOT
    ```

4.  Configure the files:
    *   `config/cameras.yaml` — Camera settings.
    *   `config/telegram.yaml` — Telegram settings.

5.  Start the system:
    ```bash
    python main.py
    ```

---

## **Configuration**

### **1. Camera Settings (`config/cameras.yaml`)**

Example camera configuration:

```yaml
cameras:
  - id: 1
    name: "Balcony"
    rtsp_url: "rtsp://192.168.2.5:554/user=video_password=video_channel=0_stream=0&onvif=0.sdp?real_stream"
    width: 1920
    height: 1080
    fps: 15
    enabled: true
    detect_motion: true  # Enable motion detection
    detect_objects: true  # Enable object detection
    object_confidence: 0.5  # Confidence threshold for objects
    telegram_chat_id: "-1001234567890"  # Telegram chat ID
    send_photo: true  # Send photos to Telegram
    send_video: true  # Send videos to Telegram
    draw_boxes: true  # Draw bounding boxes around objects
    test_frame_on_start: true  # Capture a test frame on startup
    test_video_on_start: true  # Capture a test video on startup
    motion_sensitivity: 0.2  # Motion detection sensitivity
    record_audio: false  # Record audio
    reconnect_attempts: 5  # Number of reconnection attempts
    reconnect_delay: 10  # Delay between reconnection attempts
    show_live_feed: false  # Show live video feed
    object_types: ["person", "cat", "dog"]  # Object types to detect
```
2. Telegram Settings (config/telegram.yaml)

Example Telegram configuration:
```yaml

bot_token: "123456789:ABCdefGhIJKlmNoPQRstuVWXyz"  # Bot token
```
Log Examples
```
    Motion Detection:
    text

2025-03-07 18:07:27,096 - INFO - Camera Balcony: Motion detected in the top left zone
2025-03-07 18:07:27,100 - INFO - Camera Balcony: Attempting to send photo to Telegram...

Object Detection:
text

2025-03-07 18:07:27,105 - INFO - Camera Balcony: Detected object 'person' with confidence 0.85, location: top left
2025-03-07 18:07:27,110 - INFO - Camera Balcony: Detected object 'cat' with confidence 0.72, location: bottom right

Errors:
text

2025-03-07 18:07:27,115 - ERROR - Camera Balcony: Error capturing frame with FFmpeg: Connection refused
```
Usage Examples
1. Starting the System

```bash

python main.py
```
2. Test Frame

    Upon startup, the system captures a test frame and sends it to Telegram.

3. Test Video

    Upon startup, the system captures a test video (5 seconds) and sends it to Telegram.

4. Sending Photo on Motion

    If motion is detected, the system sends a photo to Telegram indicating the motion zone.

5. Sending Video on Motion

    If motion is confirmed, the system records a video (15 seconds) and sends it to Telegram.

## **FAQ**
1. How to add a new camera?

    Add a new entry in config/cameras.yaml with a unique id and camera settings.

2. How to change motion detection sensitivity?

    Modify the motion_sensitivity parameter in config/cameras.yaml. Lower values mean higher sensitivity.

3. How to disable object detection?

    Set detect_objects: false in the camera settings.

4. How to change the types of objects to detect?

    Modify the object_types list in the camera settings. For example:
    yaml

object_types: ["person", "car"]

---

## **License**

This project is licensed under the MIT License. See the LICENSE file for details.

---

## **Author**

[Neural Networks]

[Email is too well-known to mention]

[Your ad could be here (:]
