# 基于 YOLO 和 FFmpeg 的视频监控系统

这是一个视频监控系统，使用 YOLO 进行对象检测，FFmpeg 进行视频捕获和处理。它支持运动检测、对象检测、发送 Telegram 通知和视频录制。

---

## **主要功能**

1.  **运动检测:**
    *   使用背景减除算法分析帧中的运动。
    *   记录检测到运动的区域（例如，“左上角”）。
    *   在截图上可视化运动区域。

2.  **对象检测:**
    *   使用 YOLO 检测对象（人、动物等）。
    *   记录对象类型、置信度及其在帧中的位置。
    *   支持按类型过滤对象（例如，仅人和猫）。

3.  **发送 Telegram 通知:**
    *   检测到运动或对象时，向 Telegram 发送照片和视频。
    *   支持为每个摄像头配置聊天。

4.  **视频录制:**
    *   检测到运动或对象时录制视频。
    *   支持 10 秒缓冲。

5.  **测试功能:**
    *   系统启动时捕获测试帧和视频。

---

## **要求**

1.  **硬件:**
    *   处理器: Intel i7-2700K 或更高。
    *   内存: 8 GB 或更多。
    *   硬盘: 用于存储缓冲区和录制内容。

2.  **软件:**
    *   Python 3.8 或更高版本。
    *   FFmpeg。
    *   库: `opencv-python`, `ultralytics`, `aiohttp`, `numpy`, `yaml`。

---

## **安装**

1.  安装依赖项:
    ```bash
    pip install opencv-python ultralytics aiohttp numpy yaml
    ```

2.  安装 FFmpeg:
    *   Ubuntu:
        ```bash
        sudo apt install ffmpeg
        ```
    *   Windows: 从 [官方网站](https://ffmpeg.org/download.html) 下载。

3.  克隆仓库:
    ```bash
    git clone clone evollved/MotionWatch-CCTV_SYSTEM_TELEGRAM_BOT
    cd MotionWatch-CCTV_SYSTEM_TELEGRAM_BOT
    ```

4.  配置文件:
    *   `config/cameras.yaml` — 摄像头设置。
    *   `config/telegram.yaml` — Telegram 设置。

5.  启动系统:
    ```bash
    python main.py
    ```

---

## **配置**

### **1. 摄像头设置 (`config/cameras.yaml`)**

摄像头配置示例:

```yaml
cameras:
  - id: 1
    name: "阳台"
    rtsp_url: "rtsp://192.168.2.5:554/user=video_password=video_channel=0_stream=0&onvif=0.sdp?real_stream"
    width: 1920
    height: 1080
    fps: 15
    enabled: true
    detect_motion: true  # 启用运动检测
    detect_objects: true  # 启用对象检测
    object_confidence: 0.5  # 对象置信度阈值
    telegram_chat_id: "-1001234567890"  # Telegram 聊天 ID
    send_photo: true  # 发送照片到 Telegram
    send_video: true  # 发送视频到 Telegram
    draw_boxes: true  # 在对象周围绘制方框
    test_frame_on_start: true  # 启动时捕获测试帧
    test_video_on_start: true  # 启动时捕获测试视频
    motion_sensitivity: 0.2  # 运动检测灵敏度
    record_audio: false  # 录制音频
    reconnect_attempts: 5  # 重连尝试次数
    reconnect_delay: 10  # 重连延迟（秒）
    show_live_feed: false  # 显示实时视频流
    object_types: ["person", "cat", "dog"]  # 要检测的对象类型
```
2. Telegram 设置 (config/telegram.yaml)

Telegram 配置示例:
```yaml

bot_token: "123456789:ABCdefGhIJKlmNoPQRstuVWXyz"  # 机器人令牌

```

日志示例
    运动检测:
```
2025-03-07 18:07:27,096 - INFO - 摄像头 阳台: 在 左上角 区域检测到运动
2025-03-07 18:07:27,100 - INFO - 摄像头 阳台: 尝试向 Telegram 发送照片...
```
对象检测:
```
2025-03-07 18:07:27,105 - INFO - 摄像头 阳台: 检测到对象 'person', 置信度 0.85, 位置: 左上角
2025-03-07 18:07:27,110 - INFO - 摄像头 阳台: 检测到对象 'cat', 置信度 0.72, 位置: 右下角
```
错误:
```
2025-03-07 18:07:27,115 - ERROR - 摄像头 阳台: FFmpeg 捕获帧时出错: Connection refused
```
使用示例
1. 启动系统
```bash

python main.py
```
2. 测试帧

    启动时系统捕获测试帧并发送到 Telegram。

3. 测试视频

    启动时系统捕获测试视频（5 秒）并发送到 Telegram。

4. 运动时发送照片

    如果检测到运动，系统会向 Telegram 发送带有运动区域指示的照片。

5. 运动时发送视频

    如果运动被确认，系统会录制视频（15 秒）并发送到 Telegram。

## **常见问题 (FAQ)**
1. 如何添加新摄像头？

    在 config/cameras.yaml 中添加具有唯一 id 和摄像头设置的新条目。

2. 如何更改运动检测灵敏度？

    修改 config/cameras.yaml 中的 motion_sensitivity 参数。值越小，灵敏度越高。

3. 如何禁用对象检测？

    在摄像头设置中设置 detect_objects: false。

4. 如何更改要检测的对象类型？

    修改摄像头设置中的 object_types 列表。例如:
    yaml

object_types: ["person", "car"]

许可证

本项目基于 MIT 许可证。详情请见 LICENSE 文件。

## **作者**

[神经网络]

[Email 太知名，不宜公开]

[这里可以是你的广告 (:]
