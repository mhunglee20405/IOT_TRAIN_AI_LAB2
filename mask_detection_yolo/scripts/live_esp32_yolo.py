
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT / "runs/detect/results/train"
    / "antigravity_gpu_v1/weights/best.pt"
)

# Nạp ngưỡng calibrated từ configs/thresholds.json
CONFIG_PATH = ROOT / "configs" / "thresholds.json"
DEFAULT_CONF = 0.20
DEFAULT_ACCEPT = 0.75
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as _f_cfg:
            _th = json.load(_f_cfg)
            DEFAULT_CONF = float(_th.get("conf_floor", DEFAULT_CONF))
            DEFAULT_ACCEPT = float(_th.get("accept_thresh", DEFAULT_ACCEPT))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="ESP32-CAM Live YOLO11n Detection")
    parser.add_argument("--ip", required=True)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help="Candidate detection confidence floor")
    parser.add_argument("--accept", type=float, default=DEFAULT_ACCEPT,
                        help="Acceptance threshold (below this is unknown)")
    args = parser.parse_args()

    if not (0 < args.conf <= args.accept <= 1):
        parser.error("Can 0 < conf <= accept <= 1")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    model = YOLO(str(MODEL_PATH))
    session = requests.Session()

    url = f"http://{args.ip}/capture"

    print("Camera:", url)
    print("Model:", MODEL_PATH)
    print("Press Q to exit")

    try:
        while True:
            try:
                response = session.get(url, timeout=5)
                response.raise_for_status()

                array = np.frombuffer(
                    response.content, dtype=np.uint8
                )
                frame = cv2.imdecode(
                    array, cv2.IMREAD_COLOR
                )

                if frame is None:
                    raise ValueError("Invalid JPEG")

                result = model.predict(
                    source=frame,
                    imgsz=640,
                    conf=args.conf,
                    device=0,
                    verbose=False
                )[0]

                counts = {
                    "mask": 0,
                    "no_mask": 0,
                    "unknown": 0
                }

                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    predicted = result.names[cls_id]

                    if confidence < args.accept:
                        label = "unknown"
                        color = (128, 128, 128)
                    elif predicted == "mask":
                        label = "mask"
                        color = (0, 200, 0)
                    else:
                        label = "no_mask"
                        color = (0, 0, 255)

                    counts[label] += 1

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    cv2.rectangle(
                        frame, (x1, y1),
                        (x2, y2), color, 2
                    )

                    text = f"{label} {confidence:.2f}"

                    cv2.putText(
                        frame, text,
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2
                    )

                summary = (
                    f"Mask: {counts['mask']} | "
                    f"No mask: {counts['no_mask']} | "
                    f"Unknown: {counts['unknown']}"
                )

                cv2.rectangle(
                    frame, (0, 0),
                    (frame.shape[1], 35),
                    (25, 25, 25), -1
                )
                cv2.putText(
                    frame, summary, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2
                )

                cv2.imshow("ESP32-CAM YOLO11n", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except (
                requests.RequestException,
                ValueError
            ) as error:
                print("Capture error:", error)
                time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
