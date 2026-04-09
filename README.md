"""
FYP – AI Smart Farming
Manual mobile platform control with live YOLOv8 cabbage detection display
"""

import ssl
import urllib.request
import os
from pathlib import Path

# Fix SSL certificate verification error
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import time
import numpy as np
import serial
from ultralytics import YOLO
from pymycobot import MyCobot

try:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    pygame = None
    PYGAME_AVAILABLE = False

# GUI framework
from PyQt6 import QtWidgets, QtGui, QtCore

# ===== CONFIGURATION =====
CONF_THRESHOLD = 0.5
PS5_DEADZONE = 0.30
PS5_POLL_INTERVAL = 0.05
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DETECTION_DEVICE = "cpu"


def configure_camera(cam, name="camera"):
    """Apply stable camera settings to reduce auto zoom/cropping behavior."""
    if cam is None:
        return

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # These properties are backend/hardware dependent; ignored silently if unsupported.
    cam.set(cv2.CAP_PROP_ZOOM, 0)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    print(f"[INFO] {name} configured to {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

# ===== MOBILE PLATFORM CONFIGURATION =====
PLATFORM_SPEED = 50
TURN_SPEED = 30

SCAN_SPEED = 60

# ===== TRUE LOW-HEIGHT CABBAGE INSPECTION POSTURE =====
J2_DOWN    = -100
J3_FORWARD = 80
J4         = -20
J5         = -20
J6_CAMERA  = 50   # CAMERA LOOKS DOWN (KEY FIX)

# ===== ARDUINO PLATFORM CONTROL =====
ARDUINO_PORT = '/dev/cu.usbmodem12201'  # Adjust based on your Arduino port
ARDUINO_BAUD = 9600

try:
    arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize
    print("[ARDUINO] ✓ Connected to mobile platform")
    platform_connected = True
except Exception as e:
    print(f"[ARDUINO] ⚠ Mobile platform not connected: {e}")
    print("[ARDUINO] Running in simulation mode")
    platform_connected = False

# ===== ARDUINO PLATFORM CONTROL FUNCTIONS =====
def send_arduino_command(command):
    """Send command to Arduino"""
    if platform_connected:
        arduino.write(f"{command}\n".encode())
        time.sleep(0.1)  # Small delay for command processing
        return True
    else:
        print(f"[SIMULATION] Would send: {command}")
        return False

def move_forward(distance_cm):
    """Move platform forward by specified distance"""
    print(f"[PLATFORM] Moving forward {distance_cm}cm")
    send_arduino_command("moveForward")
    time.sleep(distance_cm / 10)  # Movement time based on distance
    send_arduino_command("stopMotors")

def move_backward(distance_cm):
    """Move platform backward by specified distance"""
    print(f"[PLATFORM] Moving backward {distance_cm}cm")
    send_arduino_command("moveBackward")
    time.sleep(distance_cm / 10)
    send_arduino_command("stopMotors")

def turn_left_90():
    """Turn platform 90 degrees left"""
    print("[PLATFORM] Turning left 90°")
    send_arduino_command("turnLeft90")

def turn_right_90():
    """Turn platform 90 degrees right"""
    print("[PLATFORM] Turning right 90°")
    send_arduino_command("turnRight90")

def stop_platform():
    """Stop all platform movement"""
    print("[PLATFORM] Stopping")
    send_arduino_command("stopMotors")

def follow_line_to_next_cabbage():
    """Use camera to follow line to next cabbage position"""
    print("[PLATFORM] Following line to next cabbage...")
    send_arduino_command("followLine")

# ===== CAMERA-BASED POSITIONING =====
def detect_line_center(frame):
    """Detect line center using camera for positioning"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply threshold to find line
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find largest contour (assumed to be the line)
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            # Calculate center of line
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy

    return None

def is_at_cabbage_position(frame):
    """Check if platform is correctly positioned at cabbage using visual cues"""
    # Look for cabbage in frame center
    height, width = frame.shape[:2]
    center_region = frame[height//2-50:height//2+50, width//2-50:width//2+50]

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)

    # Define range for green cabbage color
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([80, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Check if significant green area detected
    green_pixels = cv2.countNonZero(mask)
    total_pixels = center_region.shape[0] * center_region.shape[1]

    return green_pixels > total_pixels * 0.1  # 10% green indicates cabbage presence


def get_selected_result(frame, model_choice, conf_threshold):
    if model_choice == "nano":
        return model_nano.predict(source=frame, device=DETECTION_DEVICE, verbose=False)[0]
    if model_choice == "small":
        return model_small.predict(source=frame, device=DETECTION_DEVICE, verbose=False)[0]
    if model_choice == "medium":
        return model_medium.predict(source=frame, device=DETECTION_DEVICE, verbose=False)[0]

    results_nano = model_nano.predict(source=frame, device=DETECTION_DEVICE, verbose=False)[0]
    results_small = model_small.predict(source=frame, device=DETECTION_DEVICE, verbose=False)[0]
    results_medium = model_medium.predict(source=frame, device=DETECTION_DEVICE, verbose=False)[0]

    def top_conf(result):
        if getattr(result, "probs", None) is None:
            return 0.0
        return float(result.probs.top1conf.item() * 100)

    return max(
        [results_nano, results_small, results_medium],
        key=top_conf,
    )


def get_detection_summary(result):
    probs = getattr(result, "probs", None)
    if probs is None:
        return {
            "condition": "No prediction",
            "confidence": 0.0,
            "accuracy_text": "mAP: validation metric only",
            "detections": 0,
        }

    top1_idx = int(probs.top1)
    top1_conf = float(probs.top1conf.item() * 100)
    names = result.names if hasattr(result, "names") else {}
    condition = str(names.get(top1_idx, f"Class {top1_idx}")).replace("_", " ").title()

    if top1_conf < conf_threshold_runtime:
        condition = "Uncertain"

    return {
        "condition": condition,
        "confidence": top1_conf,
        "accuracy_text": "mAP: validation metric only",
        "detections": 1,
    }


def resolve_model_path(variant_key):
    """Prefer trained classification weights; fallback to pretrained cls weights."""
    trained = Path(f"/Users/hafiyimran/Desktop/cabbage_cls_runs/cabbage_cls_yolov8{variant_key}/weights/best.pt")
    if trained.exists():
        return str(trained)
    return f"yolov8{variant_key}-cls.pt"


print("[INFO] Loading YOLOv8 detection + classification models...")
print(f"[INFO] Inference device forced to: {DETECTION_DEVICE}")
detector_model = YOLO("yolov8n.pt")
print("[INFO] Detector model: yolov8n.pt")
model_nano_path = resolve_model_path("n")
model_small_path = resolve_model_path("s")
model_medium_path = resolve_model_path("m")
print(f"[INFO] Nano model: {model_nano_path}")
print(f"[INFO] Small model: {model_small_path}")
print(f"[INFO] Medium model: {model_medium_path}")

model_nano = YOLO(model_nano_path)
model_small = YOLO(model_small_path)
model_medium = YOLO(model_medium_path)

# Runtime threshold used to mark very low-confidence predictions as uncertain.
conf_threshold_runtime = CONF_THRESHOLD * 100

# Initialize platform camera for positioning
platform_cap = cv2.VideoCapture(1)  # Second camera for platform positioning
if not platform_cap.isOpened():
    print("[WARNING] Platform positioning camera not found, using main camera")
    platform_cap = None
else:
    configure_camera(platform_cap, "Platform camera")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Main camera not found")
    exit()
configure_camera(cap, "Main camera")

print("[INFO] Cameras started")

print("[INFO] Initializing myCobot 280...")
try:
    mc = MyCobot('/dev/ttyTHS1', 1000000)

    # Move to LOW-HEIGHT inspection posture
    mc.send_angles(
        [0, J2_DOWN, J3_FORWARD, J4, J5, J6_CAMERA],
        SCAN_SPEED
    )
    time.sleep(2)

    print("[INFO] Robot in TRUE low-height inspection posture")
    robot_connected = True
except Exception as e:
    print(f"[WARNING] MyCobot not connected: {e}")
    robot_connected = False

# ===== THREAD & GUI DEFINITIONS =====

class InspectionThread(QtCore.QThread):
    frame_signal = QtCore.pyqtSignal(np.ndarray)
    status_signal = QtCore.pyqtSignal(str)
    metrics_signal = QtCore.pyqtSignal(str, str, float, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.conf_threshold = CONF_THRESHOLD
        self.model_choice = "nano"  # options: nano, small, medium, multi
        self._running = False

    def set_conf_threshold(self, value):
        # slider gives 0-100
        global conf_threshold_runtime
        self.conf_threshold = value / 100.0
        conf_threshold_runtime = self.conf_threshold * 100

    def set_model_choice(self, choice):
        self.model_choice = choice

    def run(self):
        self._running = True
        self.status_signal.emit("Live detection started")
        while not self.isInterruptionRequested():
            try:
                ret, frame = cap.read()
                if not ret:
                    self.status_signal.emit("[ERROR] Failed to read camera frame")
                    time.sleep(0.1)
                    continue
                annotated = frame.copy()

                det_result = detector_model.predict(
                    source=frame,
                    device=DETECTION_DEVICE,
                    conf=self.conf_threshold,
                    verbose=False,
                )[0]

                boxes = det_result.boxes
                names = det_result.names if hasattr(det_result, "names") else {}
                best_summary = None

                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].int().tolist()
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        det_conf = float(boxes.conf[i].item() * 100)
                        det_cls = int(boxes.cls[i].item())
                        det_name = str(names.get(det_cls, f"class_{det_cls}"))

                        crop = frame[y1:y2, x1:x2]
                        cls_result = get_selected_result(crop, self.model_choice, self.conf_threshold)
                        summary = get_detection_summary(cls_result)

                        if best_summary is None or summary["confidence"] > best_summary["confidence"]:
                            best_summary = summary

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = (
                            f"{det_name} {det_conf:.1f}% | "
                            f"{summary['condition']} {summary['confidence']:.1f}%"
                        )
                        cv2.putText(
                            annotated,
                            label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
                else:
                    cv2.putText(
                        annotated,
                        "No object detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2,
                    )
                    best_summary = {
                        "condition": "No prediction",
                        "confidence": 0.0,
                        "accuracy_text": "mAP: validation metric only",
                    }

                summary = best_summary if best_summary else {
                    "condition": "No prediction",
                    "confidence": 0.0,
                    "accuracy_text": "mAP: validation metric only",
                }

                cv2.putText(
                    annotated,
                    f"Condition: {summary['condition']}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    annotated,
                    f"Confidence: {summary['confidence']:.1f}% | Model: {self.model_choice}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
                cv2.putText(
                    annotated,
                    f"{summary['accuracy_text']}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2
                )

                self.metrics_signal.emit(
                    summary['condition'],
                    summary['condition'],
                    summary['confidence'],
                    summary['accuracy_text'],
                )
                self.frame_signal.emit(annotated)
                time.sleep(0.02)
            except Exception as e:
                self.status_signal.emit(f"[ERROR] Detection failed: {e}")
                time.sleep(0.1)

        self.status_signal.emit("Live detection stopped")
        self._running = False


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FYP Smart Farming GUI")
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640,480)
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.start_btn = QtWidgets.QPushButton("Start Detection")
        self.stop_btn = QtWidgets.QPushButton("Stop Detection")
        self.ps5_start_btn = QtWidgets.QPushButton("Start PS5 Control")
        self.ps5_stop_btn = QtWidgets.QPushButton("Stop PS5 Control")
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1,100)
        self.conf_slider.setValue(int(CONF_THRESHOLD*100))
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["nano","small","medium","multi"])
        self.model_combo.setCurrentText("nano")
        self.status_label = QtWidgets.QLabel("Status: idle")
        self.condition_label = QtWidgets.QLabel("Health Condition: waiting")
        self.detected_class_label = QtWidgets.QLabel("Detected Class: waiting")
        self.confidence_label = QtWidgets.QLabel("Confidence: 0.0%")
        self.map_label = QtWidgets.QLabel("mAP: validation metric only")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_label)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.start_btn)
        hlayout.addWidget(self.stop_btn)
        layout.addLayout(hlayout)
        ps5_layout = QtWidgets.QHBoxLayout()
        ps5_layout.addWidget(self.ps5_start_btn)
        ps5_layout.addWidget(self.ps5_stop_btn)
        layout.addLayout(ps5_layout)
        layout.addWidget(QtWidgets.QLabel("Confidence threshold:"))
        layout.addWidget(self.conf_slider)
        layout.addWidget(QtWidgets.QLabel("Model:"))
        layout.addWidget(self.model_combo)
        layout.addWidget(self.condition_label)
        layout.addWidget(self.detected_class_label)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.map_label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.worker = InspectionThread()
        self.ps5_timer = QtCore.QTimer(self)
        self.ps5_timer.setInterval(int(PS5_POLL_INTERVAL * 1000))
        self.ps5_timer.timeout.connect(self.poll_ps5_controller)
        self.ps5_active = False
        self.ps5_joystick = None
        self.ps5_active_drive_cmd = "stopMotors"
        self.ps5_last_hat_x = 0
        self.ps5_last_x_button = 0

        self.worker.frame_signal.connect(self.update_frame)
        self.worker.status_signal.connect(self.update_status)
        self.worker.metrics_signal.connect(self.update_metrics)
        self.start_btn.clicked.connect(self.start_inspection)
        self.stop_btn.clicked.connect(self.stop_inspection)
        self.ps5_start_btn.clicked.connect(self.start_ps5_control)
        self.ps5_stop_btn.clicked.connect(self.stop_ps5_control)
        self.conf_slider.valueChanged.connect(self.worker.set_conf_threshold)
        self.model_combo.currentTextChanged.connect(self.worker.set_model_choice)

    def update_frame(self, frame):
        h,w,ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def update_status(self, text):
        self.status_label.setText(f"Status: {text}")

    def update_metrics(self, condition, detected_class, confidence, accuracy_text):
        self.condition_label.setText(f"Health Condition: {condition}")
        self.detected_class_label.setText(f"Detected Class: {detected_class}")
        self.confidence_label.setText(f"Confidence: {confidence:.1f}%")
        self.map_label.setText(accuracy_text)

    def start_inspection(self):
        if not self.worker.isRunning():
            self.worker.start()

    def stop_inspection(self):
        if self.worker.isRunning():
            self.worker.requestInterruption()

    def start_ps5_control(self):
        if not PYGAME_AVAILABLE:
            self.update_status("PS5 control unavailable: install pygame")
            return

        if self.ps5_active:
            self.update_status("PS5 control already running")
            return

        try:
            if not pygame.get_init():
                pygame.init()
            pygame.joystick.init()

            if pygame.joystick.get_count() < 1:
                self.update_status("No controller detected. Pair PS5 and retry.")
                return

            self.ps5_joystick = pygame.joystick.Joystick(0)
            self.ps5_joystick.init()
            self.ps5_active = True
            self.ps5_active_drive_cmd = "stopMotors"
            self.ps5_last_hat_x = 0
            self.ps5_last_x_button = 0
            self.ps5_timer.start()
            self.update_status(f"PS5 connected: {self.ps5_joystick.get_name()}")
        except Exception as e:
            self.update_status(f"PS5 init error: {e}")

    def stop_ps5_control(self):
        if not self.ps5_active:
            return

        self.ps5_timer.stop()
        self.ps5_active = False
        self.ps5_joystick = None
        send_arduino_command("stopMotors")
        try:
            if PYGAME_AVAILABLE:
                pygame.joystick.quit()
        except Exception:
            pass
        self.update_status("PS5 control stopped")

    def poll_ps5_controller(self):
        if not self.ps5_active or self.ps5_joystick is None:
            return

        try:
            pygame.event.pump()

            left_y = self.ps5_joystick.get_axis(1) if self.ps5_joystick.get_numaxes() > 1 else 0.0
            if left_y < -PS5_DEADZONE:
                desired_drive_cmd = "moveForward"
            elif left_y > PS5_DEADZONE:
                desired_drive_cmd = "moveBackward"
            else:
                desired_drive_cmd = "stopMotors"

            if desired_drive_cmd != self.ps5_active_drive_cmd:
                send_arduino_command(desired_drive_cmd)
                self.ps5_active_drive_cmd = desired_drive_cmd
                self.update_status(f"PS5 drive command: {desired_drive_cmd}")

            hat_x = 0
            if self.ps5_joystick.get_numhats() > 0:
                hat_x = self.ps5_joystick.get_hat(0)[0]

            if hat_x == -1 and self.ps5_last_hat_x != -1:
                send_arduino_command("turnLeft90")
                self.update_status("PS5 command: turnLeft90")
            elif hat_x == 1 and self.ps5_last_hat_x != 1:
                send_arduino_command("turnRight90")
                self.update_status("PS5 command: turnRight90")
            self.ps5_last_hat_x = hat_x

            x_pressed = self.ps5_joystick.get_button(0) if self.ps5_joystick.get_numbuttons() > 0 else 0
            if x_pressed and not self.ps5_last_x_button:
                send_arduino_command("stopMotors")
                self.ps5_active_drive_cmd = "stopMotors"
                self.update_status("PS5 emergency stop")
            self.ps5_last_x_button = x_pressed

        except Exception as e:
            self.update_status(f"PS5 polling error: {e}")
            self.stop_ps5_control()

    def closeEvent(self, event):
        self.stop_ps5_control()
        if self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(2000)
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
