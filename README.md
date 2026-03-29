# Cabbage-Detection
""This project develops an AI-based smart farming mobile robot for cabbage inspection. It integrates YOLOv8 image detection, a camera system, and an Arduino Mega 2560 controlling motors via Cytron MD10A Rev2.0. A Sony DualSense Wireless Controller enables real-time manual control, allowing flexible and efficient field monitoring.""

"""
FYP – AI Smart Farming
Phase 2: Mobile Platform + YOLOv8 Detection + Sequential Cabbage Inspection
Pattern: Column 1 (right): cabbages 1-7 top-to-bottom, Column 2 (left): 7-1 bottom-to-top
"""

import ssl
import urllib.request
import os

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
# Arduino motor control functions (to be implemented)
PLATFORM_SPEED = 50
TURN_SPEED = 30
CABBAGE_DISTANCE_CM = 20  # 20cm between cabbages vertically
COLUMN_DISTANCE_CM = 5    # 5cm between columns horizontally
CABBAGES_PER_COLUMN = 7

# ===== MYCOBOT SCAN CONFIGURATION =====
# Base scan angles (left → right)
SCAN_ANGLES = [-15, -7, 0, 7, 15]
SCAN_SPEED = 60

# ===== TRUE LOW-HEIGHT CABBAGE INSPECTION POSTURE =====
J2_DOWN    = -100
J3_FORWARD = 80
J4         = -20
J5         = -20
J6_CAMERA  = 50   # CAMERA LOOKS DOWN (KEY FIX)

# ===== DETECTION SEQUENCE =====
# Column 1 (right): cabbages 1-7 (top to bottom)
# Column 2 (left): cabbages 7-1 (bottom to top)
SEQUENCE = [
    ("Column 1", "Right", list(range(1, CABBAGES_PER_COLUMN + 1))),      # 1,2,3,4,5,6,7
    ("Column 2", "Left", list(range(CABBAGES_PER_COLUMN, 0, -1)))       # 7,6,5,4,3,2,1
]

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

print("[INFO] Loading YOLOv8 pretrained models...")
model_nano = YOLO("yolov8n.pt")   # Nano - fastest, smallest
model_small = YOLO("yolov8s.pt")  # Small - balanced speed/accuracy
model_medium = YOLO("yolov8m.pt") # Medium - higher accuracy, slower

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.conf_threshold = CONF_THRESHOLD
        self.model_choice = "multi"  # options: nano, small, medium, multi
        self._running = False

    def set_conf_threshold(self, value):
        # slider gives 0-100
        self.conf_threshold = value / 100.0

    def set_model_choice(self, choice):
        self.model_choice = choice

    def run(self):
        self._running = True
        self.status_signal.emit("Inspection thread started")
        for column_name, column_side, cabbage_sequence in SEQUENCE:
            if self.isInterruptionRequested():
                break
            self.status_signal.emit(f"Starting {column_name} ({column_side})")
            for cabbage_num in cabbage_sequence:
                if self.isInterruptionRequested():
                    break
                self.status_signal.emit(f"Processing {column_name} - Cabbage {cabbage_num}")
                self.status_signal.emit("Starting multi-angle scan")
                for base_angle in SCAN_ANGLES:
                    if self.isInterruptionRequested():
                        break
                    if robot_connected:
                        mc.send_angles(
                            [base_angle, J2_DOWN, J3_FORWARD, J4, J5, J6_CAMERA],
                            SCAN_SPEED
                        )
                        time.sleep(1.2)
                    else:
                        time.sleep(0.5)
                    ret, frame = cap.read()
                    if not ret:
                        self.status_signal.emit("[ERROR] Failed to read camera frame")
                        continue
                    # choose model
                    if self.model_choice == "nano":
                        results = model_nano(frame, conf=self.conf_threshold)
                        current = results[0]
                    elif self.model_choice == "small":
                        results = model_small(frame, conf=self.conf_threshold)
                        current = results[0]
                    elif self.model_choice == "medium":
                        results = model_medium(frame, conf=self.conf_threshold)
                        current = results[0]
                    else:  # multi
                        rn = model_nano(frame, conf=self.conf_threshold)
                        rs = model_small(frame, conf=self.conf_threshold)
                        rm = model_medium(frame, conf=self.conf_threshold)
                        all_results = [rn[0], rs[0], rm[0]]
                        current = max(all_results, key=lambda r: max(r.boxes.conf.cpu().numpy()) if len(r.boxes) > 0 else 0)
                    annotated = current.plot()
                    # metrics
                    detections = current.boxes
                    num_detections = len(detections)
                    if num_detections > 0:
                        confidences = detections.conf.cpu().numpy()
                        avg_confidence = np.mean(confidences) * 100
                        min_confidence = np.min(confidences) * 100
                        max_confidence = np.max(confidences) * 100
                    else:
                        avg_confidence = min_confidence = max_confidence = 0
                    # overlay texts
                    cv2.putText(
                        annotated,
                        f"{column_name} | Cabbage {cabbage_num} | Base: {base_angle}°",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        annotated,
                        f"Phase 2: Sequential Inspection | {column_side} Column",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )
                    cv2.putText(
                        annotated,
                        f"Detections: {num_detections} | Avg Conf: {avg_confidence:.1f}% | Model:{self.model_choice}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2
                    )
                    cv2.putText(
                        annotated,
                        f"Min:{min_confidence:.1f}% Max:{max_confidence:.1f}%",
                        (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2
                    )
                    self.frame_signal.emit(annotated)
                    time.sleep(0.02)
                self.status_signal.emit(f"Completed Cabbage {cabbage_num}")
            self.status_signal.emit(f"Completed {column_name}")
        self.status_signal.emit("Inspection sequence completed")
        stop_platform()
        cap.release()
        if platform_cap:
            platform_cap.release()
        cv2.destroyAllWindows()
        if robot_connected:
            mc.send_angles([0, 0, 0, 0, 0, 0], 40)
        self._running = False


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FYP Smart Farming GUI")
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640,480)
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.ps5_start_btn = QtWidgets.QPushButton("Start PS5 Control")
        self.ps5_stop_btn = QtWidgets.QPushButton("Stop PS5 Control")
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1,100)
        self.conf_slider.setValue(int(CONF_THRESHOLD*100))
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["nano","small","medium","multi"])
        self.status_label = QtWidgets.QLabel("Status: idle")

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

    def start_inspection(self):
        if not self.worker.isRunning():
            self.worker.start()

    def stop_inspection(self):
        if self.worker.isRunning():
            self.worker.requestInterruption()

    def start_ps5_control(self):
        if self.worker.isRunning():
            self.update_status("Stop inspection before starting PS5 control")
            return

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
