# app.py
import sys
import sqlite3
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
import json
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtCore import QUrl, QSettings

import threading
import uvicorn
from api.server import api_app

# ---------------- FastAPI background server ----------------
def run_api():
    uvicorn.run(api_app, host="0.0.0.0", port=8000)

threading.Thread(target=run_api, daemon=True).start()

# ---------------- Minimal light theme (QSS) ----------------
APP_QSS = """
* { font-family: 'Segoe UI', 'Inter', Arial; font-size: 13px; }
QMainWindow { background: #f6f7fb; }
QGroupBox {
  background: #ffffff; border: 1px solid #e6e8ef; border-radius: 12px; margin-top: 10px; padding: 12px;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #171a21; font-weight: 600; }
QLabel[role="muted"] { color: #6b7280; }
QPushButton {
  min-height: 36px; padding: 0 14px; border-radius: 10px; border: 1px solid #e6e8ef; background: #ffffff; color: #111827;
}
QPushButton:hover { background: #f3f4f6; }
QPushButton:pressed { background: #e5e7eb; }
QPushButton[variant="primary"] { background: #2563eb; color: #ffffff; border: 1px solid #1e40af; }
QPushButton[variant="primary"]:hover { background: #1d4ed8; }
QPushButton[variant="danger"] { background: #ef4444; color: #ffffff; border: 1px solid #b91c1c; }
QPushButton[variant="danger"]:hover { background: #dc2626; }
#VideoFrame { background: #101316; border-radius: 12px; }
QHeaderView::section {
  background: #111827; color: #ffffff; font-weight: 600; border: none; padding: 6px 8px;
}
QTableWidget {
  background: #ffffff; border: 1px solid #e6e8ef; border-radius: 12px; gridline-color: #eef0f5;
}
QTableWidget::item { padding: 6px 8px; }
QTableWidget::item:selected { background: #e5edff; }
QStatusBar { background: #ffffff; border-top: 1px solid #e6e8ef; }
"""

# ---------------- Paths / Sounds ----------------
DB_PATH = Path(__file__).resolve().parent / "orders.db"
SOUNDS_DIR = Path("sounds")
SOUND_FILES = {
    "thumbs_up": SOUNDS_DIR / "confirm.wav",
    "open_palm": SOUNDS_DIR / "cancel.wav",
    "point": SOUNDS_DIR / "navigate.wav",
    "prev": SOUNDS_DIR / "navigate.wav",
}

def ensure_sounds():
    SOUNDS_DIR.mkdir(exist_ok=True)
    tones = {
        "thumbs_up": (880, 0.10),
        "open_palm": (440, 0.12),
        "point": (660, 0.08),
        "prev": (660, 0.08),
    }
    framerate = 44100
    for name, (freq, dur) in tones.items():
        path = SOUND_FILES[name]
        if not path.exists():
            samples = np.sin(2 * np.pi * np.arange(int(framerate * dur)) * freq / framerate)
            samples = (samples * 0.5 * (2**15 - 1)).astype(np.int16)
            with wave.open(str(path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(framerate)
                wf.writeframes(samples.tobytes())

# ---------------- SQLite helpers ----------------
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_number TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL,
        items TEXT,
        total REAL DEFAULT 0.0
    )
    """)
    conn.commit()

    c.execute("SELECT COUNT(*) FROM orders")
    if c.fetchone()[0] == 0:
        now = datetime.utcnow().isoformat()
        demos = [
            ("1001", "pending", now, json.dumps([
                {"name": "Burger", "qty": 2},
                {"name": "Fries", "qty": 1}
            ]), 120.0),
            ("1002", "pending", now, json.dumps([
                {"name": "Hotdog", "qty": 1}
            ]), 45.0),
            ("1003", "pending", now, json.dumps([
                {"name": "Pasta", "qty": 1},
                {"name": "Juice", "qty": 1}
            ]), 90.0),
            ("1004", "pending", now, json.dumps([
                {"name": "Coffee", "qty": 1}
            ]), 30.0),
        ]
        c.executemany("""
            INSERT INTO orders (order_number, status, created_at, items, total)
            VALUES (?, ?, ?, ?, ?)
        """, demos)
        conn.commit()
    conn.close()

def load_orders():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, order_number, status, items, qty, total FROM orders ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_order_status(order_id, path=DB_PATH):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("SELECT status FROM orders WHERE id = ?", (order_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def update_order_status(order_id, new_status, path=DB_PATH):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("UPDATE orders SET status = ? WHERE id = ?", (new_status, order_id))
    conn.commit()
    conn.close()

def get_order_by_id(order_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, order_number, status, items, total FROM orders WHERE id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    cols = ["id", "order_number", "status", "items", "total"]
    data = dict(zip(cols, row))

    try:
        data["items"] = json.loads(data["items"])
    except Exception:
        data["items"] = []
    return data

# ---------------- Camera Thread (robust) ----------------
class CameraThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    gesture_detected = QtCore.pyqtSignal(str)

    def __init__(self, camera_index=0, parent=None, fps=20):
        super().__init__(parent)
        self.camera_index = camera_index
        self.running = False
        self.fps = fps
        # debounce
        self.last_gesture = None
        self.same_count = 0
        self.FRAMES_REQUIRED = 12

    def run(self):
        self.running = True
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                return

            prev = 0
            # Create MediaPipe within the thread scope
            with mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            ) as hands:
                while self.running:
                    time_elapsed = time.time() - prev
                    if time_elapsed < 1 / self.fps:
                        time.sleep(max(0, 1/self.fps - time_elapsed))
                    prev = time.time()

                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    gesture = None
                    annotated = frame.copy()

                    if results.multi_hand_landmarks:
                        hand = results.multi_hand_landmarks[0]
                        mp.solutions.drawing_utils.draw_landmarks(
                            annotated, hand, mp.solutions.hands.HAND_CONNECTIONS
                        )
                        lm = [(l.x, l.y) for l in hand.landmark]

                        tip_ids = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
                        pip_ids = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

                        def is_finger_extended(f):
                            tip, pip = tip_ids[f], pip_ids[f]
                            return lm[tip][1] < lm[pip][1]

                        def is_thumb_up():
                            wrist = np.array(lm[0])
                            thumb_mcp = np.array(lm[2])
                            thumb_ip = np.array(lm[3])
                            thumb_tip = np.array(lm[4])
                            pinky_tip = np.array(lm[20])
                            dist_mcp_ip = np.linalg.norm(thumb_mcp - thumb_ip)
                            dist_ip_tip = np.linalg.norm(thumb_ip - thumb_tip)
                            dist_mcp_tip = np.linalg.norm(thumb_mcp - thumb_tip)
                            thumb_straight = dist_mcp_tip > 0.7 * (dist_mcp_ip + dist_ip_tip)
                            thumb_upward = thumb_tip[1] < wrist[1] - 0.05
                            thumb_not_near_pinky = np.linalg.norm(thumb_tip - pinky_tip) > 0.15
                            return thumb_straight and thumb_upward and thumb_not_near_pinky

                        def is_thumb_open_palm():
                            return lm[4][0] - lm[2][0] > -0.05

                        idx_ext = is_finger_extended("index")
                        mid_ext = is_finger_extended("middle")
                        ring_ext = is_finger_extended("ring")
                        pinky_ext = is_finger_extended("pinky")
                        thumb_up = is_thumb_up()
                        thumb_open = is_thumb_open_palm()

                        if thumb_up and not any([idx_ext, mid_ext, ring_ext, pinky_ext]):
                            gesture = "thumbs_up"
                            cv2.putText(annotated, "Thumbs Up", (10,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                        elif thumb_open and all([idx_ext, mid_ext, ring_ext, pinky_ext]):
                            gesture = "open_palm"
                            cv2.putText(annotated, "Open Palm", (10,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
                        elif idx_ext and not (mid_ext or ring_ext or pinky_ext):
                            gesture = "point"
                            cv2.putText(annotated, "Pointing", (10,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
                        elif idx_ext and mid_ext and not (ring_ext or pinky_ext):
                            gesture = "prev"
                            cv2.putText(annotated, "Peace Sign", (10,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,0), 2)

                    # Debounce
                    if gesture == self.last_gesture and gesture is not None:
                        self.same_count += 1
                    else:
                        self.same_count = 1 if gesture is not None else 0
                        self.last_gesture = gesture

                    if gesture and self.same_count >= self.FRAMES_REQUIRED:
                        self.gesture_detected.emit(gesture)
                        self.last_gesture = None
                        self.same_count = 0

                    # Send frame
                    h, w, ch = annotated.shape
                    qimg = QtGui.QImage(annotated.data, w, h, ch * w, QtGui.QImage.Format.Format_BGR888)
                    self.frame_ready.emit(qimg)
        finally:
            if cap is not None:
                cap.release()

    def stop(self):
        self.running = False
        self.wait(1000)

# ---------------- Main Window ----------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Order Manager (PyQt6)")
        self.resize(1200, 720)
        ensure_sounds()

        # App stylesheet
        self.setStyleSheet(APP_QSS)

        # sounds
        self.sounds = {}
        for k, p in SOUND_FILES.items():
            se = QSoundEffect()
            se.setSource(QUrl.fromLocalFile(str(p)))
            se.setVolume(0.5)
            self.sounds[k] = se

        # central widget
        main = QtWidgets.QWidget()
        self.setCentralWidget(main)

        # ===== Left Panel =====
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(15)

        # camera badge
        cam_bar = QtWidgets.QHBoxLayout()
        self.lbl_camera_badge = QtWidgets.QLabel("Camera: selecting…")
        self.lbl_camera_badge.setProperty("role", "muted")
        self.lbl_camera_badge.setStyleSheet("color:#666; font-size:12px;")
        cam_bar.addWidget(self.lbl_camera_badge)
        cam_bar.addStretch(1)
        left.addLayout(cam_bar)

        # video
        self.video_label = QtWidgets.QLabel()
        self.video_label.setObjectName("VideoFrame")
        self.video_label.setFixedSize(640, 480)
        left.addWidget(self.video_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # helper hint
        hint = QtWidgets.QLabel("Use gestures or the command bar.", parent=self)
        hint.setProperty("role", "muted")
        left.addWidget(hint)

        # Command Bar (compact)
        cmd = QtWidgets.QHBoxLayout()
        cmd.setSpacing(8)

        self.btn_prev   = QtWidgets.QPushButton("✌️ Prev")
        self.btn_next   = QtWidgets.QPushButton("👉 Next")
        self.btn_cancel = QtWidgets.QPushButton("✋ Cancel")
        self.btn_complete = QtWidgets.QPushButton("👍 Complete")

        self.btn_complete.setProperty("variant", "primary")
        self.btn_cancel.setProperty("variant", "danger")

        for b in (self.btn_prev, self.btn_next, self.btn_cancel, self.btn_complete):
            b.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            cmd.addWidget(b)
        left.addLayout(cmd)

        # ===== Right Panel =====
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(15)

        # Order Details card
        self.order_card = QtWidgets.QGroupBox("Order Details")
        self.order_card.setStyleSheet("""
            QGroupBox { background:#ffffff; border:1px solid #e6e8ef; border-radius:12px; font-weight:bold; padding:12px; }
            QLabel { font-size:13px; }
        """)
        card = QtWidgets.QFormLayout()
        card.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        card.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        card.setVerticalSpacing(8)

        self.lbl_order_number = QtWidgets.QLabel("-")
        self.lbl_order_status = QtWidgets.QLabel("-")
        self.lbl_items = QtWidgets.QLabel("-")
        self.lbl_qty = QtWidgets.QLabel("-")
        self.lbl_total = QtWidgets.QLabel("-")

        # bold important info
        self.lbl_order_number.setStyleSheet("font-weight:700;")
        self.lbl_order_status.setStyleSheet("font-weight:700;")

        card.addRow("Order #:", self.lbl_order_number)
        card.addRow("Status:", self.lbl_order_status)
        card.addRow("Items:", self.lbl_items)
        card.addRow("Quantity:", self.lbl_qty)
        card.addRow("Total:", self.lbl_total)
        self.order_card.setLayout(card)
        right.addWidget(self.order_card)

        # Orders table
       # ---- Orders Table ----
        right.addWidget(QtWidgets.QLabel("<b>All Orders</b>"))

        self.order_table = QtWidgets.QTableWidget()
        self.order_table.setColumnCount(2)
        self.order_table.setHorizontalHeaderLabels(["Order #", "Status"])

        # Per-column sizing: Order # stretches, Status fits its text
        header = self.order_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)            # Order #
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)  # Status

        # Look & behavior
        self.order_table.setStyleSheet("""
            QHeaderView::section { background:#111827; color:white; font-weight:bold; padding:6px 8px; }
            QTableWidget { background:white; border:1px solid #e6e8ef; border-radius:12px; }
        """)
        self.order_table.setAlternatingRowColors(True)
        self.order_table.setShowGrid(False)
        self.order_table.verticalHeader().setVisible(False)
        self.order_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.order_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.order_table.setEditTriggers(QtWidgets.QTableWidget.EditTrigger.NoEditTriggers)
        right.addWidget(self.order_table, 1)


        # ===== Build splitter (after left & right are defined) =====
        left_container = QtWidgets.QWidget()
        left_container.setLayout(left)
        right_container = QtWidgets.QWidget()
        right_container.setLayout(right)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(6)
        split.addWidget(left_container)
        split.addWidget(right_container)
        split.setSizes([700, 500])

        outer = QtWidgets.QVBoxLayout(main)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)
        outer.addWidget(split, 1)

        # ===== Connect Buttons =====
        self.btn_complete.clicked.connect(lambda: self.handle_action("thumbs_up"))
        self.btn_cancel.clicked.connect(lambda: self.handle_action("open_palm"))
        self.btn_next.clicked.connect(lambda: self.handle_action("point"))
        self.btn_prev.clicked.connect(lambda: self.handle_action("prev"))

        # Shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=lambda: self.handle_action("thumbs_up"))
        QtGui.QShortcut(QtGui.QKeySequence("N"), self, activated=lambda: self.handle_action("point"))
        QtGui.QShortcut(QtGui.QKeySequence("C"), self, activated=lambda: self.handle_action("open_palm"))

        # Status bar
        self.setStatusBar(QtWidgets.QStatusBar())
        self.statusBar().showMessage("Ready")

        # Camera menu + persistence
        self.settings = QSettings("BurgerFlow", "GestureOrderManager")
        self.available_cameras = []
        self.build_camera_menu()
        self.refresh_camera_list()

        remembered_idx = self.settings.value("camera_index", None, type=int)
        start_idx = None
        if remembered_idx is not None and any(ci == remembered_idx for ci, _ in self.available_cameras):
            start_idx = remembered_idx
        elif self.available_cameras:
            start_idx = self.available_cameras[0][0]
        if start_idx is None:
            start_idx = 0

        self.start_camera_thread(start_idx)
        self.mark_selected_camera_in_menu(start_idx)

        # DB + timers
        init_db()
        self.reload_orders()
        self.order_table.cellClicked.connect(self.on_order_selected)
        self.poll_timer = QtCore.QTimer()
        self.poll_timer.timeout.connect(lambda: self.reload_orders(preserve_selection=True))
        self.poll_timer.start(3000)

    # ---------- Camera menu ----------
    def build_camera_menu(self):
        menubar = self.menuBar()
        self.camera_menu = menubar.addMenu("Camera")
        self.select_camera_menu = self.camera_menu.addMenu("Select Camera")

        act_refresh = QtGui.QAction("Refresh List", self)
        act_refresh.triggered.connect(self.refresh_camera_list)
        self.camera_menu.addAction(act_refresh)

        self.camera_menu.addSeparator()
        self.current_cam_label = QtGui.QAction("Current: (none)", self)
        self.current_cam_label.setEnabled(False)
        self.camera_menu.addAction(self.current_cam_label)

    def enumerate_cameras(self, max_devices=10):
        cams = []
        for i in range(max_devices):
            cap = cv2.VideoCapture(i)
            ok = cap.isOpened()
            if ok:
                ok, _ = cap.read()
            cap.release()
            if ok:
                cams.append((i, f"Camera {i}"))
        if not cams:
            cams.append((0, "Camera 0 (fallback)"))
        return cams

    def populate_camera_select_menu(self):
        self.select_camera_menu.clear()
        self.camera_action_group = QtGui.QActionGroup(self)
        self.camera_action_group.setExclusive(True)
        for idx, label in self.available_cameras:
            act = QtGui.QAction(label, self, checkable=True)
            act.setData(idx)
            act.triggered.connect(lambda checked, cam_idx=idx: self.switch_camera(cam_idx))
            self.camera_action_group.addAction(act)
            self.select_camera_menu.addAction(act)

    def refresh_camera_list(self, max_devices=10):
        self.available_cameras = self.enumerate_cameras(max_devices)
        self.populate_camera_select_menu()

    def mark_selected_camera_in_menu(self, camera_index):
        for act in self.camera_action_group.actions():
            act.setChecked(act.data() == camera_index)
        self.current_cam_label.setText(f"Current: Camera {camera_index}")
        if hasattr(self, "lbl_camera_badge"):
            self.lbl_camera_badge.setText(f"Camera: {camera_index}")

    def _validate_camera(self, index):
        cap = cv2.VideoCapture(index)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        cap.release()
        return bool(ok)

    def start_camera_thread(self, camera_index):
        # stop & delete existing
        if hasattr(self, "cam_thread") and self.cam_thread is not None:
            if self.cam_thread.isRunning():
                self.cam_thread.stop()
            try:
                self.cam_thread.frame_ready.disconnect(self.update_frame)
                self.cam_thread.gesture_detected.disconnect(self.on_gesture)
            except Exception:
                pass
            self.cam_thread.deleteLater()
            self.cam_thread = None
            QtWidgets.QApplication.processEvents()

        # start new
        self.cam_thread = CameraThread(camera_index=camera_index)
        self.cam_thread.frame_ready.connect(self.update_frame)
        self.cam_thread.gesture_detected.connect(self.on_gesture)
        self.cam_thread.finished.connect(self.cam_thread.deleteLater)
        self.cam_thread.start()

    def switch_camera(self, camera_index):
        if not self._validate_camera(camera_index):
            QtWidgets.QMessageBox.warning(
                self, "Camera Error",
                f"Could not open Camera {camera_index}. It may be in use or unavailable."
            )
            if hasattr(self, "cam_thread") and self.cam_thread is not None:
                current_idx = getattr(self.cam_thread, "camera_index", 0)
                self.mark_selected_camera_in_menu(current_idx)
            return

        self.start_camera_thread(camera_index)
        self.mark_selected_camera_in_menu(camera_index)
        self.settings.setValue("camera_index", int(camera_index))
        self.statusBar().showMessage(f"Switched to Camera {camera_index}")

    # ---------- App helpers ----------
    def closeEvent(self, event):
        if hasattr(self, "cam_thread") and self.cam_thread is not None and self.cam_thread.isRunning():
            self.cam_thread.stop()
        self.cam_thread = None
        event.accept()

    def update_frame(self, qimg):
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )
        self.video_label.setPixmap(pix)

    def reload_orders(self, preserve_selection=True):
        selected_id = None
        if preserve_selection:
            sel = self.get_selected_order()
            if sel:
                selected_id = sel

        self.order_table.setRowCount(0)
        rows = load_orders()

        for r, row in enumerate(rows):
            id_ = row[0] if len(row) > 0 else None
            number = row[1] if len(row) > 1 else "-"
            status = row[2] if len(row) > 2 else "pending"

            self.order_table.insertRow(r)
            item_number = QtWidgets.QTableWidgetItem(str(number))
            item_number.setData(QtCore.Qt.ItemDataRole.UserRole, id_)
            status_item = QtWidgets.QTableWidgetItem(status)
            if status == "Completed":
                status_item.setForeground(QtGui.QColor("#16a34a"))
            elif status == "Cancelled":
                status_item.setForeground(QtGui.QColor("#ef4444"))
            else:
                status_item.setForeground(QtGui.QColor("#111827"))
            item_number.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
            status_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignHCenter)


            self.order_table.setItem(r, 0, item_number)
            self.order_table.setItem(r, 1, status_item)

        if preserve_selection and selected_id:
            for i in range(self.order_table.rowCount()):
                item = self.order_table.item(i, 0)
                if item and item.data(QtCore.Qt.ItemDataRole.UserRole) == selected_id:
                    self.order_table.selectRow(i)
                    self.on_order_selected(i, 0)
                    break
        elif self.order_table.rowCount() > 0:
            self.order_table.selectRow(0)
            self.on_order_selected(0, 0)

    def get_selected_order(self):
        row = self.order_table.currentRow()
        if row < 0:
            return None
        item = self.order_table.item(row, 0)
        if not item:
            return None
        return item.data(QtCore.Qt.ItemDataRole.UserRole)

    def on_order_selected(self, row, column):
        order_id = self.order_table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
        order = get_order_by_id(order_id)
        if not order:
            return
        self.lbl_order_number.setText(str(order.get("order_number", "-")))
        self.lbl_order_status.setText(order.get("status", "-"))
        items = order.get("items", [])
        if isinstance(items, str):
            import json
            try:
                items = json.loads(items)
            except Exception:
                items = []

        if not items:
            self.lbl_items.setText("-")
        else:
            lines = [f"{it['name']} — {it['qty']}x" for it in items if 'name' in it]
            self.lbl_items.setText("\n".join(lines))

        total_qty = sum(it.get("qty", 0) for it in items)
        self.lbl_qty.setText(str(total_qty))

        self.lbl_total.setText(f"{order.get('total', 0):.2f}")

        # Color status in details card
        status = order.get("status", "-")
        color = "#16a34a" if status == "Completed" else ("#ef4444" if status == "Cancelled" else "#111827")
        self.lbl_order_status.setStyleSheet(f"font-weight:700; color:{color};")

    def handle_action(self, gesture):
        order_id = self.get_selected_order()
        if not order_id:
            self.statusBar().showMessage("No order selected")
            return

        prior_status = get_order_status(order_id)
        action_changed = False

        if gesture == "thumbs_up":
            if prior_status != "Completed":
                update_order_status(order_id, "Completed")
                self.statusBar().showMessage(f"Order {order_id} Completed")
                action_changed = True

        elif gesture == "open_palm":
            if prior_status != "Cancelled":
                update_order_status(order_id, "Cancelled")
                self.statusBar().showMessage(f"Order {order_id} Cancelled")
                action_changed = True

        elif gesture == "point":
            row = self.order_table.currentRow()
            if row < self.order_table.rowCount() - 1:
                self.order_table.selectRow(row + 1)
                self.on_order_selected(row + 1, 0)
                self.statusBar().showMessage("Moved to next order")
                action_changed = True

        elif gesture == "prev":
            row = self.order_table.currentRow()
            if row > 0:
                self.order_table.selectRow(row - 1)
                self.on_order_selected(row - 1, 0)
                self.statusBar().showMessage("Moved to previous order")
                action_changed = True

        self.reload_orders(preserve_selection=True)

        if action_changed and gesture in self.sounds:
            self.sounds[gesture].stop()
            self.sounds[gesture].play()

    @QtCore.pyqtSlot(str)
    def on_gesture(self, gesture):
        self.statusBar().showMessage(f"Gesture detected: {gesture}")
        QtCore.QTimer.singleShot(1200, lambda: self.statusBar().showMessage("Ready"))
        self.handle_action(gesture)

# ---------------- entry ----------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(APP_QSS)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
