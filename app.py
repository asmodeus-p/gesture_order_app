# app.py
import sys
import sqlite3
import time
import wave
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtCore import QUrl

import threading
import uvicorn
from api.server import api_app

# Start FastAPI in the background
def run_api():
    uvicorn.run(api_app, host="0.0.0.0", port=8000)

threading.Thread(target=run_api, daemon=True).start()

# ---------- Paths ----------
DB_PATH = Path(__file__).resolve().parent / "orders.db"
SOUNDS_DIR = Path("sounds")
SOUND_FILES = {
    "thumbs_up": SOUNDS_DIR / "confirm.wav",
    "open_palm": SOUNDS_DIR / "cancel.wav",
    "point": SOUNDS_DIR / "navigate.wav",
    "prev": SOUNDS_DIR / "navigate.wav",
}

# ---------- Utility: generate simple beep WAVs if missing ----------
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

# ---------- SQLite Helpers ----------
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_number TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL
    )
    """)

    # --- Safe migration for new columns ---
    c.execute("PRAGMA table_info(orders);")
    cols = [r[1] for r in c.fetchall()]
    if "items" not in cols:
        c.execute("ALTER TABLE orders ADD COLUMN items TEXT;")
    if "qty" not in cols:
        c.execute("ALTER TABLE orders ADD COLUMN qty INTEGER DEFAULT 0;")
    if "total" not in cols:
        c.execute("ALTER TABLE orders ADD COLUMN total REAL DEFAULT 0.0;")

    conn.commit()

    # --- Seed demo data if empty ---
    c.execute("SELECT COUNT(*) FROM orders")
    if c.fetchone()[0] == 0:
        now = datetime.utcnow().isoformat()
        demos = [
            ("1001", "pending", now, "Burger, Fries", 2, 120.0),
            ("1002", "pending", now, "Hotdog", 1, 45.0),
            ("1003", "pending", now, "Pasta, Juice", 2, 90.0),
            ("1004", "pending", now, "Coffee", 1, 30.0),
        ]
        c.executemany("""
            INSERT INTO orders (order_number, status, created_at, items, qty, total)
            VALUES (?, ?, ?, ?, ?, ?)
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
    cursor.execute("SELECT id, order_number, status, items, qty, total FROM orders WHERE id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    cols = ["id", "order_number", "status", "items", "qty", "total"]
    return dict(zip(cols, row))




# ---------- Gesture detection thread (restored interface) ----------
import cv2
import time
import numpy as np
from PyQt6 import QtCore, QtGui
import mediapipe as mp

class CameraThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    gesture_detected = QtCore.pyqtSignal(str)

    def __init__(self, camera_index=0, parent=None, fps=20):
        super().__init__(parent)
        self.camera_index = camera_index
        self.running = False
        self.fps = fps

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # debounce state
        self.last_gesture = None
        self.same_count = 0
        self.FRAMES_REQUIRED = 12  # consecutive frames for confirmation

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_index)
        prev = 0
        while self.running:
            time_elapsed = time.time() - prev
            if time_elapsed < 1 / self.fps:
                time.sleep(max(0, 1/self.fps - time_elapsed))
            prev = time.time()

            ret, frame = cap.read()
            if not ret:
                continue

            # mirror-like behavior
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            gesture = None
            annotated = frame.copy()

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated, hand, mp.solutions.hands.HAND_CONNECTIONS
                )

                # normalized landmarks
                lm = [(l.x, l.y) for l in hand.landmark]

                # fingertip and pip indices
                tip_ids = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
                pip_ids = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

                # check non-thumb finger extended (strict)
                def is_finger_extended(finger):
                    tip = tip_ids[finger]
                    pip = pip_ids[finger]
                    return lm[tip][1] < lm[pip][1]  # tip above pip

                # improved thumb-up detection
                def is_thumb_up():
                    wrist = np.array(lm[0])
                    thumb_mcp = np.array(lm[2])
                    thumb_ip = np.array(lm[3])
                    thumb_tip = np.array(lm[4])
                    pinky_tip = np.array(lm[20])   # pinky fingertip landmark

                    # Distances between thumb joints
                    dist_mcp_ip = np.linalg.norm(thumb_mcp - thumb_ip)
                    dist_ip_tip = np.linalg.norm(thumb_ip - thumb_tip)
                    dist_mcp_tip = np.linalg.norm(thumb_mcp - thumb_tip)

                    # Check if thumb is straight (not curled)
                    thumb_straight = dist_mcp_tip > 0.7 * (dist_mcp_ip + dist_ip_tip)

                    # Check thumb upward direction
                    thumb_upward = thumb_tip[1] < wrist[1] - 0.05

                    # New check: thumb should NOT be close to pinky tip (to avoid curled thumb)
                    dist_thumb_pinky = np.linalg.norm(thumb_tip - pinky_tip)
                    thumb_not_near_pinky = dist_thumb_pinky > 0.15  # threshold (tune if needed)

                    return thumb_straight and thumb_upward and thumb_not_near_pinky

                # thumb relaxed for open palm
                def is_thumb_open_palm():
                    return lm[4][0] - lm[2][0] > -0.05  # allow outward angle

                # check fingers
                idx_ext = is_finger_extended("index")
                mid_ext = is_finger_extended("middle")
                ring_ext = is_finger_extended("ring")
                pinky_ext = is_finger_extended("pinky")
                thumb_up = is_thumb_up()
                thumb_open = is_thumb_open_palm()

                # --- Gesture detection ---
                # 1) Thumbs Up (thumb extended, others closed)
                if thumb_up and not any([idx_ext, mid_ext, ring_ext, pinky_ext]):
                    gesture = "thumbs_up"
                    cv2.putText(annotated, "Thumbs Up", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

                # 2) Open Palm (all extended)
                elif thumb_open and all([idx_ext, mid_ext, ring_ext, pinky_ext]):
                    gesture = "open_palm"
                    cv2.putText(annotated, "Open Palm", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

                # 3) Point (only index extended)
                elif idx_ext and not (mid_ext or ring_ext or pinky_ext):
                    gesture = "point"
                    cv2.putText(annotated, "Pointing", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
                    
                # 4) Peace Sign (index + middle)
                elif idx_ext and mid_ext and not (ring_ext or pinky_ext):
                    gesture = "prev"
                    cv2.putText(annotated, "Peace Sign", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,0), 2)

            # --- Debounce ---
            if gesture == self.last_gesture and gesture is not None:
                self.same_count += 1
            else:
                self.same_count = 1 if gesture is not None else 0
                self.last_gesture = gesture

            if gesture and self.same_count >= self.FRAMES_REQUIRED:
                self.gesture_detected.emit(gesture)
                self.last_gesture = None
                self.same_count = 0

            # --- Convert frame to QImage ---
            h, w, ch = annotated.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(annotated.data, w, h, bytes_per_line,
                                QtGui.QImage.Format.Format_BGR888)
            self.frame_ready.emit(qimg)

        cap.release()
        self.hands.close()

    def stop(self):
        self.running = False
        self.wait()


# ---------- Main Application ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Order Manager (PyQt6)")
        self.resize(1100, 700)

        ensure_sounds()

        # Load sound effects
        self.sounds = {}
        for k, p in SOUND_FILES.items():
            se = QSoundEffect()
            se.setSource(QUrl.fromLocalFile(str(p)))
            se.setVolume(0.5)
            self.sounds[k] = se

        # Central widget
        main = QtWidgets.QWidget()
        self.setCentralWidget(main)
        layout = QtWidgets.QHBoxLayout(main)

        # ===== Left: Video + Gesture Buttons =====
        left = QtWidgets.QVBoxLayout()
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        left.addWidget(self.video_label)

        self.status_label = QtWidgets.QLabel("Status: Ready")
        left.addWidget(self.status_label)

        # Gesture buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("Previous (✌️)")
        self.btn_complete = QtWidgets.QPushButton("Complete (👍)")
        self.btn_cancel = QtWidgets.QPushButton("Cancel (✋)")
        self.btn_next = QtWidgets.QPushButton("Next (👉)")
        btn_layout.addWidget(self.btn_prev)
        btn_layout.addWidget(self.btn_complete)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_next)
        left.addLayout(btn_layout)

        layout.addLayout(left)

        # ===== Right: Orders Card + Table =====
        right = QtWidgets.QVBoxLayout()

        # --- Order Details Card ---
        self.order_card = QtWidgets.QGroupBox("Order Details")
        card_layout = QtWidgets.QFormLayout()
        self.lbl_order_number = QtWidgets.QLabel("-")
        self.lbl_order_status = QtWidgets.QLabel("-")
        self.lbl_items = QtWidgets.QLabel("-")
        self.lbl_qty = QtWidgets.QLabel("-")
        self.lbl_total = QtWidgets.QLabel("-")

        card_layout.addRow("Order #:", self.lbl_order_number)
        card_layout.addRow("Status:", self.lbl_order_status)
        card_layout.addRow("Items:", self.lbl_items)
        card_layout.addRow("Quantity:", self.lbl_qty)
        card_layout.addRow("Total:", self.lbl_total)

        self.order_card.setLayout(card_layout)

        right.addWidget(self.order_card)

        # --- Orders Table ---
        right.addWidget(QtWidgets.QLabel("<b>All Orders</b>"))
        self.order_table = QtWidgets.QTableWidget()
        self.order_table.setColumnCount(2)
        self.order_table.setHorizontalHeaderLabels(["Order ID", "Status"])
        self.order_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.order_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.order_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.order_table.setEditTriggers(QtWidgets.QTableWidget.EditTrigger.NoEditTriggers)
        right.addWidget(self.order_table)

        # Legend
        legend = QtWidgets.QLabel(
            "Gestures:\n👍 Thumbs Up = Complete\n✋ Open Palm = Cancel\n👉 Point = Next order\n✌️ Peace = Previous order"
        )
        right.addWidget(legend)

        layout.addLayout(right)

        # ===== Signals =====
        self.btn_complete.clicked.connect(lambda: self.handle_action("thumbs_up"))
        self.btn_cancel.clicked.connect(lambda: self.handle_action("open_palm"))
        self.btn_next.clicked.connect(lambda: self.handle_action("point"))
        self.btn_prev.clicked.connect(lambda: self.handle_action("prev"))

        # Keyboard shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=lambda: self.handle_action("thumbs_up"))
        QtGui.QShortcut(QtGui.QKeySequence("N"), self, activated=lambda: self.handle_action("point"))
        QtGui.QShortcut(QtGui.QKeySequence("C"), self, activated=lambda: self.handle_action("open_palm"))

        # Camera thread
        self.cam_thread = CameraThread()
        self.cam_thread.frame_ready.connect(self.update_frame)
        self.cam_thread.gesture_detected.connect(self.on_gesture)
        self.cam_thread.start()

        # Load DB
        init_db()
        self.reload_orders()

        # Table selection signal
        self.order_table.cellClicked.connect(self.on_order_selected)

        # Poll timer for updates
        self.poll_timer = QtCore.QTimer()
        self.poll_timer.timeout.connect(lambda: self.reload_orders(preserve_selection=True))
        self.poll_timer.start(3000)

    # ===== Helper Methods =====
    def closeEvent(self, event):
        if hasattr(self, "cam_thread") and self.cam_thread.isRunning():
            self.cam_thread.stop()
        event.accept()

    def update_frame(self, qimg):
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )
        self.video_label.setPixmap(pix)

    # ===== Order Reload =====
    def reload_orders(self, preserve_selection=True):
        # remember selected
        selected_id = None
        if preserve_selection:
            sel = self.get_selected_order()
            if sel:
                selected_id = sel

        self.order_table.setRowCount(0)
        rows = load_orders()  # expected: (id, number, status, items, qty, total)

        for r, row in enumerate(rows):
            # Unpack safely in correct order
            id_ = row[0] if len(row) > 0 else None
            number = row[1] if len(row) > 1 else "-"
            status = row[2] if len(row) > 2 else "pending"
            items = row[3] if len(row) > 3 else "-"
            qty = row[4] if len(row) > 4 else "-"
            total = row[5] if len(row) > 5 else 0.0

            # Add to table
            self.order_table.insertRow(r)

            item_number = QtWidgets.QTableWidgetItem(str(number))
            item_number.setData(QtCore.Qt.ItemDataRole.UserRole, id_)
            status_item = QtWidgets.QTableWidgetItem(status)

            # Color by status
            if status == "completed":
                status_item.setForeground(QtGui.QColor("green"))
            elif status == "canceled":
                status_item.setForeground(QtGui.QColor("red"))
            else:
                status_item.setForeground(QtGui.QColor("black"))

            self.order_table.setItem(r, 0, item_number)
            self.order_table.setItem(r, 1, status_item)

        # restore selection
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

    # ===== Update Order Details Card =====
    def on_order_selected(self, row, column):
        order_id = self.order_table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
        order = get_order_by_id(order_id)
        if not order:
            return

        self.lbl_order_number.setText(str(order.get("order_number", "-")))
        self.lbl_order_status.setText(order.get("status", "-"))
        self.lbl_items.setText(order.get("items", "-"))
        self.lbl_qty.setText(str(order.get("qty", "-")))
        self.lbl_total.setText(f"{order.get('total', 0):.2f}")


    # ===== Handle Gesture / Button Actions =====
    def handle_action(self, gesture):
        order_id = self.get_selected_order()
        if not order_id:
            self.status_label.setText("Status: No order selected")
            return

        prior_status = get_order_status(order_id)
        action_changed = False

        if gesture == "thumbs_up":
            if prior_status != "completed":
                update_order_status(order_id, "completed")
                self.status_label.setText(f"Status: Order {order_id} completed")
                action_changed = True
        elif gesture == "open_palm":
            if prior_status != "canceled":
                update_order_status(order_id, "canceled")
                self.status_label.setText(f"Status: Order {order_id} canceled")
                action_changed = True
        elif gesture == "point":
            row = self.order_table.currentRow()
            if row < self.order_table.rowCount() - 1:
                self.order_table.selectRow(row + 1)
                self.on_order_selected(row + 1, 0)
                self.status_label.setText("Status: Moved to next order")
                action_changed = True
        elif gesture == "prev":
            row = self.order_table.currentRow()
            if row > 0:
                self.order_table.selectRow(row - 1)
                self.on_order_selected(row - 1, 0)
                self.status_label.setText("Status: Moved to previous order")
                action_changed = True

        self.reload_orders(preserve_selection=True)

        if action_changed and gesture in self.sounds:
            self.sounds[gesture].stop()
            self.sounds[gesture].play()

    # ===== Camera Gesture Slot =====
    @QtCore.pyqtSlot(str)
    def on_gesture(self, gesture):
        self.status_label.setText(f"Gesture detected: {gesture}")
        QtCore.QTimer.singleShot(1200, lambda: self.status_label.setText("Status: Ready"))
        self.handle_action(gesture)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
