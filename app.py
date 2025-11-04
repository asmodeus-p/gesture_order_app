import sys
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
from PyQt6 import QtCore, QtGui, QtWidgets

DB_PATH = "orders.db"

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
    conn.commit()
    # seed some demo orders if empty
    c.execute("SELECT COUNT(*) FROM orders")
    if c.fetchone()[0] == 0:
        now = datetime.utcnow().isoformat()
        demos = [("1001", "pending", now),
                 ("1002", "pending", now),
                 ("1003", "pending", now),
                 ("1004", "pending", now)]
        c.executemany("INSERT INTO orders (order_number, status, created_at) VALUES (?, ?, ?)", demos)
        conn.commit()
    conn.close()

def load_orders(path=DB_PATH):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("SELECT id, order_number, status FROM orders ORDER BY id")
    rows = c.fetchall()
    conn.close()
    return rows

def update_order_status(order_id, new_status, path=DB_PATH):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("UPDATE orders SET status = ? WHERE id = ?", (new_status, order_id))
    conn.commit()
    conn.close()

# ---------- Gesture detection thread ----------
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
        self.FRAMES_REQUIRED = 8  # consecutive frames for confirmation

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

                # thumb vertical for thumbs up
                def is_thumb_up():
                    return lm[4][1] < lm[3][1]  # tip above IP joint

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
                # 1) Thumbs Up
                if thumb_up and not (idx_ext or mid_ext or ring_ext or pinky_ext):
                    gesture = "thumbs_up"
                    cv2.putText(annotated, "Thumbs Up", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

                # 2) Open Palm
                elif thumb_open and all([idx_ext, mid_ext, ring_ext, pinky_ext]):
                    gesture = "open_palm"
                    cv2.putText(annotated, "Open Palm", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

                # 3) Point
                elif idx_ext and not (mid_ext or ring_ext or pinky_ext):
                    gesture = "point"
                    cv2.putText(annotated, "Pointing", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

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
        self.resize(1000, 700)

        # central widget
        main = QtWidgets.QWidget()
        self.setCentralWidget(main)
        layout = QtWidgets.QHBoxLayout(main)

        # left: video
        left = QtWidgets.QVBoxLayout()
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        left.addWidget(self.video_label)

        self.status_label = QtWidgets.QLabel("Status: Ready")
        left.addWidget(self.status_label)

        # fallback buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_complete = QtWidgets.QPushButton("Complete (👍)")
        self.btn_cancel = QtWidgets.QPushButton("Cancel (✋)")
        self.btn_next = QtWidgets.QPushButton("Next (👉)")
        btn_layout.addWidget(self.btn_complete)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_next)
        left.addLayout(btn_layout)

        # right: orders list
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("<b>Orders</b>"))
        self.order_list = QtWidgets.QListWidget()
        self.order_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        right.addWidget(self.order_list)

        # small legend
        legend = QtWidgets.QLabel("Gestures:\n👍 Thumbs Up = Complete\n✋ Open Palm = Cancel\n👉 Point = Next order")
        right.addWidget(legend)

        layout.addLayout(left)
        layout.addLayout(right)

        # wires
        self.btn_complete.clicked.connect(lambda: self.handle_action("thumbs_up"))
        self.btn_cancel.clicked.connect(lambda: self.handle_action("open_palm"))
        self.btn_next.clicked.connect(lambda: self.handle_action("point"))

        # keyboard shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=lambda: self.handle_action("thumbs_up"))
        QtGui.QShortcut(QtGui.QKeySequence("N"), self, activated=lambda: self.handle_action("point"))
        QtGui.QShortcut(QtGui.QKeySequence("C"), self, activated=lambda: self.handle_action("open_palm"))

        # camera thread
        self.cam_thread = CameraThread()
        self.cam_thread.frame_ready.connect(self.update_frame)
        self.cam_thread.gesture_detected.connect(self.on_gesture)
        self.cam_thread.start()

        # load DB
        init_db()
        self.reload_orders()

    def closeEvent(self, event):
        if hasattr(self, "cam_thread") and self.cam_thread.isRunning():
            self.cam_thread.stop()
        event.accept()

    def update_frame(self, qimg):
        # scale to label
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pix)

    def reload_orders(self, preserve_selection=True):
        # remember currently selected order
        selected_id = None
        if preserve_selection:
            item = self.order_list.currentItem()
            if item:
                selected_id = item.data(QtCore.Qt.ItemDataRole.UserRole)

        self.order_list.clear()
        rows = load_orders()
        for r in rows:
            id_, number, status = r
            item = QtWidgets.QListWidgetItem(f"#{number} — {status}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, id_)
            # style based on status
            if status == "completed":
                item.setForeground(QtGui.QColor("green"))
            elif status == "canceled":
                item.setForeground(QtGui.QColor("red"))
            self.order_list.addItem(item)

        # restore selection
        if preserve_selection and selected_id is not None:
            for i in range(self.order_list.count()):
                if self.order_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole) == selected_id:
                    self.order_list.setCurrentRow(i)
                    break
        elif self.order_list.count() > 0:
            self.order_list.setCurrentRow(0)


    def get_selected_order(self):
        item = self.order_list.currentItem()
        if not item:
            return None
        return item.data(QtCore.Qt.ItemDataRole.UserRole)

    def handle_action(self, gesture):
        order_id = self.get_selected_order()
        if not order_id:
            self.status_label.setText("Status: No order selected")
            return
        if gesture == "thumbs_up":
            update_order_status(order_id, "completed")
            self.status_label.setText(f"Status: Order {order_id} completed")
        elif gesture == "open_palm":
            update_order_status(order_id, "canceled")
            self.status_label.setText(f"Status: Order {order_id} canceled")
        elif gesture == "point":
            # next row
            row = self.order_list.currentRow()
            if row < self.order_list.count() - 1:
                self.order_list.setCurrentRow(row + 1)
                self.status_label.setText("Status: Moved to next order")
            else:
                self.status_label.setText("Status: Already at last order")
        self.reload_orders()

    # slot from camera thread when a gesture is confirmed
    @QtCore.pyqtSlot(str)
    def on_gesture(self, gesture):
        # flash UI or change label to show gesture
        self.status_label.setText(f"Gesture detected: {gesture}")
        QtCore.QTimer.singleShot(1200, lambda: self.status_label.setText("Status: Ready"))
        # map directly to action
        self.handle_action(gesture)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
