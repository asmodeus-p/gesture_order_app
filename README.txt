GESTURE ORDER MANAGER
=====================

Description
-----------
This is a local desktop application built with Python (PyQt6 and OpenCV) that allows managing orders
using hand gestures via your laptop camera. Gestures are recognized using MediaPipe Hands.

Latest Changes:
    - fixed bugs where fist is detected as thumbs up

Minimum Viable Product includes:

Gestures:
    - Open Palm: Cancel order
    - Point right: Next order
    - Peace sign: Previous Order
    - Thumbs Up: Mark order as completed
    - Fist: Hold or do nothing

UI: 
    - Frontend
    - UI that lets the cashier input order details (quantity, item type)


Key Features
------------
1. ** Current Gesture Controls:**
   - Thumbs Up: Mark the selected order as completed
   - Open Palm: Cancel the selected order
   - Point: Move to the next order
   - Peace sign: move to the previous order

2. **UI Features:**
   - Real-time video feed with hand landmarks overlay
   - Orders list with current selection highlighted
   - Status label showing detected gestures and actions
   - Optional buttons for manual order control
   - API endpoint for mobile web app POST request

3. **Debounce Mechanism:**
   - Gestures are confirmed only after being detected consistently for 12 frames to avoid flickering.

System Requirements
-------------------
- Python 3.11
- Pip package manager
- Laptop with a webcam
- Operating System: Windows, macOS, or Linux

Required Python Packages
------------------------
- opencv-python
- mediapipe
- PyQt6
- numpy (installed with OpenCV)
- matplotlib (optional, dependency of MediaPipe)

Installation
------------
1. Clone or download the project folder.
2. Open a terminal/PowerShell inside the project folder.
3. (Recommended) Create a virtual environment:


python -m venv venv

Activate the environment:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`
4. Install required packages:


python -m pip install opencv-python mediapipe PyQt6


Running the Application
-----------------------
1. Ensure your virtual environment is activated (if used).
2. Run the main application:


python app.py

3. The window will display:
- Webcam feed
- Orders list
- Status label
- Gesture legend

4. Perform gestures in front of the camera:
- Thumbs Up → completes selected order
- Open Palm → cancels selected order
- Point → selects next order
- Peace Sign -> selects previous order
- Fist -> hold/do nothing

Notes
-----
- Make sure your hand is clearly visible in front of the camera.
- Gesture detection may require slight hand adjustments for optimal accuracy.
- The app uses a debounce system to reduce false positives.
- Thumbs Up requires the thumb vertical and other fingers folded.
- Open Palm requires all four fingers extended, thumb relaxed.
- Point requires only the index finger extended.

Troubleshooting
---------------
- If gestures are not detected:
- Check camera access and make sure no other program is using the webcam.
- Ensure proper lighting and front-facing hand orientation.
- If Open Palm is not triggering:
- Ensure all four fingers are extended; thumb can be angled outward.
- If Python packages are not recognized:
- Make sure you are using Python 3.11 and the correct virtual environment.

Author
------
asmodeus-p (Marc Danielle Ipapo)

Version
-------
1.9 — Local Gesture-Based Order Manager (PyQt6 + OpenCV + MediaPipe)


