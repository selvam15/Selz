from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

app = Flask(__name__)

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Store previous position for swipe detection
prev_x, prev_y = None, None
gesture_threshold = 40 

def generate_frames():
    global prev_x, prev_y  

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            for hand in hands:
                # Draw landmarks on the frame
                drawing_utils.draw_landmarks(frame, hand)
                landmarks = hand.landmark

                # Get index finger tip position
                index_x = int(landmarks[8].x * frame_width)
                index_y = int(landmarks[8].y * frame_height)

                # Draw a circle at the index finger tip
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

                # Detect swipe gestures
                if prev_x is not None and prev_y is not None:
                    dx = index_x - prev_x
                    dy = index_y - prev_y

                    if abs(dx) > gesture_threshold:  
                        if dx > 0:
                            pyautogui.press('right')  
                            print("Swipe Right Detected! (Move Right)")
                        else:
                            pyautogui.press('left')  
                            print("Swipe Left Detected! (Move Left)")

                    if abs(dy) > gesture_threshold:  
                        if dy < 0:
                            pyautogui.press('up') 
                            print("Swipe Up Detected! (Jump)")
                        else:
                            pyautogui.press('down') 
                            print("Swipe Down Detected! (Slide)")

                # Update previous position
                prev_x, prev_y = index_x, index_y

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
