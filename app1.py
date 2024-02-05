from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

# Constants for zooming
ZOOM_FACTOR_CHANGE_THRESHOLD = 10
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 0.8
SMOOTHING_FACTOR = 0.5

#mp_hands and hands are instances of the MediaPipe hands module.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Global variables
zoom_factor = 1.0
prev_thumb_tip_y = 0

def process_frame(frame):
    global zoom_factor, prev_thumb_tip_y

    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Use MediaPipe hands module to process the frame
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Extract landmarks of the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Calculate vertical position of thumb tip and change in position
        thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]
        thumb_tip_y_change = thumb_tip_y - prev_thumb_tip_y

        # Check if thumb tip position change is significant for zooming
        if abs(thumb_tip_y_change) > ZOOM_FACTOR_CHANGE_THRESHOLD:
            # Adjust zoom factor based on thumb tip movement direction
            if thumb_tip_y_change > 0:
                zoom_factor *= ZOOM_IN_FACTOR
            else:
                zoom_factor *= ZOOM_OUT_FACTOR

        # Ensure zoom factor stays within specified limits
        zoom_factor = max(min(zoom_factor, ZOOM_IN_FACTOR), ZOOM_OUT_FACTOR)
        prev_thumb_tip_y = thumb_tip_y

        # Draw circles at each hand landmark
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw lines connecting specific hand landmarks
        connections = [(mp_hands.HandLandmark.WRIST, mp_hands.HandLandmark.THUMB_TIP),
                       (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP),
                       # Add more connections as needed
                       ]
        for connection in connections:
            start_point = (int(hand_landmarks.landmark[connection[0]].x * frame.shape[1]),
                           int(hand_landmarks.landmark[connection[0]].y * frame.shape[0]))
            end_point = (int(hand_landmarks.landmark[connection[1]].x * frame.shape[1]),
                         int(hand_landmarks.landmark[connection[1]].y * frame.shape[0]))
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

    # Resize the frame based on the calculated zoom factor
    zoomed_frame = cv2.resize(frame, None, fx=zoom_factor, fy=zoom_factor)
    return zoomed_frame

def generate_frames():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        success, frame = cap.read()
        if not success:
            break

        # Process the frame for zooming and hand tracking
        processed_frame = process_frame(frame)
        
        # Convert the processed frame to JPEG format
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        frame_bytes = buffer.tobytes()

        # Yield the frame as a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

@app.route('/input', methods=['GET', 'POST'])
def innerpage():
    return render_template('inner-page.html')

@app.route('/video_feed')
def video_feed():
    # Return a multipart response with video frames
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Run the Flask application
    app.run(debug=True, port=3050)
