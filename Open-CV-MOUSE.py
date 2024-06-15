import cv2
import mediapipe as mp
import pyautogui

# Initialize video capture from IP camera
cap = cv2.VideoCapture("http://192.168.1.3:8080/video")

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Desired window size
window_width = 1080
window_height = 720

# Create the named window with the desired size
cv2.namedWindow('Virtual Mouse', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Virtual Mouse', window_width, window_height)

# Gesture detection threshold
gesture_threshold = 0.2


# Function to detect gestures
def detect_gestures(landmarks, frame_width, frame_height):
    # Get landmarks for thumb and other fingers
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Calculate distances between thumb and other fingers
    thumb_index_dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    thumb_middle_dist = ((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2) ** 0.5
    thumb_ring_dist = ((thumb_tip.x - ring_tip.x) ** 2 + (thumb_tip.y - ring_tip.y) ** 2) ** 0.5
    thumb_pinky_dist = ((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2) ** 0.5

    # Normalize distances by frame width
    thumb_index_dist /= frame_width
    thumb_middle_dist /= frame_width
    thumb_ring_dist /= frame_width
    thumb_pinky_dist /= frame_width

    # Detect gestures
    is_open_palm = (
            thumb_index_dist > gesture_threshold and
            thumb_middle_dist > gesture_threshold and
            thumb_ring_dist > gesture_threshold and
            thumb_pinky_dist > gesture_threshold
    )

    is_closed_fist = (
            thumb_index_dist < gesture_threshold and
            thumb_middle_dist < gesture_threshold and
            thumb_ring_dist < gesture_threshold and
            thumb_pinky_dist < gesture_threshold
    )

    return is_open_palm, is_closed_fist


# Main loop
while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Resize frame to fit within the desired window size while maintaining aspect ratio
    aspect_ratio = frame_width / frame_height
    if window_width / aspect_ratio <= window_height:
        resized_frame = cv2.resize(frame, (window_width, int(window_width / aspect_ratio)))
    else:
        resized_frame = cv2.resize(frame, (int(window_height * aspect_ratio), window_height))

    # Convert the frame to RGB as required by MediaPipe
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(resized_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gestures
            is_open_palm, is_closed_fist = detect_gestures(hand_landmarks.landmark, frame_width, frame_height)

            # Get landmark positions
            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Convert normalized coordinates to pixel values
            index_x = int(index_finger_tip.x * resized_frame.shape[1])
            index_y = int(index_finger_tip.y * resized_frame.shape[0])
            thumb_x = int(thumb_tip.x * resized_frame.shape[1])
            thumb_y = int(thumb_tip.y * resized_frame.shape[0])

            # Draw circles on the detected landmarks
            cv2.circle(resized_frame, (index_x, index_y), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(resized_frame, (thumb_x, thumb_y), 10, (0, 255, 255), cv2.FILLED)

            # Convert coordinates for the screen
            screen_index_x = screen_width / frame_width * index_x
            screen_index_y = screen_height / frame_height * index_y
            screen_thumb_y = screen_height / frame_height * thumb_y

            # Move the mouse
            pyautogui.moveTo(screen_index_x, screen_index_y)

            # Click the mouse if the index and thumb are close enough
            if abs(screen_index_y - screen_thumb_y) < 20:
                pyautogui.click()
                pyautogui.sleep(1)  # Prevent multiple clicks

            # Right-click with open palm
            if is_open_palm:
                pyautogui.rightClick()
                pyautogui.sleep(1)  # Prevent multiple right-clicks

            # Double-click with closed fist
            if is_closed_fist:
                pyautogui.doubleClick()
                pyautogui.sleep(1)  # Prevent multiple double-clicks

    # Display the frame in the window
    cv2.imshow('Virtual Mouse', resized_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
