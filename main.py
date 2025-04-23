import mediapipe as mp
import cv2
import numpy as np

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

#Drawing Utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Drawing Specifications
face_mesh_drawing_spec = mp_drawing.DrawingSpec(
    color=(0, 255, 0),  # Green color in BGR
    thickness=1,
    circle_radius=1
)

face_mesh_connection_spec = mp_drawing.DrawingSpec(
    color=(255, 0, 0),  # Blue color in BGR 
    thickness=1
)

hand_landmark_spec = mp_drawing.DrawingSpec(
    color=(255, 255, 255),  # White color in BGR
    thickness=2,
    circle_radius=2
)

hand_connection_spec = mp_drawing.DrawingSpec(
    color=(255, 255, 255),  # White color in BGR
    thickness=2
)


#Video Capture (only works with webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    #! processing before handoff to mediapipe

    # Get original dimensions
    height, width = frame.shape[:2]
    # Calculate new dimensions maintaining aspect ratio
    target_height = 720
    target_width = int(width * (target_height / height))
    frame = cv2.resize(frame, (target_width, target_height))
    frame = cv2.flip(frame, 1)
    # changing color space for compatibility with mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # mediapipe processing, face mesh
    face_mesh_results = face_mesh.process(rgb_frame)
    
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=face_mesh_drawing_spec,
                connection_drawing_spec=face_mesh_connection_spec
            )
    
    #mediapipe processing, hands
    hands_results = hands.process(rgb_frame)

    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            #! drawing landmarks and connections
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_spec,
                connection_drawing_spec=hand_connection_spec
            )



    # Display the frame
    cv2.imshow('MediaPipe Face Mesh', frame)

    # Break the loop if 'q' is pressed  
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
