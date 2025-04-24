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

#! DYNAMIC VARIABLES 
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

distance_threshold = 0.05
#! DYNAMIC VARIABLES ^^

# distance formula
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


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
    
    # for use later in adding SRS title to face
    chin = 0
    if face_mesh_results.multi_face_landmarks:
        for idx, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=face_mesh_drawing_spec,
                connection_drawing_spec=face_mesh_connection_spec
            )
            #! chin landmark
            chin = face_landmarks.landmark[199]  # Changed to landmark 199 (tip of chin)
            chin_coords = [chin.x, chin.y]  # Store chin coordinates

    
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

            # Get finger tip positions for each hand, only runs if there are two hands detected
            if hands_results.multi_handedness:
                left_finger_tips = []
                right_finger_tips = []
                
                for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    # Determine if it's left or right hand
                    handedness = hands_results.multi_handedness[idx].classification[0].label
                    
                    # Finger tip landmarks are 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
                    finger_tips = [
                        (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y),  # thumb
                        (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y),  # index
                        (hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y),  # middle
                        (hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y),  # ring
                        (hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y)   # pinky
                    ]
                    
                    if handedness == "Left":
                        left_finger_tips = finger_tips
                    else:
                        right_finger_tips = finger_tips

        # left hand calculation (2)
        two = False
        if left_finger_tips:
            left_thumb_tip = [left_finger_tips[0][0], left_finger_tips[0][1]]
            left_index_tip = [left_finger_tips[1][0], left_finger_tips[1][1]]
            left_middle_tip = [left_finger_tips[2][0], left_finger_tips[2][1]]
            left_ring_tip = [left_finger_tips[3][0], left_finger_tips[3][1]]
            left_pinky_tip = [left_finger_tips[4][0], left_finger_tips[4][1]]

            # the three fingers that are down
            avg = (distance(left_thumb_tip, left_ring_tip) + distance(left_pinky_tip, left_ring_tip)) / 2
            avg2 = (distance(left_index_tip, left_ring_tip) + distance(left_middle_tip, left_ring_tip)) / 2
            if avg < distance_threshold and avg2 > distance_threshold:
                two = True
                print("two")

        # right hand calculation (5)
        five = False
        if right_finger_tips:
            right_thumb_tip = [right_finger_tips[0][0], right_finger_tips[0][1]]
            right_index_tip = [right_finger_tips[1][0], right_finger_tips[1][1]]
            right_middle_tip = [right_finger_tips[2][0], right_finger_tips[2][1]]
            right_ring_tip = [right_finger_tips[3][0], right_finger_tips[3][1]]
            right_pinky_tip = [right_finger_tips[4][0], right_finger_tips[4][1]]

            # the three fingers that are down
            # Get wrist position (landmark 0 is the wrist)
            right_wrist = [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y]  # Using actual wrist landmark
            
            # Calculate average distances from wrist to each finger tip
            wrist_to_fingers = [
                distance(right_wrist, right_thumb_tip),
                distance(right_wrist, right_index_tip), 
                distance(right_wrist, right_middle_tip),
                distance(right_wrist, right_ring_tip),
                distance(right_wrist, right_pinky_tip)
            ]
            
            print(wrist_to_fingers)


            # If all fingers are extended, their distances from wrist will be larger
            # Using a smaller threshold for five gesture
            five = all(dist > distance_threshold * 0.5 for dist in wrist_to_fingers)
            
            if five:
                print("five")

        if two and five:
            #! adding SRS title to face
            cv2.putText(frame, "SRS", (int(chin_coords[0] * width), int(chin_coords[1] * height + 50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # Display the frame

    cv2.imshow('MediaPipe Face Mesh', frame)

    # Break the loop if 'q' is pressed  
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
