import mediapipe as mp
import numpy as np
import cv2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

KEYPOINTS = {
    "left_ear": mp_pose.PoseLandmark.LEFT_EAR,
    "right_ear": mp_pose.PoseLandmark.RIGHT_EAR,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
}

def process_image(input_path=None,image=None):
    
    if input_path is not None and image is None:    
        image = cv2.imread(input_path)

    if image is None:
        print("Error: could not open image")
        return None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)
        black_background = np.zeros_like(image)
        
        if not results.pose_landmarks:
            print("No pose landmarks detected")
            return None

        if results.pose_landmarks:
            for keypoint in KEYPOINTS.values():
                landmark = results.pose_landmarks.landmark[keypoint]
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(black_background, (x, y), 5, (255, 255, 255), -1)

            connections = [
                (mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            ]

            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            midpoint_shoulder_x = int((left_shoulder.x + right_shoulder.x) / 2 * image.shape[1])
            midpoint_shoulder_y = int((left_shoulder.y + right_shoulder.y) / 2 * image.shape[0])

            left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
            lear_point = (int(left_ear.x * image.shape[1]), int(left_ear.y * image.shape[0]))
            cv2.line(black_background, lear_point, (midpoint_shoulder_x, midpoint_shoulder_y), (255, 255, 255), 2)
            
            right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
            rear_point = (int(right_ear.x * image.shape[1]), int(right_ear.y * image.shape[0]))
            cv2.line(black_background, rear_point, (midpoint_shoulder_x, midpoint_shoulder_y), (255, 255, 255), 2)

            for connection in connections:
                start = results.pose_landmarks.landmark[connection[0]]
                end = results.pose_landmarks.landmark[connection[1]]
                start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
                end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
                cv2.line(black_background, start_point, end_point, (255, 255, 255), 2)

        # Resize and pad to 640x640
        target_size = (640, 640)
        original_height, original_width = black_background.shape[:2]
        scale = min(target_size[0] / original_width, target_size[1] / original_height)
        resized_width = int(original_width * scale)
        resized_height = int(original_height * scale)
        resized_image = cv2.resize(black_background, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

        top_padding = (target_size[1] - resized_height) // 2
        bottom_padding = target_size[1] - resized_height - top_padding
        left_padding = (target_size[0] - resized_width) // 2
        right_padding = target_size[0] - resized_width - left_padding

        padded_image = cv2.copyMakeBorder(
            resized_image,
            top_padding,
            bottom_padding,
            left_padding,
            right_padding,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        return padded_image