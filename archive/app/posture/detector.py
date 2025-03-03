import cv2
import math
import mediapipe as mp
import numpy as np

class PostureDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def find_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    def find_angle(self, x1, y1, x2, y2):
        try:
            theta = math.acos((y2 - y1)*(-y1) / (math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
            return int(180/math.pi) * theta
        except:
            return 0
    
    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return frame, {
                "aligned": False,
                "neck_angle": 0,
                "torso_angle": 0,
                "posture": "not_detected"
            }
        
        lm = results.pose_landmarks
        
        # Get landmarks
        l_shldr = (int(lm.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                   int(lm.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        r_shldr = (int(lm.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                   int(lm.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        l_ear = (int(lm.landmark[self.mp_pose.PoseLandmark.LEFT_EAR].x * w),
                 int(lm.landmark[self.mp_pose.PoseLandmark.LEFT_EAR].y * h))
        l_hip = (int(lm.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * w),
                 int(lm.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * h))
        
        # Calculate metrics
        offset = self.find_distance(l_shldr[0], l_shldr[1], r_shldr[0], r_shldr[1])
        neck_inclination = self.find_angle(l_shldr[0], l_shldr[1], l_ear[0], l_ear[1])
        torso_inclination = self.find_angle(l_hip[0], l_hip[1], l_shldr[0], l_shldr[1])
        
        # Determine posture
        good_posture = neck_inclination < 40 and torso_inclination < 10
        
        # Draw visualization
        color = (0, 255, 0) if good_posture else (0, 0, 255)
        
        # Draw lines
        cv2.line(frame, l_shldr, l_ear, color, 4)
        cv2.line(frame, l_shldr, (l_shldr[0], l_shldr[1] - 100), color, 4)
        cv2.line(frame, l_hip, l_shldr, color, 4)
        cv2.line(frame, l_hip, (l_hip[0], l_hip[1] - 100), color, 4)
        
        # Add text
        cv2.putText(frame, f"Neck: {neck_inclination:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Torso: {torso_inclination:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame, {
            "aligned": offset < 100,
            "neck_angle": neck_inclination,
            "torso_angle": torso_inclination,
            "posture": "good" if good_posture else "bad"
        }