import cv2
import numpy as np

cap = cv2.VideoCapture("../Film/test_clip.mov")

H = np.array([[   0.027449,   -0.007561,     0.64209],
              [ -0.0005194,    0.033204,     -8.1587],
              [ -5.636e-06, -0.00074346,           1]])

"""
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]
    transformed_frame = cv2.warpPerspective(frame, H, (width, height))

    cv2.imshow('Transformed Video', transformed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""

pts = np.array([[350, 570], [300, 655], [346, 805]], dtype = np.float32).reshape(-1, 1, 2)
new_pts = cv2.perspectiveTransform(pts, H)

print(new_pts)

cap.release()