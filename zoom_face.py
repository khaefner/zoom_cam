import cv2
import pyvirtualcam
import numpy as np

# Enable OpenCL
cv2.ocl.setUseOpenCL(True)
print(f"OpenCL enabled: {cv2.ocl.useOpenCL()}")

# Load the pre-trained Haar Cascade classifier for face detection.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the real camera (usually device 0).
cam_id = 0
cap = cv2.VideoCapture(cam_id)

if not cap.isOpened():
    print(f"Error: Could not open camera {cam_id}.")
    exit()

# Get camera properties.
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Camera properties: width={original_width}, height={original_height}, fps={fps}")

# Parameters for smoothing transitions
smoothing_factor = 0.05  # A smaller value will slow down the transition
padding_factor = 2.5  # Increase this factor to add more space around the face

# Initial target and current positioning
target_cx = original_width // 2
target_cy = original_height // 2
current_cx = target_cx
current_cy = target_cy
target_crop_size = min(original_width, original_height) // 2
current_crop_size = target_crop_size

# Open the virtual camera.
with pyvirtualcam.Camera(width=original_width, height=original_height, fps=fps) as virtual_cam:
    print(f"Virtual camera started: {virtual_cam.device}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera.")
            break

        # Convert the frame to grayscale for face detection.
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_umat = cv2.UMat(gray_frame)

        # Detect faces in the frame.
        faces = face_cascade.detectMultiScale(gray_umat, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Find the largest face
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])  # Select face with largest area
            x, y, w, h = largest_face
            target_cx, target_cy = x + w // 2, y + h // 2
            target_crop_size = int(max(w, h) * padding_factor)  # Apply padding factor

        # Smooth transition towards the target position and crop size
        current_cx = int(current_cx + (target_cx - current_cx) * smoothing_factor)
        current_cy = int(current_cy + (target_cy - current_cy) * smoothing_factor)
        current_crop_size = int(current_crop_size + (target_crop_size - current_crop_size) * smoothing_factor)

        # Calculate aspect-ratio maintained crop box
        aspect_ratio = original_width / original_height

        # Determine crop dimensions while maintaining the aspect ratio
        if current_crop_size * aspect_ratio > current_crop_size:
            crop_width = current_crop_size
            crop_height = int(current_crop_size / aspect_ratio)
        else:
            crop_height = current_crop_size
            crop_width = int(current_crop_size * aspect_ratio)

        # Calculate crop box with clamping to frame dimensions
        x1 = max(0, current_cx - crop_width // 2)
        y1 = max(0, current_cy - crop_height // 2)
        x2 = min(original_width, current_cx + crop_width // 2)
        y2 = min(original_height, current_cy + crop_height // 2)

        # Crop the frame to the calculated region
        cropped_frame = frame[y1:y2, x1:x2]

        # Resize the cropped region to original frame size
        zoomed_frame = cv2.resize(cropped_frame, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        # Convert the frame to RGB.
        frame_rgb = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2RGB)

        # Send the frame to the virtual camera.
        virtual_cam.send(frame_rgb)

        # Wait for the next frame based on camera FPS.
        virtual_cam.sleep_until_next_frame()

# Release resources.
cap.release()
print("Camera capture and virtual camera released.")
