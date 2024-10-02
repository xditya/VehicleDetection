import sys
import cv2
from ultralytics import YOLO
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

# Load the pre-trained YOLOv10 model
model = YOLO("res/models/yolov10n.pt")

# Class names for vehicles
vehicle_classes = ["car", "truck", "bus", "motorbike"]

# Paths to the four video files
video_paths = ["video.mp4", "video2.webm", "video3.webm", "video4.webm"]
caps = [cv2.VideoCapture(f"res/videos/{path}") for path in video_paths]

# Create a PyQt application
app = QtWidgets.QApplication(sys.argv)

# Create a PyQt window for displaying vehicle counts
count_window = QtWidgets.QWidget()
count_window.setWindowTitle("Vehicle Counts")
count_window.setGeometry(100, 100, 400, 200)
count_layout = QtWidgets.QGridLayout()

# Create labels for each feed and add them to the grid layout
vehicle_count_labels = []
background_colors = ["red"] * 4  # Default background colors (all red)

for i in range(4):
    label = QtWidgets.QLabel(f"Feed {i + 1}: 0 vehicles")
    label.setStyleSheet(f"background-color: {background_colors[i]}; color: white; padding: 10px;")
    vehicle_count_labels.append(label)
    count_layout.addWidget(label, i // 2, i % 2)  # Place in a 2x2 grid

count_window.setLayout(count_layout)
count_window.show()

# Create an OpenCV window and resize it
window_name = "4 Video Feeds with Vehicle Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1200, 800)  # Resize the window to 1200x800 pixels

# Desired frames per second
desired_fps = 3
frame_interval = int(30 / desired_fps)  # Assuming 30 FPS for the input videos

# Frame counters for each video
frame_counts = [0, 0, 0, 0]

while all(cap.isOpened() for cap in caps):
    frames = []
    vehicles_detected = False  # Flag to check if any vehicle is detected
    num_vehicles_list = [0] * 4  # Initialize with zero for each feed

    # Read frames from each video file
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            break  # End of one of the videos

        # Check if the current frame should be processed
        if frame_counts[i] % frame_interval == 0:
            # Perform object detection on the current frame
            results = model(frame)
            detections = results[0]

            # Filter detections based on vehicle classes
            vehicle_detections = [
                box for box in detections.boxes if detections.names[int(box.cls)] in vehicle_classes
            ]

            # Count the number of vehicles detected
            num_vehicles = len(vehicle_detections)
            num_vehicles_list[i] = num_vehicles  # Store the count for this feed
            if num_vehicles > 0:
                vehicles_detected = True  # Set flag if vehicle detected

            # Draw bounding boxes and annotations for vehicle detections
            annotated_frame = detections.plot() if vehicle_detections else frame
            
            # Add the count of detected vehicles to the frame
            cv2.putText(annotated_frame, f'Vehicles Detected: {num_vehicles}', 
                        (10, annotated_frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            annotated_frame = frame  # Use the original frame if not processing

        frames.append(annotated_frame)
        frame_counts[i] += 1  # Increment frame counter

    # If any video ended, stop
    if len(frames) < 4:
        break

    # Determine the feed with the maximum vehicles detected
    max_vehicles = max(num_vehicles_list) if num_vehicles_list else 0
    for i in range(4):
        if num_vehicles_list[i] == max_vehicles and num_vehicles_list[i] > 0:
            background_colors[i] = "green"  # Set to green if it's the max
        elif num_vehicles_list[i] == 0:
            continue  # Keep the same color if no vehicles detected
        else:
            background_colors[i] = "red"  # Otherwise, set to red

        # Update the label with the vehicle count and the background color
        vehicle_count_labels[i].setText(f"Feed {i + 1}: {num_vehicles_list[i]} vehicles")
        vehicle_count_labels[i].setStyleSheet(f"background-color: {background_colors[i]}; color: white; padding: 10px;")

    # Resize each frame to fit a quadrant
    height, width = frames[0].shape[:2]
    new_height, new_width = height // 2, width // 2
    resized_frames = [cv2.resize(frame, (new_width, new_height)) for frame in frames]

    # Create a 2x2 grid for displaying all videos
    top_row = np.hstack((resized_frames[0], resized_frames[1]))
    bottom_row = np.hstack((resized_frames[2], resized_frames[3]))
    grid = np.vstack((top_row, bottom_row))

    # Show the combined grid in the resized window
    cv2.imshow(window_name, grid)

    # Pause for 5 seconds if any vehicle was detected
    if vehicles_detected:
        print("Vehicles detected! Pausing for 5 seconds...")
        pause_end_time = cv2.getTickCount() + (5 * cv2.getTickFrequency())

        while cv2.getTickCount() < pause_end_time:
            # Continuously update the display during the pause
            cv2.imshow(window_name, grid)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Wait for a short duration to maintain the desired FPS
    if not vehicles_detected:
        cv2.waitKey(int(1000 / desired_fps))  # Wait time to achieve ~3 FPS

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all video captures
for cap in caps:
    cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()
# Exit the PyQt application
sys.exit(app.exec_())
