# Traffic Management System

This project demonstrates a real-time Traffic Management System that detects vehicles from multiple video feeds and dynamically manages traffic lights to optimize traffic flow. The system utilizes the YOLO object detection model to count vehicles in each lane and adjusts traffic light durations accordingly based on vehicle counts in each feed.

## Features

- **Real-time Vehicle Detection:** Uses YOLO model to detect and count vehicles (e.g., cars, trucks, buses, motorbikes) from four video feeds.
- **Traffic Light Management:** Determines the optimal green light duration for each lane based on vehicle count, ensuring lanes with higher traffic receive more green light time.
- **PyQt5 Interface:** Displays vehicle counts, current traffic light status, and time remaining for each lane.
- **User-Friendly Display:** Shows all video feeds in a grid format with detected vehicle count overlay.

## Technologies Used

- **YOLO (Ultralytics)** - For vehicle detection.
- **OpenCV** - For video processing.
- **PyQt5** - For the graphical user interface.
- **NumPy** - For image and data manipulation.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have the following dependencies installed:
   - OpenCV
   - PyQt5
   - Ultralytics YOLO
   - NumPy

3. **Download the YOLO Model:**
   Place your YOLO model (`yolo11n.pt`) in the `res/models/` directory.

4. **Prepare Video Files:**
   Place video files (`video2.webm`, `video5.webm`, `video3.webm`, `video4.webm`) in the `res/videos/` directory.

## Usage

1. **Run the Application:**
   ```bash
   python main.py
   ```

2. **Interface Overview:**
   - **Traffic Count Window:** Displays each feed's vehicle count, traffic light status, and time remaining for the current green light.
   - **Video Feed Window:** Shows all video feeds in a grid with vehicle count overlays.

3. **Traffic Cycle Management:**
   - The system dynamically allocates green light times based on vehicle counts.
   - To exit, press 'q' in the video feed window.

## Code Overview

- **YOLO Model:** Detects vehicles and counts them in each feed.
- **TrafficManager Class:** Manages traffic light cycles, setting green light duration based on detected vehicle count.
- **PyQt5 Interface:** Displays real-time updates of vehicle counts, traffic light status, and time remaining for each feed.

## Customization

To add more vehicle types or change parameters like `MIN_GREEN_TIME` or `MAX_GREEN_TIME`, modify the relevant variables in the `manage_traffic` function.

## License

This project is licensed under the MIT License.