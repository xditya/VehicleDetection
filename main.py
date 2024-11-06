import sys
import cv2
from ultralytics import YOLO
import numpy as np
from PyQt5 import QtWidgets, QtCore


def manage_traffic(vehicle_counts):
    if len(vehicle_counts) != 4:
        raise ValueError("Vehicle counts must be provided for all 4 lanes")

    MIN_GREEN_TIME = 10
    MAX_GREEN_TIME = 60

    time_per_vehicle = 2

    green_times = []
    for count in vehicle_counts:
        green_time = max(MIN_GREEN_TIME, min(count * time_per_vehicle, MAX_GREEN_TIME))
        green_times.append(green_time)

    priority_lane = vehicle_counts.index(max(vehicle_counts))

    schedule = sorted(enumerate(vehicle_counts), key=lambda x: x[1], reverse=True)

    return priority_lane, green_times, schedule


def update_traffic_ui(traffic_light_labels, time_remaining_labels, lane, color, time):
    traffic_light_labels[lane].setText(color)
    traffic_light_labels[lane].setStyleSheet(
        f"background-color: {color.lower()}; color: white; padding: 10px;"
    )
    time_remaining_labels[lane].setText(f"Time: {time}s")


def main():
    model = YOLO("res/models/yolo11n.pt")

    vehicle_classes = ["car", "truck", "bus", "motorbike"]

    video_paths = ["video2.webm", "video5.webm", "video3.webm", "video4.webm"]
    caps = [cv2.VideoCapture(f"res/videos/{path}") for path in video_paths]

    app = QtWidgets.QApplication(sys.argv)

    count_window = QtWidgets.QWidget()
    count_window.setWindowTitle("Traffic Management System")
    count_window.setGeometry(100, 100, 600, 300)
    count_layout = QtWidgets.QGridLayout()

    vehicle_count_labels = []
    traffic_light_labels = []
    time_remaining_labels = []

    for i in range(4):
        label = QtWidgets.QLabel(f"Feed {i + 1}: 0 vehicles")
        label.setStyleSheet("background-color: red; color: white; padding: 10px;")
        vehicle_count_labels.append(label)
        count_layout.addWidget(label, i, 0)

        light_label = QtWidgets.QLabel("Red")
        light_label.setStyleSheet("background-color: red; color: white; padding: 10px;")
        traffic_light_labels.append(light_label)
        count_layout.addWidget(light_label, i, 1)

        time_label = QtWidgets.QLabel("Time: 0s")
        time_label.setStyleSheet(
            "background-color: white; color: black; padding: 10px;"
        )
        time_remaining_labels.append(time_label)
        count_layout.addWidget(time_label, i, 2)

    count_window.setLayout(count_layout)
    count_window.show()

    window_name = "4 Video Feeds with Vehicle Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)

    desired_fps = 3
    frame_interval = int(30 / desired_fps)

    frame_counts = [0, 0, 0, 0]

    class TrafficManager(QtCore.QObject):
        update_ui = QtCore.pyqtSignal(int, str, int)

        def __init__(self):
            super().__init__()
            self.current_lane = 0
            self.time_remaining = 0
            self.schedule = []
            self.green_times = []

        def start_traffic_cycle(self, vehicle_counts):
            _, self.green_times, self.schedule = manage_traffic(vehicle_counts)
            self.next_lane()

        def next_lane(self):
            if not self.schedule:
                return

            self.current_lane, _ = self.schedule.pop(0)
            self.time_remaining = self.green_times[self.current_lane]
            self.update_ui.emit(self.current_lane, "Green", self.time_remaining)

            for i in range(4):
                if i != self.current_lane:
                    self.update_ui.emit(i, "Red", self.time_remaining)

            QtCore.QTimer.singleShot(1000, self.update_time)

        def update_time(self):
            self.time_remaining -= 1
            self.update_ui.emit(self.current_lane, "Green", self.time_remaining)

            if self.time_remaining > 0:
                QtCore.QTimer.singleShot(1000, self.update_time)
            else:
                self.next_lane()

    traffic_manager = TrafficManager()
    traffic_manager.update_ui.connect(
        lambda lane, color, time: update_traffic_ui(
            traffic_light_labels, time_remaining_labels, lane, color, time
        )
    )

    counting_vehicles = True
    vehicle_counts = [0, 0, 0, 0]

    while all(cap.isOpened() for cap in caps):
        frames = []

        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                break

            if counting_vehicles and frame_counts[i] % frame_interval == 0:
                results = model(frame)
                detections = results[0]
                vehicle_detections = [
                    box
                    for box in detections.boxes
                    if detections.names[int(box.cls)] in vehicle_classes
                ]
                num_vehicles = len(vehicle_detections)
                vehicle_counts[i] = num_vehicles

                annotated_frame = detections.plot()
                cv2.putText(
                    annotated_frame,
                    f"Vehicles Detected: {num_vehicles}",
                    (10, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                annotated_frame = frame

            frames.append(annotated_frame)
            frame_counts[i] += 1

        if len(frames) < 4:
            break

        for i in range(4):
            vehicle_count_labels[i].setText(
                f"Feed {i + 1}: {vehicle_counts[i]} vehicles"
            )

        height, width = frames[0].shape[:2]
        new_height, new_width = height // 2, width // 2
        resized_frames = [
            cv2.resize(frame, (new_width, new_height)) for frame in frames
        ]

        top_row = np.hstack((resized_frames[0], resized_frames[1]))
        bottom_row = np.hstack((resized_frames[2], resized_frames[3]))
        grid = np.vstack((top_row, bottom_row))

        cv2.imshow(window_name, grid)

        if counting_vehicles and all(count > 0 for count in vehicle_counts):
            counting_vehicles = False
            traffic_manager.start_traffic_cycle(vehicle_counts)

        if cv2.waitKey(int(1000 / desired_fps)) & 0xFF == ord("q"):
            break

        app.processEvents()

    for cap in caps:
        cap.release()

    cv2.destroyAllWindows()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
