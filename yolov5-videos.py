import cv2
import torch


def people():
# Load pre-trained person detection model from Torch

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # Open video using OpenCV
    cap = cv2.VideoCapture("444.mp4")

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Exit the loop if the video is over
        if not ret:
            break

        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform person detection on the frame using Torch
        results = model(frame, size=640)

        # Draw bounding boxes around detected persons
        for box in results.xyxy[0]:
            if box[-1] == 0:  # Class index of person detection
                print(box)
             #   x1, y1, x2, y2, class_idx = map(int, box)
                x1, y1, x2, y2 = map(int, box[:-2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('frame', frame)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and destroy any windows created
    cap.release()
    cv2.destroyAllWindows()
