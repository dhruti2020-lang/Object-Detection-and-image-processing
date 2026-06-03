from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    results = model(frame)

    person_count = 0

    # Process detections
    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls = int(box.cls[0])

            # Class 0 = Person in COCO dataset
            if cls == 0:
                person_count += 1

    # Draw detections
    annotated_frame = results[0].plot()

    # Show count
    cv2.putText(
        annotated_frame,
        f"People Count: {person_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("People Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()