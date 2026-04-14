from ultralytics import YOLO
import cv2
import winsound   # sound ke liye

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

count = 0  # violation count

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detected_objects = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detected_objects.append(label)

    # 🚨 ALERT CONDITION
    if "person" in detected_objects and "cell phone" in detected_objects:
        
        # 🔴 Text on screen
        cv2.putText(frame, "PHONE USAGE DETECTED!", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # 🔊 STEP 7: Sound Alert
        winsound.Beep(1000, 500)

        # 📸 STEP 8: Screenshot
        
        cv2.imwrite(f"alerts/alert_{count}.jpg", frame)

        # 📊 Count
        count += 1
        print("Violations:", count)

    annotated = results[0].plot()
    cv2.imshow("AI Detector", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()