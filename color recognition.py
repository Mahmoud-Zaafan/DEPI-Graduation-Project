import cv2
from ultralytics import YOLO

def recognize_color(hue, sat, val):
    if val < 50:
        return "BLACK"
    elif sat < 50 and val > 200:
        return "WHITE"
    elif sat < 50:
        return "GREY"
    
    if 0 <= hue <= 5 or 170 <= hue <= 180:
        return "RED"
    elif 6 <= hue <= 15:
        return "ORANGE"
    elif 16 <= hue <= 35:
        return "YELLOW"
    elif 36 <= hue <= 85:
        return "GREEN"
    elif 86 <= hue <= 125:
        return "BLUE"
    elif 126 <= hue <= 145:
        return "VIOLET"
    elif 146 <= hue <= 169:
        return "PINK"
    elif 10 <= hue <= 20:
        return "BROWN"
    elif 0 <= hue <= 180 and 0 <= sat <= 30 and 160 <= val <= 255:
        return "SILVER"
    elif 81 <= hue <= 95 and 50 <= sat <= 255 and 50 <= val <= 255:
        return "TURQUOISE"
    else:
        return "Undefined"

def get_object_color(frame, model, conf_threshold=0.5):
    results = model(frame, verbose=False, conf=conf_threshold)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    object_colors = []

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pixel_center = hsv_frame[cy, cx]
        hue, sat, val = pixel_center
        color = recognize_color(hue, sat, val)
        object_colors.append((result.cls, color, (x1, y1, x2, y2)))
        
    return object_colors

# Example usage
model = YOLO('model/yolov8n.pt')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    success, frame = cap.read()
    if not success:
        break

    object_colors = get_object_color(frame, model)

    for obj_class, color, bbox in object_colors:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{color}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('YOLOv8 Object Detection and Color Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()