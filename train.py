from ultralytics import YOLO
import cv2

model = YOLO('yolov8x.pt')

image_path = 'kitchen.png'
image = cv2.imread(image_path)

results = model(image)

annotated_image = results[0].plot()

output_path = 'train.jpg'
cv2.imwrite(output_path, annotated_image)

cv2.imshow('Detected', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()