import torch
import cv2
import requests
from torchvision import transforms
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(best.pt)  # Use "cuda" if you have a GPU

# Set the model to evaluation mode
model.eval()

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((416, 416)),  # Adjust to your model's input size
    transforms.ToTensor(),
])

# Load an image from a URL
url = "https://example.com/fire_image.jpg"
response = requests.get(url)
img = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)

# Preprocess the image
img = transform(img).unsqueeze(0)  # Add batch dimension

# Make predictions
with torch.no_grad():
    predictions = model(img)

# Postprocess predictions (apply NMS and confidence threshold)
predictions = non_max_suppression(predictions, conf_thres=0.5, iou_thres=0.5)

# Visualize or save the results
if predictions[0] is not None:
    # Draw bounding boxes on the image
    for pred in predictions[0]:
        x1, y1, x2, y2, conf, cls = pred
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f'Class: {int(cls)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Fire Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No fire detected.")
