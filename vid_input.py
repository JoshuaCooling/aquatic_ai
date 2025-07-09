import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Define model architecture
model = models.densenet121(weights=None)
num_ftrs = model.classifier.in_features
model.classifier = torch.nn.Linear(num_ftrs, 12)

# Load trained model
model.load_state_dict(torch.load("invasive_species_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Class labels
class_labels = ["bladderwort", "canadian_waterweed", "carolina_fanwort", "coontail", 
                "curly_leaf_pondweed", "eurasian_watermilfoil", "european_frogbit",
                "hydrilla", "parrotfeather", "richards_pondweed", "siberian_watermilfoil",
                "starry_stonewort"]

# Function to classify a single frame
def classify_frame(frame, threshold=0.50):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    max_prob, predicted_class = torch.max(probabilities, 0)

    if max_prob.item() < threshold:
        return "Other", max_prob.item()

    return class_labels[predicted_class], max_prob.item()

# Open video file or camera
video_path = r"C:\Users\coolj\OneDrive\Documents\Winter25\Capstone\Video\Little Glen Lake Eurasian Watermilfoil Site LGL 39.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_skip = 100
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip == 0: 
        predicted_class, confidence = classify_frame(frame)
        
        print(f"Frame {frame_count}: Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

    cv2.imshow("Video Classification", frame)

    key = cv2.waitKey(1)

    if cv2.getWindowProperty("Video Classification", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
