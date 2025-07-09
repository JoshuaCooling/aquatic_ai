import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

# Define class names
class_names = [
    "bladderwort", "canadian_waterweed", "carolina_fanwort", "coontail",
    "curly_leaf_pondweed", "eurasian_watermilfoil", "european_frogbit",
    "hydrilla", "parrotfeather", "richards_pondweed", "siberian_watermilfoil",
    "starry_stonewort"
]

# Load model
def load_model(model_path):
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Preprocessing function to match training conditions
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Predict function
def predict_image(model, image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
    return class_names[predicted_idx], probabilities[predicted_idx].item()

# Classify a single image
def classify_single_image(model_path, image_path):
    model = load_model(model_path)
    predicted_class, confidence = predict_image(model, image_path)
    print(f"Predicted class: {predicted_class} (Confidence: {confidence:.2f})")

# Batch classify all images in a folder
def classify_images_in_folder(model_path, folder_path):
    model = load_model(model_path)
    for img in os.listdir(folder_path):
        if img.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, img)
            predicted_class, confidence = predict_image(model, img_path)
            print(f"{img}: {predicted_class} (Confidence: {confidence:.2f})")


model_path = "invasive_species_model.pth"
test_image = r"C:\Users\coolj\OneDrive\Documents\Winter25\Capstone\total\parrotfeather_images\image_3.JPG"
test_folder = r"C:\Users\coolj\OneDrive\Documents\Winter25\Capstone\Video\frames"

# Run single image classification
classify_single_image(model_path, test_image)

# Run batch classification
classify_images_in_folder(model_path, test_folder)
