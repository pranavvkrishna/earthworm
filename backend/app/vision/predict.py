# Disease Detection Inference
# Loads the trained ResNet18 model and runs predictions on crop images.

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "earthworm_disease_model.pth")

# same normalization used during training
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

_model = None
_class_names = None


def load_model():
    # load the model once and cache it (avoids reloading on every request)
    global _model, _class_names

    if _model is not None:
        return _model, _class_names

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _model = model
    _class_names = class_names

    return model, class_names


def predict_disease(image_path: str, top_k: int = 3):
    # Run inference on a single image, return top_k predictions with confidence
    model, class_names = load_model()

    image = Image.open(image_path).convert("RGB")
    input_tensor = inference_transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_probs, top_indices = torch.topk(probabilities, top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "disease": class_names[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    return results


def format_prediction(results):
    # format prediction results into a readable string for the agent
    top = results[0]
    disease_name = top["disease"].replace("_", " ").replace("__", " - ")
    confidence = top["confidence"]

    output = f"Predicted: {disease_name} ({confidence}% confidence)\n"
    if len(results) > 1:
        output += "Other possibilities: "
        others = [f"{r['disease'].replace('_', ' ')} ({r['confidence']}%)" for r in results[1:]]
        output += ", ".join(others)

    return output


if __name__ == "__main__":
    # quick manual test — replace with a real image path to test
    test_image = "test_leaf.jpg"

    if os.path.exists(test_image):
        results = predict_disease(test_image)
        print(format_prediction(results))
    else:
        print(f"No test image found at {test_image}.")
        print("Put a sample leaf image there to test, or call predict_disease() with your own path.")