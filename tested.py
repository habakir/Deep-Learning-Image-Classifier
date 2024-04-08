import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import torchvision.models as models
import random

model_path = "best_resnet34_model.pth"
state_dict = torch.load(model_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

directories = ["do", "re", "mi", "fa", "sol", "la", "ti"]

model = models.resnet34(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 7)
model.load_state_dict(state_dict)
model.eval()

ensemble_size = 5
class_labels = ["do", "re", "mi", "fa", "sol", "la", "ti"]


for directory in directories:
    files = os.listdir(directory)

    image_files = [file for file in files if file.lower().endswith(('.jpeg'))]

    if not image_files:
        print(f"No image files found in directory: {directory}")
        continue


    image_name = random.choice(image_files)
    image_path = os.path.join(directory, image_name)
    test_image = Image.open(image_path)
    test_image_tensor = transform(test_image).unsqueeze(0)
    
    predictions = []
    
    for _ in range(ensemble_size):
        random_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        augmented_image_tensor = random_transform(test_image).unsqueeze(0)
        
  
        with torch.no_grad():
            output = model(augmented_image_tensor)
        
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_label = class_labels[predicted_class]
        predictions.append(predicted_label)
    
    predicted_label = max(set(predictions), key=predictions.count)
    
    print("Actual label:", directory)
    print("Predicted label:", predicted_label)
    print("Image path:", image_path)
